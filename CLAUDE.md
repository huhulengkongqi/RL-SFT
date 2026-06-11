# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository builds an Agentic RL SFT data synthesis pipeline. The main flow is:

```text
raw data → seed prompts → evolved tasks → reference/test enrichment → AgentLoop trajectories → SFT JSON
```

The project is not a model-training repo; it prepares high-quality task and trajectory data for agent cold-start training.

## Common commands

### Setup

```bash
uv sync
uv sync --extra sandbox   # Docker sandbox test/dev dependencies
uv sync --extra dataset   # HuggingFace datasets/pyarrow export dependencies
cp .env.example .env      # if local credentials/config are needed
```

Important env vars used by scripts:

- `ANTHROPIC_AUTH_TOKEN`: Volcano/Claude-compatible API key used by trajectory generation, evolution, judge calls, and some tests.
- `VOLCANO_CLAUDE_BASE_URL`: defaults to `https://ark.cn-beijing.volces.com/api/coding/v3` in scripts that use the native Volcano Claude client.
- `VLLM_BASE_URL`: default local vLLM/OpenAI-compatible endpoint, usually `http://localhost:8000/v1`; some judge scripts default to Volcano `coding/v3` if unset.
- `VLLM_MODEL`: default local/judge model name; `.env.example` uses `Qwen/Qwen2.5-7B-Instruct-AWQ` for local vLLM, while several scripts default to `doubao-seed-2.0-lite`.

### Tests

```bash
uv run pytest
uv run pytest tests/validation/ -v                         # offline validation tests
uv run pytest tests/validation/test_seed_pool.py -v         # one test file
uv run pytest tests/validation/test_seed_pool.py::TestSeedPromptPool::test_weighted_sampling -v
uv run pytest tests/test_agent_loop.py -v                  # mocked Environment, no Docker required
uv run pytest tests/test_trajectory_sample.py -v           # best-of-N utilities, no real API required
uv run pytest tests/test_quality_filter.py -v              # quality funnel unit coverage
uv run pytest tests/test_data_formatter.py -v              # SFT formatter; export tests are skipped without optional deps
uv run pytest tests/infra/environment/test_environment.py -v # requires Docker sandbox
```

`pytest.ini` sets `testpaths = tests`, `pythonpath = src`, verbose short tracebacks, and `asyncio_mode = auto`.

### Lint, format, type check

```bash
uv run ruff format src/ tests/ scripts/
uv run ruff check src/ tests/ scripts/
uv run mypy src/
```

Ruff uses `ruff.toml`: 120-character line length, double quotes, isort first-party packages `agent_sft` and `infra`, and lint groups `E/W/F/I/N/UP/B/SIM/TCH` with `E501` and `B008` ignored.

### vLLM / sandbox utilities

```bash
./scripts/start_vllm_docker.sh
./scripts/verify_vllm.sh
./scripts/stop_vllm_docker.sh
uv run python scripts/start_local_vllm.py
```

Docker is required for the real environment sandbox tests and demos. GPU/vLLM workflows additionally require NVIDIA Docker support and a compatible CUDA driver.

### Quality filter utilities

```bash
# Run trajectory quality filter pipeline over *_raw.json trajectory files
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/run_quality_filter.py \
  --input-dir data/sft_trajectories \
  --raw-glob "*_raw.json" \
  --output-dir data/quality_filter \
  --level2-judge-sleep-min 8 --level2-judge-sleep-max 12

# Analyze a timestamped filter report and generate summary statistics
uv run python scripts/analyze_quality_filter_report.py data/quality_filter/quality_filter_report_YYYYMMDD_HHMMSS.json
```

## Pipeline scripts

```bash
# Generate/assess seed and evolved prompt data
uv run python scripts/generate_seed_prompts.py
uv run python scripts/run_evolution.py --use-mock
uv run python scripts/run_evolution.py --seed-file data/final_seed_pool_181_real.json --generations 4 --evolutions-per-seed 3 --output-dir data/evolved
uv run python scripts/assess_evolution.py data/evolved/final_evolved.json
uv run python scripts/quality_assessment.py --input data/seed_prompts.json --stats-only

# Use Volcano/Claude-compatible endpoint for evolution
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/run_evolution.py --use-claude --claude-model ark-code-latest --min-sleep 12 --max-sleep 18

# Repair math references with an LLM
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/fix_math_references_with_llm.py --input data/claude_evolved_4gen/final_evolved_v1.0_complete.json --sleep-min 10 --sleep-max 18

# Generate trajectories in the real Environment
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/generate_single_trajectory_real_env.py --domain code_debug --max-steps 20 --sleep-min 10 --sleep-max 18
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/generate_all_trajectories_real_env.py --limit 3
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/generate_all_trajectories_real_env.py --domain code_debug --limit 5
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/generate_all_trajectories_real_env.py --resume data/sft_trajectories/batch_progress_YYYYMMDD_HHMMSS.jsonl

# Best-of-N trajectory sampling and benchmarking
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/trajectory_sample.py --domain code_debug --limit 1 --n 4
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/trajectory_sample.py --benchmark --benchmark-tasks 100 --n 16 --task-concurrency 1

# Trajectory quality filter funnel
ANTHROPIC_AUTH_TOKEN=your_api_key uv run python scripts/run_quality_filter.py \
  --input-dir data/sft_trajectories \
  --raw-glob "*_raw.json" \
  --output-dir data/quality_filter \
  --level2-judge-sleep-min 8 --level2-judge-sleep-max 12
uv run python scripts/analyze_quality_filter_report.py data/quality_filter/quality_filter_report_YYYYMMDD_HHMMSS.json

# SFT data formatting (convert raw trajectories to training-ready chat format)
uv run python scripts/sft_formatter_full_test.py                              # Test with synthetic data
uv run python scripts/sft_formatter_full_test.py --data-dir data/sft_trajectories --limit 50 \
  --strategy middle --max-tokens 3277 --output-dir data/formatted_sft --export-format both
uv run python scripts/sft_formatter_full_test.py --quality-report data/quality_filter/quality_filter_report_YYYYMMDD_HHMMSS.json \
  --output-dir data/formatted_sft --export-format both
uv run python scripts/sft_formatter_full_test.py --format function_json       # Use JSON tool call format instead of ReAct
```

`run_evolution.py --use-mock` is the fastest smoke test because it exercises pipeline logic without real LLM calls. Scripts in `scripts/archive/` are historical data-prep utilities rather than current entry points.

### Pipeline script reference

| Script | Purpose |
|---|---|
| `run_evolution.py` | Multi-generation Evol-Instruct task evolution |
| `assess_evolution.py` | Evolution result statistics and quality metrics |
| `generate_seed_prompts.py` | Base seed pool generation across 4 domains |
| `fix_math_references_with_llm.py` | Batch-correct math reference solutions |
| `generate_single_trajectory_real_env.py` | Single trajectory generation in real Environment |
| `generate_all_trajectories_real_env.py` | Batch trajectory generation with resume support |
| `quality_assessment.py` | Seed/evolved data quality scoring |
| `trajectory_sample.py` | Best-of-N concurrent trajectory sampling |
| `run_quality_filter.py` | Trajectory quality filter funnel pipeline |
| `analyze_quality_filter_report.py` | Filter report analysis and summary stats |

## Architecture

### `src/agent_sft`: data synthesis pipeline

- `task_generator/` defines seed/task Pydantic models, seed-pool sampling/versioning, LLM-based task generation, AST function-call parsing, and task validation.
- `evol_instruct/` implements the multi-generation Evol-Instruct pipeline. `evolver.py` applies evolution strategies; `pipeline.py` orchestrates evolve → deduplicate → quality filter → stats.
- `quality_filter/` contains the **trajectory quality filter funnel**: Level 1 load/validation, Level 2 PRM-style scoring with offline Monte Carlo and optional LLM judge, Level 3 MinHash/optional embedding deduplication plus diversity metrics, Level 4 difficulty-aware sampling, AgentHER relabeling for useful failed trajectories, and comprehensive report generation.
- `trajectory_sampler/agent_loop.py` contains the teacher-agent harness: immutable `AgentState`, termination detection, trajectory recording, ReAct/function-JSON formatting, and layered Observation→Thought→Action generation.
- `trajectory_sampler/trajectory_sample.py` implements best-of-N concurrent sampling, trajectory ranking, failure summaries, and sandbox failure detection.
- `dataset_builder/` contains the SFT training data formatter with 4-role chat template support, token counting, trajectory truncation strategies, loss mask generation for scratchpad masking, and HuggingFace-compatible Parquet/JSONL export.

#### Four task domains and sources

| Domain | Source examples | Target task type |
|---|---|---|
| `code_debug` | StackOverflow, real Python bugs, SDK examples | Debugging, fixing, explaining bugs |
| `math_reasoning` | Math reasoning datasets | Step-by-step reasoning, numeric answers |
| `api_orchestration` | OpenAPI, SDK, FastAPI examples | API call sequencing, error handling, auth flows |
| `multi_step_planning` | Ansible, CI/CD workflows, deployment samples | Multi-phase planning, dependencies, risk control |

### `src/infra`: execution and model clients

- `vllm_client/` provides an OpenAI-compatible client used for local vLLM and the Volcano `coding/v3` endpoint in some scripts.
- `anthropic_client/` contains a native-protocol Volcano Claude client.
- `local_transformers/` supports direct HuggingFace inference for local development.
- `sandbox/` provides Docker-isolated code execution and execution result models.
- `environment/` implements the Gym-style agent environment: `Environment.step(action)` executes tool calls or verifies final answers, `SandboxPool` manages reusable Docker containers, and `AnswerVerifier` handles code, math, format, and judge-backed verification.

### Core runtime flow

```text
SeedPromptPool
  → TaskGenerator / EvolutionPipeline
  → quality filters and discriminator
  → enriched task JSON with test/reference data
  → AgentLoop
  → Environment.step(ToolCallAction | FinalAnswerAction)
  → SandboxPool / AnswerVerifier
  → TrajectoryRecorder
  → raw trajectory JSON + SFT JSON
  → QualityFilterFunnel (load/validate → score → deduplicate/diversity → difficulty sampling → optional AgentHER)
  → final curated SFT dataset
```

LLM clients are duck-typed around `chat()`, `achat()`, and `achat_stream()` where available. Most pipeline code accepts an injected client rather than constructing one internally. Trajectory generation scripts wrap clients with explicit randomized sleep before each API/Judge call.

### Environment requirements

- Python 3.11+
- `uv` package manager
- Docker (required for Environment / sandbox tests)
- Optional: NVIDIA GPU + vLLM Docker for local inference

## Verification modes

`AnswerVerifier` and `Environment` use different validation paths by task domain/data shape:

- `code_debug`: tries executable code validation against test cases; for debugging-report style expected outputs, combines format checks, LLM-as-judge, and evidence from prior successful `exec` tool calls.
- `math_reasoning`: extracts final numeric/math answers and uses symbolic or numeric equivalence with tolerance.
- `api_orchestration` and `multi_step_planning`: primarily use format validation plus optional LLM-as-judge for completeness and correctness.

## Data paths

- `data/raw/`: raw collected sources such as StackOverflow XML, parquet math data, OpenAPI/SDK examples, Ansible examples, and workflow examples.
- `data/seed_prompts.json`: 200 seed prompts documented in `data/README.md`.
- `data/final_seed_pool_181_real.json`: real curated seed-pool example used by newer pipeline commands.
- `data/evolved/`: Evol-Instruct outputs.
- `data/claude_evolved_4gen/final_evolved_v1.0_complete.json`: complete 4-generation task dataset with reference/test enrichment.
- `data/reference_checks/`: math reference audit/fix outputs and checkpoints.
- `data/sft_trajectories/`: raw and SFT-format trajectories, batch progress JSONL, best-of-N summaries, and benchmark reports generated by AgentLoop scripts.
- `data/quality_filter/`: timestamped `quality_filter_report_*.json` reports and `filtered_sft_trajectories_*.json` exports from the quality filter funnel.
- `data/formatted_sft/`: training-ready SFT data in standard chat template format (JSONL/Parquet) with loss masks. The formatter can read either raw trajectories from `--data-dir` or kept raw paths from `--quality-report`.

Evolved prompt files may not have exactly the same structure as seed prompts: they can include `evolution_metadata`, omit or inherit `source`, and have `validator_code = null`.

## Testing boundaries

- `tests/validation/` is offline and should be the first target for quick checks; sandbox-related validation tests skip when Docker is unavailable.
- `tests/test_agent_loop.py`, `tests/test_trajectory_sample.py`, `tests/test_quality_filter.py`, and most of `tests/test_data_formatter.py` use mocks/synthetic data and do not require real API calls.
- `tests/infra/environment/test_environment.py` and sandbox validation paths require Docker.
- `tests/test_vllm_client.py` mostly unit-tests client/server config, but live vLLM checks need a running endpoint.
- `tests/test_volcano_claude*.py`, `tests/test_unified_client.py`, and trajectory-generation scripts require API credentials for live calls.

## Development notes

- Python source lives under `src/`; scripts explicitly add `src` to `sys.path` when run directly.
- Data models use Pydantic `BaseModel`; config-style objects commonly use dataclasses; string enums subclass `str, Enum`.
- Keep API-calling scripts rate-limited. Existing scripts use serialized or low-concurrency calls with random sleeps such as 10–18s or 12–18s.
- Prefer `uv run ...` for commands so the project environment is used consistently.
