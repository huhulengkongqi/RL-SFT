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
cp .env.example .env  # if local credentials/config are needed
```

Important env vars used by scripts:

- `ANTHROPIC_AUTH_TOKEN`: Volcano/Claude-compatible API key.
- `VOLCANO_CLAUDE_BASE_URL`: defaults to `https://ark.cn-beijing.volces.com/api/coding/v3` in scripts.
- `VLLM_BASE_URL`: default local vLLM OpenAI-compatible endpoint, usually `http://localhost:8000/v1`.
- `VLLM_MODEL`: default local model, usually `Qwen/Qwen2.5-7B-Instruct-AWQ`.

### Tests

```bash
uv run pytest
uv run pytest tests/validation/ -v                         # offline validation tests
uv run pytest tests/validation/test_seed_pool.py -v         # one test file
uv run pytest tests/validation/test_seed_pool.py::TestSeedPromptPool::test_weighted_sampling -v
uv run pytest tests/infra/environment/test_environment.py -v # requires Docker sandbox
uv run pytest tests/test_agent_loop.py -v
uv run pytest tests/test_trajectory_sample.py -v
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
```

`run_evolution.py --use-mock` is the fastest smoke test because it exercises pipeline logic without real LLM calls. Scripts in `scripts/archive/` are historical data-prep utilities rather than current entry points.

## Architecture

### `src/agent_sft`: data synthesis pipeline

- `task_generator/` defines seed/task Pydantic models, seed-pool sampling/versioning, LLM-based task generation, AST function-call parsing, and task validation.
- `evol_instruct/` implements the multi-generation Evol-Instruct pipeline. `evolver.py` applies evolution strategies; `pipeline.py` orchestrates evolve → deduplicate → quality filter → stats.
- `quality_filter/` contains embedding deduplication, LLM quality discrimination, and diversity metrics.
- `trajectory_sampler/agent_loop.py` contains the teacher-agent harness: immutable `AgentState`, termination detection, trajectory recording, ReAct/function-JSON formatting, and layered Observation→Thought→Action generation.
- `trajectory_sampler/trajectory_sample.py` implements best-of-N concurrent sampling, trajectory ranking, failure summaries, and sandbox failure detection.
- `dataset_builder/` is currently only a package stub.

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
```

LLM clients are duck-typed around `chat()`, `achat()`, and `achat_stream()` where available. Most pipeline code accepts an injected client rather than constructing one internally. Trajectory generation scripts wrap clients with explicit randomized sleep before each API/Judge call.

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

Evolved prompt files may not have exactly the same structure as seed prompts: they can include `evolution_metadata`, omit or inherit `source`, and have `validator_code = null`.

## Testing boundaries

- `tests/validation/` is offline and should be the first target for quick checks.
- `tests/infra/environment/test_environment.py` and sandbox validation paths require Docker.
- `tests/test_trajectory_sample.py` covers best-of-N sampling utilities without requiring real API calls.
- `tests/test_vllm_client.py` requires a vLLM/Docker setup.
- `tests/test_volcano_claude*.py` require API credentials.

## Development notes

- Python source lives under `src/`; scripts explicitly add `src` to `sys.path` when run directly.
- Data models use Pydantic `BaseModel`; config-style objects commonly use dataclasses; string enums subclass `str, Enum`.
- Keep API-calling scripts rate-limited. Existing scripts use serialized or low-concurrency calls with random sleeps such as 10–18s or 12–18s.
- Prefer `uv run ...` for commands so the project environment is used consistently.
