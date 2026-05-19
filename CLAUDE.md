# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agentic RL SFT Data Synthesis Pipeline — generates SFT training data for agentic RL cold-start via:
1. **Task Generation** - Seed-based task generation with function call validation
2. **Evol-Instruct Pipeline** - Multi-generational prompt evolution with 7 strategies
3. **Quality Filtering** - Embedding deduplication + LLM-based quality discrimination
4. **Sandbox Execution** - Docker-isolated code execution with test case validation

Core modules (`task_generator`, `evol_instruct`, `quality_filter`, `sandbox`) are fully implemented.

## Prerequisites

- Python 3.11+
- `uv` package manager
- NVIDIA driver ≥ 550.x (CUDA 12.4+) for GPU inference
- Docker + NVIDIA Container Toolkit for vLLM Docker deployment

## Commands

### Installation & Setup
```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN and API keys
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_task_generator.py -v
uv run pytest tests/test_vllm_client.py -v

# pytest configuration:
# - testpaths = tests
# - pythonpath = src
# - addopts = -v --tb=short
# - asyncio_mode = auto
```

### Linting & Formatting
```bash
# Code formatting (ruff)
uv run ruff format src/ tests/

# Linting (ruff)
uv run ruff check src/ tests/

# Type checking (mypy)
uv run mypy src/
```

### vLLM Deployment (Docker)

#### Fastest Mode (Recommended)
```bash
# Pre-pull image (~30 seconds)
./scripts/pull_vllm_image.sh

# Start vLLM Docker service
./scripts/start_vllm_docker.sh

# Run 6-point validation suite
./scripts/verify_vllm.sh

# Stop service
./scripts/stop_vllm_docker.sh
```

#### Direct Docker Compose
```bash
# Pull mode (fastest)
docker compose --profile pull up -d

# With custom env
HF_TOKEN=<token> docker compose --profile pull up
VLLM_PORT=8001 docker compose --profile pull up

# Build modes
docker compose --profile build up --build  # ~2min
docker compose --profile full up --build   # ~10min

# View logs
docker compose logs -f vllm

# Stop
docker compose down
```

#### Local Development vLLM
```bash
# Start local vLLM with small model for testing
uv run python scripts/start_local_vllm.py
```

### Evol-Instruct Pipeline
```bash
# Run full evolution pipeline (4 generations, 3 evolutions per seed)
uv run python scripts/run_evolution.py

# Custom evolution parameters
uv run python scripts/run_evolution.py --generations 8 --evolutions-per-seed 5

# Test with mock LLM (no real LLM required)
uv run python scripts/run_evolution.py --use-mock

# Output: data/evolved/final_evolved.json + statistics
```

### Demo & Utility Scripts
```bash
# Demo task generation
uv run python scripts/demo_task_generator.py

# Demo task validation with sandbox execution
uv run python scripts/demo_task_validation.py

# Regenerate 200 seed prompts
uv run python scripts/generate_seed_prompts.py

# Quality assessment for evolved data / seed pools
uv run python scripts/assess_evolution.py data/evolved/generation_1.json  # For evolved generation data
uv run python scripts/quality_assessment.py --input data/seed_prompts.json --stats-only  # For seed pool data
uv run python scripts/quality_assessment.py --input data/seed_prompts.json --output data/filtered_seeds.json --min-quality 0.70

# Enable LLM quality discriminator during evolution (filters out low-quality prompts)
uv run python scripts/run_evolution.py --discriminator-min-score 0.6  # Filter out prompts with score < 0.6
uv run python scripts/run_evolution.py --disable-discriminator  # Skip quality filtering for speed (original behavior)
```

### Manual Verification Commands
```bash
# Health check
curl http://localhost:8000/health

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages":[{"role":"user","content":"Hello!"}]
  }'

# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Architecture

### Core Component Structure
```
src/
├── agent_sft/
│   ├── task_generator/      # IMPLEMENTED — seed pool + LLM-based task generation
│   │   ├── models.py        # Pydantic data models
│   │   ├── seed_pool.py     # Weighted sampling, versioning, JSON I/O
│   │   ├── generator.py     # Async batch generation with retries
│   │   ├── validator.py     # Task validation pipeline
│   │   └── function_parser.py  # AST-based function validation
│   ├── evol_instruct/       # IMPLEMENTED — multi-gen evolution pipeline
│   │   ├── evolver.py       # 7 evolution strategies (deepening, widening, etc.)
│   │   ├── models.py        # Evolution data models
│   │   └── pipeline.py      # End-to-end: evolve → dedup → filter → stats
│   ├── quality_filter/      # IMPLEMENTED — quality & diversity filtering
│   │   ├── embedding_deduplicator.py  # Sentence-BERT similarity deduplication
│   │   ├── lm_discriminator.py        # LLM-based quality judging
│   │   └── metrics.py       # Self-BLEU, diversity statistics
│   ├── function_registry.py # Available function library
│   ├── trajectory_sampler/  # STUB
│   └── dataset_builder/     # STUB
└── infra/
    ├── vllm_client/         # VLLMClient + Docker lifecycle (OpenAI-compatible)
    ├── anthropic_client/    # Volcano Engine Claude endpoint (native protocol)
    ├── local_transformers/  # Direct HuggingFace inference (dev/CPU)
    └── sandbox/             # IMPLEMENTED — Docker-isolated code execution
        ├── execution_manager.py  # Container lifecycle + test execution
        └── models.py        # Execution result data models
```

### LLM Client Interface
All three clients (`VLLMClient`, `AnthropicClient`, `LocalLLMClient`) share a duck-typed interface: `chat()`, `achat()`, `achat_stream()`. `TaskGenerator` accepts any of them via dependency injection — it only calls `achat()`.

### LLM Client Selection Guide
| Client | Use Case | Pros | Cons |
|--------|----------|------|------|
| **VLLMClient** | Local GPU inference with vLLM Docker | Fast, no rate limits, OpenAI-compatible | Requires GPU ≥ 16GB VRAM |
| **AnthropicClient** | Volcano Claude API (China region) | High quality, thinking mode support | Rate limited, requires API token |
| **LocalLLMClient** | Direct HuggingFace transformers | CPU/GPU compatible, no server needed | Slower, memory intensive |

**AnthropicClient Special Features:**
- `create_volcano_claude_client()` factory function for quick initialization
- `sleep_before_request` / `sleep_after_request` parameters for rate limiting
- `achat_stream()` async streaming support
- Thinking mode extraction (handles `<thinking>` blocks in responses)

### Task Generation Flow
```
SeedPromptPool
      ↓ (weighted sampling by quality score)
SeedPrompt[]
      ↓
TaskGenerator.generate_batch(mode="seed_based")
      ↓
┌─ Async Task Generation with Semaphore (5 concurrent) ─┐
│   • LLM mutation via client.achat()                   │
│   • Tenacity retry (3 attempts, exponential backoff)  │
└────────────────────────────────────────────────────────┘
      ↓
TaskValidator.validate_batch()
      ↓
┌─ Validation Pipeline ─────────────────────────────────┐
│   1. FunctionSignatureParser (AST validation)         │
│   2. SandboxExecutor (code execution + test cases)    │
└────────────────────────────────────────────────────────┘
      ↓
Task[] with ValidationReport
```

Three generation modes: `seed_only` (pass-through), `seed_based` (optional LLM mutation), `full_generation`.

### Evol-Instruct Pipeline Flow
```
Seed Prompts (N)
      ↓
┌─ Evolution Generation ────────────────────────────────────┐
│  7 strategies: Deepen (constraints/reasoning/concretize), │
│  Complex Input, CoT, Breadth, In-context Learning         │
│  3 variants per seed → N×3 prompts                        │
└────────────────────────────────────────────────────────────┘
      ↓
┌─ Embedding Deduplication ─────────────────────────────────┐
│  Sentence-BERT embeddings + cosine similarity (0.85 thresh)│
└────────────────────────────────────────────────────────────┘
      ↓
┌─ LLM Quality Discrimination ──────────────────────────────┐
│  Judge: originality, clarity, complexity, value-add       │
│  Minimum score: 0.5                                        │
└────────────────────────────────────────────────────────────┘
      ↓
┌─ Diversity Metrics ───────────────────────────────────────┐
│  Self-BLEU score, strategy distribution, filter rates     │
└────────────────────────────────────────────────────────────┘
      ↓
Evolved Prompts (M) → seeds for next generation (4 gens total)
```

Output: `data/evolved/final_evolved.json` + per-generation statistics.

### Key Architectural Patterns
- **Dependency Injection**: `TaskGenerator` accepts `llm_client`, `validators`, `task_validator` as parameters
- **Duck Typing**: All LLM clients share the same interface - no abstract base class needed
- **Async Semaphores**: Rate limiting for concurrent LLM calls
- **Retry Pattern**: Tenacity with exponential backoff
- **Strategy Pattern**: Domain-specific validators

### Function Registry System
The `FunctionRegistry` provides dynamic function resolution and validation:
- **Categories**: `BUILTIN` (Python builtins), `STANDARD_LIBRARY` (stdlib modules), `CUSTOM` (user-defined)
- Used by `FunctionSignatureParser` to AST-validate function calls in generated tasks
- Supports category-based filtering during task generation

### Sandbox Execution
The Docker-based sandbox provides isolated code execution:
- **SandboxConfig**: `memory_limit`, `disk_limit`, `network_access`, `image`, `timeout`
- **Lifecycle**: Async context manager creates/destroys containers automatically
- **Validation**: Runs test cases against generated code and returns `TestCaseExecution` results

### Rate Limiting Patterns
For Claude API evolution runs:
- Wrap `client.achat` with random sleep (12-18s by default) before each request
- Set `max_concurrent_requests=1` for serialized execution
- Use `discriminator_min_score` filtering to reduce unnecessary API calls

### Data Structure
Seed prompts are stored in `data/seed_prompts.json` with 200 high-quality prompts across 4 domains:
- `code_debug` (50) - Code debugging tasks
- `api_orchestration` (50) - API orchestration tasks  
- `math_reasoning` (50) - Math reasoning tasks
- `multi_step_planning` (50) - Multi-step planning tasks

Each seed includes test cases, validator code, difficulty (Easy/Medium/Hard), and quality score.

**Evolved Data Format Note:**
Evolved prompts (`data/evolved/generation_*.json`) have a slightly different structure:
- `evolution_metadata` - Contains generation number, parent_id, strategy, evolution prompt
- `source` field may be missing (inherited from parent)
- `validator_code` may be `null` (inherited from parent)
- Test cases are preserved but may not match the evolved prompt

Use `quality_assessment.py` with evolved data for pipeline statistics, but note it expects full seed structure.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_AUTH_TOKEN` | Volcano Engine Claude API auth key | - |
| `VOLCANO_CLAUDE_BASE_URL` | Volcano Claude API endpoint | `https://ark.cn-beijing.volces.com/api/coding/v3` |
| `HF_TOKEN` | HuggingFace token for private models | - |
| `VLLM_BASE_URL` | vLLM server endpoint | `http://localhost:8000/v1` |
| `VLLM_MODEL` | vLLM model name | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `VLLM_PORT` | vLLM server port | `8000` |
| `VLLM_GPU_MEMORY` | GPU memory utilization | `0.85` |
| `VLLM_MODE` | Docker mode: `pull`/`build`/`full` | `pull` |
| `VLLM_IMAGE` | Docker image name | `ghcr.io/huhulengkongqi/rl-sft-vllm:v0.6.3-rlsft.1` |
| `VLLM_QUANTIZATION` | Quantization mode | `awq` |
| `VLLM_USE_DOCKER` | Use Docker for vLLM | `true` |
| `OPENAI_API_KEY` | For external OpenAI API calls | - |
| `API_TIMEOUT_MS` | API call timeout in milliseconds | - |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Disable non-essential traffic flag | - |
| `ANTHROPIC_API_KEY` | For Anthropic API calls (not used by volcano client) | - |
| `E2B_API_KEY` | For E2B sandbox (optional) | - |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_CONCURRENT_REQUESTS` | Max concurrent LLM requests | `5` |

## Key Files Reference

| File | Purpose |
|------|---------|
| **Task Generation** | |
| `src/agent_sft/task_generator/generator.py` | Main task generation logic |
| `src/agent_sft/task_generator/validator.py` | Task validation pipeline |
| `src/agent_sft/task_generator/function_parser.py` | AST-based function call validation |
| **Evol-Instruct** | |
| `src/agent_sft/evol_instruct/pipeline.py` | End-to-end evolution pipeline |
| `src/agent_sft/evol_instruct/evolver.py` | 7 evolution strategies engine |
| **Quality Filtering** | |
| `src/agent_sft/quality_filter/embedding_deduplicator.py` | Sentence-BERT deduplication |
| `src/agent_sft/quality_filter/lm_discriminator.py` | LLM quality judge |
| `scripts/quality_assessment.py` | Standalone seed quality assessment pipeline |
| **Infrastructure** | |
| `src/agent_sft/function_registry.py` | Available function library |
| `src/infra/anthropic_client/client.py` | Volcano Claude native API client |
| `src/infra/vllm_client/client.py` | vLLM OpenAI-compatible client |
| `src/infra/sandbox/execution_manager.py` | Docker sandbox code execution |

## Development Rules (Project-Specific)

### Code Style
- **Line length**: 120 characters
- **Quote style**: Double quotes (`"`) preferred
- **isort**: First-party packages `["agent_sft", "infra"]`
- **Module docstrings**: All files have docstrings at the top
- **Type hints**: Used throughout; mypy is used for type checking
- **Logging**: Use `logger = logging.getLogger(__name__)` pattern

### Data Models
- Use Pydantic `BaseModel` for data models
- Use `@dataclass` for config objects
- Enums subclass `str, Enum` for type-safe string enums

### Linting Rules (Ruff)
Enabled categories: `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `SIM`, `TCH`
Specific ignores: `E501` (line too long), `B008` (allows mutable defaults)

## Testing Structure
- **`tests/validation/`** - Unit tests for core modules, can run offline
- **`tests/test_*.py`** - Integration tests, may require Docker/GPU/API tokens
- **`MockLLMClient`** - Built into `run_evolution.py` for offline testing without real LLM calls

Use `--use-mock` flag with `run_evolution.py` to validate pipeline logic without API calls.

## Reproducibility Guarantees
1. **SHA256 Digest Locked Base Images** - All Docker base images pinned by digest
2. **Hash-Locked Python Dependencies** - `requirements-vllm.txt` includes full SHA256 hashes
3. **GHCR Provenance Attestation** - GitHub Actions auto-publish with build attestation
4. **Deterministic Builds** - All pip installs use `--require-hashes`

## Troubleshooting
- **NVIDIA Driver Mismatch**: Ensure driver version ≥ 550.x for CUDA 12.4
- **Docker GPU Passthrough**: Run `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` to verify
- **vLLM Startup Failures**: Check `docker compose logs vllm` for OOM or model download issues
- **Claude API Rate Limiting**: Increase `--min-sleep` and `--max-sleep` values, reduce concurrency to 1
- **HF Mirror in China**: Set `HF_ENDPOINT=https://hf-mirror.com` for faster model downloads
