#!/bin/bash
#
# One-click SFT pipeline runner.
#
# Usage:
#   ./scripts/run_pipeline.sh --test                # Quick test with synthetic data
#   ./scripts/run_pipeline.sh --full                # Full pipeline
#   ./scripts/run_pipeline.sh --config my.yaml      # Custom config
#   ./scripts/run_pipeline.sh --limit 100           # Limit to 100 tasks
#   ./scripts/run_pipeline.sh --flywheel            # Run data flywheel
#
# Environment variables:
#   ANTHROPIC_AUTH_TOKEN - Required for LLM calls
#   VOLCANO_CLAUDE_BASE_URL - Optional endpoint override

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

MODE="default"
CONFIG_FILE=""
LIMIT=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            MODE="test"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --flywheel)
            EXTRA_ARGS="$EXTRA_ARGS --flywheel"
            shift
            ;;
        --verbose|-v)
            EXTRA_ARGS="$EXTRA_ARGS --verbose"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "========================================"
log_info "Agent SFT Pipeline Orchestrator"
log_info "========================================"

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
    log_warn "ANTHROPIC_AUTH_TOKEN is not set"
    log_warn "LLM calls will fail without a valid token"
    if [[ "$MODE" != "test" ]]; then
        log_error "Set ANTHROPIC_AUTH_TOKEN before running in full mode"
        exit 1
    fi
fi

if [[ ! -d ".venv" ]]; then
    log_info "Creating virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true

log_info "Syncing dependencies..."
uv sync

case "$MODE" in
    test)
        log_info "Running in TEST mode (synthetic data)"
        CONFIG_FILE="${CONFIG_FILE:-config/pipeline_test.yaml}"
        LIMIT="0"
        ;;
    full)
        log_info "Running in FULL mode"
        CONFIG_FILE="${CONFIG_FILE:-config/pipeline_default.yaml}"
        ;;
    default)
        log_info "Running with custom configuration"
        CONFIG_FILE="${CONFIG_FILE:-config/pipeline_default.yaml}"
        ;;
esac

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

log_info "Using config: $CONFIG_FILE"

CMD="python scripts/run_orchestrator.py --config $CONFIG_FILE"
if [[ -n "$LIMIT" ]]; then
    CMD="$CMD --limit $LIMIT"
fi
CMD="$CMD $EXTRA_ARGS"

log_info "Command: $CMD"
log_info "----------------------------------------"

eval $CMD
EXIT_CODE=$?

log_info "----------------------------------------"
if [[ $EXIT_CODE -eq 0 ]]; then
    log_info "Pipeline completed successfully!"
else
    log_error "Pipeline failed with exit code $EXIT_CODE"
fi
log_info "========================================"

exit $EXIT_CODE
