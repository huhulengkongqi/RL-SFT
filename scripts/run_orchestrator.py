#!/usr/bin/env python3
"""
Run the SFT pipeline orchestrator.

Usage:
    ANTHROPIC_AUTH_TOKEN=xxx python scripts/run_orchestrator.py --config config/pipeline_default.yaml
    ANTHROPIC_AUTH_TOKEN=xxx python scripts/run_orchestrator.py --config config/pipeline_test.yaml --limit 5
    python scripts/run_orchestrator.py --help
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.orchestrator import PipelineOrchestrator, load_config, DataFlywheel


def generate_test_seeds(limit: int) -> list:
    """Generate test seed tasks for quick testing."""
    domains = ["code_debug", "math_reasoning", "api_orchestration", "multi_step_planning"]
    seeds = []

    for i in range(limit):
        domain = domains[i % len(domains)]
        seeds.append({
            "task_id": f"test_seed_{i:04d}",
            "id": f"test_seed_{i:04d}",
            "domain": domain,
            "difficulty": "medium",
            "prompt": f"Test prompt {i} for {domain} domain. This is a synthetic test prompt for pipeline validation.",
            "test_cases": [],
            "source": "synthetic_test",
            "quality_score": 0.8,
            "tags": ["test", "synthetic"],
        })

    return seeds


async def main():
    parser = argparse.ArgumentParser(description="Run SFT pipeline orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_default.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        help="Override seed file path from config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of seed tasks (0 = generate synthetic tests)",
    )
    parser.add_argument(
        "--flywheel",
        action="store_true",
        help="Run in data flywheel mode (requires config.flywheel.enabled=true)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run ID (for resuming)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path), override_run_id=args.run_id)

    if args.seed_file:
        config.seed_generation.seed_file = args.seed_file

    if args.limit is not None and args.limit == 0:
        print("Generating synthetic test seeds...")
        orchestrator = PipelineOrchestrator(config)
        test_seeds = generate_test_seeds(10)
        await orchestrator.seed_tasks(test_seeds)
    else:
        seed_file = Path(config.seed_generation.seed_file)
        if not seed_file.exists():
            print(f"Seed file not found: {seed_file}")
            print("Run with --limit 0 to use synthetic test seeds")
            return 1

        orchestrator = PipelineOrchestrator(config)
        seed_count = await orchestrator.load_seed_file(str(seed_file), limit=args.limit)

        if args.limit:
            print(f"Loaded {seed_count} seed tasks (limit={args.limit})")

    print(f"Starting pipeline: {config.name} v{config.version}")
    print(f"Run ID: {config.run_id}")
    print(f"Output directory: {orchestrator.output_dir}")
    print("-" * 60)

    if args.flywheel:
        config.flywheel.enabled = True
        flywheel = DataFlywheel(config)
        success = await flywheel.run()
    else:
        success = await orchestrator.run()

    if success:
        print("\nPipeline completed successfully!")
        return 0
    else:
        print("\nPipeline failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
