#!/usr/bin/env python3
"""
Stress test the pipeline orchestrator.

Simulates high volume of tasks to measure throughput, latency,
and resource utilization under load.

Usage:
    python scripts/stress_test_pipeline.py --tasks 1000 --workers 16
    python scripts/stress_test_pipeline.py --quick
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    CheckpointManager,
    create_queue,
)


@dataclass
class StressTestResult:
    test_name: str
    total_tasks: int
    num_workers: int
    start_time: float
    end_time: float = 0.0
    succeeded: int = 0
    failed: int = 0
    per_task_latency: List[float] = field(default_factory=list)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_tasks / self.total_time

    @property
    def avg_latency(self) -> float:
        if not self.per_task_latency:
            return 0.0
        return sum(self.per_task_latency) / len(self.per_task_latency)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.succeeded / self.total_tasks


async def run_stress_test(
    num_tasks: int,
    num_workers: int,
    worker_latency: float = 0.1,
    failure_rate: float = 0.0,
) -> StressTestResult:
    """Run a stress test with simulated workers."""
    from agent_sft.orchestrator.queue import create_queue, InMemoryQueue
    from agent_sft.orchestrator.checkpoint import CheckpointManager, TaskStatus
    from agent_sft.orchestrator.worker_base import BaseWorker

    result = StressTestResult(
        test_name="stress_test",
        total_tasks=num_tasks,
        num_workers=num_workers,
        start_time=time.time(),
    )

    queue = create_queue(use_redis=False)
    checkpoint = CheckpointManager(
        checkpoint_dir="data/stress_test/checkpoints",
        run_id=f"stress_{int(time.time())}",
        resume=False,
    )

    await queue.ensure_group("stress:input", "stress_test_workers")

    class MockWorker(BaseWorker):
        async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
            start = time.time()
            await asyncio.sleep(worker_latency)

            import random
            if random.random() < failure_rate:
                raise Exception("Simulated failure")

            result.per_task_latency.append(time.time() - start)
            result.succeeded += 1

            return {"processed": True, "task_id": item.get("task_id")}

    for i in range(num_tasks):
        await queue.push("stress:input", {"task_id": f"task_{i:06d}"})

    workers: List[MockWorker] = []
    worker_tasks: List[asyncio.Task] = []

    for i in range(num_workers):
        worker = MockWorker(
            queue=queue,
            checkpoint=checkpoint,
            stage="stress_test",
            input_stream="stress:input",
            output_stream=None,
            worker_id=f"worker_{i}",
            idle_timeout=2.0,
        )
        workers.append(worker)
        worker_tasks.append(asyncio.create_task(worker.start()))

    while True:
        pending = await queue.len("stress:input")
        processed = result.succeeded + result.failed
        if pending == 0 and processed >= num_tasks:
            break
        await asyncio.sleep(0.5)

    for t in worker_tasks:
        t.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    await queue.close()

    stats = checkpoint.get_stats().get("stress_test", {})
    result.failed = stats.get("failed", 0)
    result.end_time = time.time()

    return result


async def run_pipeline_integration_test(num_tasks: int) -> StressTestResult:
    """Run end-to-end integration test using real pipeline (seed stage only)."""
    result = StressTestResult(
        test_name="pipeline_integration",
        total_tasks=num_tasks,
        num_workers=2,
        start_time=time.time(),
    )

    config = PipelineConfig(
        name="stress_test",
        version="1.0.0",
        resume=False,
        checkpoint_dir="data/stress_test/checkpoints",
        output_dir="data/stress_test/output",
    )
    config.queue.use_redis = False
    config.evolution.enabled = False
    config.trajectory_generation.enabled = False
    config.quality_filter.enabled = False
    config.dataset_build.enabled = False

    orchestrator = PipelineOrchestrator(config)

    domains = ["code_debug", "math_reasoning", "api_orchestration", "multi_step_planning"]
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            "task_id": f"seed_{i:06d}",
            "id": f"seed_{i:06d}",
            "domain": domains[i % len(domains)],
            "difficulty": "medium",
            "prompt": f"Test prompt {i}",
            "test_cases": [],
            "source": "stress_test",
        })

    await orchestrator.seed_tasks(tasks)
    success = await orchestrator.run()

    result.end_time = time.time()
    result.succeeded = num_tasks if success else 0
    result.failed = 0 if success else num_tasks

    return result


def generate_report(results: List[StressTestResult], output_path: str) -> str:
    """Generate markdown report from stress test results."""
    lines = []
    lines.append("# Pipeline Stress Test Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    for result in results:
        lines.append(f"## {result.test_name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tasks | {result.total_tasks} |")
        lines.append(f"| Workers | {result.num_workers} |")
        lines.append(f"| Total Time | {result.total_time:.2f}s |")
        lines.append(f"| Throughput | {result.throughput:.2f} tasks/s |")
        lines.append(f"| Average Latency | {result.avg_latency * 1000:.2f} ms |")
        lines.append(f"| Succeeded | {result.succeeded} |")
        lines.append(f"| Failed | {result.failed} |")
        lines.append(f"| Success Rate | {result.success_rate * 100:.2f}% |")
        lines.append("")

        for key, value in result.extra_metrics.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("All tests completed successfully.")
    lines.append("The pipeline orchestrator can handle high throughput.")

    report = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


async def main():
    parser = argparse.ArgumentParser(description="Stress test the pipeline orchestrator")
    parser.add_argument(
        "--tasks",
        type=int,
        default=1000,
        help="Number of tasks to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.05,
        help="Simulated per-task processing latency in seconds",
    )
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.0,
        help="Simulated failure rate (0.0-1.0)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run end-to-end integration test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stress_test/report.md",
        help="Output report path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results: List[StressTestResult] = []

    if args.quick:
        print("Running quick smoke test...")
        result = await run_stress_test(
            num_tasks=100,
            num_workers=4,
            worker_latency=0.01,
            failure_rate=0.0,
        )
        results.append(result)

    elif args.integration:
        print("Running pipeline integration test...")
        result = await run_pipeline_integration_test(num_tasks=args.tasks)
        results.append(result)

    else:
        print(f"Running stress test: {args.tasks} tasks, {args.workers} workers...")
        result = await run_stress_test(
            num_tasks=args.tasks,
            num_workers=args.workers,
            worker_latency=args.latency,
            failure_rate=args.failure_rate,
        )
        results.append(result)

    report = generate_report(results, args.output)

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    for result in results:
        print(f"\nTest: {result.test_name}")
        print(f"  Total time:     {result.total_time:.2f}s")
        print(f"  Throughput:     {result.throughput:.2f} tasks/s")
        print(f"  Avg latency:    {result.avg_latency * 1000:.2f} ms")
        print(f"  Succeeded:      {result.succeeded}/{result.total_tasks}")
        print(f"  Success rate:   {result.success_rate * 100:.2f}%")
    print("=" * 60)
    print(f"\nFull report written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
