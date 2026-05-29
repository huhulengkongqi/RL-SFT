"""Best-of-N trajectory sampling with real Environment and Volcano VLLMClient.

Defaults mirror generate_all_trajectories_real_env.py:
  - input: data/final_evolved_v1.0_complete_math_fixed_20260522_154556.json
  - model: doubao-seed-2.0-lite
  - base URL: https://ark.cn-beijing.volces.com/api/coding/v3
  - sleep: 10-18 seconds before every LLM/Judge call
  - API key: read from ANTHROPIC_AUTH_TOKEN only
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.trajectory_sampler import SamplingConfig, sample_trajectories, select_best_trajectory, summarize_sample_results
from agent_sft.trajectory_sampler.trajectory_sample import TrajectorySampleResult
from infra.vllm_client.client import VLLMClient

DEFAULT_INPUT = Path("data/final_evolved_v1.0_complete_math_fixed_20260522_154556.json")
DEFAULT_OUTPUT_DIR = Path("data/sft_trajectories")
DEFAULT_VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"


class RateLimitedLLM:
    def __init__(self, inner: VLLMClient, min_sleep: float, max_sleep: float):
        self.inner = inner
        self.model = inner.model
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        self.call_count = 0

    async def achat(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.call_count += 1
        sleep_time = random.uniform(self.min_sleep, self.max_sleep)
        print(f"  [RateLimit] Sleeping {sleep_time:.1f}s before API call #{self.call_count}")
        await asyncio.sleep(sleep_time)
        response = await self.inner.achat(model=model, messages=messages, **kwargs)
        print(f"  [LLM] Response preview: {response[:160].replace(chr(10), ' ')}...")
        return response


def load_tasks(path: Path, domain: Optional[str], limit: Optional[int], task_id: Optional[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts") or data.get("tasks") or data.get("data") or []
    if domain:
        data = [task for task in data if task.get("domain") == domain]
    if task_id:
        data = [task for task in data if task.get("id") == task_id]
    if limit is not None:
        data = data[:limit]
    return data


def save_trajectory_outputs(
    result: TrajectorySampleResult,
    output_dir: Path,
    task_id: str,
    trajectory_format: str,
) -> Dict[str, Optional[str]]:
    if result.trajectory is None:
        return {"raw_path": None, "sft_path": None}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_short = task_id[:8]
    raw_path = output_dir / f"bestofn_{task_short}_sample{result.sample_id}_{timestamp}_raw.json"
    sft_path = output_dir / f"bestofn_{task_short}_sample{result.sample_id}_{timestamp}_sft.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result.trajectory.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(result.trajectory.to_sft_format(trajectory_format), f, ensure_ascii=False, indent=2, default=str)
    return {"raw_path": str(raw_path), "sft_path": str(sft_path)}


def final_verification_summary(result: TrajectorySampleResult) -> Dict[str, Any]:
    trajectory = result.trajectory
    if not trajectory or not trajectory.steps:
        return {}
    last = trajectory.steps[-1]
    obs = last.observation
    details = obs.content if isinstance(obs.content, dict) else {}
    metadata = obs.metadata or {}
    return {
        "final_observation_success": obs.success,
        "verification_mode": metadata.get("verification_mode"),
        "verification_score": metadata.get("verification_score"),
        "format_checks": details.get("checks"),
        "code_execution": details.get("code_execution"),
        "successful_exec_evidence": details.get("successful_exec_evidence"),
        "llm_judge": details.get("llm_judge"),
        "error": obs.error,
    }


class GpuMonitor:
    def __init__(self, interval_seconds: float = 2.0):
        self.interval_seconds = interval_seconds
        self.samples: List[Dict[str, float]] = []
        self.available = shutil.which("nvidia-smi") is not None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "GpuMonitor":
        if self.available:
            self._running = True
            self._task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll(self) -> None:
        while self._running:
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 2:
                        self.samples.append({"gpu_utilization": float(parts[0]), "memory_used_mb": float(parts[1])})
            except Exception:
                self.available = False
                return
            await asyncio.sleep(self.interval_seconds)

    def summary(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": "nvidia-smi not found or failed"}
        if not self.samples:
            return {"available": True, "samples": 0}
        gpu_values = [sample["gpu_utilization"] for sample in self.samples]
        memory_values = [sample["memory_used_mb"] for sample in self.samples]
        return {
            "available": True,
            "samples": len(self.samples),
            "mean_gpu_utilization": sum(gpu_values) / len(gpu_values),
            "max_gpu_utilization": max(gpu_values),
            "mean_memory_used_mb": sum(memory_values) / len(memory_values),
            "max_memory_used_mb": max(memory_values),
            "note": "Local GPU metrics are not model-serving utilization when using a remote API endpoint.",
        }


async def run_task(
    task: Dict[str, Any],
    llm: RateLimitedLLM,
    config: SamplingConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    task_id = task.get("id", "unknown")
    started = time.perf_counter()
    results = await sample_trajectories(task, n=config.n, llm_client=llm, config=config)
    best = select_best_trajectory(results)
    saved: List[Dict[str, Optional[str]]] = []
    best_sample_id: Optional[int] = None
    best_raw_path: Optional[str] = None
    best_sft_path: Optional[str] = None

    for result in results:
        paths = save_trajectory_outputs(result, output_dir, task_id, config.trajectory_format)
        saved.append({"sample_id": result.sample_id, **paths})
        if best is not None and result.trajectory is best:
            best_sample_id = result.sample_id
            best_raw_path = paths["raw_path"]
            best_sft_path = paths["sft_path"]

    stats = summarize_sample_results(results)
    return {
        "task_id": task_id,
        "domain": task.get("domain"),
        "difficulty": task.get("difficulty"),
        "n": config.n,
        "elapsed_seconds": time.perf_counter() - started,
        "best_sample_id": best_sample_id,
        "best_raw_path": best_raw_path,
        "best_sft_path": best_sft_path,
        "best_success": bool(best and best.success),
        "all_paths": saved,
        "verification": next((final_verification_summary(result) for result in results if result.sample_id == best_sample_id), {}),
        **stats,
    }


async def run_tasks_with_concurrency(
    tasks: List[Dict[str, Any]],
    llm: RateLimitedLLM,
    config: SamplingConfig,
    output_dir: Path,
    task_concurrency: int,
    progress_path: Path,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(task_concurrency)
    progress_lock = asyncio.Lock()
    results: List[Dict[str, Any]] = []

    async def run_one(index: int, task: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            print(f"[{index}/{len(tasks)}] Sampling task {task.get('id', 'unknown')} n={config.n}")
            try:
                result = await run_task(task, llm, config, output_dir)
            except Exception as exc:
                result = {
                    "task_id": task.get("id", "unknown"),
                    "domain": task.get("domain"),
                    "difficulty": task.get("difficulty"),
                    "n": config.n,
                    "best_success": False,
                    "error": repr(exc),
                }
            async with progress_lock:
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                results.append(result)
            return result

    await asyncio.gather(*(run_one(index, task) for index, task in enumerate(tasks, 1)))
    return results


def aggregate_benchmark(results: List[Dict[str, Any]], elapsed_seconds: float, gpu_summary: Dict[str, Any]) -> Dict[str, Any]:
    total_samples = sum(result.get("total_samples", result.get("n", 0)) for result in results)
    successful_samples = sum(result.get("successful_samples", 0) for result in results)
    completed_samples = sum(result.get("completed_samples", 0) for result in results)
    failure_reasons: Dict[str, int] = {}
    for result in results:
        for reason, count in result.get("failure_reasons", {}).items():
            failure_reasons[reason] = failure_reasons.get(reason, 0) + count
    sandbox_failures = sum(result.get("sandbox_failures", 0) for result in results)
    return {
        "tasks": len(results),
        "total_samples": total_samples,
        "completed_samples": completed_samples,
        "successful_samples": successful_samples,
        "sample_success_rate": successful_samples / total_samples if total_samples else 0.0,
        "best_of_n_task_success_rate": sum(1 for result in results if result.get("best_success")) / len(results) if results else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "trajectories_per_second": total_samples / elapsed_seconds if elapsed_seconds else 0.0,
        "tasks_per_hour": len(results) * 3600 / elapsed_seconds if elapsed_seconds else 0.0,
        "avg_steps": sum(result.get("avg_steps", 0.0) for result in results) / len(results) if results else 0.0,
        "avg_tokens": sum(result.get("avg_tokens", 0.0) for result in results) / len(results) if results else 0.0,
        "failure_reasons": failure_reasons,
        "sandbox_failures": sandbox_failures,
        "sandbox_failure_rate": sandbox_failures / total_samples if total_samples else 0.0,
        "gpu_utilization": gpu_summary,
    }


def write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    benchmark = summary["benchmark"]
    lines = [
        "# Trajectory Sampling Benchmark Report",
        "",
        f"- Tasks: {benchmark['tasks']}",
        f"- Total samples: {benchmark['total_samples']}",
        f"- Trajectories/sec: {benchmark['trajectories_per_second']:.4f}",
        f"- Tasks/hour: {benchmark['tasks_per_hour']:.2f}",
        f"- Sample success rate: {benchmark['sample_success_rate']:.2%}",
        f"- Best-of-N task success rate: {benchmark['best_of_n_task_success_rate']:.2%}",
        f"- Average steps: {benchmark['avg_steps']:.2f}",
        f"- Average tokens: {benchmark['avg_tokens']:.2f}",
        f"- Sandbox failure rate: {benchmark['sandbox_failure_rate']:.2%}",
        "",
        "## Failure reasons",
        "",
    ]
    for reason, count in benchmark["failure_reasons"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## GPU utilization", "", f"```json\n{json.dumps(benchmark['gpu_utilization'], indent=2)}\n```"])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-N trajectory sampling")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="doubao-seed-2.0-lite")
    parser.add_argument("--base-url", default=DEFAULT_VOLCANO_BASE_URL)
    parser.add_argument("--domain", default=None, choices=[None, "math_reasoning", "code_debug", "api_orchestration", "multi_step_planning"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--trajectory-format", choices=["react", "function_json"], default="react")
    parser.add_argument("--thought-temperature", type=float, default=0.8)
    parser.add_argument("--reasoning-temperature", type=float, default=None, help="Deprecated alias for --thought-temperature")
    parser.add_argument("--action-temperature", type=float, default=0.2)
    parser.add_argument("--single-call", action="store_true", help="Disable Observation->Thought->Action two-call loop")
    parser.add_argument("--sleep-min", type=float, default=10.0)
    parser.add_argument("--sleep-max", type=float, default=18.0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-tasks", type=int, default=100)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("ERROR: ANTHROPIC_AUTH_TOKEN is not set.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = output_dir / f"bestofn_progress_{timestamp}.jsonl"
    summary_path = Path(args.report_json) if args.report_json else output_dir / f"bestofn_summary_{timestamp}.json"
    report_md_path = Path(args.report_md) if args.report_md else output_dir / f"bestofn_report_{timestamp}.md"

    limit = args.benchmark_tasks if args.benchmark and args.limit is None else args.limit
    tasks = load_tasks(Path(args.input), args.domain, limit, args.task_id)
    thought_temperature = args.reasoning_temperature if args.reasoning_temperature is not None else args.thought_temperature
    config = SamplingConfig(
        n=args.n,
        max_steps=args.max_steps,
        trajectory_format=args.trajectory_format,
        thought_temperature=thought_temperature,
        action_temperature=args.action_temperature,
        layered_temperature=not args.single_call,
    )
    base_llm = VLLMClient(base_url=args.base_url, api_key=api_key, timeout=120, model=args.model)
    llm = RateLimitedLLM(base_llm, args.sleep_min, args.sleep_max)

    print("=" * 90)
    print("BEST-OF-N TRAJECTORY SAMPLING")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"API key: ANTHROPIC_AUTH_TOKEN")
    print(f"Sleep: {args.sleep_min}-{args.sleep_max}s before every API call")
    print(f"Tasks: {len(tasks)} | n={args.n} | task_concurrency={args.task_concurrency}")
    print(f"Format: {args.trajectory_format} | Observation->Thought->Action: {config.layered_temperature}")
    print(f"Thought T: {config.thought_temperature}")
    print(f"Action T: {config.action_temperature}")
    print("=" * 90)

    started = time.perf_counter()
    async with GpuMonitor() as gpu_monitor:
        results = await run_tasks_with_concurrency(tasks, llm, config, output_dir, args.task_concurrency, progress_path)
    elapsed = time.perf_counter() - started
    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "progress_path": str(progress_path),
        "model": args.model,
        "base_url": args.base_url,
        "n": args.n,
        "max_steps": args.max_steps,
        "trajectory_format": args.trajectory_format,
        "observation_thought_action": config.layered_temperature,
        "thought_temperature": config.thought_temperature,
        "action_temperature": config.action_temperature,
        "task_concurrency": args.task_concurrency,
        "llm_call_count": llm.call_count,
        "benchmark": aggregate_benchmark(results, elapsed, gpu_monitor.summary()),
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_markdown_report(report_md_path, summary)

    print("=" * 90)
    print(f"Summary: {summary_path}")
    print(f"Report: {report_md_path}")
    print(f"Progress: {progress_path}")
    print(f"Trajectories/sec: {summary['benchmark']['trajectories_per_second']:.4f}")
    print(f"Sample success rate: {summary['benchmark']['sample_success_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
