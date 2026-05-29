"""
Generate trajectories for all evolved tasks with real Environment and Volcano API.

Defaults match the production run requested:
  - input: data/final_evolved_v1.0_complete_math_fixed_20260522_154556.json
  - model: doubao-seed-2.0-lite
  - max steps: 20
  - sleep: 10-18 seconds before every LLM/Judge call
  - API key: read from ANTHROPIC_AUTH_TOKEN only

Usage:
  set ANTHROPIC_AUTH_TOKEN=your_api_key
  uv run python scripts/generate_all_trajectories_real_env.py

Useful testing options:
  uv run python scripts/generate_all_trajectories_real_env.py --limit 3
  uv run python scripts/generate_all_trajectories_real_env.py --domain code_debug --limit 5
  uv run python scripts/generate_all_trajectories_real_env.py --resume data/sft_trajectories/batch_progress_xxx.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.trajectory_sampler import AgentLoop, LayeredTemperatureConfig, Trajectory
from infra.environment.environment import Environment
from infra.vllm_client.client import VLLMClient


DEFAULT_INPUT = Path("data/final_evolved_v1.0_complete_math_fixed_20260522_154556.json")
DEFAULT_OUTPUT_DIR = Path("data/sft_trajectories")


class RateLimitedLLM:
    """VLLMClient wrapper with random sleep before each API call."""

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


def load_tasks(path: Path, domain: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts") or data.get("tasks") or data.get("data") or []
    if domain:
        data = [task for task in data if task.get("domain") == domain]
    if limit is not None:
        data = data[:limit]
    return data


def load_completed(progress_path: Optional[Path]) -> set[str]:
    completed: set[str] = set()
    if not progress_path or not progress_path.exists():
        return completed
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("status") == "completed":
                    completed.add(item["task_id"])
            except json.JSONDecodeError:
                pass
    return completed


def save_outputs(trajectory: Trajectory, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id_short = trajectory.task_id[:8]

    raw_path = output_dir / f"realenv_{task_id_short}_{timestamp}_raw.json"
    sft_path = output_dir / f"realenv_{task_id_short}_{timestamp}_sft.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.model_dump(), f, indent=2, ensure_ascii=False, default=str)

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.to_sft_format(), f, indent=2, ensure_ascii=False, default=str)

    return raw_path, sft_path


def final_verification_summary(trajectory: Trajectory) -> Dict[str, Any]:
    if not trajectory.steps:
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


async def run_one(
    task: Dict[str, Any],
    llm: RateLimitedLLM,
    output_dir: Path,
    max_steps: int,
    model: str,
    thought_temperature: float,
    action_temperature: float,
    single_call: bool,
) -> Dict[str, Any]:
    task_id = task.get("id", "unknown")
    print("=" * 90)
    print(f"Task {task_id} | domain={task.get('domain')} | difficulty={task.get('difficulty')}")
    print(f"Prompt: {task.get('prompt', '')[:200]}...")

    started = time.time()
    async with Environment(max_steps=max_steps, judge_client=llm, judge_model=model) as env:
        loop = AgentLoop(
            env=env,
            llm_client=llm,
            max_steps=max_steps,
            token_budget=20000,
            temperature_config=LayeredTemperatureConfig(
                enabled=not single_call,
                thought_temperature=thought_temperature,
                action_temperature=action_temperature,
            ),
        )
        trajectory = await loop.run(task)

    elapsed = time.time() - started
    raw_path, sft_path = save_outputs(trajectory, output_dir)
    verification = final_verification_summary(trajectory)

    result = {
        "task_id": task_id,
        "domain": task.get("domain"),
        "difficulty": task.get("difficulty"),
        "status": "completed",
        "success": trajectory.success,
        "termination_reason": trajectory.termination_reason,
        "steps": len(trajectory.steps),
        "elapsed_seconds": elapsed,
        "raw_path": str(raw_path),
        "sft_path": str(sft_path),
        "verification": verification,
    }

    print(
        f"Done task={task_id} success={trajectory.success} "
        f"steps={len(trajectory.steps)} elapsed={elapsed:.1f}s raw={raw_path}"
    )
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trajectories for all evolved tasks with real Environment")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="doubao-seed-2.0-lite")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--sleep-min", type=float, default=10.0)
    parser.add_argument("--sleep-max", type=float, default=18.0)
    parser.add_argument("--thought-temperature", type=float, default=0.8)
    parser.add_argument("--action-temperature", type=float, default=0.2)
    parser.add_argument("--single-call", action="store_true", help="Disable Observation->Thought->Action two-call loop")
    parser.add_argument("--domain", default=None, choices=[None, "math_reasoning", "code_debug", "api_orchestration", "multi_step_planning"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Existing progress JSONL path to resume")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("ERROR: ANTHROPIC_AUTH_TOKEN is not set.")
        print("Run: set ANTHROPIC_AUTH_TOKEN=your_api_key")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = Path(args.resume) if args.resume else output_dir / f"batch_progress_{timestamp}.jsonl"
    summary_path = output_dir / f"batch_summary_{timestamp}.json"

    tasks = load_tasks(input_path, args.domain, args.limit)
    completed = load_completed(progress_path)

    print("=" * 90)
    print("BATCH TRAJECTORY GENERATION")
    print("=" * 90)
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Progress: {progress_path}")
    print(f"Model: {args.model}")
    print(f"Max steps: {args.max_steps}")
    print(f"Sleep: {args.sleep_min}-{args.sleep_max}s before every API call")
    print(f"Observation->Thought->Action: {not args.single_call}")
    print(f"Thought T: {args.thought_temperature}")
    print(f"Action T: {args.action_temperature}")
    print(f"Tasks selected: {len(tasks)}")
    print(f"Already completed: {len(completed)}")

    base_llm = VLLMClient(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        api_key=api_key,
        timeout=120,
        model=args.model,
    )
    llm = RateLimitedLLM(base_llm, args.sleep_min, args.sleep_max)

    results: List[Dict[str, Any]] = []
    with open(progress_path, "a", encoding="utf-8") as progress_file:
        for index, task in enumerate(tasks, 1):
            task_id = task.get("id", "unknown")
            if task_id in completed:
                print(f"[{index}/{len(tasks)}] Skip completed task {task_id}")
                continue

            print(f"[{index}/{len(tasks)}] Running task {task_id}")
            try:
                result = await run_one(
                    task,
                    llm,
                    output_dir,
                    args.max_steps,
                    args.model,
                    args.thought_temperature,
                    args.action_temperature,
                    args.single_call,
                )
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "domain": task.get("domain"),
                    "difficulty": task.get("difficulty"),
                    "status": "failed",
                    "success": False,
                    "error": repr(exc),
                }
                print(f"FAILED task={task_id}: {exc!r}")

            results.append(result)
            progress_file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            progress_file.flush()

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "progress_path": str(progress_path),
        "model": args.model,
        "max_steps": args.max_steps,
        "sleep_min": args.sleep_min,
        "sleep_max": args.sleep_max,
        "thought_temperature": args.thought_temperature,
        "action_temperature": args.action_temperature,
        "observation_thought_action": not args.single_call,
        "domain": args.domain,
        "limit": args.limit,
        "new_results": len(results),
        "completed": sum(1 for r in results if r.get("status") == "completed"),
        "successful_trajectories": sum(1 for r in results if r.get("success") is True),
        "failed_tasks": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 90)
    print(f"Batch finished. Summary: {summary_path}")
    print(f"Progress: {progress_path}")
    print(f"Successful trajectories: {summary['successful_trajectories']}/{summary['new_results']}")


if __name__ == "__main__":
    asyncio.run(main())
