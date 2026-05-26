"""
Generate one trajectory with REAL Environment (Docker sandbox) and Volcano API.

This script uses:
  - Random real task from specified domain
  - Real Volcano LLM API via VLLMClient
  - Real infra.environment.Environment with SandboxPool/Docker
  - Rate-limit sleep before every LLM call
  - Full raw + SFT trajectory recording with timestamp

Usage:
  set ANTHROPIC_AUTH_TOKEN=your_api_key
  uv run python scripts/generate_single_trajectory_real_env.py --domain math_reasoning

Domains:
  math_reasoning, code_debug, api_orchestration, multi_step_planning
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
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.trajectory_sampler import AgentLoop, Trajectory
from infra.environment.environment import Environment
from infra.vllm_client.client import VLLMClient


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


def load_random_task(domain: str) -> dict:
    data_path = Path("data/reference_checks/final_evolved_v1.0_complete_math_fixed_20260522_154556.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    candidates = [t for t in all_tasks if t.get("domain") == domain]
    if not candidates:
        raise ValueError(f"No tasks found for domain={domain}")

    task = random.choice(candidates)
    print(f"Random task #{candidates.index(task)} / {len(candidates)}")
    print(f"  ID: {task.get('id')}")
    print(f"  Domain: {task.get('domain')}")
    print(f"  Difficulty: {task.get('difficulty')}")
    print(f"  Test cases: {len(task.get('test_cases', []))}")
    print(f"  Prompt: {task.get('prompt', '')[:180]}...")
    return task


def save_outputs(trajectory: Trajectory, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id_short = trajectory.task_id[:8]

    raw_path = output_dir / f"realenv_{task_id_short}_{timestamp}_raw.json"
    sft_path = output_dir / f"realenv_{task_id_short}_{timestamp}_sft.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.model_dump(), f, indent=2, ensure_ascii=False, default=str)

    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.to_sft_format(), f, indent=2, ensure_ascii=False, default=str)

    print(f"  Raw saved: {raw_path}")
    print(f"  SFT saved: {sft_path}")


def _print_verification_checklist(metadata: dict, content: object) -> None:
    mode = metadata.get("verification_mode")
    if not mode:
        return
    print(f"    verification_mode: {mode}")
    print(f"    verification_score: {metadata.get('verification_score')}")
    answer_for_verification = metadata.get("answer_for_verification")
    if answer_for_verification is not None:
        print(f"    answer_for_verification: {str(answer_for_verification)[:160]}")

    details = content if isinstance(content, dict) else {}
    checks = details.get("checks")
    if checks:
        print("    format checks:")
        for name, passed in checks.items():
            print(f"      {'PASS' if passed else 'FAIL'} {name}")

    code_execution = details.get("code_execution")
    if code_execution:
        print("    code_execution:")
        print(f"      tests_passed: {code_execution.get('tests_passed')}/{code_execution.get('tests_total')}")
        print(f"      execution_status: {code_execution.get('execution_status')}")
        print(f"      error: {str(code_execution.get('error'))[:160]}")

    evidence = details.get("successful_exec_evidence")
    if evidence:
        print("    successful_exec_evidence:")
        print(f"      matched: {evidence.get('matched')}")
        print(f"      step: {evidence.get('step')}")
        print(f"      functions: {evidence.get('executed_functions')}")

    judge = details.get("llm_judge")
    if judge:
        judge_payload = judge.get("llm_judge", judge)
        print("    llm_judge:")
        print(f"      passed: {judge_payload.get('passed')}")
        print(f"      score: {judge_payload.get('score')}")
        print(f"      reason: {str(judge_payload.get('reason'))[:200]}")


def print_summary(trajectory: Trajectory) -> None:
    print("\n" + "=" * 70)
    print("TRAJECTORY SUMMARY")
    print("=" * 70)
    print(f"  Task ID:      {trajectory.task_id}")
    print(f"  Domain:       {trajectory.domain}")
    print(f"  Steps:        {len(trajectory.steps)}")
    print(f"  Termination:  {trajectory.termination_reason}")
    print(f"  Success:      {trajectory.success}")
    print(f"  Final score:  {trajectory.final_score}")
    print()
    for i, step in enumerate(trajectory.steps, 1):
        action = step.action
        observation = step.observation
        action_name = getattr(action, "name", "final_answer")
        print(f"  Step {i}: {action.action_type} / {action_name}")
        print(f"    obs.success: {observation.success}")
        if observation.error:
            print(f"    obs.error:   {observation.error[:160]}")
        else:
            print(f"    obs.content: {str(observation.content)[:160]}")
        _print_verification_checklist(observation.metadata or {}, observation.content)


def print_conversation_preview(trajectory: Trajectory) -> None:
    sft = trajectory.to_sft_format()
    print("\n" + "=" * 70)
    print("SFT CONVERSATION PREVIEW")
    print("=" * 70)
    for i, message in enumerate(sft["messages"], 1):
        content = message["content"][:120].replace("\n", " ")
        print(f"  [{i:02d}] {message['role'].upper():9}: {content}...")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one trajectory with real Environment")
    parser.add_argument("--domain", default="math_reasoning", choices=[
        "math_reasoning", "code_debug", "api_orchestration", "multi_step_planning"
    ])
    parser.add_argument("--model", default="doubao-seed-2.0-lite")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--sleep-min", type=float, default=3.0)
    parser.add_argument("--sleep-max", type=float, default=8.0)
    parser.add_argument("--output-dir", default="data/sft_trajectories")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("ERROR: ANTHROPIC_AUTH_TOKEN is not set.")
        print("Run: set ANTHROPIC_AUTH_TOKEN=your_api_key")
        sys.exit(1)

    print("=" * 70)
    print("GENERATE SINGLE TRAJECTORY WITH REAL ENVIRONMENT")
    print("=" * 70)
    print(f"API key: {api_key[:12]}...{api_key[-4:]}")
    print(f"Domain: {args.domain}")
    print(f"Model: {args.model}")
    print(f"Max steps: {args.max_steps}")
    print(f"Sleep: {args.sleep_min}s - {args.sleep_max}s")
    print()

    task = load_random_task(args.domain)

    base_llm = VLLMClient(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        api_key=api_key,
        timeout=120,
        model=args.model,
    )
    llm = RateLimitedLLM(base_llm, args.sleep_min, args.sleep_max)

    started = time.time()

    async with Environment(max_steps=args.max_steps, judge_client=llm, judge_model=args.model) as env:
        loop = AgentLoop(
            env=env,
            llm_client=llm,
            max_steps=args.max_steps,
            token_budget=20000,
        )
        trajectory = await loop.run(task)

    elapsed = time.time() - started
    print_summary(trajectory)
    print(f"\nElapsed: {elapsed:.1f}s")
    save_outputs(trajectory, Path(args.output_dir))
    print_conversation_preview(trajectory)


if __name__ == "__main__":
    asyncio.run(main())
