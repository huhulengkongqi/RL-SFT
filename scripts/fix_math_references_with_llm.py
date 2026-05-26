"""
Batch-check and fix all math_reasoning reference answers using Volcano LLM.

This script does NOT modify the original dataset in place. It writes:
  1. A new complete dataset JSON with corrected math references
  2. A detailed audit report JSON
  3. A progress checkpoint JSONL for resume/debugging

Usage:
    set ANTHROPIC_AUTH_TOKEN=your_api_key

    # Dry run first 5 math tasks, output corrected full JSON + report
    uv run python scripts/fix_math_references_with_llm.py --limit 5

    # Full run for all math_reasoning tasks
    uv run python scripts/fix_math_references_with_llm.py --sleep-min 10 --sleep-max 18

    # Resume using existing checkpoint path
    uv run python scripts/fix_math_references_with_llm.py --resume data/reference_checks/math_reference_fix_progress_xxx.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from infra.vllm_client.client import VLLMClient


DEFAULT_INPUT = Path("data/claude_evolved_4gen/final_evolved_v1.0_complete.json")
DEFAULT_OUTPUT_DIR = Path("data/reference_checks")


def extract_reference(task: Dict[str, Any]) -> Any:
    if task.get("final_answer") is not None:
        return task["final_answer"]
    test_cases = task.get("test_cases") or []
    if test_cases:
        expected = test_cases[0].get("expected_output")
        if isinstance(expected, dict):
            if expected.get("final_answer") is not None:
                return expected["final_answer"]
            if expected.get("answer") is not None:
                return expected["answer"]
        return expected
    return None


def normalize_numeric_answer(value: Any) -> Optional[str]:
    """Return one clean numeric string if possible. Integers have no .0; non-integers keep decimals."""
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    num = numbers[-1]
    try:
        as_float = float(num)
        if as_float.is_integer():
            return str(int(as_float))
        return ("%.12f" % as_float).rstrip("0").rstrip(".")
    except ValueError:
        return num


def update_reference(task: Dict[str, Any], corrected_answer: str) -> None:
    """Update math reference in-place, preserving existing structure."""
    corrected_answer = normalize_numeric_answer(corrected_answer) or str(corrected_answer)
    task["final_answer"] = str(corrected_answer)
    test_cases = task.get("test_cases") or []
    if test_cases:
        expected = test_cases[0].setdefault("expected_output", {})
        if isinstance(expected, dict):
            expected["final_answer"] = str(corrected_answer)
            expected["answer"] = str(corrected_answer)
        else:
            test_cases[0]["expected_output"] = {"final_answer": str(corrected_answer), "answer": str(corrected_answer)}


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_prompt(task: Dict[str, Any], reference: Any) -> str:
    prompt = task.get("prompt", "")
    test_cases = task.get("test_cases", [])
    return f"""You are an independent math reference-answer auditor.

You must independently solve the math problem and decide if the stored reference answer is correct.

Important requirements:
- Do NOT assume the stored reference is correct.
- Independently solve the problem and extract the final answer.
- The final answer MUST be a single numeric string whenever possible.
- If the correct answer is an integer, output it as digits only, e.g. "42".
- If the correct answer is not an integer, output a decimal, e.g. "15.4" or "0.154".
- Do NOT include units, percent signs, commas, equations, explanations, or words in independent_answer/corrected_answer.
- If the stored reference is wrong, provide the corrected final numeric answer.
- If the task is ambiguous or impossible to verify, mark cannot_verify.
- Output ONLY valid JSON.

Task prompt:
{prompt}

Stored reference answer:
{json.dumps(reference, ensure_ascii=False, indent=2)}

Test cases / metadata:
{json.dumps(test_cases[:2], ensure_ascii=False, indent=2)}

Return exactly this JSON schema:
{{
  "reference_correct": true or false,
  "confidence": 0.0 to 1.0,
  "independent_answer": "single numeric answer only, no units or explanation",
  "corrected_answer": "single numeric answer that should be stored; same as stored reference if correct",
  "stored_reference": "the stored reference you checked",
  "reason": "concise explanation",
  "error_type": "none | wrong_reference | ambiguous_task | insufficient_information | cannot_verify"
}}
"""


class RateLimitedLLM:
    def __init__(self, client: VLLMClient, sleep_min: float, sleep_max: float):
        self.client = client
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.call_count = 0

    async def ask(self, prompt: str) -> str:
        self.call_count += 1
        sleep_time = random.uniform(self.sleep_min, self.sleep_max)
        print(f"  [RateLimit] Sleeping {sleep_time:.1f}s before LLM call #{self.call_count}")
        await asyncio.sleep(sleep_time)
        return await self.client.achat(
            model=self.client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful math auditor. You output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )


async def audit_one(llm: RateLimitedLLM, task: Dict[str, Any]) -> Dict[str, Any]:
    reference = extract_reference(task)
    result = {
        "task_id": task.get("id"),
        "domain": task.get("domain"),
        "difficulty": task.get("difficulty"),
        "stored_reference": reference,
    }

    if reference is None:
        result.update(
            {
                "reference_correct": False,
                "confidence": 1.0,
                "independent_answer": None,
                "corrected_answer": None,
                "reason": "No stored reference answer found.",
                "error_type": "insufficient_information",
                "raw_response": None,
                "updated": False,
            }
        )
        return result

    raw = await llm.ask(build_prompt(task, reference))
    result["raw_response"] = raw

    try:
        parsed = parse_json_response(raw)
    except Exception as e:
        result.update(
            {
                "reference_correct": False,
                "confidence": 0.0,
                "independent_answer": None,
                "corrected_answer": None,
                "reason": f"Failed to parse LLM JSON response: {e}",
                "error_type": "cannot_verify",
                "updated": False,
            }
        )
        return result

    parsed["independent_answer"] = normalize_numeric_answer(parsed.get("independent_answer")) or parsed.get("independent_answer")
    parsed["corrected_answer"] = normalize_numeric_answer(parsed.get("corrected_answer")) or parsed.get("corrected_answer")
    result.update(parsed)
    corrected = result.get("corrected_answer") or result.get("independent_answer")
    should_update = result.get("reference_correct") is False and result.get("error_type") == "wrong_reference" and corrected
    result["updated"] = bool(should_update)
    result["new_reference"] = str(corrected) if should_update else normalize_numeric_answer(reference) or reference
    return result


def load_completed(checkpoint: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not checkpoint or not checkpoint.exists():
        return completed
    with open(checkpoint, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                completed[item["task_id"]] = item
    return completed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fix math_reasoning references with LLM audit")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="doubao-seed-2.0-lite")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N math tasks for testing")
    parser.add_argument("--sleep-min", type=float, default=10.0)
    parser.add_argument("--sleep-max", type=float, default=18.0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint JSONL to resume from")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("ERROR: ANTHROPIC_AUTH_TOKEN is not set")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(input_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    fixed_tasks = deepcopy(all_tasks)
    math_indices = [i for i, task in enumerate(fixed_tasks) if task.get("domain") == "math_reasoning"]
    if args.limit is not None:
        math_indices = math_indices[: args.limit]

    checkpoint_path = Path(args.resume) if args.resume else output_dir / f"math_reference_fix_progress_{timestamp}.jsonl"
    completed = load_completed(checkpoint_path)

    client = VLLMClient(
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        api_key=api_key,
        timeout=120,
        model=args.model,
    )
    llm = RateLimitedLLM(client, args.sleep_min, args.sleep_max)

    results: List[Dict[str, Any]] = list(completed.values())
    print(f"Total math tasks selected: {len(math_indices)}")
    print(f"Already completed from checkpoint: {len(completed)}")
    print(f"Checkpoint: {checkpoint_path}")

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint:
        for position, idx in enumerate(math_indices, 1):
            task = fixed_tasks[idx]
            task_id = task.get("id")
            if task_id in completed:
                audit = completed[task_id]
            else:
                print("=" * 90)
                print(f"[{position}/{len(math_indices)}] Auditing task {task_id}")
                print(f"Stored reference: {extract_reference(task)}")
                audit = await audit_one(llm, task)
                checkpoint.write(json.dumps(audit, ensure_ascii=False, default=str) + "\n")
                checkpoint.flush()
                results.append(audit)

            if audit.get("updated"):
                print(f"  UPDATE reference: {audit.get('stored_reference')} -> {audit.get('new_reference')}")
                update_reference(task, audit["new_reference"])
            else:
                print(f"  KEEP reference: correct={audit.get('reference_correct')} error_type={audit.get('error_type')}")

    output_path = output_dir / f"final_evolved_v1.0_complete_math_fixed_{timestamp}.json"
    report_path = output_dir / f"math_reference_fix_report_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_tasks, f, ensure_ascii=False, indent=2, default=str)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "model": args.model,
        "math_tasks_selected": len(math_indices),
        "audited": len(results),
        "updated": sum(1 for item in results if item.get("updated")),
        "kept_correct": sum(1 for item in results if item.get("reference_correct") is True),
        "cannot_verify": sum(1 for item in results if item.get("error_type") == "cannot_verify"),
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 90)
    print(f"Updated references: {summary['updated']}")
    print(f"New full dataset: {output_path}")
    print(f"Audit report: {report_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    asyncio.run(main())
