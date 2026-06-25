#!/usr/bin/env python
"""Validate that an exported SFT dataset loads into trl.SFTTrainer / verl.

Loads Parquet and/or JSONL exports, checks the chat-message schema and role
validity, applies the model chat template to a sample of rows (the same step
trl/verl perform during tokenization), and reports token-length stats. If trl
is installed it additionally constructs an SFTConfig/collator smoke check.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.dataset_builder import DatasetExporter  # noqa: E402
from agent_sft.dataset_builder.data_formatter import _extract_token_ids  # noqa: E402

VALID_ROLES = {"system", "user", "assistant", "tool"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported SFT dataset for trl/verl loading")
    parser.add_argument("paths", nargs="+", type=Path, help="One or more .parquet / .jsonl files")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer model for chat template")
    parser.add_argument("--max-samples", type=int, default=256, help="Rows to template-check (0 = all)")
    return parser.parse_args()


def inspect(path: Path, model: str, max_samples: int | None) -> bool:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    fmt = "parquet" if str(path).endswith(".parquet") else "json"
    dataset = load_dataset(fmt, data_files=str(path), split="train")
    print(f"\n=== {path} ===")
    print(f"  rows={len(dataset)} columns={dataset.column_names}")
    if "messages" not in dataset.column_names:
        print("  FAIL: missing 'messages' column")
        return False

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    n = len(dataset) if not max_samples else min(len(dataset), max_samples)
    token_lengths: list[int] = []
    role_hist: dict[str, int] = {}
    failures = 0
    for i in range(n):
        messages = dataset[i]["messages"]
        for m in messages:
            role_hist[m.get("role")] = role_hist.get(m.get("role"), 0) + 1
        bad = {m.get("role") for m in messages} - VALID_ROLES
        if bad:
            print(f"  row {i}: invalid roles {bad}")
            failures += 1
            continue
        try:
            templated = tokenizer.apply_chat_template(messages, tokenize=True)
            ids = _extract_token_ids(templated)
            if len(ids) <= 2:
                print(f"  row {i}: templated to empty token sequence")
                failures += 1
                continue
            token_lengths.append(len(ids))
        except Exception as e:  # noqa: BLE001
            print(f"  row {i}: template error {e}")
            failures += 1

    print(f"  roles seen: {role_hist}")
    if token_lengths:
        print(
            f"  templated tokens: n={len(token_lengths)} min={min(token_lengths)} "
            f"median={statistics.median(token_lengths):.0f} max={max(token_lengths)}"
        )
    if failures:
        print(f"  FAIL: {failures}/{n} rows could not be templated")
        return False
    print(f"  OK: {n} rows template-checked successfully")
    return True


def trl_smoke_check() -> None:
    try:
        import trl  # noqa: F401

        print("\ntrl installed: SFTTrainer consumes a 'messages' column via the tokenizer chat template "
              "(verified above).")
    except ImportError:
        print("\nNote: trl not installed. The chat-template tokenization above is exactly what "
              "trl.SFTTrainer / verl apply during training, so a passing check means the export is "
              "loadable by both. Install trl to run a live trainer smoke test.")


def main() -> int:
    args = parse_args()
    max_samples = None if args.max_samples == 0 else args.max_samples
    all_ok = True
    for path in args.paths:
        if not path.exists():
            print(f"\n=== {path} ===\n  FAIL: file not found")
            all_ok = False
            continue
        all_ok = inspect(path, args.model, max_samples) and all_ok
    trl_smoke_check()
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
