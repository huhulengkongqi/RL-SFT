#!/usr/bin/env python
"""Summarize observed pass rates from raw trajectory files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BUCKET_PRESETS = {
    "baseline": {
        "easy": (0.75, 1.01),
        "medium": (0.35, 0.75),
        "hard": (0.05, 0.35),
        "extreme": (0.0, 0.05),
    },
    "strict_easy": {
        "easy": (0.85, 1.01),
        "medium": (0.40, 0.85),
        "hard": (0.05, 0.40),
        "extreme": (0.0, 0.05),
    },
    "balanced_candidate": {
        "easy": (0.65, 1.01),
        "medium": (0.30, 0.65),
        "hard": (0.05, 0.30),
        "extreme": (0.0, 0.05),
    },
    "wider_extreme": {
        "easy": (0.65, 1.01),
        "medium": (0.30, 0.65),
        "hard": (0.10, 0.30),
        "extreme": (0.0, 0.10),
    },
}

BUCKET_ORDER = ("easy", "medium", "hard", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze observed pass rate by task_id")
    parser.add_argument("--data-dir", default="data/sft_trajectories", help="Directory containing *_raw.json files")
    parser.add_argument("--raw-glob", default="*_raw.json", help="Raw trajectory glob pattern")
    parser.add_argument("--limit", type=int, default=None, help="Optional max raw files to read")
    parser.add_argument(
        "--bucket-preset",
        choices=[*BUCKET_PRESETS.keys(), "all"],
        default="all",
        help="Difficulty bucket preset to apply",
    )
    parser.add_argument("--show-boundaries", action="store_true", help="Print bucket interval boundaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_files = sorted(Path(args.data_dir).glob(args.raw_glob))
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for raw_file in raw_files:
        try:
            record = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        task_id = str(record.get("task_id") or raw_file.stem)
        grouped[task_id].append(record)

    task_stats = []
    attempts_counts = []
    for task_id, records in grouped.items():
        attempts = len(records)
        successes = sum(1 for record in records if is_success(record))
        pass_rate = successes / attempts if attempts else 0.0
        attempts_counts.append(attempts)
        task_stats.append({
            "task_id": task_id,
            "attempts": attempts,
            "successes": successes,
            "observed_pass_rate": pass_rate,
        })

    rates = [stat["observed_pass_rate"] for stat in task_stats]
    print(f"Raw files read: {len(raw_files)}")
    print(f"Skipped files: {skipped}")
    print(f"Tasks: {len(task_stats)}")
    if not task_stats:
        return
    print(f"Observed pass rate min: {min(rates):.4f}")
    print(f"Observed pass rate mean: {statistics.mean(rates):.4f}")
    print(f"Observed pass rate median: {statistics.median(rates):.4f}")
    print(f"Observed pass rate max: {max(rates):.4f}")
    print(f"Attempts per task min/mean/median/max: {min(attempts_counts)}/{statistics.mean(attempts_counts):.2f}/{statistics.median(attempts_counts):.1f}/{max(attempts_counts)}")

    preset_names = list(BUCKET_PRESETS) if args.bucket_preset == "all" else [args.bucket_preset]
    for preset_name in preset_names:
        buckets = BUCKET_PRESETS[preset_name]
        print(f"\nDifficulty buckets by observed pass rate [{preset_name}]:")
        if args.show_boundaries:
            for bucket in BUCKET_ORDER:
                lower, upper = buckets[bucket]
                print(f"  {bucket}: [{lower:.2f}, {upper:.2f})")
        bucket_counts = summarize_buckets(task_stats, buckets)
        print("  distribution:")
        for bucket in BUCKET_ORDER:
            count = bucket_counts[bucket]
            print(f"    {bucket}: {count} ({count / len(task_stats):.2%})")

    lowest_preset = preset_names[0]
    lowest_buckets = BUCKET_PRESETS[lowest_preset]
    print(f"\nLowest pass-rate tasks (bucket preset: {lowest_preset}):")
    for stat in sorted(task_stats, key=lambda item: (item["observed_pass_rate"], -item["attempts"], item["task_id"]))[:10]:
        bucket = bucket_for_rate(stat["observed_pass_rate"], lowest_buckets)
        print(
            f"  {stat['task_id']} attempts={stat['attempts']} successes={stat['successes']} "
            f"rate={stat['observed_pass_rate']:.4f} bucket={bucket}"
        )


def is_success(record: dict[str, Any]) -> bool:
    return bool(record.get("success")) or float(record.get("final_score") or 0.0) >= 1.0


def summarize_buckets(task_stats: list[dict[str, Any]], buckets: dict[str, tuple[float, float]]) -> Counter:
    return Counter(bucket_for_rate(stat["observed_pass_rate"], buckets) for stat in task_stats)


def bucket_for_rate(pass_rate: float, buckets: dict[str, tuple[float, float]]) -> str:
    for bucket in BUCKET_ORDER:
        lower, upper = buckets[bucket]
        if lower <= pass_rate < upper:
            return bucket
    return "medium"


if __name__ == "__main__":
    main()
