"""Rebucket difficulty by adjusting pass@32 bucket thresholds, then re-run Level 4 sampling.

Works offline from an existing quality filter report, no re-running L1-L3.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.quality_filter import DifficultyAwareSampler  # noqa: E402
from agent_sft.quality_filter.quality_filter import write_quality_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebucket difficulty by adjusting pass@32 thresholds, then re-run Level 4 sampling"
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        required=True,
        help="Path to existing quality filter report JSON",
    )
    parser.add_argument(
        "--easy-min",
        type=float,
        default=0.75,
        help="Minimum pass@32 for easy bucket (default: 0.75)",
    )
    parser.add_argument(
        "--medium-min",
        type=float,
        default=0.35,
        help="Minimum pass@32 for medium bucket (default: 0.35)",
    )
    parser.add_argument(
        "--hard-min",
        type=float,
        default=0.05,
        help="Minimum pass@32 for hard bucket (default: 0.05)",
    )
    parser.add_argument(
        "--level4-target-count",
        type=int,
        default=None,
        help="Target sample count for Level 4 (default: keep all Level 3 survivors)",
    )
    parser.add_argument(
        "--difficulty-sampling-ratio",
        default="1:3:4:2",
        help="Difficulty sampling ratio easy:medium:hard:extreme (default: 1:3:4:2)",
    )
    parser.add_argument(
        "--level4-seed",
        type=int,
        default=0,
        help="Random seed for Level 4 sampling (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/quality_filter"),
        help="Output directory (default: data/quality_filter)",
    )
    return parser.parse_args()


def _pass_rate_bucket(pass_rate: float, easy_min: float, medium_min: float, hard_min: float) -> str:
    """Determine difficulty bucket from pass rate, matching DifficultyClassifier logic.

    Bounds are [min, max), left inclusive, right exclusive.
    """
    if pass_rate >= easy_min:
        return "easy"
    elif pass_rate >= medium_min:
        return "medium"
    elif pass_rate >= hard_min:
        return "hard"
    return "extreme"


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load report
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    all_records = report.get("kept", []) + report.get("filtered", [])

    # Validate thresholds are in correct order
    if not (args.easy_min > args.medium_min > args.hard_min >= 0):
        raise SystemExit(f"Thresholds must be decreasing: easy_min({args.easy_min}) > medium_min({args.medium_min}) > hard_min({args.hard_min}) >= 0")

    print(f"Input report: {args.input_report.name}")
    print(f"Total records: {len(all_records)}")
    print(f"New thresholds: easy >= {args.easy_min}, medium >= {args.medium_min}, hard >= {args.hard_min}")

    # Group by task_id, compute pass@32 per task
    task_stats = {}
    for rec in all_records:
        task_id = rec.get("task_id")
        if task_id not in task_stats:
            task_stats[task_id] = {"attempts": 0, "successes": 0}
        task_stats[task_id]["attempts"] += 1
        if rec.get("metadata", {}).get("level1", {}).get("passed", False):
            task_stats[task_id]["successes"] += 1

    for task_id in task_stats:
        attempts = task_stats[task_id]["attempts"]
        successes = task_stats[task_id]["successes"]
        task_stats[task_id]["pass_at_32"] = successes / attempts if attempts else 0.0

    print(f"Unique tasks: {len(task_stats)}")

    # Rebucket
    task_buckets = {
        task_id: _pass_rate_bucket(
            stats["pass_at_32"], args.easy_min, args.medium_min, args.hard_min
        )
        for task_id, stats in task_stats.items()
    }

    bucket_counts = Counter(task_buckets.values())
    print("New bucket distribution by task:")
    for bucket in ["easy", "medium", "hard", "extreme"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100 * count / len(task_buckets)
        print(f"  {bucket:8s}: {count:4d} ({pct:4.1f}%)")

    # Update records' metadata.difficulty.bucket
    for rec in all_records:
        task_id = rec.get("task_id")
        new_bucket = task_buckets.get(task_id, "medium")
        if "difficulty" not in rec["metadata"]:
            rec["metadata"]["difficulty"] = {}
        rec["metadata"]["difficulty"]["bucket"] = new_bucket

    # Update difficulty_diagnostics in report
    report["difficulty_diagnostics"] = {
        "task_count": len(task_buckets),
        "bucket_counts_by_task": dict(bucket_counts),
        "bucket_thresholds": {
            "easy": [args.easy_min, 1.01],
            "medium": [args.medium_min, args.easy_min],
            "hard": [args.hard_min, args.medium_min],
            "extreme": [0.0, args.hard_min],
        },
    }

    # Reconstruct Level 3 survivor pool (records with level4 metadata
    level3_records = [rec for rec in all_records if rec.get("metadata", {}).get("level4")]
    print(f"Level 3 survivors: {len(level3_records)}")

    # Parse ratio
    ratio_parts = [int(p) for p in args.difficulty_sampling_ratio.split(":")]
    if len(ratio_parts) != 4:
        raise SystemExit("Ratio must have four parts, e.g. 1:3:4:2")
    ratio = dict(zip(["easy", "medium", "hard", "extreme"], ratio_parts))

    # Run Level 4 sampling
    sampler = DifficultyAwareSampler(ratio=ratio, seed=args.level4_seed)
    target_count = args.level4_target_count or len(level3_records)
    print(f"Running Level 4 sampling: target={target_count}, ratio={args.difficulty_sampling_ratio}")

    level4_bucket_counts_before = Counter(
        rec["metadata"]["difficulty"]["bucket"] for rec in level3_records
    )
    print(f"Level 3 bucket distribution: {dict(level4_bucket_counts_before)}")

    # Need to load raw file for step count tie-break
    from dataclasses import dataclass

    @dataclass
    class FakeRecord:
        id: str
        task_id: str
        domain: str
        difficulty: str
        quality_score: float
        metadata: dict
        raw: dict

    selected = []
    for rec in level3_records:
        raw = {}
        raw_path = rec.get("raw_path")
        if raw_path and Path(raw_path).exists():
            try:
                raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        fake = FakeRecord(
            id=rec["id"],
            task_id=rec.get("task_id", ""),
            domain=rec.get("domain", ""),
            difficulty=rec["metadata"]["difficulty"]["bucket"],
            quality_score=rec.get("quality_score", 0.0),
            metadata=rec["metadata"],
            raw=raw,
        )
        selected.append(fake)

    selected, level4_meta = sampler.sample(selected, target_count)
    print(f"Level 4 selected: {len(selected)} records")
    print(f"Level 4 selected_by_bucket: {level4_meta.get('selected_by_bucket', {})}")
    print(f"Level 4 deficit: {level4_meta.get('deficit', 0)}")

    # Update funnel.level4
    for stage in report.get("funnel", []):
        if stage.get("stage") == "level4_difficulty_aware_sampling":
            stage["output_count"] = len(selected)
            stage["metadata"] = level4_meta

    # Build new kept list (recover summaries from the original records by id)
    selected_ids = {rec.id for rec in selected}
    new_kept = [rec for rec in all_records if rec["id"] in selected_ids]
    new_filtered = [rec for rec in all_records if rec["id"] not in selected_ids]

    # Reload SFT content for selected records
    filtered_sft = []
    for rec in new_kept:
        sft_path = rec.get("sft_path")
        if not sft_path:
            continue
        sft_file = Path(sft_path)
        if not sft_file.exists():
            continue
        try:
            sft = json.loads(sft_file.read_text(encoding="utf-8"))
            filtered_sft.append(sft)
        except (json.JSONDecodeError, OSError):
            continue

    print(f"SFT entries loaded: {len(filtered_sft)}")

    # Update summary
    total_input = report.get("summary", {}).get("input_count", len(all_records))
    report["summary"] = {
        "input_count": total_input,
        "final_count": len(new_kept),
        "overall_pass_rate": len(new_kept) / total_input if total_input else 0.0,
        "her_count": report.get("summary", {}).get("her_count", 0),
        "filtered_sft_count": len(filtered_sft),
        "filtered_sft_with_her_count": len(filtered_sft),
    }

    # Update kept/filtered
    report["kept"] = new_kept
    report["filtered"] = new_filtered
    report["filtered_sft"] = filtered_sft

    # Store rebucket parameters for traceability
    report["rebucket_params"] = {
        "easy_min": args.easy_min,
        "medium_min": args.medium_min,
        "hard_min": args.hard_min,
        "difficulty_sampling_ratio": args.difficulty_sampling_ratio,
        "level4_target_count": args.level4_target_count,
        "level4_seed": args.level4_seed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"quality_filter_report_rebucket_{timestamp}.json"
    filtered_path = args.output_dir / f"filtered_sft_trajectories_rebucket_{timestamp}.json"

    write_quality_outputs(report, report_path, filtered_path)

    print(f"\nDONE!")
    print(f"Report:   {report_path}")
    print(f"Filtered: {filtered_path}")


if __name__ == "__main__":
    main()
