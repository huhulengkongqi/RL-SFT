"""Re-run only Level 4 (difficulty-aware sampling) of the quality filter funnel.

Rebuilds the Level 3 survivor pool from an existing quality-filter report and
re-samples with a new target count, without re-running Levels 1-3. Final SFT
payloads for newly selected records are reloaded from their original sft_path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.quality_filter.quality_filter import (  # noqa: E402
    DifficultyAwareSampler,
    TrajectoryRecord,
    write_quality_outputs,
)

LEVEL4_STAGE = "level4_difficulty_aware_sampling"


def parse_ratio(value: str) -> dict[str, int]:
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ratio must have four parts, e.g. 1:3:4:2")
    return dict(zip(["easy", "medium", "hard", "extreme"], parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run only Level 4 sampling from an existing quality-filter report")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/quality_filter/quality_filter_report_20260603_134839.json"),
    )
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--difficulty-sampling-ratio", type=parse_ratio, default=parse_ratio("1:3:4:2"))
    parser.add_argument("--level4-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_filter"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--filtered-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_record(summary: dict) -> TrajectoryRecord:
    """Reconstruct a TrajectoryRecord from a report summary; load raw for step-count tie-break."""
    raw_path = Path(summary["raw_path"]) if summary.get("raw_path") else None
    sft_path = Path(summary["sft_path"]) if summary.get("sft_path") else None

    raw: dict = {}
    if raw_path and raw_path.exists():
        try:
            raw = load_json(raw_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] failed to read raw {raw_path}: {exc}", flush=True)
    elif raw_path:
        print(f"[warn] missing raw file: {raw_path}", flush=True)

    return TrajectoryRecord(
        id=summary["id"],
        task_id=summary.get("task_id", ""),
        domain=summary.get("domain", ""),
        difficulty=summary.get("difficulty", "medium"),
        raw_path=raw_path,
        sft_path=sft_path,
        raw=raw,
        sft=None,
        quality_score=summary.get("quality_score", 0.0) or 0.0,
        quality_score_source=summary.get("quality_score_source", "unset"),
        metadata=summary.get("metadata", {}) or {},
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_json or args.output_dir / f"quality_filter_report_{timestamp}.json"
    filtered_path = args.filtered_json or args.output_dir / f"filtered_sft_trajectories_{timestamp}.json"

    print(f"[rerun_level4] loading report {args.report}", flush=True)
    report = load_json(args.report)

    # Level 3 survivors = the records that were fed into Level 4 last run.
    # Every such record received metadata.level4; reconstruct the pool from kept + filtered.
    kept = report.get("kept", [])
    filtered = report.get("filtered", [])
    pool_summaries = list(kept) + [f for f in filtered if "level4" in (f.get("metadata") or {})]
    print(f"[rerun_level4] level3 survivor pool: {len(pool_summaries)} (kept={len(kept)})", flush=True)

    records = [build_record(s) for s in pool_summaries]

    sampler = DifficultyAwareSampler(ratio=args.difficulty_sampling_ratio, seed=args.level4_seed)
    selected, level4_meta = sampler.sample(records, args.target_count)
    print(f"[rerun_level4] selected {len(selected)} records", flush=True)
    print(f"[rerun_level4] level4 metadata: {json.dumps(level4_meta, ensure_ascii=False)}", flush=True)

    # Load SFT payloads for selected records from their original sft_path.
    filtered_sft: list = []
    missing_sft = 0
    for record in selected:
        if not record.sft_path:
            continue
        if not record.sft_path.exists():
            missing_sft += 1
            print(f"[warn] missing sft file: {record.sft_path}", flush=True)
            continue
        try:
            record.sft = load_json(record.sft_path)
            filtered_sft.append(record.sft)
        except (json.JSONDecodeError, OSError) as exc:
            missing_sft += 1
            print(f"[warn] failed to read sft {record.sft_path}: {exc}", flush=True)
    print(f"[rerun_level4] filtered_sft payloads: {len(filtered_sft)} (missing/failed sft: {missing_sft})", flush=True)

    # Build the new report from the original, replacing only Level 4 outputs.
    new_report = copy.deepcopy(report)
    selected_ids = {record.id for record in selected}

    for stage in new_report.get("funnel", []):
        if stage.get("stage") == LEVEL4_STAGE:
            stage["output_count"] = len(selected)
            stage["pass_rate"] = (len(selected) / stage["input_count"]) if stage.get("input_count") else 0.0
            stage["metadata"] = level4_meta

    input_count = new_report.get("summary", {}).get("input_count", len(records))
    new_report["summary"] = {
        **new_report.get("summary", {}),
        "final_count": len(selected),
        "overall_pass_rate": (len(selected) / input_count) if input_count else 0.0,
        "filtered_sft_count": len(filtered_sft),
        "filtered_sft_with_her_count": len(filtered_sft),
    }
    new_report["kept"] = [record.summary() for record in selected]
    new_report["filtered"] = [s for s in (kept + filtered) if s["id"] not in selected_ids]
    new_report["filtered_sft"] = filtered_sft
    new_report["created_at"] = datetime.now().isoformat()
    new_report["rerun_level4"] = {
        "source_report": str(args.report),
        "target_count": args.target_count,
        "difficulty_sampling_ratio": args.difficulty_sampling_ratio,
        "level4_seed": args.level4_seed,
    }

    write_quality_outputs(new_report, report_path, filtered_path)
    print(f"[rerun_level4] wrote report  -> {report_path}", flush=True)
    print(f"[rerun_level4] wrote filtered -> {filtered_path}", flush=True)


if __name__ == "__main__":
    main()
