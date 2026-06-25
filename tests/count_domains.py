"""Count trajectories per domain: raw data dir vs latest quality-filter report.

Usage:
    uv run python tests/count_domains.py
"""

import glob
import json
from collections import Counter
from pathlib import Path

DOMAINS = ["code_debug", "math_reasoning", "api_orchestration", "multi_step_planning"]
RAW_DIR = Path("data/sft_trajectories")
QF_DIR = Path("data/quality_filter")


def count_raw_dir() -> Counter:
    counts: Counter = Counter()
    for path in RAW_DIR.glob("*_raw.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        counts[str(record.get("domain", "unknown"))] += 1
    return counts


def latest_report() -> Path | None:
    reports = sorted(QF_DIR.glob("quality_filter_report_*.json"), reverse=True)
    return reports[0] if reports else None


def count_report_kept(report_path: Path) -> Counter:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return Counter(str(entry.get("domain", "unknown")) for entry in report.get("kept", []))


def print_table(title: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n{title} (total={total})")
    for domain in DOMAINS:
        print(f"  {domain:22s}: {counts.get(domain, 0)}")
    extra = set(counts) - set(DOMAINS)
    for domain in sorted(extra):
        print(f"  {domain:22s}: {counts[domain]}  (unexpected)")


def main() -> None:
    raw_counts = count_raw_dir()
    print_table(f"Raw trajectories in {RAW_DIR}", raw_counts)

    report_path = latest_report()
    if report_path is None:
        print(f"\nNo quality_filter report found in {QF_DIR}")
        return
    kept_counts = count_report_kept(report_path)
    print_table(f"Filtered (kept) in {report_path.name}", kept_counts)


if __name__ == "__main__":
    main()
