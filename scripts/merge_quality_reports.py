"""Merge a domain-specific quality-filter report into a base report.

Use case: the base report was produced over all domains, but one domain (e.g.
code_debug) was re-run separately after a verifier fix. This stitches the
re-run domain's results into the base report, replacing that domain's entries.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

QF_DIR = Path("data/quality_filter")
DOMAINS = ["code_debug", "math_reasoning", "api_orchestration", "multi_step_planning"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def auto_domain_report(domain: str) -> Path | None:
    for path in sorted(QF_DIR.glob("quality_filter_report_*.json"), reverse=True):
        try:
            cfg = load(path).get("config", {})
        except Exception:
            continue
        if cfg.get("domain") == domain:
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a domain-specific quality report into a base report")
    parser.add_argument("--base", type=Path, default=QF_DIR / "quality_filter_report_20260622_093217.json")
    parser.add_argument("--domain-report", type=Path, default=None, help="Path to the re-run domain report (default: auto-detect)")
    parser.add_argument("--domain", default="code_debug", help="Domain being merged in")
    parser.add_argument("--output-dir", type=Path, default=QF_DIR)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--filtered-json", type=Path, default=None)
    return parser.parse_args()


def domain_counts(entries: list[dict]) -> dict[str, int]:
    c = Counter(str(e.get("domain", "unknown")) for e in entries)
    return {d: c.get(d, 0) for d in DOMAINS} | {k: v for k, v in c.items() if k not in DOMAINS}


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    domain_report = args.domain_report or auto_domain_report(args.domain)
    if domain_report is None:
        raise SystemExit(f"No re-run report found for domain={args.domain} in {QF_DIR}")

    base = load(args.base)
    dom = load(domain_report)
    print(f"base report : {args.base.name}")
    print(f"domain report: {domain_report.name} (domain={args.domain})")

    d = args.domain
    base_kept_other = [e for e in base.get("kept", []) if e.get("domain") != d]
    base_filtered_other = [e for e in base.get("filtered", []) if e.get("domain") != d]

    merged_kept = base_kept_other + dom.get("kept", [])
    merged_filtered = base_filtered_other + dom.get("filtered", [])
    merged_filtered_sft = list(base.get("filtered_sft", [])) + list(dom.get("filtered_sft", []))
    merged_her_sft = list(base.get("her_sft", [])) + list(dom.get("her_sft", []))

    base_her = base.get("her_sft", [])
    dom_her = dom.get("her_sft", [])
    her_count = len(base_her) + len(dom_her)
    input_count = base.get("summary", {}).get("input_count", 0)
    # 结构与原始报告完全匹配，不添加任何额外字段
    merged = {
        "created_at": datetime.now().isoformat(),
        "config": base.get("config"),
        "summary": {
            "input_count": input_count,
            "final_count": len(merged_kept),
            "overall_pass_rate": (len(merged_kept) / input_count) if input_count else 0.0,
            "her_count": her_count,
            "filtered_sft_count": len(merged_filtered_sft),
            "filtered_sft_with_her_count": len(merged_filtered_sft) + len(merged_her_sft),
        },
        "funnel": base.get("funnel"),
        "difficulty_diagnostics": base.get("difficulty_diagnostics"),
        "her_relabeling": base.get("her_relabeling"),
        "her_sft": merged_her_sft,
        "kept": merged_kept,
        "filtered": merged_filtered,
        "failures": base.get("failures", []),
        "filtered_sft": merged_filtered_sft,
    }

    report_path = args.report_json or args.output_dir / f"quality_filter_report_merged_{timestamp}.json"
    filtered_path = args.filtered_json or args.output_dir / f"filtered_sft_trajectories_merged_{timestamp}.json"
    report_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    filtered_path.write_text(json.dumps(merged_filtered_sft, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmerged kept: {len(merged_kept)} (was {len(base.get('kept', []))})")
    print(f"merged filtered_sft: {len(merged_filtered_sft)}")
    print("kept domain breakdown:")
    for dom_name, n in domain_counts(merged_kept).items():
        print(f"  {dom_name:22s}: {n}")
    print(f"\nreport  -> {report_path}")
    print(f"filtered -> {filtered_path}")


if __name__ == "__main__":
    main()
