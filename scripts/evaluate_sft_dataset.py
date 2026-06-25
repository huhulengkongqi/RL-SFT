#!/usr/bin/env python
"""Build a mixed SFT dataset and generate an automated evaluation report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.dataset_builder import (  # noqa: E402
    AutoEvaluator,
    DatasetBuilder,
    DatasetExporter,
    DatasetMixConfig,
    DatasetSourceConfig,
    EvaluationReportGenerator,
    ExportConfig,
    SFTDataPipeline,
    SFTFormatterConfig,
    TokenShapingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a general-agent SFT dataset")
    parser.add_argument("--data-dir", default="data/sft_trajectories", help="Directory containing *_raw.json files")
    parser.add_argument(
        "--quality-report",
        default="auto",
        help="Quality filter report path, 'auto' for latest completed report, or 'none' to scan raw data",
    )
    parser.add_argument("--output-dir", default="data/formatted_sft", help="Dataset output directory")
    parser.add_argument("--report-dir", default="data/evaluation", help="Evaluation report output directory")
    parser.add_argument("--dataset-name", default="general_agent_sft_v1", help="Dataset base name")
    parser.add_argument("--version", default=None, help="Dataset/report version; defaults to timestamp")
    parser.add_argument("--target-size", type=int, default=None, help="Target mixed dataset size")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed")
    parser.add_argument("--export-format", choices=["jsonl", "parquet", "both"], default="both")
    parser.add_argument("--token-shaping", action="store_true", help="Filter formatted records by token-length band")
    parser.add_argument("--token-min", type=int, default=None, help="Min token_count to keep (token shaping)")
    parser.add_argument("--token-max", type=int, default=None, help="Max token_count to keep (token shaping)")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate inputs without formatting/exporting a dataset")
    parser.add_argument("--no-build", action="store_true", help="Alias for --evaluate-only")
    parser.add_argument("--validate", action="store_true", help="After building, verify the export loads via chat template (trl/verl)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    mix_config = DatasetMixConfig(target_size=args.target_size, seed=args.seed)
    token_shaping = TokenShapingConfig(
        enabled=args.token_shaping,
        min_tokens=args.token_min,
        max_tokens=args.token_max,
    )
    pipeline = SFTDataPipeline(
        SFTFormatterConfig(export=ExportConfig(format=args.export_format), token_shaping=token_shaping)
    )
    builder = DatasetBuilder(mix_config=mix_config, pipeline=pipeline)
    sources, selected_report = resolve_sources(args, builder)

    raw_records, source_summaries = builder.load_sources(sources)
    if not raw_records and selected_report is not None:
        fallback_source = DatasetSourceConfig(path=args.data_dir, kind="raw_dir", label="raw_trajectories_fallback")
        sources = [fallback_source]
        raw_records, source_summaries = builder.load_sources(sources)
    selected_records, selected_counts, ratio_gaps = builder.select_records(raw_records)
    evaluator = AutoEvaluator()

    building = not args.evaluate_only and not args.no_build
    manifest_dict: dict[str, Any] | None = None
    export_paths: dict[str, str] = {}

    if building:
        result = builder.build(
            sources, output_dir=args.output_dir, dataset_name=args.dataset_name, version=version
        )
        formatted_records = result.formatted_records
        manifest_dict = result.manifest.to_dict()
        export_paths = {key: str(path) for key, path in result.export_paths.items()}
    else:
        # Format (with optional token shaping) to obtain token counts, without exporting.
        formatted_records, _ = pipeline.process_batch(selected_records)
        manifest_dict = {
            "dataset_name": args.dataset_name,
            "version": version,
            "source_summaries": source_summaries,
            "selected_counts": selected_counts,
            "ratio_gaps": ratio_gaps,
            "output_files": {},
        }

    token_counts = [int(r["token_count"]) for r in formatted_records if r.get("token_count")]
    metrics = evaluator.evaluate(
        formatted_records, dataset_name=args.dataset_name, token_counts=token_counts
    )
    metrics["sources"] = source_summaries
    metrics["selected_quality_report"] = str(selected_report) if selected_report else None
    metrics["selected_counts"] = selected_counts
    metrics["ratio_gaps"] = ratio_gaps
    metrics["token_shaping"] = {
        "enabled": token_shaping.enabled,
        "min_tokens": token_shaping.min_tokens,
        "max_tokens": token_shaping.max_tokens,
        "selected_before_shaping": len(selected_records),
        "records_after_shaping": len(formatted_records),
    }
    if building:
        metrics["manifest_path"] = str(result.manifest_path)
        metrics["export_paths"] = export_paths

    metrics_path = report_dir / f"eval_metrics_{version}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = EvaluationReportGenerator().render_markdown(metrics, manifest=manifest_dict)
    report_path = report_dir / f"eval_report_{version}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Loaded records: {len(raw_records)}")
    print(f"Selected records: {len(selected_records)}")
    print(f"Formatted/evaluated records: {len(formatted_records)}")
    td = metrics.get("token_distribution") or {}
    if td.get("count"):
        print(
            f"Token median: {td.get('median', 0):.0f} "
            f"(target {td['target_median_range'][0]:.0f}-{td['target_median_range'][1]:.0f}, "
            f"in_band={td.get('median_in_target_band')}, lognormal={td.get('is_lognormal')})"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    if building and metrics.get("manifest_path"):
        print(f"Manifest: {metrics['manifest_path']}")

    if args.validate and building:
        target = export_paths.get("parquet") or export_paths.get("jsonl")
        if target:
            ok = DatasetExporter.validate_with_trl(target)
            print(f"Validation (trl/verl loadable): {'PASS' if ok else 'FAIL'} -> {target}")


def resolve_sources(args: argparse.Namespace, builder: DatasetBuilder) -> tuple[list[DatasetSourceConfig], Path | None]:
    quality_report = str(args.quality_report).strip()
    if quality_report.lower() == "none":
        return [DatasetSourceConfig(path=args.data_dir, kind="raw_dir", label="raw_trajectories")], None
    if quality_report.lower() == "auto":
        report_path = builder.latest_quality_report()
        if report_path is not None:
            return [DatasetSourceConfig(path=report_path, kind="quality_report", label="latest_completed_quality_report")], report_path
        return [DatasetSourceConfig(path=args.data_dir, kind="raw_dir", label="raw_trajectories")], None
    report_path = Path(quality_report)
    return [DatasetSourceConfig(path=report_path, kind="quality_report", label="quality_report")], report_path


if __name__ == "__main__":
    main()
