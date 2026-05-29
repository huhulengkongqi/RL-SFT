"""Run the trajectory quality-filtering funnel."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.quality_filter import (  # noqa: E402
    DEDUP_TEXT_PRESETS,
    DedupTextConfig,
    QualityFilter,
    QualityFilterConfig,
)
from agent_sft.quality_filter.quality_filter import write_quality_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QualityFilter over raw/SFT trajectory outputs")
    parser.add_argument("--input-dir", type=Path, default=Path("data/sft_trajectories"))
    parser.add_argument("--raw-glob", default="*_raw.json")
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_filter"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--level1-concurrency", type=int, default=2)
    parser.add_argument("--dedup-text-mode", choices=sorted(DEDUP_TEXT_PRESETS), default="trajectory")
    for field_name in ["prompt", "thought", "action", "observation", "error", "final-answer"]:
        parser.add_argument(f"--include-{field_name}", action="store_true", default=None)
        parser.add_argument(
            f"--exclude-{field_name}",
            action="store_false",
            dest=f"include_{field_name.replace('-', '_')}",
            default=None,
        )
    parser.add_argument("--minhash-ngram", type=int, default=5)
    parser.add_argument("--minhash-num-perm", type=int, default=128)
    parser.add_argument("--lsh-bands", type=int, default=32)
    parser.add_argument("--minhash-jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--embedding-threshold", type=float, default=0.9)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-diagnostics-top-k", type=int, default=50)
    parser.add_argument("--enable-embedding-dedup", action="store_true")
    parser.add_argument("--fail-open-missing-task", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--filtered-json", type=Path, default=None)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_json or args.output_dir / f"quality_filter_report_{timestamp}.json"
    filtered_path = args.filtered_json or args.output_dir / f"filtered_sft_trajectories_{timestamp}.json"
    rows = args.minhash_num_perm // args.lsh_bands

    if args.minhash_num_perm % args.lsh_bands != 0:
        raise ValueError("--minhash-num-perm must be divisible by --lsh-bands")

    dedup_text_config = DedupTextConfig(
        mode=args.dedup_text_mode,
        include_prompt=args.include_prompt,
        include_thought=args.include_thought,
        include_action=args.include_action,
        include_observation=args.include_observation,
        include_error=args.include_error,
        include_final_answer=args.include_final_answer,
    )

    config = QualityFilterConfig(
        input_dir=args.input_dir,
        raw_glob=args.raw_glob,
        task_file=args.task_file,
        limit=args.limit,
        domain=args.domain,
        level1_concurrency=args.level1_concurrency,
        dedup_text_config=dedup_text_config,
        minhash_num_perm=args.minhash_num_perm,
        minhash_ngram=args.minhash_ngram,
        lsh_bands=args.lsh_bands,
        lsh_rows=rows,
        minhash_jaccard_threshold=args.minhash_jaccard_threshold,
        embedding_similarity_threshold=args.embedding_threshold,
        enable_embedding_dedup=args.enable_embedding_dedup,
        embedding_model=args.embedding_model,
        embedding_diagnostics_top_k=args.embedding_diagnostics_top_k,
        fail_open_missing_task=args.fail_open_missing_task,
    )

    quality_filter = QualityFilter(config)
    report = await quality_filter.run()
    write_quality_outputs(report, report_path, filtered_path)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nFunnel pass rates:")
    for stage in report["funnel"]:
        print(
            f"- {stage['stage']}: {stage['output_count']}/{stage['input_count']} "
            f"({stage['pass_rate']:.2%})"
        )
    print(f"\nReport: {report_path}")
    print(f"Filtered SFT: {filtered_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
