"""Summarize QualityFilter report diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a QualityFilter report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--top-groups", type=int, default=10)
    parser.add_argument("--top-pairs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    level3 = next(stage for stage in report["funnel"] if stage["stage"] == "level3_deduplication_diversity")
    metadata = level3["metadata"]

    print("Summary")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nFunnel")
    for stage in report["funnel"]:
        print(f"- {stage['stage']}: {stage['output_count']}/{stage['input_count']} ({stage['pass_rate']:.2%})")

    print("\nEmbedding diagnostics")
    for key in [
        "embedding_enabled",
        "embedding_pairs_above_threshold",
        "embedding_same_task_pairs_above_threshold",
        "embedding_cross_task_pairs_above_threshold",
        "embedding_error",
        "duplicate_groups",
        "filtered_duplicates",
    ]:
        print(f"- {key}: {metadata.get(key)}")

    print("\nSimilarity distribution")
    for bucket, count in (metadata.get("embedding_similarity_distribution") or {}).items():
        print(f"- {bucket}: {count}")

    print("\nTop duplicate groups")
    for group in (metadata.get("duplicate_group_summaries") or [])[: args.top_groups]:
        print(
            f"- size={group['size']} unique_tasks={group['unique_task_count']} "
            f"same_task={group['same_task_group']} kept={group['kept_id']} tasks={group['task_ids']}"
        )

    print("\nTop embedding pairs")
    for pair in (metadata.get("embedding_top_pairs") or [])[: args.top_pairs]:
        print(
            f"- sim={pair['similarity']} same_task={pair['same_task']} "
            f"left={pair['left_id']} right={pair['right_id']}"
        )


if __name__ == "__main__":
    main()
