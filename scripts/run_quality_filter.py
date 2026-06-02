"""Run the trajectory quality-filtering funnel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft.quality_filter import (  # noqa: E402
    DEDUP_TEXT_PRESETS,
    DedupTextConfig,
    LLMJudgeConsensus,
    LLMMCEstimator,
    ProcessRewardScorer,
    QualityFilter,
    QualityFilterConfig,
)
from agent_sft.quality_filter.quality_filter import write_quality_outputs  # noqa: E402
from infra.vllm_client.client import VLLMClient  # noqa: E402


class RateLimitedJudge:
    def __init__(self, inner: VLLMClient, sleep_min: float, sleep_max: float):
        self.inner = inner
        self.model = inner.model
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max

    async def achat(self, *args, **kwargs):
        if self.sleep_max > 0:
            await asyncio.sleep(random.uniform(self.sleep_min, self.sleep_max))
        return await self.inner.achat(*args, **kwargs)


def print_progress(event: str, **payload) -> None:
    if payload:
        details = " ".join(f"{key}={value}" for key, value in payload.items())
        print(f"[quality_filter] {event} {details}", flush=True)
    else:
        print(f"[quality_filter] {event}", flush=True)


def parse_ratio(value: str) -> dict[str, int]:
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ratio must have four parts, e.g. 1:3:4:2")
    return dict(zip(["easy", "medium", "hard", "extreme"], parts))


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
    parser.add_argument("--enable-level2-prm", dest="enable_level2_prm", action="store_true", default=True)
    parser.add_argument("--disable-level2-prm", dest="enable_level2_prm", action="store_false")
    parser.add_argument("--level2-min-score", type=float, default=0.5)
    parser.add_argument("--level2-mc-weight", type=float, default=0.5)
    parser.add_argument("--level2-judge-weight", type=float, default=0.5)
    parser.add_argument("--level2-judge-consensus-k", type=int, default=3)
    parser.add_argument("--level2-offline-mc-mode", default="hybrid")
    parser.add_argument("--enable-level2-llm-judge", dest="enable_level2_llm_judge", action="store_true", default=True)
    parser.add_argument("--disable-level2-llm-judge", dest="enable_level2_llm_judge", action="store_false")
    parser.add_argument("--enable-level2-llm-mc", dest="enable_level2_llm_mc", action="store_true", default=True)
    parser.add_argument("--disable-level2-llm-mc", dest="enable_level2_llm_mc", action="store_false")
    parser.add_argument("--level2-mc-rollouts", type=int, default=5)
    parser.add_argument(
        "--level2-judge-base-url",
        default=os.getenv("VLLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"),
    )
    parser.add_argument("--level2-judge-model", default=os.getenv("VLLM_MODEL", "doubao-seed-2.0-lite"))
    parser.add_argument("--level2-judge-api-key", default=os.getenv("ANTHROPIC_AUTH_TOKEN"))
    parser.add_argument("--level2-judge-sleep-min", type=float, default=0.0)
    parser.add_argument("--level2-judge-sleep-max", type=float, default=0.0)
    parser.add_argument("--level2-judge-fail-closed", action="store_true")
    parser.add_argument("--enable-level4-sampling", dest="enable_level4_sampling", action="store_true", default=True)
    parser.add_argument("--disable-level4-sampling", dest="enable_level4_sampling", action="store_false")
    parser.add_argument("--difficulty-sampling-ratio", type=parse_ratio, default=parse_ratio("1:3:4:2"))
    parser.add_argument("--level4-target-count", type=int, default=None)
    parser.add_argument("--level4-seed", type=int, default=0)
    parser.add_argument("--enable-her-relabeling", dest="enable_her_relabeling", action="store_true", default=True)
    parser.add_argument("--disable-her-relabeling", dest="enable_her_relabeling", action="store_false")
    parser.add_argument("--her-include-in-filtered-sft", action="store_true")
    parser.add_argument("--her-mode", choices=["heuristic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--her-max-per-failed-task", type=int, default=2)
    parser.add_argument("--her-min-partial-score", type=float, default=0.2)
    parser.add_argument("--her-json", type=Path, default=None)
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
        enable_level2_prm=args.enable_level2_prm,
        level2_min_score=args.level2_min_score,
        level2_mc_weight=args.level2_mc_weight,
        level2_judge_weight=args.level2_judge_weight,
        level2_judge_consensus_k=args.level2_judge_consensus_k,
        level2_fail_closed_on_judge_error=args.level2_judge_fail_closed,
        level2_offline_mc_mode=args.level2_offline_mc_mode,
        enable_level4_sampling=args.enable_level4_sampling,
        difficulty_sampling_ratio=args.difficulty_sampling_ratio,
        level4_target_count=args.level4_target_count,
        level4_seed=args.level4_seed,
        enable_her_relabeling=args.enable_her_relabeling,
        her_include_in_filtered_sft=args.her_include_in_filtered_sft,
        her_mode=args.her_mode,
        her_max_per_failed_task=args.her_max_per_failed_task,
        her_min_partial_score=args.her_min_partial_score,
        fail_open_missing_task=args.fail_open_missing_task,
    )

    process_reward_scorer = None
    judge_client = None
    if args.enable_level2_llm_judge or args.enable_level2_llm_mc or args.her_mode in {"llm", "hybrid"}:
        judge_client = RateLimitedJudge(
            VLLMClient(
                base_url=args.level2_judge_base_url,
                api_key=args.level2_judge_api_key,
                timeout=120,
                model=args.level2_judge_model,
            ),
            args.level2_judge_sleep_min,
            args.level2_judge_sleep_max,
        )
        if args.enable_level2_llm_judge or args.enable_level2_llm_mc:
            process_reward_scorer = ProcessRewardScorer(
                mc_estimator=LLMMCEstimator(
                    judge_client,
                    rollouts=args.level2_mc_rollouts,
                    fail_closed=args.level2_judge_fail_closed,
                )
                if args.enable_level2_llm_mc
                else None,
                judge_consensus=LLMJudgeConsensus(
                    judge_client=judge_client if args.enable_level2_llm_judge else None,
                    consensus_k=args.level2_judge_consensus_k,
                    fail_closed=args.level2_judge_fail_closed,
                ),
                min_score=args.level2_min_score,
                mc_weight=args.level2_mc_weight,
                judge_weight=args.level2_judge_weight,
            )

    quality_filter = QualityFilter(
        config,
        process_reward_scorer=process_reward_scorer,
        her_llm_client=judge_client,
        progress=print_progress,
    )
    report = await quality_filter.run()
    write_quality_outputs(report, report_path, filtered_path)
    if args.her_json is not None:
        args.her_json.parent.mkdir(parents=True, exist_ok=True)
        args.her_json.write_text(json.dumps(report.get("her_sft", []), ensure_ascii=False, indent=2), encoding="utf-8")

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
