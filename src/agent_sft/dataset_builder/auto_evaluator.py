"""Automated metric matrix for agent SFT datasets."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_sft.quality_filter.metrics import DiversityMetrics

from .dataset_builder import DEFAULT_DOMAIN_CATEGORY_MAP
from .token_distribution import TokenDistributionConfig, analyze_token_distribution


@dataclass
class EvaluationThresholds:
    pass_at_1: float = 0.60
    pass_at_8: float = 0.85
    diversity_score: float = 0.70
    min_avg_steps: float = 3.0
    max_avg_steps: float = 8.0
    min_difficulty_share: float = 0.10
    general_share_min: float = 0.30
    general_share_max: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoEvaluatorConfig:
    thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)
    domain_category_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DOMAIN_CATEGORY_MAP))
    diversity_sample_size: int = 500
    token_distribution: TokenDistributionConfig = field(default_factory=TokenDistributionConfig)
    difficulty_buckets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "easy": (0.75, 1.01),
            "medium": (0.35, 0.75),
            "hard": (0.05, 0.35),
            "extreme": (0.0, 0.05),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["difficulty_buckets"] = {key: list(value) for key, value in self.difficulty_buckets.items()}
        return data

class AutoEvaluator:
    def __init__(self, config: AutoEvaluatorConfig | None = None, diversity_metrics: DiversityMetrics | None = None):
        self.config = config or AutoEvaluatorConfig()
        self.diversity_metrics = diversity_metrics or DiversityMetrics()

    def evaluate(
        self,
        trajectories: list[dict[str, Any]],
        dataset_name: str | None = None,
        token_counts: list[int] | None = None,
    ) -> dict[str, Any]:
        task_stats = self._task_pass_stats(trajectories)
        pass_at_1 = self._mean([stats["pass_at_1"] for stats in task_stats.values()])
        pass_at_8 = self._mean([stats["pass_at_8"] for stats in task_stats.values()])
        under_sampled = sum(1 for stats in task_stats.values() if stats["attempts"] < 8)

        diversity = self._diversity(trajectories)
        steps = [self._step_count(record) for record in trajectories]
        avg_steps = self._mean(steps)
        median_steps = statistics.median(steps) if steps else 0.0
        distributions = self._distributions(trajectories)
        observed_difficulty = self._observed_difficulty_distribution(task_stats)

        token_distribution: dict[str, Any] | None = None
        if token_counts is None:
            token_counts = [int(t) for record in trajectories if (t := record.get("token_count"))]
        if token_counts:
            token_distribution = analyze_token_distribution(token_counts, self.config.token_distribution)

        matrix = self._metric_matrix(
            pass_at_1=pass_at_1,
            pass_at_8=pass_at_8,
            diversity_score=diversity["diversity_score"],
            avg_steps=avg_steps,
            difficulty_distribution=observed_difficulty["share"],
            general_share=distributions["category_share"].get("general_instruction", 0.0),
            under_sampled_tasks=under_sampled,
            token_distribution=token_distribution,
        )
        weak_areas = [entry for entry in matrix.values() if not entry["passed"]]

        return {
            "dataset_name": dataset_name,
            "record_count": len(trajectories),
            "task_count": len(task_stats),
            "config": self.config.to_dict(),
            "metrics": matrix,
            "pass_at": {
                "pass_at_1": pass_at_1,
                "pass_at_8": pass_at_8,
                "under_sampled_tasks_lt_8": under_sampled,
                "task_stats": task_stats,
            },
            "diversity": diversity,
            "trajectory_length": {
                "avg_steps": avg_steps,
                "median_steps": median_steps,
                "min_steps": min(steps) if steps else 0,
                "max_steps": max(steps) if steps else 0,
            },
            "token_distribution": token_distribution,
            "distribution": distributions,
            "observed_difficulty": observed_difficulty,
            "weak_areas": weak_areas,
        }

    @staticmethod
    def _step_count(record: dict[str, Any]) -> int:
        steps = record.get("steps")
        if isinstance(steps, list):
            return len(steps)
        return int(record.get("num_steps", 0) or 0)

    def _task_pass_stats(self, trajectories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in trajectories:
            grouped[str(record.get("task_id", "unknown"))].append(record)

        stats = {}
        for task_id, records in grouped.items():
            attempts = len(records)
            successes = sum(1 for record in records if self._is_success(record))
            k = min(8, attempts)
            stats[task_id] = {
                "attempts": attempts,
                "successes": successes,
                "pass_at_1": successes / attempts if attempts else 0.0,
                "pass_at_8": self._pass_at_k(attempts, successes, k),
                "pass_at_8_observed_k": k,
                "difficulty_bucket": self._bucket_for_rate(successes / attempts if attempts else 0.0),
            }
        return stats

    @staticmethod
    def _pass_at_k(n: int, c: int, k: int) -> float:
        if n <= 0 or k <= 0:
            return 0.0
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    @staticmethod
    def _is_success(record: dict[str, Any]) -> bool:
        return bool(record.get("success")) or float(record.get("final_score") or 0.0) >= 1.0

    def _bucket_for_rate(self, pass_rate: float) -> str:
        for bucket, (lower, upper) in self.config.difficulty_buckets.items():
            if lower <= pass_rate < upper:
                return bucket
        return "medium"

    def _diversity(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [self._trajectory_text(record) for record in trajectories]
        texts = [text for text in texts if text]
        self_bleu = self.diversity_metrics.self_bleu(texts, sample_size=self.config.diversity_sample_size) if texts else 0.0
        return {
            "self_bleu": self_bleu,
            "diversity_score": 1.0 - self_bleu,
            "num_texts": len(texts),
        }

    @staticmethod
    def _trajectory_text(record: dict[str, Any]) -> str:
        if record.get("messages"):
            return "\n".join(str(message.get("content", "")) for message in record.get("messages", []))
        parts = []
        for step in record.get("steps") or []:
            if step.get("thought"):
                parts.append(str(step["thought"]))
            action = step.get("action") or {}
            if action:
                parts.append(str(action))
            observation = step.get("observation") or {}
            if observation:
                parts.append(str(observation.get("content") or observation.get("error") or observation))
        return "\n".join(parts)

    def _distributions(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        domains = Counter(str(record.get("domain", "unknown")) for record in trajectories)
        categories = Counter(self.config.domain_category_map.get(str(record.get("domain", "")), "unknown") for record in trajectories)
        raw_difficulty = Counter(str(record.get("difficulty", "unknown")) for record in trajectories)
        total = len(trajectories)
        return {
            "domain_counts": dict(domains),
            "domain_share": self._shares(domains, total),
            "category_counts": dict(categories),
            "category_share": self._shares(categories, total),
            "raw_difficulty_counts": dict(raw_difficulty),
            "raw_difficulty_share": self._shares(raw_difficulty, total),
        }

    def _observed_difficulty_distribution(self, task_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(stats["difficulty_bucket"] for stats in task_stats.values())
        total = len(task_stats)
        for bucket in self.config.difficulty_buckets:
            counts.setdefault(bucket, 0)
        return {"counts_by_task": dict(counts), "share": self._shares(counts, total), "task_count": total}

    def _metric_matrix(
        self,
        *,
        pass_at_1: float,
        pass_at_8: float,
        diversity_score: float,
        avg_steps: float,
        difficulty_distribution: dict[str, float],
        general_share: float,
        under_sampled_tasks: int,
        token_distribution: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        thresholds = self.config.thresholds
        min_difficulty = min((difficulty_distribution.get(bucket, 0.0) for bucket in self.config.difficulty_buckets), default=0.0)
        matrix = {
            "pass_at_1": self._threshold_entry(pass_at_1, f">= {thresholds.pass_at_1:.2f}", pass_at_1 >= thresholds.pass_at_1),
            "pass_at_8": self._threshold_entry(
                pass_at_8,
                f">= {thresholds.pass_at_8:.2f}",
                pass_at_8 >= thresholds.pass_at_8,
                note=f"{under_sampled_tasks} tasks have fewer than 8 attempts" if under_sampled_tasks else "",
            ),
            "diversity_score": self._threshold_entry(
                diversity_score,
                f">= {thresholds.diversity_score:.2f}",
                diversity_score >= thresholds.diversity_score,
            ),
            "avg_steps": self._threshold_entry(
                avg_steps,
                f"{thresholds.min_avg_steps:.1f} - {thresholds.max_avg_steps:.1f}",
                thresholds.min_avg_steps <= avg_steps <= thresholds.max_avg_steps,
            ),
            "difficulty_coverage": self._threshold_entry(
                min_difficulty,
                f"each >= {thresholds.min_difficulty_share:.2f}",
                min_difficulty >= thresholds.min_difficulty_share,
                note="; ".join(
                    f"{bucket}={difficulty_distribution.get(bucket, 0.0):.1%}"
                    for bucket in self.config.difficulty_buckets
                ),
            ),
            "general_instruction_share": self._threshold_entry(
                general_share,
                f"{thresholds.general_share_min:.2f} - {thresholds.general_share_max:.2f}",
                thresholds.general_share_min <= general_share <= thresholds.general_share_max,
            ),
        }
        if token_distribution and token_distribution.get("count"):
            tcfg = self.config.token_distribution
            matrix["token_median_in_band"] = self._threshold_entry(
                token_distribution["median"],
                f"{tcfg.target_median_min:.0f} - {tcfg.target_median_max:.0f} tokens",
                bool(token_distribution["median_in_target_band"]),
                note=(
                    f"median={token_distribution['median']:.0f}, "
                    f"p50={token_distribution['percentiles']['p50']}, "
                    f"p90={token_distribution['percentiles']['p90']}"
                ),
            )
            matrix["token_lognormal"] = self._threshold_entry(
                token_distribution["log_skewness"],
                f"|log-skew| <= {tcfg.lognormal_skew_tolerance:.2f}",
                bool(token_distribution["is_lognormal"]),
                note=f"log_skewness={token_distribution['log_skewness']:.3f}",
            )
        return matrix

    @staticmethod
    def _threshold_entry(value: float, threshold: str, passed: bool, note: str = "") -> dict[str, Any]:
        return {"value": value, "threshold": threshold, "passed": passed, "note": note}

    @staticmethod
    def _shares(counts: Counter[str], total: int) -> dict[str, float]:
        if total == 0:
            return {key: 0.0 for key in counts}
        return {key: value / total for key, value in counts.items()}

    @staticmethod
    def _mean(values: list[float | int]) -> float:
        return sum(values) / len(values) if values else 0.0
