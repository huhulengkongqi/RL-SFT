"""Markdown report generator for dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationReportGenerator:
    bar_width: int = 20

    def render_markdown(self, metrics: dict[str, Any], manifest: dict[str, Any] | None = None) -> str:
        lines = [
            f"# 自动化数据集评估报告 / Dataset Evaluation Report",
            "",
            f"- Dataset: `{metrics.get('dataset_name') or (manifest or {}).get('dataset_name') or 'unknown'}`",
            f"- Records: {metrics.get('record_count', 0)}",
            f"- Tasks: {metrics.get('task_count', 0)}",
        ]
        if manifest:
            lines.extend([
                f"- Version: `{manifest.get('version', 'unknown')}`",
                f"- Created at: {manifest.get('created_at', 'unknown')}",
                f"- Config hash: `{manifest.get('config_hash', 'unknown')}`",
            ])
        lines.extend(["", "## 指标矩阵 / Metric Matrix", "", "| Metric | Value | Threshold | Status | Note |", "|---|---:|---|---|---|"])
        for name, entry in metrics.get("metrics", {}).items():
            status = "PASS" if entry.get("passed") else "FAIL"
            value = entry.get("value")
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| `{name}` | {formatted} | {entry.get('threshold', '')} | {status} | {entry.get('note', '')} |")

        lines.extend(["", "## 数据配比 / Dataset Mix", "", "### Category Share", ""])
        category_share = metrics.get("distribution", {}).get("category_share", {})
        lines.extend(self._bar_lines(category_share))
        lines.extend(["", "### Domain Share", ""])
        lines.extend(self._bar_lines(metrics.get("distribution", {}).get("domain_share", {})))

        lines.extend(["", "## 难度覆盖 / Difficulty Coverage", ""])
        difficulty_share = metrics.get("observed_difficulty", {}).get("share", {})
        lines.extend(self._bar_lines(difficulty_share))

        pass_at = metrics.get("pass_at", {})
        lines.extend([
            "",
            "## Pass@k Diagnostics",
            "",
            f"- Pass@1: {pass_at.get('pass_at_1', 0.0):.4f}",
            f"- Pass@8: {pass_at.get('pass_at_8', 0.0):.4f}",
            f"- Tasks with fewer than 8 attempts: {pass_at.get('under_sampled_tasks_lt_8', 0)}",
        ])

        length = metrics.get("trajectory_length", {})
        diversity = metrics.get("diversity", {})
        lines.extend([
            "",
            "## Trajectory and Diversity",
            "",
            f"- Average steps: {length.get('avg_steps', 0.0):.2f}",
            f"- Median steps: {length.get('median_steps', 0.0):.2f}",
            f"- Step range: {length.get('min_steps', 0)} - {length.get('max_steps', 0)}",
            f"- Self-BLEU: {diversity.get('self_bleu', 0.0):.4f}",
            f"- Diversity Score (1 - Self-BLEU): {diversity.get('diversity_score', 0.0):.4f}",
        ])

        token_dist = metrics.get("token_distribution")
        if token_dist and token_dist.get("count"):
            pct = token_dist.get("percentiles", {})
            band = token_dist.get("target_median_range", [0, 0])
            lines.extend([
                "",
                "## Token 长度分布 / Token Length Distribution",
                "",
                f"- Records counted: {token_dist.get('count', 0)}",
                f"- Median: {token_dist.get('median', 0.0):.0f} "
                f"(target {band[0]:.0f}-{band[1]:.0f}, "
                f"{'IN BAND' if token_dist.get('median_in_target_band') else 'OUT OF BAND'})",
                f"- Mean: {token_dist.get('mean', 0.0):.0f} | Std: {token_dist.get('std', 0.0):.0f}",
                f"- Range: {token_dist.get('min', 0)} - {token_dist.get('max', 0)}",
                f"- Percentiles: p10={pct.get('p10', 0)}, p25={pct.get('p25', 0)}, "
                f"p50={pct.get('p50', 0)}, p75={pct.get('p75', 0)}, p90={pct.get('p90', 0)}, "
                f"p95={pct.get('p95', 0)}, p99={pct.get('p99', 0)}",
                f"- Log-normal: {'YES' if token_dist.get('is_lognormal') else 'NO'} "
                f"(log-skewness={token_dist.get('log_skewness', 0.0):.3f}, "
                f"log-mean={token_dist.get('log_mean', 0.0):.3f}, log-std={token_dist.get('log_std', 0.0):.3f})",
            ])
            if "lognormal_normaltest_p" in token_dist:
                lines.append(f"- Log-space normaltest p-value: {token_dist['lognormal_normaltest_p']:.4f}")

        lines.extend(["", "## 弱项与补充建议 / Weak Areas", ""])
        weak_areas = metrics.get("weak_areas", [])
        if not weak_areas:
            lines.append("- No weak areas found against the configured thresholds.")
        else:
            for entry in weak_areas:
                recommendation = self._recommendation(entry)
                lines.append(f"- `{self._metric_name(metrics, entry)}` failed: {entry.get('note') or entry.get('threshold')}. {recommendation}")

        if manifest:
            lines.extend(["", "## 输出文件 / Output Files", ""])
            for key, path in (manifest.get("output_files") or {}).items():
                lines.append(f"- {key}: `{path}`")
            if manifest.get("ratio_gaps"):
                lines.extend(["", "## 配比缺口 / Ratio Gaps", ""])
                for gap in manifest["ratio_gaps"]:
                    lines.append(
                        f"- {gap.get('category')}: target={gap.get('target')}, available={gap.get('available')}, shortfall={gap.get('shortfall')}"
                    )
        lines.append("")
        return "\n".join(lines)

    def _bar_lines(self, shares: dict[str, float]) -> list[str]:
        if not shares:
            return ["No data."]
        lines = ["| Bucket | Share | Chart |", "|---|---:|---|"]
        for key, value in sorted(shares.items()):
            lines.append(f"| {key} | {value:.1%} | {self._bar(value)} |")
        return lines

    def _bar(self, value: float) -> str:
        filled = round(max(0.0, min(1.0, value)) * self.bar_width)
        return "█" * filled + "░" * (self.bar_width - filled)

    @staticmethod
    def _metric_name(metrics: dict[str, Any], entry: dict[str, Any]) -> str:
        for name, candidate in metrics.get("metrics", {}).items():
            if candidate is entry:
                return name
        for name, candidate in metrics.get("metrics", {}).items():
            if candidate == entry:
                return name
        return "metric"

    @staticmethod
    def _recommendation(entry: dict[str, Any]) -> str:
        note = str(entry.get("note", ""))
        if "extreme" in note or "difficulty" in note:
            return "建议补采低通过率任务，优先补齐 hard/extreme 难度覆盖。"
        if "fewer than 8" in note:
            return "建议对样本不足任务补齐至少 8 条独立轨迹。"
        if "median" in note:
            return "建议调整 token 塑形区间（--token-min/--token-max）或截断上限，使中位数落入目标带。"
        if "log_skewness" in note:
            return "建议通过 token 塑形过滤极端长短样本，使长度更接近对数正态分布。"
        return "建议按该指标对应数据桶追加样本后重新评估。"
