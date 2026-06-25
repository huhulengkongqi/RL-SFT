"""Token-length distribution analysis for SFT datasets.

Measures how close a dataset's token-length distribution is to the target:
log-normal shape with the median in the 1024-2048 band (leaving room for RL
generation under a 4096 context window).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


@dataclass
class TokenDistributionConfig:
    """Target distribution thresholds for token-length analysis."""

    target_median_min: float = 1024.0
    target_median_max: float = 2048.0
    # |skewness of log(tokens)| below this => approximately log-normal.
    lognormal_skew_tolerance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    n = len(sorted_values)
    idx = int(round(pct / 100.0 * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return sorted_values[idx]


def analyze_token_distribution(
    token_counts: list[int],
    config: TokenDistributionConfig | None = None,
) -> dict[str, Any]:
    """Compute token-length distribution statistics and target checks.

    Returns count/min/max/mean/median/std, percentiles, log-space stats and a
    Fisher-Pearson skewness of log(tokens). A near-zero log-skew indicates an
    approximately log-normal distribution. Also flags whether the median falls
    inside the configured target band.
    """
    cfg = config or TokenDistributionConfig()
    counts = sorted(int(t) for t in token_counts if t and int(t) > 0)
    n = len(counts)

    if n == 0:
        return {
            "count": 0,
            "target_median_range": [cfg.target_median_min, cfg.target_median_max],
            "median_in_target_band": False,
            "is_lognormal": False,
        }

    mean = statistics.mean(counts)
    median = statistics.median(counts)
    std = statistics.pstdev(counts) if n > 1 else 0.0

    logs = [math.log(t) for t in counts]
    log_mean = statistics.mean(logs)
    log_std = statistics.pstdev(logs) if n > 1 else 0.0

    if n > 2 and log_std > 0:
        log_skew = sum(((x - log_mean) / log_std) ** 3 for x in logs) / n
    else:
        log_skew = 0.0

    median_in_band = cfg.target_median_min <= median <= cfg.target_median_max
    is_lognormal = abs(log_skew) <= cfg.lognormal_skew_tolerance

    result: dict[str, Any] = {
        "count": n,
        "min": counts[0],
        "max": counts[-1],
        "mean": mean,
        "median": median,
        "std": std,
        "percentiles": {f"p{p}": _percentile(counts, p) for p in _PERCENTILES},
        "log_mean": log_mean,
        "log_std": log_std,
        "log_skewness": log_skew,
        "is_lognormal": is_lognormal,
        "median_in_target_band": median_in_band,
        "target_median_range": [cfg.target_median_min, cfg.target_median_max],
        "lognormal_skew_tolerance": cfg.lognormal_skew_tolerance,
    }

    # Optional, more rigorous normality test on log(tokens) if scipy is present.
    try:
        from scipy import stats as _scipy_stats

        if n >= 8:
            result["lognormal_normaltest_p"] = float(_scipy_stats.normaltest(logs).pvalue)
    except Exception:
        pass

    return result
