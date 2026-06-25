"""Tests for token-length distribution analysis."""

import math
import random

from agent_sft.dataset_builder import analyze_token_distribution
from agent_sft.dataset_builder.token_distribution import TokenDistributionConfig


def test_empty_returns_zero_count():
    result = analyze_token_distribution([])
    assert result["count"] == 0
    assert result["median_in_target_band"] is False
    assert result["is_lognormal"] is False


def test_median_in_target_band():
    # Values centered around ~1500 tokens.
    counts = [1024, 1200, 1400, 1500, 1600, 1800, 2048]
    result = analyze_token_distribution(counts)
    assert 1024 <= result["median"] <= 2048
    assert result["median_in_target_band"] is True
    assert result["percentiles"]["p50"] == result["median"] or result["count"] == len(counts)


def test_median_out_of_band_low():
    counts = [50, 80, 100, 120, 150]
    result = analyze_token_distribution(counts)
    assert result["median_in_target_band"] is False


def test_lognormal_sample_is_detected():
    rng = random.Random(0)
    # Draw from a log-normal centered so the median ~ e^7.3 ~ 1480 tokens.
    counts = [int(math.exp(rng.normalvariate(7.3, 0.4))) for _ in range(2000)]
    result = analyze_token_distribution(counts)
    assert result["count"] > 1900
    assert result["is_lognormal"] is True
    assert abs(result["log_skewness"]) <= result["lognormal_skew_tolerance"]


def test_custom_config_band():
    cfg = TokenDistributionConfig(target_median_min=100, target_median_max=200)
    result = analyze_token_distribution([120, 150, 180], cfg)
    assert result["median_in_target_band"] is True
    assert result["target_median_range"] == [100, 200]
