"""Dataset building and formatting module for SFT training.

This module converts raw agent trajectories to standard HuggingFace/trl-compatible
format with chat template support, truncation strategies, and loss masking.

Main components:
- DataFormatter: Converts trajectories to standard 4-role chat format
- TokenCounter: Accurate token counting with HuggingFace tokenizers
- TrajectoryTruncator: Middle/head/tail truncation strategies
- LossMaskBuilder: Scratchpad-aware loss masking for efficient training
- DatasetExporter: Parquet/JSONL export for HuggingFace Dataset
- SFTDataPipeline: End-to-end processing pipeline
"""

from .auto_evaluator import AutoEvaluator, AutoEvaluatorConfig, EvaluationThresholds
from .data_formatter import (
    ChatMessage,
    DataFormatter,
    DatasetExporter,
    ExportConfig,
    FormatConfig,
    FormattedTrajectory,
    LossMaskBuilder,
    LossMaskConfig,
    RawTrajectory,
    SFTDataPipeline,
    SFTFormatterConfig,
    TokenCounter,
    TokenShapingConfig,
    TrajectoryTruncator,
    TruncationConfig,
    TruncationStrategy,
)
from .dataset_builder import (
    DatasetBuilder,
    DatasetBuildResult,
    DatasetManifest,
    DatasetMixConfig,
    DatasetSourceConfig,
    make_default_pipeline,
)
from .report_generator import EvaluationReportGenerator
from .token_distribution import TokenDistributionConfig, analyze_token_distribution

__all__ = [
    "AutoEvaluator",
    "AutoEvaluatorConfig",
    "ChatMessage",
    "DataFormatter",
    "DatasetBuilder",
    "DatasetBuildResult",
    "DatasetExporter",
    "DatasetManifest",
    "DatasetMixConfig",
    "DatasetSourceConfig",
    "EvaluationReportGenerator",
    "EvaluationThresholds",
    "ExportConfig",
    "FormatConfig",
    "FormattedTrajectory",
    "LossMaskBuilder",
    "LossMaskConfig",
    "RawTrajectory",
    "SFTDataPipeline",
    "SFTFormatterConfig",
    "TokenCounter",
    "TokenDistributionConfig",
    "TokenShapingConfig",
    "TrajectoryTruncator",
    "TruncationConfig",
    "TruncationStrategy",
    "analyze_token_distribution",
    "make_default_pipeline",
]

