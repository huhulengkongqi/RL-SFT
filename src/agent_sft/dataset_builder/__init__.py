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
    TrajectoryTruncator,
    TruncationConfig,
    TruncationStrategy,
)

__all__ = [
    "ChatMessage",
    "DataFormatter",
    "DatasetExporter",
    "ExportConfig",
    "FormatConfig",
    "FormattedTrajectory",
    "LossMaskBuilder",
    "LossMaskConfig",
    "RawTrajectory",
    "SFTDataPipeline",
    "SFTFormatterConfig",
    "TokenCounter",
    "TrajectoryTruncator",
    "TruncationConfig",
    "TruncationStrategy",
]

