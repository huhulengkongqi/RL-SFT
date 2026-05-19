"""Quality filtering and diversity metrics module."""

from .lm_discriminator import LMDiscriminator, LMDiscriminatorResult
from .metrics import DiversityMetrics, calculate_diversity_stats

__all__ = [
    "LMDiscriminator",
    "LMDiscriminatorResult",
    "DiversityMetrics",
    "calculate_diversity_stats",
]

# Lazy import embedding deduplicators to avoid Windows segfault issues
def __getattr__(name):
    if name in ("EmbeddingDeduplicator", "IncrementalDeduplicator"):
        try:
            from . import embedding_deduplicator
            return getattr(embedding_deduplicator, name)
        except ImportError:
            raise ImportError(f"{name} requires sentence-transformers which couldn't be loaded")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
