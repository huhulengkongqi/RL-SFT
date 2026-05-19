"""Embedding-based deduplication using sentence-transformers."""

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingDeduplicator:
    """
    Deduplicates prompts using sentence embeddings and cosine similarity.

    Filters out prompts with cosine similarity > threshold to any existing prompt.
    When duplicates are found, keeps the one with higher quality score.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding deduplicator.

        Args:
            model_name: sentence-transformers model name
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
            device: Device for model inference ('cpu', 'cuda', None for auto)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        """Lazy load the sentence-transformer model."""
        if self._model is None:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        self._load_model()
        assert self._model is not None

        embeddings = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings)

    def deduplicate(
        self,
        prompts: List[Dict],
        prompt_key: str = "prompt",
        quality_key: str = "quality_score",
        id_key: str = "id",
    ) -> List[Dict]:
        """
        Deduplicate prompts based on embedding similarity.
        Falls back to hash-based deduplication if sentence-transformers not available.

        Args:
            prompts: List of prompt dictionaries
            prompt_key: Key for prompt text in dict
            quality_key: Key for quality score (used to decide which to keep)
            id_key: Key for unique ID

        Returns:
            List of deduplicated prompts
        """
        if len(prompts) <= 1:
            return prompts

        # Fallback to hash-based deduplication
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using hash-based deduplication")
            seen_hashes = set()
            kept_prompts = []
            # Sort by quality to keep higher quality first
            sorted_prompts = sorted(prompts, key=lambda x: x.get(quality_key, 0.0), reverse=True)
            for prompt in sorted_prompts:
                content_hash = hashlib.md5(prompt[prompt_key].encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    kept_prompts.append(prompt)
            logger.info(f"Hash deduplication: {len(prompts)} → {len(kept_prompts)}")
            return kept_prompts

        self._load_model()
        assert self._model is not None

        # Sort by quality score descending - higher quality kept in case of duplicates
        sorted_prompts = sorted(prompts, key=lambda x: x.get(quality_key, 0.0), reverse=True)

        texts = [p[prompt_key] for p in sorted_prompts]
        embeddings = self.encode(texts)

        keep_mask = np.ones(len(sorted_prompts), dtype=bool)
        similarity_matrix = cosine_similarity(embeddings)

        for i in range(len(sorted_prompts)):
            if not keep_mask[i]:
                continue

            # Mark duplicates of i as not to keep (j > i since sorted by quality)
            for j in range(i + 1, len(sorted_prompts)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    keep_mask[j] = False
                    logger.debug(
                        f"Filtered duplicate prompt: id={sorted_prompts[j][id_key]}, "
                        f"similarity={similarity_matrix[i, j]:.4f}"
                    )

        kept_prompts = [p for p, keep in zip(sorted_prompts, keep_mask) if keep]
        logger.info(f"Deduplication: {len(prompts)} → {len(kept_prompts)} (filtered {len(prompts) - len(kept_prompts)})")

        return kept_prompts

    def calculate_similarity_matrix(self, prompts: List[Dict], prompt_key: str = "prompt") -> np.ndarray:
        """Calculate pairwise similarity matrix for prompts."""
        texts = [p[prompt_key] for p in prompts]
        embeddings = self.encode(texts)
        return cosine_similarity(embeddings)


class IncrementalDeduplicator(EmbeddingDeduplicator):
    """
    Deduplicator that maintains state for incremental processing.
    Useful for adding new batches without duplicates across batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._known_embeddings: List[np.ndarray] = []
        self._known_ids: List[str] = []

    def add_to_known(self, prompts: List[Dict], prompt_key: str = "prompt", id_key: str = "id") -> None:
        """Add prompts to known set without deduplication."""
        texts = [p[prompt_key] for p in prompts]
        ids = [p[id_key] for p in prompts]
        embeddings = self.encode(texts)

        self._known_ids.extend(ids)
        self._known_embeddings.extend(embeddings)

    def deduplicate_new(self, prompts: List[Dict], prompt_key: str = "prompt", id_key: str = "id") -> List[Dict]:
        """
        Deduplicate new prompts against known set and within themselves.

        Args:
            prompts: New prompts to deduplicate
            prompt_key: Key for prompt text
            id_key: Key for ID

        Returns:
            Deduplicated prompts
        """
        if not prompts:
            return []

        # First deduplicate within the new batch
        deduped_internal = self.deduplicate(prompts, prompt_key, id_key)

        if not self._known_embeddings:
            # No existing embeddings - just add all
            self.add_to_known(deduped_internal, prompt_key, id_key)
            return deduped_internal

        # Check against known embeddings
        new_texts = [p[prompt_key] for p in deduped_internal]
        new_embeddings = self.encode(new_texts)
        known_embeddings_array = np.array(self._known_embeddings)

        similarities = cosine_similarity(new_embeddings, known_embeddings_array)
        max_similarities = similarities.max(axis=1)

        kept = []
        for prompt, sim in zip(deduped_internal, max_similarities):
            if sim <= self.similarity_threshold:
                kept.append(prompt)
            else:
                logger.debug(
                    f"Filtered prompt (duplicate of known): id={prompt[id_key]}, similarity={sim:.4f}"
                )

        # Add kept to known
        self.add_to_known(kept, prompt_key, id_key)
        return kept

    def reset(self) -> None:
        """Reset known embeddings state."""
        self._known_embeddings = []
        self._known_ids = []
