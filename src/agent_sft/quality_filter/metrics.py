"""Diversity metrics for evolved prompts - Self-BLEU and more."""

import logging
import math
from collections import Counter
from typing import List

import nltk
from nltk.util import ngrams

logger = logging.getLogger(__name__)


class DiversityMetrics:
    """Calculate diversity metrics for a corpus of prompts."""

    def __init__(self, max_ngram: int = 4):
        """
        Initialize diversity metrics calculator.

        Args:
            max_ngram: Maximum n-gram size for BLEU calculation
        """
        self.max_ngram = max_ngram
        self._download_nltk_data()

    @staticmethod
    def _download_nltk_data() -> None:
        """Download required nltk data if not present."""
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text - handle both Chinese and English."""
        # Simple mixed-language tokenization
        # For English, use word tokenization; for Chinese just split chars
        tokens = []
        for word in nltk.word_tokenize(text.lower()):
            if any("一" <= c <= "鿿" for c in word):
                # Chinese - character-level
                tokens.extend(list(word))
            else:
                tokens.append(word)
        return tokens

    def _calculate_bleu(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        n: int = 4,
    ) -> float:
        """
        Calculate BLEU score between hypothesis and references.

        Args:
            hypothesis: Tokenized hypothesis
            references: List of tokenized references
            n: Max n-gram order

        Returns:
            BLEU score (0-1)
        """
        if not hypothesis or not references:
            return 0.0

        # Calculate n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            hyp_ngrams = Counter(ngrams(hypothesis, i))
            if not hyp_ngrams:
                precisions.append(0.0)
                continue

            max_ref_counts = Counter()
            for ref in references:
                ref_ngrams = Counter(ngrams(ref, i))
                for ng, count in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), count)

            clipped = sum(min(hyp_ngrams[ng], max_ref_counts.get(ng, 0)) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())

            if total > 0:
                precisions.append(clipped / total)
            else:
                precisions.append(0.0)

        # Geometric mean of precisions
        if all(p == 0 for p in precisions):
            return 0.0

        geo_mean = math.exp(sum(math.log(p + 1e-10) for p in precisions if p > 0) / n)

        # Brevity penalty
        hyp_len = len(hypothesis)
        ref_len = min((abs(len(ref) - hyp_len), len(ref)) for ref in references)[1]

        if hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

        return bp * geo_mean

    def self_bleu(self, texts: List[str], sample_size: int = 500) -> float:
        """
        Calculate Self-BLEU score for a corpus.

        Self-BLEU measures the average similarity between each text
        and all other texts in the corpus. Lower score = higher diversity.

        Args:
            texts: List of text strings
            sample_size: Max texts to use (to save computation time)

        Returns:
            Self-BLEU score (0-1), lower = more diverse
        """
        if len(texts) < 2:
            return 0.0

        # Sample if too many texts
        if len(texts) > sample_size:
            import random

            sampled = random.sample(texts, sample_size)
        else:
            sampled = texts

        tokenized = [self._tokenize(text) for text in sampled]
        n = min(self.max_ngram, min(len(t) for t in tokenized if t))

        if n < 1:
            return 0.0

        bleu_scores = []
        for i, hyp in enumerate(tokenized):
            references = [tokenized[j] for j in range(len(tokenized)) if j != i]
            if references:
                bleu = self._calculate_bleu(hyp, references, n=n)
                bleu_scores.append(bleu)

        if not bleu_scores:
            return 0.0

        mean_bleu = sum(bleu_scores) / len(bleu_scores)
        logger.info(f"Self-BLEU ({len(bleu_scores)} samples): {mean_bleu:.4f}")
        return mean_bleu

    def distinct_n_grams(self, texts: List[str], n: int = 2) -> float:
        """
        Calculate distinct n-gram ratio.

        Args:
            texts: List of text strings
            n: N-gram order

        Returns:
            distinct-ngram ratio (0-1), higher = more diverse
        """
        all_ngrams = set()
        total_ngrams = 0

        for text in texts:
            tokens = self._tokenize(text)
            ngs = list(ngrams(tokens, n))
            all_ngrams.update(ngs)
            total_ngrams += len(ngs)

        if total_ngrams == 0:
            return 0.0

        ratio = len(all_ngrams) / total_ngrams
        logger.info(f"Distinct-{n}: {ratio:.4f} ({len(all_ngrams)} unique / {total_ngrams} total)")
        return ratio

    def vocabulary_size(self, texts: List[str]) -> int:
        """Count unique words/tokens in the corpus."""
        vocab = set()
        for text in texts:
            vocab.update(self._tokenize(text))
        return len(vocab)

    def get_all_metrics(self, texts: List[str]) -> dict:
        """Calculate all diversity metrics."""
        if not texts:
            return {}

        return {
            "self_bleu": self.self_bleu(texts),
            "distinct_1": self.distinct_n_grams(texts, 1),
            "distinct_2": self.distinct_n_grams(texts, 2),
            "distinct_3": self.distinct_n_grams(texts, 3),
            "vocab_size": self.vocabulary_size(texts),
            "num_texts": len(texts),
        }


def calculate_diversity_stats(texts: List[str]) -> dict:
    """Convenience function to calculate diversity stats."""
    metrics = DiversityMetrics()
    return metrics.get_all_metrics(texts)
