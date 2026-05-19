"""Seed prompt pool management with sampling and versioning."""
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import Difficulty, Domain, SeedPrompt, SourceType


class SeedPromptPool:
    """Manage and sample seed prompts with domain, difficulty, and version support."""

    def __init__(self) -> None:
        self._prompts: Dict[str, SeedPrompt] = {}
        self._version_history: List[Dict[str, str]] = []
        self._sampled_ids: Set[str] = set()

    def add(self, prompt: SeedPrompt) -> None:
        """Add a single seed prompt to the pool."""
        if prompt.id in self._prompts:
            raise ValueError(f"Prompt with id {prompt.id} already exists")
        self._prompts[prompt.id] = prompt

    def add_batch(self, prompts: List[SeedPrompt]) -> None:
        """Add multiple seed prompts to the pool."""
        for prompt in prompts:
            self.add(prompt)

    def get(self, prompt_id: str) -> Optional[SeedPrompt]:
        """Get a prompt by ID."""
        return self._prompts.get(prompt_id)

    def filter_by_domain(self, domain: Domain) -> List[SeedPrompt]:
        """Filter prompts by domain."""
        return [p for p in self._prompts.values() if p.domain == domain]

    def filter_by_difficulty(self, difficulty: Difficulty) -> List[SeedPrompt]:
        """Filter prompts by difficulty level."""
        return [p for p in self._prompts.values() if p.difficulty == difficulty]

    def filter_by_version(self, version: str) -> List[SeedPrompt]:
        """Filter prompts by version tag."""
        return [p for p in self._prompts.values() if p.version == version]

    def filter_by_source(self, source: SourceType) -> List[SeedPrompt]:
        """Filter prompts by source type."""
        return [p for p in self._prompts.values() if p.source == source]

    def sample(
        self,
        count: int,
        domain: Optional[Domain] = None,
        difficulty: Optional[Difficulty] = None,
        source: Optional[SourceType] = None,
        weight_by_quality: bool = True,
        avoid_duplicates: bool = True,
    ) -> List[SeedPrompt]:
        """
        Sample prompts from the pool with optional filtering.

        Args:
            count: Number of prompts to sample
            domain: Optional domain filter
            difficulty: Optional difficulty filter
            source: Optional source filter
            weight_by_quality: If True, higher quality prompts are more likely to be sampled
            avoid_duplicates: If True, avoid sampling previously returned prompts in this session

        Returns:
            List of sampled SeedPrompt objects
        """
        candidates = list(self._prompts.values())

        if domain:
            candidates = [p for p in candidates if p.domain == domain]
        if difficulty:
            candidates = [p for p in candidates if p.difficulty == difficulty]
        if source:
            candidates = [p for p in candidates if p.source == source]
        if avoid_duplicates:
            candidates = [p for p in candidates if p.id not in self._sampled_ids]

        if not candidates:
            return []

        if weight_by_quality:
            weights = [p.quality_score or 0.5 for p in candidates]
            total = sum(weights)
            if total > 0:
                sampled = []
                remaining_candidates = candidates.copy()
                remaining_weights = weights.copy()
                while len(sampled) < count and remaining_candidates:
                    selected = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
                    sampled.append(selected)
                    idx = remaining_candidates.index(selected)
                    remaining_candidates.pop(idx)
                    remaining_weights.pop(idx)
            else:
                sampled = random.sample(candidates, min(count, len(candidates)))
        else:
            sampled = random.sample(candidates, min(count, len(candidates)))

        if avoid_duplicates:
            self._sampled_ids.update(p.id for p in sampled)

        return sampled

    def reset_sampling(self) -> None:
        """Reset the sampled IDs tracking to allow re-sampling prompts."""
        self._sampled_ids.clear()

    def bump_version(self, new_version: str) -> None:
        """Bump the version of all prompts and record in history."""
        self._version_history.append(
            {
                "version": new_version,
                "timestamp": datetime.now().isoformat(),
                "prompt_count": len(self._prompts),
            }
        )
        for prompt in self._prompts.values():
            prompt.version = new_version

    def get_version_history(self) -> List[Dict[str, str]]:
        """Get the version history of the prompt pool."""
        return list(self._version_history)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the prompt pool."""
        domain_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        source_counts = defaultdict(int)
        version_counts = defaultdict(int)

        for prompt in self._prompts.values():
            domain_counts[prompt.domain.value] += 1
            difficulty_counts[prompt.difficulty.value] += 1
            source_counts[prompt.source.value] += 1
            version_counts[prompt.version] += 1

        quality_scores = [p.quality_score for p in self._prompts.values() if p.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

        return {
            "total_prompts": len(self._prompts),
            "by_domain": dict(domain_counts),
            "by_difficulty": dict(difficulty_counts),
            "by_source": dict(source_counts),
            "by_version": dict(version_counts),
            "avg_quality_score": avg_quality,
        }

    def save(self, path: str) -> None:
        """Save the prompt pool to a JSON file."""
        data = {
            "prompts": [p.model_dump(mode="json") for p in self._prompts.values()],
            "version_history": self._version_history,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SeedPromptPool":
        """Load a prompt pool from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pool = cls()
        for prompt_data in data["prompts"]:
            prompt = SeedPrompt.model_validate(prompt_data)
            pool.add(prompt)
        pool._version_history = data.get("version_history", [])
        return pool

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self):
        return iter(self._prompts.values())
