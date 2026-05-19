"""Data models for EvolInstruct instruction evolution."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvolutionStrategy(str, Enum):
    """Types of evolution strategies."""

    # Deep evolution - increase complexity and depth
    ADD_CONSTRAINTS = "add_constraints"
    DEEPEN_REASONING = "deepen_reasoning"
    CONCRETIZE = "concretize"
    COMPLEX_INPUT = "complex_input"
    ADD_COT = "add_cot"

    # Broad evolution - create diverse variations
    BROAD_MUTATION = "broad_mutation"


class EvolutionMetadata(BaseModel):
    """Metadata for an evolved prompt."""

    generation: int = Field(description="Evolution generation number")
    parent_id: Optional[str] = Field(description="ID of parent prompt")
    strategy: EvolutionStrategy = Field(description="Evolution strategy used")
    evolution_prompt: Optional[str] = Field(description="Prompt used for evolution")
    original_quality_score: float = Field(description="Quality score from parent")
    discriminator_score: Optional[float] = Field(default=None, description="LLM discriminator score")
    discriminator_reason: Optional[str] = Field(default=None, description="Discriminator judgment reason")


class EvolvedPrompt(BaseModel):
    """An evolved prompt with evolution metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(description="The evolved prompt text")
    domain: str = Field(description="Domain of the prompt")
    difficulty: str = Field(description="Difficulty level")
    quality_score: float = Field(description="Quality score of this prompt")
    evolution_metadata: EvolutionMetadata = Field(description="Evolution tracking metadata")
    test_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Test cases")
    validator_code: Optional[str] = Field(default=None, description="Validation code")
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

    def to_seed_dict(self) -> Dict[str, Any]:
        """Convert to seed prompt dict format for SeedPromptPool."""
        return {
            "id": self.id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "test_cases": self.test_cases,
            "validator_code": self.validator_code or "def validate(sol): return True",
            "source": "evolved",
            "version": f"evo-gen-{self.evolution_metadata.generation}",
            "created_at": self.created_at.isoformat(),
            "quality_score": self.quality_score,
            "tags": self.tags + ["evolved", self.evolution_metadata.strategy],
        }


class EvolutionResult(BaseModel):
    """Result of a single evolution step."""

    original_id: str
    evolved_prompt: Optional[EvolvedPrompt] = None
    success: bool
    error: Optional[str] = None
    strategy: EvolutionStrategy


class GenerationStats(BaseModel):
    """Statistics for a single evolution generation."""

    generation: int
    initial_count: int
    evolved_count: int
    after_deduplication_count: int
    final_valid_count: int
    deduplication_filter_rate: float
    discriminator_filter_rate: float
    total_filter_rate: float
    self_bleu: float
    strategies_used: Dict[EvolutionStrategy, int]
    timestamp: datetime = Field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"Generation {self.generation}: "
            f"init={self.initial_count} → "
            f"evolved={self.evolved_count} → "
            f"deduped={self.after_deduplication_count} → "
            f"valid={self.final_valid_count} | "
            f"filter_rate={self.total_filter_rate:.2%} | "
            f"self_bleu={self.self_bleu:.4f}"
        )
