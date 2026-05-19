"""Data models for task generation."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ValidationError(BaseModel):
    """Validation error details"""
    type: str
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None


class ValidationWarning(BaseModel):
    """Validation warning details"""
    type: str
    message: str
    suggestion: Optional[str] = None


class ValidationReport(BaseModel):
    """Complete validation report for a task"""
    function_signature_valid: bool
    sandbox_execution_passed: bool
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationWarning] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    tested_at: datetime = Field(default_factory=datetime.now)


class Domain(str, Enum):
    """Task domains for SFT training."""

    CODE_DEBUG = "code_debug"
    API_ORCHESTRATION = "api_orchestration"
    MATH_REASONING = "math_reasoning"
    MULTI_STEP_PLANNING = "multi_step_planning"


class Difficulty(str, Enum):
    """Difficulty levels for tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SourceType(str, Enum):
    """Source types of seed prompts."""

    HUMAN_CURATED = "human_curated"
    CRAWLED = "crawled"
    LLM_GENERATED = "llm_generated"


class TaskTestCase(BaseModel):
    """Test case for validating task solutions."""

    input: Any
    expected_output: Any
    is_public: bool = True


class SeedPrompt(BaseModel):
    """Seed prompt model with metadata and validation artifacts."""

    id: str = Field(description="Unique UUID identifier")
    domain: Domain
    difficulty: Difficulty
    prompt: str = Field(min_length=10)
    test_cases: List[TaskTestCase] = Field(min_length=1)
    validator_code: str = Field(min_length=10, description="Python code for validating solutions")
    source: SourceType
    version: str = Field(default="v1.0")
    created_at: datetime = Field(default_factory=datetime.now)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)

    @field_validator("validator_code")
    @classmethod
    def validate_validator_code(cls, v: str) -> str:
        """Ensure validator code contains a function definition."""
        if "def " not in v and "lambda" not in v:
            raise ValueError("validator_code must contain a function definition")
        return v


class Task(SeedPrompt):
    """Generated task model, extending SeedPrompt with generation metadata."""

    generation_trace: Optional[Dict[str, Any]] = None
    parent_seed_id: Optional[str] = None
    generation_mode: str = Field(default="seed_based")
    validation_passed: bool = False
    reference_solution: Optional[str] = Field(
        default=None,
        description="Reference solution code for validation"
    )
    function_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted function calls from reference solution"
    )
    validation_report: Optional['ValidationReport'] = Field(
        default=None,
        description="Validation report for the task"
    )
