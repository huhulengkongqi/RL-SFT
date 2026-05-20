"""Data models for Environment action-observation loop."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Type of action the agent can take."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class VerificationMode(str, Enum):
    """Verification strategy for answer validation."""

    CODE_EXECUTION = "code_execution"
    MATH_EQUATION = "math_equation"
    FORMAT_VALIDATION = "format_validation"


class BaseAction(BaseModel):
    """Base class for all actions."""

    action_type: ActionType
    thought: Optional[str] = Field(None, description="LLM's reasoning for this action")


class ToolCallAction(BaseAction):
    """Represents a tool/function call action."""

    name: str = Field(..., description="Function name to call")
    args: List[Any] = Field(default_factory=list, description="Positional arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")
    id: Optional[str] = Field(None, description="Unique call ID for tracking")


class FinalAnswerAction(BaseAction):
    """Represents a final answer submission."""

    answer: Any = Field(..., description="Final answer content")
    reasoning_steps: List[str] = Field(
        default_factory=list, description="Step-by-step reasoning"
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


Action = Union[ToolCallAction, FinalAnswerAction]


class Observation(BaseModel):
    """Result of an action execution."""

    success: bool
    content: Any = Field(..., description="Observation content/output")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(0.0, description="Time taken in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Complete result of a single environment step."""

    action: Action
    observation: Observation
    done: bool = Field(..., description="Whether episode is complete")
    step: int = Field(..., description="Step number in episode")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional info"
    )


class EpisodeInfo(BaseModel):
    """Metadata about the current episode/task."""

    task_id: str
    domain: str
    difficulty: str
    max_steps: int
    current_step: int
    test_cases_passed: int
    test_cases_total: int


class VerificationResult(BaseModel):
    """Result of answer verification."""

    mode: VerificationMode
    passed: bool
    score: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PooledContainerStats(BaseModel):
    """Statistics for a pooled container."""

    container_id: str
    use_count: int
    last_used: datetime
    created_at: datetime
    is_busy: bool


class PoolMetrics(BaseModel):
    """Overall SandboxPool metrics."""

    pool_size: int
    active_containers: int
    busy_containers: int
    total_executions: int
    container_reuse_rate: float
    avg_container_lifetime: float
