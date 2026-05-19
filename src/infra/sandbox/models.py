"""
Models for sandbox execution and validation results.
"""

import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class ExecutionStatus(Enum):
    """Execution status codes"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class ValidationErrorType(Enum):
    """Types of validation errors"""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    ASSERTION_FAILURE = "assertion_failure"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    UNKNOWN = "unknown"


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox"""
    status: ExecutionStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None  # in MB
    traceback: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    files_created: List[str] = None
    files_modified: List[str] = None

    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.files_modified is None:
            self.files_modified = []

    @classmethod
    def success(cls, output: str, execution_time: float, memory_usage: Optional[float] = None):
        return cls(
            status=ExecutionStatus.SUCCESS,
            output=output,
            execution_time=execution_time,
            memory_usage=memory_usage
        )

    @classmethod
    def failure(cls, error: str, execution_time: float, traceback: Optional[str] = None):
        return cls(
            status=ExecutionStatus.FAILURE,
            error=error,
            execution_time=execution_time,
            traceback=traceback
        )

    @classmethod
    def timeout(cls, execution_time: float):
        return cls(
            status=ExecutionStatus.TIMEOUT,
            error="Execution timed out",
            execution_time=execution_time
        )

    @classmethod
    def error(cls, error: str, execution_time: float):
        return cls(
            status=ExecutionStatus.ERROR,
            error=error,
            execution_time=execution_time
        )


@dataclass
class TestCaseExecution:
    """Execution result for a single test case"""
    __test__ = False  # Tell pytest this is not a test class

    passed: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    test_case_index: int = 0
    error_type: Optional[ValidationErrorType] = None

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    image: str = "python:3.11-slim"
    timeout: int = 30  # seconds
    memory_limit: int = 512  # MB
    disk_limit: int = 1024  # MB
    network_access: bool = False
    allowed_imports: List[str] = None

    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = [
                'json', 'os', 'sys', 'datetime', 'math', 'random',
                'collections', 'itertools', 'base64', 'urllib', 'requests'
            ]


class SandboxExecutionRequest(BaseModel):
    """Request for sandbox execution"""
    code: str = Field(description="Python code to execute")
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    config: SandboxConfig = Field(default_factory=SandboxConfig)
    requirements: List[str] = Field(default_factory=list)


class SandboxExecutionResponse(BaseModel):
    """Response from sandbox execution"""
    execution_result: ExecutionResult
    test_results: List[TestCaseExecution] = Field(default_factory=list)
    passed_tests: int = 0
    total_tests: int = 0
    overall_passed: bool = False
    execution_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None