"""Environment core interface for agent action-observation loop."""

import asyncio
import logging
from typing import Any, Dict, Optional

from .answer_verifier import AnswerVerifier
from .models import (
    Action,
    ActionType,
    EpisodeInfo,
    FinalAnswerAction,
    Observation,
    StepResult,
    ToolCallAction,
    VerificationMode,
)
from .sandbox_pool import SandboxPool

logger = logging.getLogger(__name__)


class Environment:
    """
    Unified execution environment for agent interaction.

    Supports two action types:
    - ToolCallAction: Call a tool/function
    - FinalAnswerAction: Submit a final answer for verification

    The environment handles sandboxed code execution and answer verification.
    """

    def __init__(
        self,
        sandbox_pool: Optional[SandboxPool] = None,
        max_steps: int = 10,
    ):
        self.sandbox_pool = sandbox_pool or SandboxPool()
        self.answer_verifier = AnswerVerifier(self.sandbox_pool)
        self.max_steps = max_steps
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal episode state."""
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_step = 0
        self.done = False
        self.history: list[StepResult] = []
        self._test_cases_passed = 0
        self._test_cases_total = 0

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset environment for a new task.

        Args:
            task: Task dict with prompt, domain, test cases, etc.

        Returns:
            Initial observation with task context
        """
        self._reset_state()
        self.current_task = task

        # Initialize test case counts
        test_cases = task.get("test_cases", [])
        self._test_cases_total = len(test_cases)

        logger.info(
            f"Environment reset: task={task.get('id', 'unknown')}, "
            f"domain={task.get('domain', 'unknown')}, "
            f"test_cases={self._test_cases_total}"
        )

        return {
            "task_id": task.get("id", ""),
            "domain": task.get("domain", ""),
            "prompt": task.get("prompt", ""),
            "max_steps": self.max_steps,
            "test_cases_count": self._test_cases_total,
        }

    async def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: Either ToolCallAction or FinalAnswerAction

        Returns:
            StepResult with observation and done flag
        """
        if self.done:
            return StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error="Episode already complete",
                ),
                done=True,
                step=self.current_step,
                info={"error": "already_done"},
            )

        self.current_step += 1

        # Handle action type
        if isinstance(action, FinalAnswerAction):
            result = await self._handle_final_answer(action)
        elif isinstance(action, ToolCallAction):
            result = await self._handle_tool_call(action)
        else:
            result = StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error=f"Unknown action type: {type(action)}",
                ),
                done=self.current_step >= self.max_steps,
                step=self.current_step,
                info={"error": "unknown_action_type"},
            )

        # Enforce max steps termination
        if self.current_step >= self.max_steps and not result.done:
            result.done = True
            result.observation.error = (
                f"Max steps ({self.max_steps}) reached. "
                f"Task truncated before final answer."
            )
            result.info["truncated"] = True

        self.history.append(result)

        if result.done:
            self.done = True
            logger.info(
                f"Episode complete: steps={self.current_step}, "
                f"passed={self._test_cases_passed}/{self._test_cases_total}"
            )

        return result

    async def _handle_final_answer(self, action: FinalAnswerAction) -> StepResult:
        """Handle final answer submission with verification."""
        task_domain = self.current_task.get("domain", "") if self.current_task else ""
        test_cases = self.current_task.get("test_cases", []) if self.current_task else []

        # Determine verification mode based on task domain
        verification_mode: Optional[VerificationMode] = None
        verify_kwargs = {}

        if task_domain in {"math", "math_reasoning", "arithmetic", "algebra"}:
            verification_mode = VerificationMode.MATH_EQUATION
            # Extract ground truth from first test case
            if test_cases:
                ground_truth = test_cases[0].get("expected_output", "")
                if isinstance(ground_truth, dict):
                    ground_truth = ground_truth.get("final_answer", "")
                verify_kwargs["ground_truth"] = ground_truth
                verify_kwargs["expression"] = True

        elif task_domain in {"code", "coding", "code_debug", "programming"}:
            verification_mode = VerificationMode.CODE_EXECUTION
            verify_kwargs["test_cases"] = test_cases
            # Try to extract function name from the solution code
            # Default to 'solution' if not specified
            verify_kwargs["function_name"] = "solution"
        else:
            # For open-ended tasks, use format validation
            verification_mode = VerificationMode.FORMAT_VALIDATION
            if test_cases and isinstance(test_cases[0].get("expected_output"), dict):
                verify_kwargs["required_fields"] = list(
                    test_cases[0]["expected_output"].keys()
                )

        # Run verification
        verification_result = await self.answer_verifier.verify(
            action.answer,
            mode=verification_mode,
            **verify_kwargs,
        )

        # Update test case tracking
        if verification_result.details.get("tests_passed") is not None:
            self._test_cases_passed = verification_result.details["tests_passed"]
            self._test_cases_total = verification_result.details.get(
                "tests_total", self._test_cases_total
            )

        return StepResult(
            action=action,
            observation=Observation(
                success=verification_result.passed,
                content=verification_result.details,
                error=verification_result.error,
                metadata={
                    "verification_mode": verification_mode,
                    "verification_score": verification_result.score,
                },
            ),
            done=True,
            step=self.current_step,
            info={
                "verification_result": verification_result,
                "episode_info": EpisodeInfo(
                    task_id=self.current_task.get("id", "") if self.current_task else "",
                    domain=task_domain,
                    difficulty=self.current_task.get("difficulty", "")
                    if self.current_task
                    else "",
                    max_steps=self.max_steps,
                    current_step=self.current_step,
                    test_cases_passed=self._test_cases_passed,
                    test_cases_total=self._test_cases_total,
                ),
            },
        )

    async def _handle_tool_call(self, action: ToolCallAction) -> StepResult:
        """Handle tool function call.

        Routes supported tools include:
        - exec: Execute code in sandbox
        - eval: Evaluate a Python expression
        - check_solution: Run solution verification
        """
        tool_name = action.name.lower()

        try:
            if tool_name == "exec":
                # Execute code in sandbox
                return await self._tool_exec(action)
            elif tool_name == "eval":
                # Evaluate expression
                return await self._tool_eval(action)
            elif tool_name == "check_solution":
                # Check a solution against test cases
                return await self._tool_check_solution(action)
            else:
                return StepResult(
                    action=action,
                    observation=Observation(
                        success=False,
                        content=None,
                        error=f"Unknown tool: {tool_name}. "
                        f"Available tools: exec, eval, check_solution",
                    ),
                    done=False,
                    step=self.current_step,
                    info={"error": "unknown_tool"},
                )
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error=str(e),
                ),
                done=False,
                step=self.current_step,
                info={"error": "tool_execution_failed"},
            )

    async def _tool_exec(self, action: ToolCallAction) -> StepResult:
        """Execute code in sandbox."""
        code = action.kwargs.get("code") or (action.args[0] if action.args else None)

        if not code:
            return StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error="No code provided for exec tool",
                ),
                done=False,
                step=self.current_step,
                info={"error": "missing_code"},
            )

        from infra.sandbox.models import SandboxExecutionRequest

        request = SandboxExecutionRequest(
            code=code,
            test_cases=[],
            requirements=action.kwargs.get("requirements", []),
        )

        response = await self.sandbox_pool.execute(request)

        return StepResult(
            action=action,
            observation=Observation(
                success=response.overall_passed,
                content=response.execution_result.output,
                error=response.execution_result.error
                if response.execution_result.error
                else None,
                execution_time=response.execution_result.execution_time,
                metadata={
                    "status": response.execution_result.status.value,
                },
            ),
            done=False,
            step=self.current_step,
            info={"sandbox_response": response},
        )

    async def _tool_eval(self, action: ToolCallAction) -> StepResult:
        """Evaluate a Python expression in sandbox."""
        expr = action.kwargs.get("expr") or (action.args[0] if action.args else None)

        if not expr:
            return StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error="No expression provided for eval tool",
                ),
                done=False,
                step=self.current_step,
                info={"error": "missing_expression"},
            )

        # Wrap expression in print statement
        code = f"print({expr})"

        from infra.sandbox.models import SandboxExecutionRequest

        request = SandboxExecutionRequest(
            code=code,
            test_cases=[],
            requirements=[],
        )

        response = await self.sandbox_pool.execute(request)

        return StepResult(
            action=action,
            observation=Observation(
                success=response.overall_passed,
                content=response.execution_result.output.strip()
                if response.execution_result.output
                else "",
                error=response.execution_result.error,
            ),
            done=False,
            step=self.current_step,
            info={"sandbox_response": response},
        )

    async def _tool_check_solution(self, action: ToolCallAction) -> StepResult:
        """Check a solution against test cases without ending the episode."""
        solution = action.kwargs.get("solution") or (
            action.args[0] if action.args else None)

        if not solution:
            return StepResult(
                action=action,
                observation=Observation(
                    success=False,
                    content=None,
                    error="No solution provided for check_solution tool",
                ),
                done=False,
                step=self.current_step,
                info={"error": "missing_solution"},
            )

        test_cases = self.current_task.get("test_cases", []) if self.current_task else []

        verification_result = await self.answer_verifier.verify(
            solution,
            mode=VerificationMode.CODE_EXECUTION,
            test_cases=test_cases,
        )

        return StepResult(
            action=action,
            observation=Observation(
                success=verification_result.passed,
                content=verification_result.details,
                error=verification_result.error,
            ),
            done=False,
            step=self.current_step,
            info={
                "verification_mode": verification_result.mode,
                "verification_score": verification_result.score,
            },
        )

    def get_episode_info(self) -> EpisodeInfo:
        """Get current episode metadata."""
        return EpisodeInfo(
            task_id=self.current_task.get("id", "") if self.current_task else "",
            domain=self.current_task.get("domain", "") if self.current_task else "",
            difficulty=self.current_task.get("difficulty", "")
            if self.current_task
            else "",
            max_steps=self.max_steps,
            current_step=self.current_step,
            test_cases_passed=self._test_cases_passed,
            test_cases_total=self._test_cases_total,
        )

    async def __aenter__(self) -> "Environment":
        await self.sandbox_pool.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.sandbox_pool.shutdown()
