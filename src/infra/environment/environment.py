"""Environment core interface for agent action-observation loop."""

import ast
import asyncio
import json
import logging
import re
import textwrap
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
    VerificationResult,
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
        judge_client: Optional[Any] = None,
        judge_model: Optional[str] = None,
    ):
        self.sandbox_pool = sandbox_pool or SandboxPool()
        self.answer_verifier = AnswerVerifier(self.sandbox_pool)
        self.max_steps = max_steps
        self.judge_client = judge_client
        self.judge_model = judge_model
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal episode state."""
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_step = 0
        self.done = False
        self.history: list[StepResult] = []
        self._test_cases_passed = 0
        self._test_cases_total = 0

    @staticmethod
    def _is_parseable_python(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @classmethod
    def _extract_python_code(cls, answer: Any) -> Any:
        """Extract executable Python code from a final answer if it contains fenced code blocks."""
        if not isinstance(answer, str):
            return answer
        fenced = re.findall(r"```([^\n`]*)\n(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
        cleaned_blocks = [
            textwrap.dedent(block).strip()
            for language, block in fenced
            if block.strip() and language.strip().lower() in {"", "python", "py"}
        ]
        parseable_blocks = [block for block in cleaned_blocks if cls._is_parseable_python(block)]
        if parseable_blocks:
            return "\n\n".join(parseable_blocks)
        if cleaned_blocks:
            return cleaned_blocks[0]

        stripped = textwrap.dedent(answer).strip()
        code_indicators = (
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "if __name__",
            "= ",
        )
        if any(indicator in stripped for indicator in code_indicators) and cls._is_parseable_python(stripped):
            return stripped
        return ""

    @staticmethod
    def _extract_math_answer(answer: Any) -> Any:
        """Extract the final numeric value from a verbose math answer."""
        if not isinstance(answer, str):
            return answer
        text = answer.replace(",", "")
        explicit = re.findall(r"(?:answer is|answer:|=)\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if explicit:
            return explicit[-1]
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else answer

    async def _llm_judge_answer(self, task_domain: str, answer: Any) -> VerificationResult:
        """Use optional LLM-as-Judge for open-ended final answers."""
        if self.judge_client is None:
            return VerificationResult(
                mode=VerificationMode.FORMAT_VALIDATION,
                passed=False,
                score=0.0,
                details={"judge_skipped": "no_judge_client"},
                error="LLM judge client is not configured",
            )

        prompt = f"""You are a strict answer judge for agent trajectory data.

Task domain: {task_domain}

Task prompt:
{self.current_task.get('prompt', '') if self.current_task else ''}

Candidate final answer:
{answer}

Judge whether the candidate answer correctly solves the task.
For code_debug: check root cause, fixed code, and explanation.
For api_orchestration: check endpoint/order/auth/error handling/completeness.
For multi_step_planning: check ordered steps/dependencies/risks/completeness.

Return ONLY valid JSON:
{{
  "passed": true or false,
  "score": 0.0 to 1.0,
  "reason": "short reason",
  "missing_or_wrong": ["item1", "item2"]
}}
"""
        raw = await self.judge_client.achat(
            model=self.judge_model or getattr(self.judge_client, "model", "default"),
            messages=[
                {"role": "system", "content": "You are a strict verification judge. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        try:
            text = raw.strip()
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = json.loads(match.group(0) if match else text)
            score = float(parsed.get("score", 0.0))
            passed = bool(parsed.get("passed", False)) and score >= 0.7
            return VerificationResult(
                mode=VerificationMode.FORMAT_VALIDATION,
                passed=passed,
                score=score,
                details={"llm_judge": parsed, "raw_response": raw},
                error=None if passed else parsed.get("reason", "LLM judge rejected answer"),
            )
        except Exception as e:
            return VerificationResult(
                mode=VerificationMode.FORMAT_VALIDATION,
                passed=False,
                score=0.0,
                details={"raw_response": raw, "parse_error": str(e)},
                error=f"LLM judge response parse failed: {e}",
            )

    @staticmethod
    def _wrap_script_as_solution(code: str) -> str:
        """Wrap executable script-style final code into a solution(**kwargs) function."""
        indented = "\n".join(f"    {line}" if line.strip() else "" for line in str(code).splitlines())
        return (
            "def solution(**kwargs):\n"
            "    globals().update(kwargs)\n"
            f"{indented}\n"
            "    for _name in ('result', 'answer', 'output', 'customer_spending'):\n"
            "        if _name in locals():\n"
            "            return locals()[_name]\n"
            "    return None\n"
        )

    def _find_successful_exec_evidence(self, final_answer: Any, extracted_code: Any) -> Optional[Dict[str, Any]]:
        """Find prior successful exec tool calls as evidence for code_debug final answer."""
        answer_text = str(final_answer)
        extracted_text = str(extracted_code or "")
        for step in reversed(self.history):
            action = step.action
            observation = step.observation
            if not isinstance(action, ToolCallAction) or action.name != "exec" or not observation.success:
                continue
            executed_code = str(action.kwargs.get("code") or (action.args[0] if action.args else ""))
            if not executed_code.strip():
                continue
            executed_functions = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", executed_code)
            shared_function = any(f"def {name}" in answer_text or f"def {name}" in extracted_text for name in executed_functions)
            substantial_overlap = len(set(executed_code.split()) & set(extracted_text.split())) >= 20 if extracted_text else False
            if shared_function or substantial_overlap:
                return {
                    "matched": True,
                    "step": step.step,
                    "executed_functions": executed_functions,
                    "observation_content": observation.content,
                    "observation_metadata": observation.metadata,
                }
        return None

    @staticmethod
    def _infer_function_name(code: Any, answer: Any = None) -> str:
        """Infer target function name from extracted code/final answer for code_debug verification."""
        code_text = str(code or "")
        try:
            tree = ast.parse(code_text)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if "solution" in functions:
                return "solution"
            if functions:
                return functions[-1]
        except SyntaxError:
            pass

        text = f"{answer or ''}\n{code_text}"
        matches = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", text)
        if "solution" in matches:
            return "solution"
        return matches[-1] if matches else "solution"

    @staticmethod
    def _verify_code_debug_answer(answer: Any, expected_output: Any) -> tuple[bool, float, Dict[str, Any]]:
        """Validate code_debug final answers as debugging reports, not always executable solution() functions."""
        text = str(answer)
        expected_keys = list(expected_output.keys()) if isinstance(expected_output, dict) else []
        required = expected_keys or ["root_cause", "fixed_code", "explanation"]
        checks: Dict[str, bool] = {}

        lower = text.lower()
        if "root_cause" in required:
            checks["root_cause"] = any(token in lower for token in ["root cause", "cause", "bug", "issue", "原因", "根因", "问题"])
        if "fixed_code" in required:
            checks["fixed_code"] = bool(re.search(r"```(?:python)?\s*\n.*?```", text, flags=re.IGNORECASE | re.DOTALL) or "def " in text or "class " in text)
        if "explanation" in required:
            checks["explanation"] = any(token in lower for token in ["explanation", "explain", "because", "fix", "修复", "说明", "解释"])

        for key in required:
            checks.setdefault(key, key in lower or key in text)

        passed_count = sum(1 for value in checks.values() if value)
        total = len(checks) or 1
        score = passed_count / total
        return score >= 1.0, score, {"checks": checks, "required_fields": required}

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
        answer_for_verification = action.answer
        verification_result = None

        if task_domain in {"math", "math_reasoning", "arithmetic", "algebra"}:
            verification_mode = VerificationMode.MATH_EQUATION
            answer_for_verification = self._extract_math_answer(action.answer)
            # Extract ground truth from first test case
            if test_cases:
                ground_truth = test_cases[0].get("expected_output", "")
                if isinstance(ground_truth, dict):
                    ground_truth = ground_truth.get("final_answer", "")
                verify_kwargs["ground_truth"] = ground_truth
                verify_kwargs["expression"] = True

        elif task_domain in {"code", "coding", "code_debug", "programming"}:
            expected_output = test_cases[0].get("expected_output", {}) if test_cases else {}
            extracted_code = self._extract_python_code(action.answer)

            inferred_function_name = self._infer_function_name(extracted_code, action.answer)
            code_result = await self.answer_verifier.verify(
                extracted_code,
                mode=VerificationMode.CODE_EXECUTION,
                test_cases=test_cases,
                function_name=inferred_function_name,
            )

            wrapped_function_name = None
            if not code_result.passed:
                error_text = str(code_result.details.get("error", "")) + str(code_result.details.get("test_details", ""))
                if "not defined" in error_text or inferred_function_name == "solution":
                    wrapped_code = self._wrap_script_as_solution(extracted_code)
                    wrapped_function_name = "solution"
                    wrapped_result = await self.answer_verifier.verify(
                        wrapped_code,
                        mode=VerificationMode.CODE_EXECUTION,
                        test_cases=test_cases,
                        function_name=wrapped_function_name,
                    )
                    if wrapped_result.score >= code_result.score:
                        code_result = wrapped_result

            exec_evidence = self._find_successful_exec_evidence(action.answer, extracted_code)
            report_fields = {"root_cause", "fixed_code", "explanation"}
            expected_keys = set(expected_output.keys()) if isinstance(expected_output, dict) else set()
            can_fallback_to_report = bool(expected_keys & report_fields)

            if code_result.passed:
                verification_mode = VerificationMode.CODE_EXECUTION
                details = dict(code_result.details)
                details["inferred_function_name"] = inferred_function_name
                details["wrapped_function_name"] = wrapped_function_name
                verification_result = VerificationResult(
                    mode=code_result.mode,
                    passed=code_result.passed,
                    score=code_result.score,
                    details=details,
                    error=code_result.error,
                )
                answer_for_verification = extracted_code
            elif can_fallback_to_report:
                verification_mode = VerificationMode.FORMAT_VALIDATION
                passed, score, details = self._verify_code_debug_answer(action.answer, expected_output)
                judge_result = await self._llm_judge_answer(task_domain, action.answer)
                details["inferred_function_name"] = inferred_function_name
                details["wrapped_function_name"] = wrapped_function_name
                details["code_execution"] = code_result.details
                details["successful_exec_evidence"] = exec_evidence
                details["llm_judge"] = judge_result.details
                evidence_passed = exec_evidence is not None
                combined_score = min(score, judge_result.score)
                if evidence_passed:
                    combined_score = max(combined_score, min(1.0, (score + judge_result.score + 1.0) / 3))
                fallback_passed = passed and judge_result.passed and evidence_passed
                verification_result = VerificationResult(
                    mode=verification_mode,
                    passed=fallback_passed,
                    score=combined_score,
                    details=details,
                    error=None if fallback_passed else "code execution failed and fallback format/judge/evidence validation did not fully pass",
                )
                answer_for_verification = action.answer
            else:
                verification_mode = VerificationMode.CODE_EXECUTION
                details = dict(code_result.details)
                details["inferred_function_name"] = inferred_function_name
                details["wrapped_function_name"] = wrapped_function_name
                verification_result = VerificationResult(
                    mode=code_result.mode,
                    passed=code_result.passed,
                    score=code_result.score,
                    details=details,
                    error=code_result.error,
                )
                answer_for_verification = extracted_code
        else:
            # For open-ended tasks, use format validation + optional LLM judge
            verification_mode = VerificationMode.FORMAT_VALIDATION
            if test_cases and isinstance(test_cases[0].get("expected_output"), dict):
                verify_kwargs["required_fields"] = list(
                    test_cases[0]["expected_output"].keys()
                )

        # Run verification unless domain-specific logic already produced a result
        if verification_result is None:
            verification_result = await self.answer_verifier.verify(
                answer_for_verification,
                mode=verification_mode,
                **verify_kwargs,
            )
            if task_domain in {"api_orchestration", "multi_step_planning"}:
                judge_result = await self._llm_judge_answer(task_domain, action.answer)
                details = dict(verification_result.details)
                details["llm_judge"] = judge_result.details
                verification_result = VerificationResult(
                    mode=verification_mode,
                    passed=verification_result.passed and judge_result.passed,
                    score=min(verification_result.score, judge_result.score),
                    details=details,
                    error=None if verification_result.passed and judge_result.passed else "format validation or LLM judge failed",
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
                    "answer_for_verification": answer_for_verification,
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
