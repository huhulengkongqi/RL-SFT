"""Multi-mode answer verification for different task types."""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from infra.sandbox.models import SandboxExecutionRequest

from .models import (
    VerificationMode,
    VerificationResult,
)
from .sandbox_pool import SandboxPool

logger = logging.getLogger(__name__)

try:
    import sympy

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("SymPy not available, math verification will use numeric comparison only")


class BaseVerifier(ABC):
    """Abstract base class for answer verifiers."""

    @abstractmethod
    async def verify(
        self, answer: Any, ground_truth: Optional[Any] = None, **kwargs
    ) -> VerificationResult:
        """Verify an answer against ground truth or validation rules."""
        pass


class CodeExecutionVerifier(BaseVerifier):
    """
    Verifies code solutions by executing them in a sandbox with test cases.

    Generates test harness code from provided test cases and runs it
    in the Docker sandbox environment.
    """

    def __init__(self, sandbox_pool: SandboxPool):
        self.sandbox_pool = sandbox_pool

    async def verify(
        self,
        answer: Any,
        ground_truth: Optional[Any] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        function_name: str = "solution",
        **kwargs,
    ) -> VerificationResult:
        """
        Verify code by running test cases.

        Args:
            answer: The code to verify (string or dict with 'fixed_code' key)
            ground_truth: Optional reference solution code
            test_cases: List of test cases with 'input' and 'expected_output'
            function_name: Name of the function to test in the code

        Returns:
            VerificationResult with pass/fail status and test details
        """
        # Extract code from answer (handle both string and dict formats)
        if isinstance(answer, dict):
            code = answer.get("fixed_code") or answer.get("code") or answer.get("solution")
        elif isinstance(answer, str):
            code = answer
        else:
            code = str(answer)

        if not code:
            return VerificationResult(
                mode=VerificationMode.CODE_EXECUTION,
                passed=False,
                score=0.0,
                error="No code provided for verification",
                details={"error": "empty_code"},
            )

        test_cases = test_cases or []

        # Generate test harness code
        test_harness = self._generate_test_harness(code, test_cases, function_name)

        # Create execution request
        request = SandboxExecutionRequest(
            code=test_harness,
            test_cases=[],  # Test cases already in the harness
            requirements=[],
        )

        # Execute in sandbox
        response = await self.sandbox_pool.execute(request)

        # Parse results
        passed_count = 0
        total_count = len(test_cases)
        test_details = []

        if response.overall_passed:
            try:
                # Parse JSON output from test harness
                if response.execution_result.output:
                    output_lines = response.execution_result.output.strip().split("\n")
                    for line in output_lines:
                        try:
                            result = json.loads(line)
                            if isinstance(result, dict) and "passed" in result:
                                test_details.append(result)
                                if result["passed"]:
                                    passed_count += 1
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                logger.debug(f"Error parsing test results: {e}")

        # If no results parsed but execution succeeded, assume basic syntax check passed
        if not test_details and response.execution_result.status.value == "success":
            if total_count == 0:
                # No test cases, just check syntax
                passed_count = 1
                total_count = 1
                test_details.append({"passed": True, "message": "Syntax check passed"})

        score = passed_count / total_count if total_count > 0 else 0.0

        return VerificationResult(
            mode=VerificationMode.CODE_EXECUTION,
            passed=score >= 1.0,
            score=score,
            details={
                "tests_passed": passed_count,
                "tests_total": total_count,
                "test_details": test_details,
                "execution_status": response.execution_result.status.value,
                "output": response.execution_result.output,
                "error": response.execution_result.error,
            },
        )

    def _generate_test_harness(
        self, code: str, test_cases: List[Dict[str, Any]], function_name: str
    ) -> str:
        """Generate a test harness that runs all test cases and outputs JSON results."""
        # Escape any triple quotes in the code
        code_escaped = code.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")

        harness = f"""
import json
import sys

# User's solution code
{code}

# Test runner
def run_tests():
    test_results = []
"""

        for i, tc in enumerate(test_cases):
            input_val = tc.get("input")
            expected = tc.get("expected_output")

            # Handle different input formats
            if isinstance(input_val, dict) and "args" in input_val:
                args = input_val.get("args", [])
                kwargs = input_val.get("kwargs", {})
                call_str = f"{function_name}(*{args}, **{kwargs})"
            elif isinstance(input_val, dict):
                args_str = ", ".join(f"{k}={v!r}" for k, v in input_val.items())
                call_str = f"{function_name}({args_str})"
            else:
                call_str = f"{function_name}({input_val!r})"

            harness += f"""
    try:
        result = {call_str}
        expected = {expected!r}
        passed = result == expected
        test_results.append({{
            "test_index": {i},
            "passed": passed,
            "result": str(result) if not isinstance(result, (int, float, str, bool, type(None))) else result,
            "expected": str(expected) if not isinstance(expected, (int, float, str, bool, type(None))) else expected,
        }})
    except Exception as e:
        test_results.append({{
            "test_index": {i},
            "passed": False,
            "error": str(e),
        }})
"""

        harness += f"""
    for result in test_results:
        print(json.dumps(result))

if __name__ == "__main__":
    run_tests()
"""
        return harness


class MathEquationVerifier(BaseVerifier):
    """
    Verifies mathematical answers using SymPy symbolic equivalence
    or numeric comparison with tolerance.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    async def verify(
        self,
        answer: Any,
        ground_truth: Optional[Any] = None,
        expression: bool = False,
        **kwargs,
    ) -> VerificationResult:
        """
        Verify a mathematical answer.

        Args:
            answer: The answer to verify (number, string, or expression)
            ground_truth: The correct answer for comparison
            expression: If True, attempt symbolic equivalence check
        """
        if ground_truth is None:
            return VerificationResult(
                mode=VerificationMode.MATH_EQUATION,
                passed=False,
                score=0.0,
                error="No ground truth provided for math verification",
                details={"error": "missing_ground_truth"},
            )

        try:
            # Convert both to strings for parsing
            answer_str = str(answer).strip()
            truth_str = str(ground_truth).strip()

            # Exact string match first (fast path)
            if answer_str == truth_str:
                return VerificationResult(
                    mode=VerificationMode.MATH_EQUATION,
                    passed=True,
                    score=1.0,
                    details={"match_type": "exact_string"},
                )

            # Try numeric comparison
            try:
                answer_num = float(answer_str)
                truth_num = float(truth_str)
                diff = abs(answer_num - truth_num)
                if diff <= self.tolerance:
                    return VerificationResult(
                        mode=VerificationMode.MATH_EQUATION,
                        passed=True,
                        score=1.0,
                        details={
                            "match_type": "numeric",
                            "difference": diff,
                            "tolerance": self.tolerance,
                        },
                    )
            except (ValueError, TypeError):
                pass

            # Try symbolic equivalence with SymPy if available
            if SYMPY_AVAILABLE and expression:
                try:
                    answer_expr = sympy.sympify(answer_str)
                    truth_expr = sympy.sympify(truth_str)

                    # Simplify and check equality
                    simplified_diff = sympy.simplify(answer_expr - truth_expr)
                    if simplified_diff == 0:
                        return VerificationResult(
                            mode=VerificationMode.MATH_EQUATION,
                            passed=True,
                            score=1.0,
                            details={"match_type": "symbolic_equivalence"},
                        )
                except Exception as e:
                    logger.debug(f"SymPy verification failed: {e}")

            # If all checks failed
            return VerificationResult(
                mode=VerificationMode.MATH_EQUATION,
                passed=False,
                score=0.0,
                details={
                    "answer": answer_str,
                    "expected": truth_str,
                    "match_type": "none",
                },
            )

        except Exception as e:
            return VerificationResult(
                mode=VerificationMode.MATH_EQUATION,
                passed=False,
                score=0.0,
                error=str(e),
                details={"error": "verification_exception", "message": str(e)},
            )


class FormatValidationVerifier(BaseVerifier):
    """
    Verifies answers against format rules:
    - Required fields for dict outputs
    - Regex pattern matching for strings
    - Type validation
    """

    async def verify(
        self,
        answer: Any,
        ground_truth: Optional[Any] = None,
        required_fields: Optional[List[str]] = None,
        regex_pattern: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ) -> VerificationResult:
        """
        Verify answer format.

        Args:
            answer: The answer to verify
            ground_truth: Optional reference for structure comparison
            required_fields: List of required field names for dict answers
            regex_pattern: Regex pattern to match string answers
            expected_type: Expected type name ('dict', 'str', 'int', etc.)
        """
        checks_passed = 0
        total_checks = 0
        details = {}

        # Type check
        if expected_type:
            total_checks += 1
            type_mapping = {
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "bool": bool,
            }
            expected_pytype = type_mapping.get(expected_type.lower())
            if expected_pytype and isinstance(answer, expected_pytype):
                checks_passed += 1
                details["type_check"] = "passed"
            else:
                details["type_check"] = f"failed (expected {expected_type}, got {type(answer).__name__})"

        # Required fields check
        if required_fields and isinstance(answer, dict):
            total_checks += len(required_fields)
            missing_fields = []
            for field in required_fields:
                if field in answer:
                    checks_passed += 1
                else:
                    missing_fields.append(field)

            if missing_fields:
                details["required_fields"] = f"missing: {missing_fields}"
            else:
                details["required_fields"] = "passed"

        # Regex check
        if regex_pattern and isinstance(answer, str):
            total_checks += 1
            if re.match(regex_pattern, answer):
                checks_passed += 1
                details["regex_check"] = "passed"
            else:
                details["regex_check"] = f"failed (pattern: {regex_pattern})"

        score = checks_passed / total_checks if total_checks > 0 else 1.0

        return VerificationResult(
            mode=VerificationMode.FORMAT_VALIDATION,
            passed=score >= 1.0,
            score=score,
            details=details,
        )


class AnswerVerifier:
    """
    Main answer verifier that dispatches to appropriate verification mode.

    Supports:
    - CODE_EXECUTION: Run code with test cases in sandbox
    - MATH_EQUATION: Symbolic or numeric math verification
    - FORMAT_VALIDATION: Structural and format checking
    """

    def __init__(self, sandbox_pool: SandboxPool):
        self.code_verifier = CodeExecutionVerifier(sandbox_pool)
        self.math_verifier = MathEquationVerifier()
        self.format_verifier = FormatValidationVerifier()

    async def verify(
        self,
        answer: Any,
        mode: VerificationMode,
        **kwargs,
    ) -> VerificationResult:
        """
        Verify an answer using the specified mode.

        Args:
            answer: The answer to verify
            mode: Which verification mode to use
            **kwargs: Mode-specific parameters
        """
        if mode == VerificationMode.CODE_EXECUTION:
            return await self.code_verifier.verify(answer, **kwargs)
        elif mode == VerificationMode.MATH_EQUATION:
            return await self.math_verifier.verify(answer, **kwargs)
        elif mode == VerificationMode.FORMAT_VALIDATION:
            return await self.format_verifier.verify(answer, **kwargs)
        else:
            raise ValueError(f"Unknown verification mode: {mode}")

    def auto_verify(
        self,
        answer: Any,
        task_domain: str,
        **kwargs,
    ) -> VerificationResult:
        """
        Automatically select verification mode based on task domain.

        Args:
            answer: The answer to verify
            task_domain: Task domain (math, code, reasoning, etc.)
        """
        # Auto-select mode based on task type
        if task_domain in {"math", "math_reasoning", "arithmetic"}:
            return self.verify(answer, VerificationMode.MATH_EQUATION, **kwargs)
        elif task_domain in {"code", "coding", "code_debug", "programming"}:
            return self.verify(answer, VerificationMode.CODE_EXECUTION, **kwargs)
        else:
            # For open-ended tasks, just do format validation
            return self.verify(answer, VerificationMode.FORMAT_VALIDATION, **kwargs)
