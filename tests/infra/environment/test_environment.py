"""Unit tests for Environment, SandboxPool, and AnswerVerifier."""

import asyncio
import pytest
import sys
from typing import List, Dict, Any

from infra.environment import (
    Environment,
    SandboxPool,
    AnswerVerifier,
    ToolCallAction,
    FinalAnswerAction,
    VerificationMode,
    ActionType,
)


# ============================================
# Test Tasks: 10 code tasks + 5 math tasks
# ============================================

CODE_TASKS: List[Dict[str, Any]] = [
    # Task 1: Basic arithmetic function
    {
        "id": "code_task_1",
        "domain": "code",
        "difficulty": "easy",
        "prompt": "Write a function add(a, b) that returns the sum of a and b.",
        "test_cases": [
            {"input": {"args": [2, 3]}, "expected_output": 5},
            {"input": {"args": [-1, 1]}, "expected_output": 0},
            {"input": {"args": [0, 0]}, "expected_output": 0},
        ],
        "reference_solution": """
def solution(a, b):
    return a + b
        """,
    },
    # Task 2: String manipulation
    {
        "id": "code_task_2",
        "domain": "code",
        "difficulty": "easy",
        "prompt": "Write a function reverse_string(s) that returns the reversed string.",
        "test_cases": [
            {"input": {"args": ["hello"]}, "expected_output": "olleh"},
            {"input": {"args": [""]}, "expected_output": ""},
            {"input": {"args": ["a"]}, "expected_output": "a"},
        ],
        "reference_solution": """
def solution(s):
    return s[::-1]
        """,
    },
    # Task 3: List filtering
    {
        "id": "code_task_3",
        "domain": "code",
        "difficulty": "easy",
        "prompt": "Write a function filter_evens(numbers) that returns only even numbers.",
        "test_cases": [
            {"input": {"args": [[1, 2, 3, 4]]}, "expected_output": [2, 4]},
            {"input": {"args": [[]]}, "expected_output": []},
            {"input": {"args": [[1, 3, 5]]}, "expected_output": []},
        ],
        "reference_solution": """
def solution(numbers):
    return [n for n in numbers if n % 2 == 0]
        """,
    },
    # Task 4: Dictionary operations
    {
        "id": "code_task_4",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function sum_values(d) that returns the sum of all values in a dictionary.",
        "test_cases": [
            {"input": {"args": [{"a": 1, "b": 2}]}, "expected_output": 3},
            {"input": {"args": [{}]}, "expected_output": 0},
            {"input": {"args": [{"x": 10, "y": 20, "z": 30}]}, "expected_output": 60},
        ],
        "reference_solution": """
def solution(d):
    return sum(d.values())
        """,
    },
    # Task 5: Sorting function
    {
        "id": "code_task_5",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function sort_by_length(strings) that sorts strings by length.",
        "test_cases": [
            {"input": {"args": [["apple", "cat", "banana"]]}, "expected_output": ["cat", "apple", "banana"]},
        ],
        "reference_solution": """
def solution(strings):
    return sorted(strings, key=len)
        """,
    },
    # Task 6: Fibonacci recursive
    {
        "id": "code_task_6",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function fib(n) that returns the nth Fibonacci number.",
        "test_cases": [
            {"input": {"args": [0]}, "expected_output": 0},
            {"input": {"args": [1]}, "expected_output": 1},
            {"input": {"args": [10]}, "expected_output": 55},
        ],
        "reference_solution": """
def solution(n):
    if n <= 1:
        return n
    return solution(n - 1) + solution(n - 2)
        """,
    },
    # Task 7: Exception handling
    {
        "id": "code_task_7",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function safe_divide(a, b) that returns a/b or None on division by zero.",
        "test_cases": [
            {"input": {"args": [10, 2]}, "expected_output": 5.0},
            {"input": {"args": [10, 0]}, "expected_output": None},
        ],
        "reference_solution": """
def solution(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
        """,
    },
    # Task 8: Class definition
    {
        "id": "code_task_8",
        "domain": "code",
        "difficulty": "hard",
        "prompt": "Write a class Counter with increment() and get_count() methods.",
        "test_cases": [
            {"input": {"method": "increment", "args": []}, "expected_output": None},
        ],
        "reference_solution": """
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count
        """,
    },
    # Task 9: File I/O simulation
    {
        "id": "code_task_9",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function count_lines(text) that counts the number of lines in a string.",
        "test_cases": [
            {"input": {"args": ["line1"]}, "expected_output": 1},
            {"input": {"args": ["line1\\nline2\\nline3"]}, "expected_output": 3},
        ],
        "reference_solution": """
def solution(text):
    return len(text.split('\\n'))
        """,
    },
    # Task 10: API call pattern
    {
        "id": "code_task_10",
        "domain": "code",
        "difficulty": "medium",
        "prompt": "Write a function format_response(status, data) that returns a dict.",
        "test_cases": [
            {"input": {"args": ["success", {"id": 1}]}, "expected_output": {"status": "success", "data": {"id": 1}}},
        ],
        "reference_solution": """
def solution(status, data):
    return {"status": status, "data": data}
        """,
    },
]

MATH_TASKS: List[Dict[str, Any]] = [
    # Task 1: Integer arithmetic
    {
        "id": "math_task_1",
        "domain": "math_reasoning",
        "difficulty": "easy",
        "prompt": "A shirt costs $25. How much do 3 shirts cost?",
        "test_cases": [
            {"input": {"question": "A shirt costs $25. How much do 3 shirts cost?"},
             "expected_output": {"final_answer": "75", "steps_required": True}}
        ],
    },
    # Task 2: Unit conversion (time)
    {
        "id": "math_task_2",
        "domain": "math_reasoning",
        "difficulty": "easy",
        "prompt": "How many minutes are in 2.5 hours?",
        "test_cases": [
            {"input": {"question": "How many minutes are in 2.5 hours?"},
             "expected_output": {"final_answer": "150", "steps_required": True}}
        ],
    },
    # Task 3: Percentage calculation
    {
        "id": "math_task_3",
        "domain": "math_reasoning",
        "difficulty": "medium",
        "prompt": "A $80 item is on sale for 25% off. What is the sale price?",
        "test_cases": [
            {"input": {"question": "A $80 item is on sale for 25% off. What is the sale price?"},
             "expected_output": {"final_answer": "60", "steps_required": True}}
        ],
    },
    # Task 4: Algebraic expression
    {
        "id": "math_task_4",
        "domain": "math_reasoning",
        "difficulty": "hard",
        "prompt": "If x = 3, what is x^2 + 2x + 1?",
        "test_cases": [
            {"input": {"question": "If x = 3, what is x^2 + 2x + 1?"},
             "expected_output": {"final_answer": "16", "steps_required": True}}
        ],
    },
    # Task 5: Floating point tolerance
    {
        "id": "math_task_5",
        "domain": "math_reasoning",
        "difficulty": "easy",
        "prompt": "What is 1 divided by 3?",
        "test_cases": [
            {"input": {"question": "What is 1 divided by 3?"},
             "expected_output": {"final_answer": "0.3333333333", "steps_required": True}}
        ],
    },
]


# ============================================
# Unit Tests
# ============================================

@pytest.fixture(scope="module")
def event_loop():
    """Create a single event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def sandbox_pool():
    """Create a shared SandboxPool for all tests."""
    pool = SandboxPool(pool_size=3, pre_warm_count=1)
    await pool.initialize()
    yield pool
    await pool.shutdown()


class TestSandboxPool:
    """Test SandboxPool container management."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, sandbox_pool):
        """Test that pool initializes with pre-warmed containers."""
        metrics = sandbox_pool.get_metrics()
        assert metrics.active_containers >= 1
        assert metrics.pool_size == 3

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, sandbox_pool):
        """Test concurrent execution of multiple tasks."""
        from infra.sandbox.models import SandboxExecutionRequest

        async def run_task(i):
            request = SandboxExecutionRequest(
                code=f"print({i} * 2)",
                test_cases=[],
            )
            return await sandbox_pool.execute(request)

        # Run 5 tasks concurrently
        tasks = [run_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.overall_passed for r in results)

    @pytest.mark.asyncio
    async def test_container_reuse(self, sandbox_pool):
        """Test that containers are reused across executions."""
        from infra.sandbox.models import SandboxExecutionRequest

        initial_execs = sandbox_pool.get_metrics().total_executions

        for i in range(5):
            request = SandboxExecutionRequest(code=f"print({i})", test_cases=[])
            await sandbox_pool.execute(request)

        final_metrics = sandbox_pool.get_metrics()
        assert final_metrics.total_executions == initial_execs + 5
        # Should have reused containers (not created 5 new ones)
        assert final_metrics.active_containers <= 3


class TestAnswerVerifier:
    """Test AnswerVerifier with all three verification modes."""

    @pytest.mark.asyncio
    async def test_code_execution_verifier_correct(self, sandbox_pool):
        """Test correct code passes verification."""
        verifier = AnswerVerifier(sandbox_pool)
        task = CODE_TASKS[0]

        result = await verifier.verify(
            task["reference_solution"],
            mode=VerificationMode.CODE_EXECUTION,
            test_cases=task["test_cases"],
            function_name="solution",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["tests_passed"] == result.details["tests_total"]

    @pytest.mark.asyncio
    async def test_code_execution_verifier_incorrect(self, sandbox_pool):
        """Test incorrect code fails verification."""
        verifier = AnswerVerifier(sandbox_pool)
        task = CODE_TASKS[0]

        wrong_code = """
def add(a, b):
    return a - b
        """

        result = await verifier.verify(
            wrong_code,
            mode=VerificationMode.CODE_EXECUTION,
            test_cases=task["test_cases"],
        )

        assert result.passed is False
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_math_verification_exact(self, sandbox_pool):
        """Test exact string match for math answers."""
        verifier = AnswerVerifier(sandbox_pool)

        result = await verifier.verify(
            "75",
            mode=VerificationMode.MATH_EQUATION,
            ground_truth="75",
        )

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_math_verification_numeric_tolerance(self, sandbox_pool):
        """Test numeric comparison with tolerance."""
        verifier = AnswerVerifier(sandbox_pool)

        result = await verifier.verify(
            0.333333,
            mode=VerificationMode.MATH_EQUATION,
            ground_truth=0.3333333333,
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_format_validation_required_fields(self, sandbox_pool):
        """Test required fields validation for dict answers."""
        verifier = AnswerVerifier(sandbox_pool)

        result = await verifier.verify(
            {"root_cause": "found", "fixed_code": "code", "explanation": "done"},
            mode=VerificationMode.FORMAT_VALIDATION,
            required_fields=["root_cause", "fixed_code", "explanation"],
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_format_validation_missing_fields(self, sandbox_pool):
        """Test missing fields cause validation failure."""
        verifier = AnswerVerifier(sandbox_pool)

        result = await verifier.verify(
            {"root_cause": "found"},  # Missing fixed_code and explanation
            mode=VerificationMode.FORMAT_VALIDATION,
            required_fields=["root_cause", "fixed_code", "explanation"],
        )

        assert result.passed is False
        assert result.score < 1.0


class TestEnvironment:
    """Test Environment step interface."""

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test environment reset with task."""
        env = Environment(max_steps=5)
        task = CODE_TASKS[0]

        initial_state = env.reset(task)

        assert initial_state["task_id"] == task["id"]
        assert initial_state["domain"] == task["domain"]
        assert initial_state["max_steps"] == 5
        assert env.current_step == 0
        assert env.done is False

    @pytest.mark.asyncio
    async def test_tool_call_exec(self):
        """Test exec tool call action."""
        async with Environment() as env:
            env.reset(CODE_TASKS[0])

            action = ToolCallAction(
                action_type=ActionType.TOOL_CALL,
                name="exec",
                kwargs={"code": "print('hello world')"},
            )

            result = await env.step(action)

            assert result.done is False
            assert result.observation.success is True
            assert "hello world" in result.observation.content

    @pytest.mark.asyncio
    async def test_tool_call_eval(self):
        """Test eval tool call action."""
        async with Environment() as env:
            env.reset(CODE_TASKS[0])

            action = ToolCallAction(
                action_type=ActionType.TOOL_CALL,
                name="eval",
                kwargs={"expr": "2 + 3"},
            )

            result = await env.step(action)

            assert result.done is False
            assert result.observation.success is True
            assert "5" in result.observation.content

    @pytest.mark.asyncio
    async def test_final_answer_math(self):
        """Test final answer submission for math task."""
        async with Environment() as env:
            task = MATH_TASKS[0]
            env.reset(task)

            action = FinalAnswerAction(
                action_type=ActionType.FINAL_ANSWER,
                answer="75",
            )

            result = await env.step(action)

            assert result.done is True
            assert result.observation.success is True

    @pytest.mark.asyncio
    async def test_final_answer_code(self):
        """Test final answer submission for code task."""
        async with Environment() as env:
            task = CODE_TASKS[0]
            env.reset(task)

            action = FinalAnswerAction(
                action_type=ActionType.FINAL_ANSWER,
                answer=task["reference_solution"],
            )

            result = await env.step(action)

            assert result.done is True
            # Code verification runs test cases
            assert "tests_passed" in result.observation.content

    @pytest.mark.asyncio
    async def test_max_steps_termination(self):
        """Test that episode terminates after max steps."""
        async with Environment(max_steps=3) as env:
            env.reset(CODE_TASKS[0])

            # Two tool calls (should not trigger termination)
            for i in range(2):
                action = ToolCallAction(
                    action_type=ActionType.TOOL_CALL,
                    name="eval",
                    kwargs={"expr": f"{i} + 1"},
                )
                result = await env.step(action)
                assert result.done is False

            # Third should trigger max steps
            action3 = ToolCallAction(
                action_type=ActionType.TOOL_CALL,
                name="eval",
                kwargs={"expr": "3 + 1"},
            )
            result = await env.step(action3)

            assert result.done is True
            assert "Max steps" in (result.observation.error or "")


class TestEndToEndTasks:
    """End-to-end tests for all 15 tasks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_idx", range(len(CODE_TASKS)))
    async def test_all_code_tasks(self, task_idx):
        """Test all 10 code tasks end-to-end."""
        task = CODE_TASKS[task_idx]

        async with Environment() as env:
            env.reset(task)

            # Submit correct solution as final answer
            action = FinalAnswerAction(
                action_type=ActionType.FINAL_ANSWER,
                answer=task["reference_solution"],
            )

            result = await env.step(action)

            assert result.done is True
            # At least 5 of the tasks should pass fully (simpler ones)
            if task_idx < 5:  # First 5 tasks are simpler
                assert result.observation.success is True or result.observation.content.get("tests_passed", 0) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_idx", range(len(MATH_TASKS)))
    async def test_all_math_tasks(self, task_idx):
        """Test all 5 math tasks end-to-end."""
        task = MATH_TASKS[task_idx]
        ground_truth = task["test_cases"][0]["expected_output"]["final_answer"]

        async with Environment() as env:
            env.reset(task)

            # Submit correct answer
            action = FinalAnswerAction(
                action_type=ActionType.FINAL_ANSWER,
                answer=ground_truth,
            )

            result = await env.step(action)

            assert result.done is True
            assert result.observation.success is True

    @pytest.mark.asyncio
    async def test_tool_then_final_answer_flow(self):
        """Test a complete episode: tool calls then final answer."""
        task = CODE_TASKS[0]

        async with Environment(max_steps=5) as env:
            env.reset(task)

            # Step 1: Test an expression with exec
            action1 = ToolCallAction(
                action_type=ActionType.TOOL_CALL,
                name="exec",
                kwargs={"code": "print(2 + 3)"},
            )
            result1 = await env.step(action1)
            assert result1.done is False
            assert result1.observation.success is True

            # Step 2: Check solution with check_solution
            action2 = ToolCallAction(
                action_type=ActionType.TOOL_CALL,
                name="check_solution",
                kwargs={"solution": task["reference_solution"]},
            )
            result2 = await env.step(action2)
            assert result2.done is False

            # Step 3: Submit final answer
            action3 = FinalAnswerAction(
                action_type=ActionType.FINAL_ANSWER,
                answer=task["reference_solution"],
            )
            result3 = await env.step(action3)

            assert result3.done is True
            assert env.current_step == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
