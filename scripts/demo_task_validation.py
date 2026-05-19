"""
Demo script for task validation functionality.
"""

import asyncio
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import Task, TaskTestCase, ValidationReport, SourceType
from src.agent_sft.function_registry import FunctionRegistry
from src.agent_sft.task_generator.validator import TaskValidator
from src.infra.sandbox.execution_manager import SandboxExecutor
from src.infra.sandbox.models import SandboxConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_function_validation():
    """Demo function signature validation."""
    print("\n=== Function Signature Validation Demo ===")

    # Create test tasks
    tasks = [
        # Valid task
        Task(
            id="valid-task",
            domain="code_debug",
            difficulty="easy",
            prompt="Calculate the sum of a list",
            test_cases=[TaskTestCase(input=[1, 2, 3], expected_output=6)],
            validator_code="def validate(result): return isinstance(result, int)",
            source=SourceType.HUMAN_CURATED
        ),
        # Task with invalid function
        Task(
            id="invalid-task",
            domain="code_debug",
            difficulty="easy",
            prompt="Use a non-existent function",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        ),
        # Task with valid but complex functions
        Task(
            id="complex-task",
            domain="code_debug",
            difficulty="medium",
            prompt="Process JSON data",
            test_cases=[TaskTestCase(input='{"key": "value"}', expected_output="value")],
            validator_code="def validate(result): return result == 'value'",
            source=SourceType.HUMAN_CURATED
        )
    ]

    # Add reference solutions
    tasks[0].reference_solution = """
def sum_list(numbers):
    return sum(numbers)
result = sum_list([1, 2, 3])
"""

    tasks[1].reference_solution = """
# This uses a non-existent function
invalid_function()
"""

    tasks[2].reference_solution = """
import json
data = json.loads('{"key": "value"}')
result = data["key"]
"""

    # Initialize validator without sandbox (for demo purposes)
    registry = FunctionRegistry()
    validator = TaskValidator(
        function_registry=registry,
        enable_sandbox=False
    )

    # Validate tasks
    for task in tasks:
        print(f"\nValidating task: {task.id}")
        print(f"Prompt: {task.prompt}")

        # Parse functions
        function_calls = validator.parser.parse_functions(task)
        print(f"Found {len(function_calls)} function calls:")
        for call in function_calls:
            print(f"  - {call.name}() at line {call.line_number}")

        # Validate signatures
        signature_result = validator.parser.validate_signatures(function_calls)
        print(f"Function signatures valid: {signature_result.valid}")

        if signature_result.errors:
            print("Errors:")
            for error in signature_result.errors:
                print(f"  - {error.message}")

        if signature_result.suggestions:
            print("Suggestions:")
            for suggestion in signature_result.suggestions:
                print(f"  - {suggestion}")


async def demo_sandbox_validation():
    """Demo sandbox execution validation (requires Docker)."""
    print("\n=== Sandbox Validation Demo ===")

    # Check if Docker is available
    try:
        import docker
        docker_client = docker.from_env()
        docker_client.ping()
        print("Docker is available - proceeding with sandbox demo")

        # Create task with reference solution
        task = Task(
            id="sandbox-task",
            domain="api_orchestration",
            difficulty="medium",
            prompt="Fetch data from API and process",
            test_cases=[
                TaskTestCase(
                    input={"url": "https://api.example.com/data"},
                    expected_output={"status": "success"}
                )
            ],
            validator_code="""
def validate(result):
    return isinstance(result, dict) and result.get("status") == "success"
""",
            source=SourceType.HUMAN_CURATED
        )

        task.reference_solution = """
import requests

def fetch_and_process(url):
    response = requests.get(url, timeout=10)
    data = response.json()
    result = {
        "status": "success",
        "data": data
    }
    return result

url = "https://api.example.com/data"
result = fetch_and_process(url)
"""

        # Initialize sandbox
        config = SandboxConfig(
            timeout=30,
            memory_limit=256,
            network_access=True,
            allowed_imports=["requests", "json", "os"]
        )
        sandbox = SandboxExecutor(config)

        # Initialize validator with sandbox
        validator = TaskValidator(
            sandbox_executor=sandbox,
            enable_sandbox=True
        )

        # Validate task
        report = await validator.validate_task(task)
        print(f"\nValidation Report:")
        print(f"Function signatures valid: {report.function_signature_valid}")
        print(f"Sandbox execution passed: {report.sandbox_execution_passed}")
        print(f"Execution time: {report.execution_time:.2f}s")
        print(f"Memory usage: {report.memory_usage} MB")

        if report.errors:
            print("\nErrors:")
            for error in report.errors:
                print(f"  - {error.message}")

        sandbox.close()

    except Exception as e:
        print(f"Docker not available: {e}")
        print("Skipping sandbox demo")


async def demo_batch_validation():
    """Demo batch validation of multiple tasks."""
    print("\n=== Batch Validation Demo ===")

    # Create multiple tasks
    tasks = []
    for i in range(5):
        task = Task(
            id=f"batch-task-{i}",
            domain="math_reasoning",
            difficulty="easy" if i < 3 else "medium",
            prompt=f"Calculate factorial of {i+5}",
            test_cases=[TaskTestCase(input=i+5, expected_output=i+5)],
            validator_code="def validate(result): return isinstance(result, int)",
            source=SourceType.HUMAN_CURATED
        )

        if i == 0:
            # Valid solution
            task.reference_solution = f"""
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

result = factorial({i+5})
"""
        elif i == 1:
            # Invalid function call
            task.reference_solution = f"""
invalid_function({i+5})
"""
        else:
            # Valid solution
            task.reference_solution = f"""
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

result = factorial({i+5})
"""

        tasks.append(task)

    # Initialize validator
    registry = FunctionRegistry()
    validator = TaskValidator(
        function_registry=registry,
        enable_sandbox=False
    )

    # Validate batch
    print("Validating 5 tasks...")
    reports = await validator.validate_batch(tasks)

    for i, (task, report) in enumerate(zip(tasks, reports)):
        status = "PASS" if report.function_signature_valid else "FAIL"
        print(f"{status} Task {i+1}: {task.domain.value} - {task.difficulty.value}")
        print(f"   Signature valid: {report.function_signature_valid}")
        print(f"   Execution passed: {report.sandbox_execution_passed}")
        print(f"   Errors: {len(report.errors)}")
        print(f"   Suggestions: {len(report.suggestions)}")


async def main():
    """Run all demo functions."""
    print("Task Validation Demo")
    print("====================")

    try:
        await demo_function_validation()
        await demo_sandbox_validation()
        await demo_batch_validation()

        print("\n=== Demo Complete ===")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())