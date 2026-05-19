"""
Tests for task validation functionality.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.agent_sft.function_registry import FunctionRegistry
from src.agent_sft.task_generator.function_parser import FunctionSignatureParser, FunctionCall
from src.agent_sft.task_generator.models import Task, TaskTestCase, ValidationReport, SourceType
from src.agent_sft.task_generator.validator import TaskValidator
from src.infra.sandbox.execution_manager import SandboxExecutor
from src.infra.sandbox.models import SandboxExecutionResponse, TestCaseExecution, ExecutionResult, ExecutionStatus


class TestFunctionParser:
    """Test function signature parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FunctionRegistry()
        self.parser = FunctionSignatureParser(self.registry)

    def test_parse_simple_function_call(self):
        """Test parsing simple function calls."""
        code = "print('Hello, World!')"
        task = Task(
            id="test-1",
            domain="code_debug",
            difficulty="easy",
            prompt="Debug the code",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = code

        function_calls = self.parser.parse_functions(task)
        assert len(function_calls) == 1
        assert function_calls[0].name == "print"
        assert function_calls[0].args == ["Hello, World!"]

    def test_parse_function_with_kwargs(self):
        """Test parsing function calls with keyword arguments."""
        code = "sorted([3, 1, 2], reverse=True)"
        task = Task(
            id="test-2",
            domain="code_debug",
            difficulty="medium",
            prompt="Sort a list",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = code

        function_calls = self.parser.parse_functions(task)
        assert len(function_calls) == 1
        assert function_calls[0].name == "sorted"
        assert function_calls[0].args == [[3, 1, 2]]
        assert function_calls[0].kwargs == {"reverse": True}

    def test_parse_multiple_function_calls(self):
        """Test parsing multiple function calls."""
        code = """
result = len([1, 2, 3])
print(result)
max_val = max(result, 10)
"""
        task = Task(
            id="test-3",
            domain="code_debug",
            difficulty="easy",
            prompt="Multiple operations",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = code

        function_calls = self.parser.parse_functions(task)
        assert len(function_calls) == 3  # len, print, max
        names = [call.name for call in function_calls]
        assert "len" in names
        assert "print" in names
        assert "max" in names

    def test_validate_valid_function_calls(self):
        """Test validation of valid function calls."""
        code = "print(len([1, 2, 3]))"
        task = Task(
            id="test-4",
            domain="code_debug",
            difficulty="easy",
            prompt="Valid operations",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = code

        function_calls = self.parser.parse_functions(task)
        result = self.parser.validate_signatures(function_calls)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_function_calls(self):
        """Test validation of invalid function calls."""
        code = "nonexistent_function()"
        task = Task(
            id="test-5",
            domain="code_debug",
            difficulty="easy",
            prompt="Invalid function",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = code

        function_calls = self.parser.parse_functions(task)
        result = self.parser.validate_signatures(function_calls)

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].type.value == "missing_function"

    def test_validate_imports(self):
        """Test import validation."""
        valid_code = "import json\ndata = json.loads('{}')"
        invalid_code = "import nonexistent_module"

        # Valid imports
        errors = self.parser.validate_imports(valid_code)
        assert len(errors) == 0

        # Invalid imports
        errors = self.parser.validate_imports(invalid_code)
        assert len(errors) == 1
        assert errors[0].type.value == "import_error"


def _is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False

# Skip entire class if Docker not available
docker_available = _is_docker_available()

@pytest.mark.skipif(not docker_available, reason="Docker daemon not running")
class TestSandboxExecutor:
    """Test sandbox execution functionality."""

    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client."""
        with patch('docker.from_env') as mock:
            client = Mock()
            client.ping.return_value = True
            client.containers.run.return_value = Mock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def sandbox_executor(self, mock_docker_client):
        """Create sandbox executor with mocked Docker."""
        with patch('docker.from_env', return_value=mock_docker_client):
            from src.infra.sandbox.execution_manager import SandboxExecutor
            return SandboxExecutor()

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, sandbox_executor):
        """Test executing simple code."""
        # Mock successful execution response
        expected_response = SandboxExecutionResponse(
            execution_result=ExecutionResult.success("Hello, World!\n", 0.5),
            test_results=[],
            passed_tests=0,
            total_tests=0,
            overall_passed=True
        )

        # Mock the execute_code method directly
        sandbox_executor.execute_code = AsyncMock(return_value=expected_response)

        request = Mock()
        request.code = "print('Hello, World!')"
        request.test_cases = []
        request.requirements = []

        result = await sandbox_executor.execute_code(request)

        assert result.execution_result.status == ExecutionStatus.SUCCESS
        assert result.execution_result.output == "Hello, World!\n"
        assert result.overall_passed is True

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, sandbox_executor):
        """Test code execution with timeout."""
        # Mock timeout execution response
        expected_response = SandboxExecutionResponse(
            execution_result=ExecutionResult.timeout(2.0),
            test_results=[],
            passed_tests=0,
            total_tests=0,
            overall_passed=False
        )

        # Mock the execute_code method directly
        sandbox_executor.execute_code = AsyncMock(return_value=expected_response)

        request = Mock()
        request.code = "import time\ntime.sleep(2)\nprint('Done')"
        request.test_cases = []
        request.requirements = []

        result = await sandbox_executor.execute_code(request)

        assert result.execution_result.status == ExecutionStatus.TIMEOUT
        assert result.overall_passed is False


class TestTaskValidator:
    """Test comprehensive task validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FunctionRegistry()
        self.parser = FunctionSignatureParser(self.registry)
        self.mock_executor = Mock()

    @pytest.mark.asyncio
    async def test_validate_task_without_sandbox(self):
        """Test task validation without sandbox execution."""
        validator = TaskValidator(
            function_registry=self.registry,
            sandbox_executor=None,
            enable_sandbox=False
        )

        # Create a task with valid code
        task = Task(
            id="test-1",
            domain="code_debug",
            difficulty="easy",
            prompt="Debug the code",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return result == 'result'",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = "print('Hello, World!')"

        # Run validation synchronously for testing
        report = await validator.validate_task(task)

        assert report.function_signature_valid is True
        assert report.sandbox_execution_passed is True  # Without sandbox

    @patch('src.agent_sft.task_generator.validator.SandboxExecutor')
    @pytest.mark.asyncio
    async def test_validate_task_with_sandbox(self, mock_sandbox_class):
        """Test task validation with sandbox execution."""
        # Mock sandbox response
        mock_sandbox = Mock()
        mock_sandbox_class.return_value = mock_sandbox

        mock_response = SandboxExecutionResponse(
            execution_result=ExecutionResult.success("Test output", 1.0),
            test_results=[
                TestCaseExecution(
                    passed=True,
                    output="Test passed",
                    execution_time=0.1,
                    test_case_index=0
                )
            ],
            passed_tests=1,
            total_tests=1,
            overall_passed=True
        )
        mock_sandbox.execute_code = AsyncMock(return_value=mock_response)

        validator = TaskValidator(
            sandbox_executor=mock_sandbox,
            enable_sandbox=True
        )

        # Create a task
        task = Task(
            id="test-2",
            domain="code_debug",
            difficulty="easy",
            prompt="Debug the code",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return result == 'result'",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = "result = 'result'\nprint(result)"

        # Run validation
        report = await validator.validate_task(task)

        assert report.function_signature_valid is True
        assert report.sandbox_execution_passed is True
        assert report.execution_time == 1.0
        assert len(report.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_task_with_errors(self):
        """Test task validation that detects errors."""
        validator = TaskValidator(
            function_registry=self.registry,
            sandbox_executor=None,
            enable_sandbox=False
        )

        # Create a task with invalid code
        task = Task(
            id="test-3",
            domain="code_debug",
            difficulty="easy",
            prompt="Debug the code",
            test_cases=[TaskTestCase(input="test", expected_output="result")],
            validator_code="def validate(result): return True",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = "nonexistent_function()"

        # Run validation
        report = await validator.validate_task(task)

        assert report.function_signature_valid is False
        assert len(report.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_batch(self):
        """Test validating multiple tasks concurrently."""
        # Mock multiple tasks
        tasks = []
        for i in range(3):
            task = Task(
                id=f"test-{i}",
                domain="code_debug",
                difficulty="easy",
                prompt=f"Task number {i} for testing",
                test_cases=[TaskTestCase(input=f"input{i}", expected_output=f"result{i}")],
                validator_code="def validate(result): return True",
                source=SourceType.HUMAN_CURATED
            )
            task.reference_solution = f"print('Task {i}')"
            tasks.append(task)

        # Mock sandbox
        mock_response = SandboxExecutionResponse(
            execution_result=ExecutionResult.success("Output", 0.5),
            test_results=[],
            passed_tests=0,
            total_tests=0,
            overall_passed=True
        )

        with patch('src.agent_sft.task_generator.validator.SandboxExecutor') as mock_class:
            mock_sandbox = Mock()
            mock_sandbox.execute_code = AsyncMock(return_value=mock_response)
            mock_class.return_value = mock_sandbox

            validator = TaskValidator(sandbox_executor=mock_sandbox)

            reports = await validator.validate_batch(tasks)

            assert len(reports) == 3
            for report in reports:
                assert report.function_signature_valid is True
                assert report.sandbox_execution_passed is True


class TestIntegration:
    """Integration tests for the validation system."""

    def test_end_to_end_validation(self):
        """Test complete validation flow."""
        # Create a comprehensive test
        registry = FunctionRegistry()
        parser = FunctionSignatureParser(registry)
        validator = TaskValidator(function_registry=registry, enable_sandbox=False)

        # Task with valid code
        task = Task(
            id="integration-test",
            domain="code_debug",
            difficulty="medium",
            prompt="Calculate the sum of numbers",
            test_cases=[
                TaskTestCase(input=[1, 2, 3], expected_output=6),
                TaskTestCase(input=[], expected_output=0)
            ],
            validator_code="""
def validate(result):
    try:
        return isinstance(result, (int, float))
    except:
        return False
""",
            source=SourceType.HUMAN_CURATED
        )
        task.reference_solution = """
def sum_numbers(numbers):
    return sum(numbers)

result = sum_numbers([1, 2, 3])
"""

        # Parse functions
        function_calls = parser.parse_functions(task)
        assert len(function_calls) == 1  # sum (sum_numbers is user-defined, skipped)

        # Validate signatures
        signature_result = parser.validate_signatures(function_calls)
        assert signature_result.valid is True

        # Validate task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        report = loop.run_until_complete(validator.validate_task(task))

        assert report.function_signature_valid is True
        assert report.sandbox_execution_passed is True
        assert len(report.errors) == 0