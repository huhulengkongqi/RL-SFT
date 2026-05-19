"""
Comprehensive task validator that integrates function signature parsing and sandbox execution.
"""

import asyncio
import time
from typing import List, Optional, Tuple

from ..function_registry import FunctionRegistry
from .function_parser import FunctionSignatureParser, FunctionCall, ValidationResult as FunctionValidationResult
from .models import Task, ValidationReport, ValidationError, ValidationWarning
from src.infra.sandbox.models import (
    SandboxExecutionRequest, SandboxExecutionResponse,
    TestCaseExecution, ValidationErrorType, ExecutionResult
)
from src.infra.sandbox.execution_manager import SandboxExecutor


class TaskValidator:
    """Comprehensive task validator that combines signature parsing and sandbox execution"""

    def __init__(
        self,
        function_registry: Optional[FunctionRegistry] = None,
        sandbox_executor: Optional[SandboxExecutor] = None,
        enable_sandbox: bool = True
    ):
        self.function_registry = function_registry or FunctionRegistry()
        self.parser = FunctionSignatureParser(self.function_registry)
        self.sandbox_executor = sandbox_executor
        self.enable_sandbox = enable_sandbox and sandbox_executor is not None

    async def validate_task(self, task: Task) -> ValidationReport:
        """Perform complete task validation"""
        start_time = time.time()

        # 1. Function signature validation
        function_calls = self.parser.parse_functions(task)
        signature_result = self.parser.validate_signatures(function_calls)

        # 2. Sandbox execution validation
        sandbox_result = None
        execution_time = 0

        if self.enable_sandbox and task.reference_solution:
            sandbox_request = SandboxExecutionRequest(
                code=task.reference_solution,
                test_cases=[
                    {
                        'input_data': tc.input,
                        'expected_output': tc.expected_output,
                        'validator_code': task.validator_code
                    }
                    for tc in task.test_cases
                ],
                requirements=self._extract_requirements(task.reference_solution)
            )

            sandbox_result = await self.sandbox_executor.execute_code(sandbox_request)
            execution_time = sandbox_result.execution_result.execution_time
        else:
            # Without sandbox, we can still validate the code syntax
            syntax_check = self._check_syntax(task.reference_solution)
            if syntax_check:
                sandbox_result = SandboxExecutionResponse(
                    execution_result=syntax_check,
                    test_results=[],
                    passed_tests=0,
                    total_tests=0,
                    overall_passed=syntax_check.status == "success"
                )
                execution_time = syntax_check.execution_time
            else:
                # No syntax issues found - assume it's valid
                from ...infra.sandbox.models import ExecutionResult
                sandbox_result = SandboxExecutionResponse(
                    execution_result=ExecutionResult.success("No syntax errors", 0.0),
                    test_results=[],
                    passed_tests=0,
                    total_tests=0,
                    overall_passed=True
                )
                execution_time = 0.0

        # 3. Generate validation report
        report = ValidationReport(
            function_signature_valid=signature_result.valid,
            sandbox_execution_passed=sandbox_result.overall_passed if sandbox_result else False,
            execution_time=execution_time,
            memory_usage=sandbox_result.execution_result.memory_usage if sandbox_result else None,
            errors=self._collect_errors(signature_result, sandbox_result),
            warnings=self._collect_warnings(signature_result, sandbox_result),
            suggestions=self._generate_suggestions(signature_result, sandbox_result)
        )

        return report

    def _check_syntax(self, code: str) -> Optional['ExecutionResult']:
        """Check code syntax without full execution."""
        try:
            import ast
            ast.parse(code)
            return None  # No syntax errors
        except SyntaxError as e:
            from src.infra.sandbox.models import ExecutionResult, ExecutionStatus
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Syntax error: {e}",
                execution_time=0,
                line_number=e.lineno,
                column_number=e.offset
            )

    def _collect_errors(
        self,
        signature_result: FunctionValidationResult,
        sandbox_result: Optional[SandboxExecutionResponse]
    ) -> List[ValidationError]:
        """Collect validation errors from all sources"""
        errors = []

        # Add function signature errors
        for error in signature_result.errors:
            errors.append(ValidationError(
                type=error.type.value,
                message=error.message,
                line_number=error.line_number,
                column_number=error.column_number,
                suggestion=error.suggestion
            ))

        # Add sandbox execution errors
        if sandbox_result:
            exec_result = sandbox_result.execution_result
            if exec_result.status == "failure" or exec_result.status == "error":
                error_type = "runtime_error" if exec_result.error else "unknown"
                errors.append(ValidationError(
                    type=error_type,
                    message=exec_result.error or "Unknown execution error",
                    line_number=exec_result.traceback and self._extract_line_number(exec_result.traceback),
                    suggestion=self._suggest_runtime_fix(exec_result.error)
                ))

            # Add test case errors
            for test_result in sandbox_result.test_results:
                if not test_result.passed:
                    errors.append(ValidationError(
                        type=test_result.error_type.value if test_result.error_type else "test_failure",
                        message=test_result.error or "Test failed",
                        line_number=test_result.test_case_index,
                        suggestion=f"Check test case {test_result.test_case_index}"
                    ))

        return errors

    def _collect_warnings(
        self,
        signature_result: FunctionValidationResult,
        sandbox_result: Optional[SandboxExecutionResponse]
    ) -> List[ValidationWarning]:
        """Collect validation warnings"""
        warnings = []

        # Add function signature warnings
        for warning in signature_result.warnings:
            warnings.append(ValidationWarning(
                type="function_warning",
                message=warning,
                suggestion="Review function calls in the code"
            ))

        # Add sandbox warnings
        if sandbox_result:
            if sandbox_result.execution_result.execution_time > 10:
                warnings.append(ValidationWarning(
                    type="performance",
                    message="Code execution took longer than expected",
                    suggestion="Optimize the code for better performance"
                ))

            memory_usage = sandbox_result.execution_result.memory_usage
            if memory_usage and memory_usage > 100:  # 100MB
                warnings.append(ValidationWarning(
                    type="memory",
                    message="High memory usage detected",
                    suggestion="Reduce memory consumption"
                ))

        return warnings

    def _generate_suggestions(
        self,
        signature_result: FunctionValidationResult,
        sandbox_result: Optional[SandboxExecutionResponse]
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Add function signature suggestions
        suggestions.extend(signature_result.suggestions)

        # Add execution suggestions
        if sandbox_result and not sandbox_result.overall_passed:
            if sandbox_result.execution_result.status == "timeout":
                suggestions.append("Increase timeout or optimize the code")
            elif sandbox_result.passed_tests < sandbox_result.total_tests:
                suggestions.append(
                    f"Fix {sandbox_result.total_tests - sandbox_result.passed_tests} failing test(s)"
                )

        # General suggestions
        if not signature_result.valid and (sandbox_result and not sandbox_result.overall_passed):
            suggestions.append("Review both function signatures and execution logic")

        return suggestions

    def _extract_requirements(self, code: str) -> List[str]:
        """Extract required packages from code"""
        requirements = []
        import_statements = []

        # Common import to package mapping
        import_package_map = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'requests': 'requests',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'sklearn': 'scikit-learn',
            'tensorflow': 'tensorflow',
            'torch': 'torch',
            'opencv-python': 'opencv-python',
            'PIL': 'Pillow',
            'flask': 'flask',
            'django': 'django',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'sqlalchemy': 'sqlalchemy',
            'psycopg2': 'psycopg2-binary',
            'mysql-connector': 'mysql-connector-python',
        }

        # Extract import statements
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                module = line.split('import ')[1].split()[0]
                if '.' in module:
                    module = module.split('.')[0]
                import_statements.append(module)
            elif line.startswith('from '):
                module = line.split('from ')[1].split(' import ')[0]
                import_statements.append(module)

        # Map to packages
        for module in import_statements:
            for import_name, package_name in import_package_map.items():
                if module.startswith(import_name):
                    requirements.append(package_name)
                    break

        return list(set(requirements))  # Remove duplicates

    def _extract_line_number(self, traceback_str: str) -> Optional[int]:
        """Extract line number from traceback"""
        try:
            lines = traceback_str.split('\n')
            for line in lines:
                if 'line ' in line.lower():
                    parts = line.split('line ')
                    if len(parts) > 1:
                        return int(parts[1].split()[0])
        except:
            pass
        return None

    def _suggest_runtime_fix(self, error: str) -> Optional[str]:
        """Suggest fixes for common runtime errors"""
        error_lower = error.lower()

        if 'nameerror' in error_lower and 'is not defined' in error_lower:
            return "Check for typos in variable names"
        elif 'attributeerror' in error_lower and 'has no attribute' in error_lower:
            return "Verify the object has the required method/attribute"
        elif 'module not found' in error_lower or 'importerror' in error_lower:
            return "Install required packages or check import statements"
        elif 'zero division' in error_lower:
            return "Add check for division by zero"
        elif 'index out of range' in error_lower or 'list index out of range' in error_lower:
            return "Check array bounds before accessing elements"

        return None

    async def validate_batch(self, tasks: List[Task]) -> List[ValidationReport]:
        """Validate multiple tasks concurrently"""
        semaphore = asyncio.Semaphore(5)  # Limit concurrent validations

        async def validate_with_semaphore(task):
            async with semaphore:
                return await self.validate_task(task)

        # Create tasks for concurrent execution
        validation_tasks = [validate_with_semaphore(task) for task in tasks]

        # Execute concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Process results
        reports = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle validation errors
                reports.append(ValidationReport(
                    function_signature_valid=False,
                    sandbox_execution_passed=False,
                    errors=[ValidationError(
                        type="validation_error",
                        message=f"Validation failed: {str(result)}",
                        line_number=None
                    )],
                    warnings=[],
                    suggestions=[]
                ))
            else:
                reports.append(result)

        return reports