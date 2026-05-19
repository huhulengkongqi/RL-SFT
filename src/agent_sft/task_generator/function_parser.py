"""
Function Signature Parser for extracting and validating function calls in tasks.
"""

import ast
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..function_registry import FunctionRegistry, FunctionInfo, FunctionCategory


class SignatureErrorType(Enum):
    """Types of signature validation errors"""
    MISSING_FUNCTION = "missing_function"
    WRONG_ARGUMENTS = "wrong_arguments"
    WRONG_RETURN_TYPE = "wrong_return_type"
    TYPE_MISMATCH = "type_mismatch"
    INVALID_SYNTAX = "invalid_syntax"
    IMPORT_ERROR = "import_error"


@dataclass
class FunctionCall:
    """Represents a function call in the task"""
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    line_number: int
    column_number: int
    source_code: str

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.name}({all_args}) at line {self.line_number}"


@dataclass
class SignatureError:
    """Represents a signature validation error"""
    type: SignatureErrorType
    message: str
    function_name: str
    line_number: int
    column_number: int
    suggestion: Optional[str] = None

    def __str__(self):
        return f"Line {self.line_number}: {self.message}"


@dataclass
class ValidationResult:
    """Result of function signature validation"""
    valid: bool
    function_calls: List[FunctionCall]
    errors: List[SignatureError]
    warnings: List[str]
    suggestions: List[str]


class FunctionSignatureParser:
    """Parse and validate function calls in task code"""

    def __init__(self, function_registry: Optional[FunctionRegistry] = None):
        self.function_registry = function_registry or FunctionRegistry()

    def parse_functions(self, task: 'Task') -> List[FunctionCall]:
        """Extract all function calls from task code"""
        function_calls = []

        # Parse the task's reference solution code
        if hasattr(task, 'reference_solution') and task.reference_solution:
            try:
                tree = ast.parse(task.reference_solution)
                # First, collect all user-defined function names
                user_defined_funcs = self._collect_user_defined_functions(tree)
                # Then extract function calls, excluding user-defined ones
                function_calls = self._extract_function_calls(tree, user_defined_funcs)
            except SyntaxError as e:
                raise ValueError(f"Invalid syntax in reference solution: {e}")

        return function_calls

    def _collect_user_defined_functions(self, tree: ast.AST) -> set:
        """Collect all function names defined by the user in the code"""
        func_names = set()

        class FuncDefVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                func_names.add(node.name)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                func_names.add(node.name)
                self.generic_visit(node)

        visitor = FuncDefVisitor()
        visitor.visit(tree)
        return func_names

    def _extract_function_calls(self, tree: ast.AST, user_defined_funcs: set) -> List[FunctionCall]:
        """Extract function calls from AST, excluding user-defined functions"""
        function_calls = []

        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser

            def visit_Call(self, node: ast.Call):
                # Extract function name
                func_name = self._get_function_name(node.func)

                # Skip user-defined functions
                if func_name and func_name not in user_defined_funcs:
                    # Extract arguments
                    args = self._extract_arguments(node.args, node.keywords)

                    # Create FunctionCall object
                    function_call = FunctionCall(
                        name=func_name,
                        args=args['args'],
                        kwargs=args['kwargs'],
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        source_code=ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                    )
                    function_calls.append(function_call)

                self.generic_visit(node)

            def _get_function_name(self, node: ast.AST) -> Optional[str]:
                """Get function name from AST node"""
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    # Handle attribute calls like obj.method()
                    return f"{node.value.id}.{node.attr}"
                return None

            def _extract_arguments(self, args: List[ast.AST], keywords: List[ast.keyword]) -> Dict[str, Any]:
                """Extract positional and keyword arguments"""
                positional_args = []
                keyword_args = {}

                # Extract positional arguments
                for arg in args:
                    value = self._extract_value(arg)
                    positional_args.append(value)

                # Extract keyword arguments
                for keyword in keywords:
                    value = self._extract_value(keyword.value)
                    keyword_args[keyword.arg] = value

                return {'args': positional_args, 'kwargs': keyword_args}

            def _extract_value(self, node: ast.AST) -> Any:
                """Extract value from AST node"""
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    return f"${node.id}"  # Reference to a variable
                elif isinstance(node, ast.List):
                    return [self._extract_value(elt) for elt in node.elts]
                elif isinstance(node, ast.Dict):
                    return {
                        self._extract_value(k): self._extract_value(v)
                        for k, v in zip(node.keys, node.values)
                    }
                else:
                    return f"<expression>"

        visitor = FunctionCallVisitor(self)
        visitor.visit(tree)

        return function_calls

    def validate_signatures(self, function_calls: List[FunctionCall]) -> ValidationResult:
        """Validate function call signatures"""
        errors = []
        warnings = []
        suggestions = []

        for func_call in function_calls:
            error = self._validate_single_call(func_call)
            if error:
                errors.append(error)

                # Generate suggestions for common errors
                if error.type == SignatureErrorType.MISSING_FUNCTION:
                    suggestion = self._suggest_alternative_function(func_call.name)
                    if suggestion:
                        error.suggestion = suggestion
                        suggestions.append(f"Try using: {suggestion}")

                elif error.type == SignatureErrorType.WRONG_ARGUMENTS:
                    func_info = self.function_registry.get_function(func_call.name)
                    if func_info and hasattr(func_info.signature, '__annotations__'):
                        suggestion = self._suggest_argument_fix(
                            func_call, func_info.signature.__annotations__
                        )
                        if suggestion:
                            error.suggestion = suggestion
                            suggestions.append(suggestion)

        # Add general warnings
        if function_calls and errors:
            warnings.append(f"Found {len(function_calls)} function calls with {len(errors)} errors")

        return ValidationResult(
            valid=len(errors) == 0,
            function_calls=function_calls,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_single_call(self, func_call: FunctionCall) -> Optional[SignatureError]:
        """Validate a single function call"""
        # Check if function exists
        func_info = self.function_registry.get_function(func_call.name)
        if not func_info:
            return SignatureError(
                type=SignatureErrorType.MISSING_FUNCTION,
                message=f"Function '{func_call.name}' not found",
                function_name=func_call.name,
                line_number=func_call.line_number,
                column_number=func_call.column_number
            )

        # Validate number of arguments
        expected_args = self._get_expected_args(func_info.signature)
        actual_args = len(func_call.args) + len(func_call.kwargs)

        # Check minimum arguments
        if expected_args and actual_args < expected_args.get('min', 0):
            return SignatureError(
                type=SignatureErrorType.WRONG_ARGUMENTS,
                message=f"Not enough arguments for '{func_call.name}'. "
                       f"Expected at least {expected_args.get('min', 0)}, got {actual_args}",
                function_name=func_call.name,
                line_number=func_call.line_number,
                column_number=func_call.column_number,
                suggestion=f"Check function signature: {func_call.name}(*args, **kwargs)"
            )

        # Check maximum arguments (for functions with fixed arity)
        if expected_args and expected_args.get('fixed'):
            if actual_args != expected_args['fixed']:
                return SignatureError(
                    type=SignatureErrorType.WRONG_ARGUMENTS,
                    message=f"Wrong number of arguments for '{func_call.name}'. "
                           f"Expected {expected_args['fixed']}, got {actual_args}",
                    function_name=func_call.name,
                    line_number=func_call.line_number,
                    column_number=func_call.column_number,
                    suggestion=f"Use exactly {expected_args['fixed']} arguments"
                )

        return None

    def _get_expected_args(self, func: callable) -> Optional[Dict[str, int]]:
        """Get expected number of arguments from function signature"""
        if not hasattr(func, '__code__'):
            return None

        code = func.__code__
        arg_count = code.co_argcount

        # Handle default arguments
        defaults = func.__defaults__ or ()
        defaults_count = len(defaults)

        return {
            'fixed': arg_count if not defaults else None,
            'min': arg_count - defaults_count,
            'max': arg_count
        }

    def _suggest_alternative_function(self, func_name: str) -> Optional[str]:
        """Suggest an alternative function name"""
        # Simple fuzzy matching
        available_funcs = self.function_registry.get_available_functions()

        # Try to find similar function names
        suggestions = []
        for available_func in available_funcs:
            if func_name in available_func or available_func in func_name:
                suggestions.append(available_func)

        # Return first reasonable suggestion
        for suggestion in suggestions:
            if suggestion != func_name:
                return suggestion

        return None

    def _suggest_argument_fix(self, func_call: FunctionCall, annotations: Dict[str, Any]) -> Optional[str]:
        """Suggest fixes for argument issues based on type annotations"""
        if not annotations:
            return None

        # Simple suggestion based on parameter names
        param_names = list(annotations.keys())
        if 'self' in param_names:
            param_names.remove('self')

        if len(param_names) > 0:
            return f"Expected parameters: {', '.join(param_names)}"

        return None

    def validate_imports(self, code: str) -> List[SignatureError]:
        """Validate import statements in code"""
        errors = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self.function_registry.validate_import(alias.name):
                            errors.append(SignatureError(
                                type=SignatureErrorType.IMPORT_ERROR,
                                message=f"Cannot import module: {alias.name}",
                                function_name=f"import {alias.name}",
                                line_number=node.lineno,
                                column_number=node.col_offset
                            ))
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if not self.function_registry.validate_import(module_name):
                        errors.append(SignatureError(
                            type=SignatureErrorType.IMPORT_ERROR,
                            message=f"Cannot import from module: {module_name}",
                            function_name=f"from {module_name} import",
                            line_number=node.lineno,
                            column_number=node.col_offset
                        ))

        except SyntaxError as e:
            errors.append(SignatureError(
                type=SignatureErrorType.INVALID_SYNTAX,
                message=f"Syntax error in import: {e}",
                function_name="import",
                line_number=e.lineno,
                column_number=e.offset
            ))

        return errors