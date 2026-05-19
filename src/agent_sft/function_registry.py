"""
Function Registry for managing available functions and their signatures.
"""

import ast
import importlib
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass
from enum import Enum


class FunctionCategory(Enum):
    """Categories of functions in the registry"""
    BUILTIN = "builtin"
    STANDARD_LIBRARY = "standard_library"
    CUSTOM = "custom"


@dataclass
class FunctionInfo:
    """Information about a registered function"""
    name: str
    signature: Callable
    category: FunctionCategory
    description: str = ""
    module: Optional[str] = None


class FunctionRegistry:
    """Registry for managing available functions and their signatures"""

    def __init__(self):
        self._functions: Dict[str, FunctionInfo] = {}
        self._builtin_functions = self._initialize_builtin_functions()
        self._standard_library_functions = self._initialize_standard_library_functions()
        self._initialize_registry()

    def _initialize_builtin_functions(self) -> Dict[str, FunctionInfo]:
        """Initialize Python built-in functions"""
        builtins = {}

        # Common built-in functions
        builtin_mapping = {
            'print': (lambda *args, **kwargs: None, "Print function"),
            'len': (lambda x: len(x), "Get length of object"),
            'type': (lambda x: type(x), "Get type of object"),
            'str': (lambda x: str(x), "Convert to string"),
            'int': (lambda x: int(x), "Convert to integer"),
            'float': (lambda x: float(x), "Convert to float"),
            'list': (lambda x: list(x), "Convert to list"),
            'dict': (lambda x: dict(x), "Convert to dictionary"),
            'range': (lambda *args: range(*args), "Generate range of numbers"),
            'enumerate': (lambda iterable, start=0: enumerate(iterable, start), "Enumerate iterable"),
            'zip': (lambda *iterables: zip(*iterables), "Zip iterables together"),
            'sum': (lambda iterable: sum(iterable), "Sum of iterable"),
            'max': (lambda *args: max(*args), "Maximum value"),
            'min': (lambda *args: min(*args), "Minimum value"),
            'abs': (lambda x: abs(x), "Absolute value"),
            'round': (lambda x, n=0: round(x, n), "Round number"),
            'sorted': (lambda iterable, *args, **kwargs: sorted(iterable, *args, **kwargs), "Sort iterable"),
            'any': (lambda iterable: any(iterable), "Check if any element is True"),
            'all': (lambda iterable: all(iterable), "Check if all elements are True"),
            'isinstance': (lambda obj, classinfo: isinstance(obj, classinfo), "Check instance type"),
            'hasattr': (lambda obj, name: hasattr(obj, name), "Check attribute existence"),
            'getattr': (lambda obj, name, default=None: getattr(obj, name, default), "Get attribute"),
            'setattr': (lambda obj, name, value: setattr(obj, name, value), "Set attribute"),
        }

        for name, (func, desc) in builtin_mapping.items():
            builtins[name] = FunctionInfo(
                name=name,
                signature=func,
                category=FunctionCategory.BUILTIN,
                description=desc
            )

        return builtins

    def _initialize_standard_library_functions(self) -> Dict[str, FunctionInfo]:
        """Initialize common standard library functions"""
        stdlib = {}

        # Common standard library modules
        stdlib_modules = {
            'json': ['dump', 'dumps', 'load', 'loads'],
            'os': ['listdir', 'path.exists', 'path.join', 'mkdir'],
            'sys': ['exit', 'argv'],
            'datetime': ['datetime.now', 'timedelta'],
            'collections': ['defaultdict', 'Counter'],
            'itertools': ['chain', 'product', 'combinations'],
            'math': ['sqrt', 'pow', 'log', 'exp', 'sin', 'cos'],
            'random': ['choice', 'shuffle', 'random'],
            'requests': ['get', 'post', 'put', 'delete'],  # Common HTTP operations
            'urllib.request': ['urlopen'],
            'base64': ['b64encode', 'b64decode'],
        }

        for module_name, functions in stdlib_modules.items():
            for func_name in functions:
                if '.' in func_name:
                    # Handle nested functions like os.path.exists
                    parent_func, child_func = func_name.split('.', 1)
                    full_name = f"{module_name}.{func_name}"
                else:
                    parent_func = func_name
                    full_name = f"{module_name}.{func_name}"

                stdlib[full_name] = FunctionInfo(
                    name=full_name,
                    signature=lambda: None,  # Will be resolved dynamically
                    category=FunctionCategory.STANDARD_LIBRARY,
                    module=module_name,
                    description=f"Function from {module_name}"
                )

        return stdlib

    def _initialize_registry(self):
        """Initialize the registry with all functions"""
        # Add built-in functions
        self._functions.update(self._builtin_functions)
        # Add standard library functions
        self._functions.update(self._standard_library_functions)
        # Add custom functions (can be extended later)
        self._add_custom_functions()

    def _add_custom_functions(self):
        """Add custom function library"""
        custom_functions = {
            'api': {
                'fetch_data': (
                    lambda url, params=None: None,
                    "Fetch data from API endpoint",
                    "api"
                ),
                'save_to_db': (
                    lambda data, table=None: None,
                    "Save data to database",
                    "api"
                ),
                'call_api': (
                    lambda method, url, headers=None, data=None: None,
                    "Generic API caller",
                    "api"
                ),
            },
            'math': {
                'calculate': (
                    lambda a, b, operation: None,
                    "Mathematical calculation",
                    "math"
                ),
                'optimize': (
                    lambda params: None,
                    "Optimization function",
                    "math"
                ),
                'stats': (
                    lambda data: None,
                    "Calculate statistics",
                    "math"
                ),
            },
            'file': {
                'read_file': (
                    lambda path: None,
                    "Read file content",
                    "file"
                ),
                'write_file': (
                    lambda path, content: None,
                    "Write content to file",
                    "file"
                ),
                'list_files': (
                    lambda directory: None,
                    "List files in directory",
                    "file"
                ),
            },
        }

        for category, functions in custom_functions.items():
            for name, (func, desc, module) in functions.items():
                full_name = f"{category}.{name}"
                self._functions[full_name] = FunctionInfo(
                    name=full_name,
                    signature=func,
                    category=FunctionCategory.CUSTOM,
                    description=desc,
                    module=module
                )

    def register_function(self, name: str, signature: Callable, description: str = "",
                         category: FunctionCategory = FunctionCategory.CUSTOM):
        """Register a new function"""
        self._functions[name] = FunctionInfo(
            name=name,
            signature=signature,
            category=category,
            description=description
        )

    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Get function by name"""
        # Try exact match first
        if name in self._functions:
            return self._functions[name]

        # Try to resolve standard library functions dynamically
        if '.' in name:
            module_path, func_name = name.rsplit('.', 1)
            return self._resolve_dynamic_function(module_path, func_name)

        return None

    def _resolve_dynamic_function(self, module_path: str, func_name: str) -> Optional[FunctionInfo]:
        """Resolve standard library functions dynamically"""
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return FunctionInfo(
                name=f"{module_path}.{func_name}",
                signature=func,
                category=FunctionCategory.STANDARD_LIBRARY,
                module=module_path,
                description=f"Function from {module_path}"
            )
        except (ImportError, AttributeError):
            return None

    def validate_import(self, module_name: str) -> bool:
        """Validate if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def get_available_functions(self, category: Optional[FunctionCategory] = None) -> List[str]:
        """Get list of available functions"""
        if category:
            return [name for name, info in self._functions.items()
                   if info.category == category]
        return list(self._functions.keys())

    def get_functions_by_category(self) -> Dict[FunctionCategory, List[str]]:
        """Get functions grouped by category"""
        categories = {}
        for name, info in self._functions.items():
            if info.category not in categories:
                categories[info.category] = []
            categories[info.category].append(name)
        return categories

    def has_function(self, name: str) -> bool:
        """Check if function exists"""
        return self.get_function(name) is not None

    def get_function_info(self, name: str) -> Optional[FunctionInfo]:
        """Get complete function information"""
        return self._functions.get(name) or self.get_function(name)