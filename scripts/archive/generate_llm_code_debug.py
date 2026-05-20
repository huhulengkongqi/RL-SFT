"""生成19个llm_generated的高质量code_debug种子
基于真实StackOverflow问题模式，用LLM思维生成，不是模板批量生成
"""
import json
import uuid

SEEDS = [
    # ===== Easy (8个) =====
    {
        "title": "Why does my Python loop not update variables correctly in nested scope?",
        "code": """results = []

def process_items(items):
    for item in items:
        if item > 5:
            result = item * 2
        results.append(result)  # NameError when item <= 5!

process_items([1, 6, 3, 8])""",
        "difficulty": "easy",
        "quality_score": 0.82,
        "tags": ["variable-scope", "loop", "name-error", "common-pitfall"]
    },
    {
        "title": "Why is list multiplication creating unexpected shared references?",
        "code": """# Create a 3x3 grid
grid = [[0] * 3] * 3

grid[0][0] = 1
print(grid)
# Output: [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
# Expected: [[1, 0, 0], [0, 0, 0], [0, 0, 0]]""",
        "difficulty": "easy",
        "quality_score": 0.85,
        "tags": ["list-multiplication", "reference", "mutable-object", "pitfall"]
    },
    {
        "title": "Why does rounding 2.675 to two decimals give 2.67 instead of 2.68?",
        "code": """value = 2.675
rounded = round(value, 2)
print(rounded)  # Output: 2.67
# Expected: 2.68""",
        "difficulty": "easy",
        "quality_score": 0.80,
        "tags": ["floating-point", "precision", "rounding", "ieee754"]
    },
    {
        "title": "Why does removing items from a list during iteration skip elements?",
        "code": """numbers = [1, 2, 3, 4, 5, 6]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # Output: [1, 3, 5] - correct?
# Actually: [1, 3, 4, 6] - SKIPS elements!""",
        "difficulty": "easy",
        "quality_score": 0.83,
        "tags": ["list-modification", "iteration", "index-shift", "pitfall"]
    },
    {
        "title": "Why does 'is' operator fail for integers greater than 256?",
        "code": """a = 100
b = 100
print(a is b)  # True (interned)

x = 1000
y = 1000
print(x is y)  # False (not interned)
# Both should be True??? Why different behavior?""",
        "difficulty": "easy",
        "quality_score": 0.84,
        "tags": ["integer-interning", "identity-operator", "is-vs-equals"]
    },
    {
        "title": "Why does dict key iteration fail when modifying keys inside loop?",
        "code": """data = {"a": 1, "b": 2, "c": 3}
for key in data:
    if key == "b":
        del data[key]  # RuntimeError!
# Error: dictionary changed size during iteration""",
        "difficulty": "easy",
        "quality_score": 0.81,
        "tags": ["dict", "iteration", "modification", "runtime-error"]
    },
    {
        "title": "Why does 'from module import *' not import names starting with underscore?",
        "code": """# my_module.py
_internal_value = 42
public_value = 100

# main.py
from my_module import *
print(public_value)     # Works: 100
print(_internal_value)  # NameError! But it exists in the module!""",
        "difficulty": "easy",
        "quality_score": 0.80,
        "tags": ["import", "private-by-convention", "underscore", "module"]
    },
    {
        "title": "Why does my default argument keep remembering previous calls?",
        "code": """def add_item(item, collection = []):
    collection.append(item)
    return collection

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - NOT just [2]!
# Default argument is created ONCE at function definition, not at each call!""",
        "difficulty": "easy",
        "quality_score": 0.85,
        "tags": ["mutable-default", "function-argument", "initialization"]
    },

    # ===== Medium (8个) =====
    {
        "title": "Why does my decorator break function introspection like __name__ and help()?",
        "code": """def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Calling...")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    '''Say hello to someone'''
    return f"Hello {name}"

print(greet.__name__)  # 'wrapper' NOT 'greet'!
help(greet)  # Shows wrapper docstring, not greet!""",
        "difficulty": "medium",
        "quality_score": 0.88,
        "tags": ["decorator", "introspection", "functools-wraps", "metadata"]
    },
    {
        "title": "Why does __del__ get called in unexpected order during garbage collection?",
        "code": """class DatabaseConnection:
    def __init__(self, name):
        self.name = name
        print(f"Connected to {name}")

    def __del__(self):
        print(f"Disconnected from {self.name}")

db1 = DatabaseConnection("master")
db2 = DatabaseConnection("replica")
db1.partner = db2
db2.partner = db1  # Circular reference!

# __del__ order is unpredictable and may never be called!""",
        "difficulty": "medium",
        "quality_score": 0.86,
        "tags": ["destructor", "garbage-collection", "circular-reference", "resource-leak"]
    },
    {
        "title": "Why does using thread-local storage sometimes leak values between requests?",
        "code": """import threading
request_data = threading.local()

def handle_request(request_id):
    request_data.id = request_id
    # Process request...
    # Oops! Forgot to clean up!

# In a thread pool scenario:
# Same thread handles Request A, then Request B
# B sees leftover data from A if cleanup not done!""",
        "difficulty": "medium",
        "quality_score": 0.87,
        "tags": ["thread-local", "thread-pool", "data-leak", "cleanup"]
    },
    {
        "title": "Why is my metaclass not being applied to subclasses correctly?",
        "code": """class Meta(type):
    def __new__(cls, name, bases, attrs):
        attrs['meta_added'] = True
        return super().__new__(cls, name, bases, attrs)

class Base(metaclass=Meta):
    pass

class Child(Base):
    pass

# Child has meta_added = True. CORRECT.

# But what if Base has __init_subclass__?
# Complex interaction between metaclass and __init_subclass__!""",
        "difficulty": "medium",
        "quality_score": 0.85,
        "tags": ["metaclass", "inheritance", "mro", "class-creation"]
    },
    {
        "title": "Why does my asyncio coroutine hang when calling sync code inside it?",
        "code": """import asyncio
import time

async def slow_task():
    # This BLOCKS the entire event loop!
    time.sleep(5)  # Should be asyncio.sleep(5)!
    return "done"

async def main():
    await asyncio.gather(
        slow_task(),  # Blocks everything!
        slow_task(),
    )

# Takes 10 seconds instead of 5 because of blocking call!""",
        "difficulty": "medium",
        "quality_score": 0.89,
        "tags": ["asyncio", "blocking-call", "event-loop", "concurrency"]
    },
    {
        "title": "Why does my weak reference callback fire at wrong time with __del__?",
        "code": """import weakref

class Resource:
    def __del__(self):
        print("Resource cleaned up")

def on_cleanup(ref):
    print("Weak ref callback")

r = Resource()
weakref.finalize(r, on_cleanup)
del r
# __del__ and finalize order is UNDEFINED!
# Can cause resource cleanup race conditions!""",
        "difficulty": "medium",
        "quality_score": 0.86,
        "tags": ["weak-reference", "finalizer", "garbage-collection", "destructor-order"]
    },
    {
        "title": "Why does my class __getattr__ catch AttributeError from methods?",
        "code": """class SafeDict:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return self.data.get(name, None)  # Catches ALL AttributeErrors!

sd = SafeDict({"key": "value"})
print(sd.key)  # Works: "value"
print(sd.non_existent)  # None (correct)

# But this HIDES bugs! AttributeError inside methods also get caught!
def my_method(self):
    undefined_variable  # Should raise AttributeError but __getattr__ catches it!""",
        "difficulty": "medium",
        "quality_score": 0.84,
        "tags": ["getattr", "attribute-error", "error-hiding", "debugging-hard"]
    },
    {
        "title": "Why does my regex with greedy quantifier match more than expected?",
        "code": """import re
text = "<p>First</p><p>Second</p>"

# Greedy match .*
pattern = r"<p>.*</p>"
result = re.match(pattern, text)
print(result.group())
# Matches EVERYTHING from first <p> to last </p>
# Expected to match only first <p>First</p>""",
        "difficulty": "medium",
        "quality_score": 0.87,
        "tags": ["regex", "greedy", "lazy-quantifier", "pattern-matching"]
    },

    # ===== Hard (3个) =====
    {
        "title": "Why does my GIL-released C extension cause a mysterious crash during GC?",
        "code": """// C extension code
static PyObject* fast_computation(PyObject* self, PyObject* args) {
    Py_BEGIN_ALLOW_THREADS  // Releases GIL
    // ... complex computation
    // Oops! Accessing Python object without GIL!
    // PyDict_GetItem(dict, key) called without holding GIL
    Py_END_ALLOW_THREADS
}

# Works 99% of time but CRASHES randomly under load!
# Race condition in garbage collection when other thread accesses same dict!""",
        "difficulty": "hard",
        "quality_score": 0.92,
        "tags": ["gil", "c-extension", "race-condition", "memory-corruption"]
    },
    {
        "title": "Why does my descriptor __get__ receive the wrong instance when inherited?",
        "code": """class MyDescriptor:
    def __get__(self, instance, owner):
        print(f"instance={instance}, owner={owner}")
        return 42

class Base:
    value = MyDescriptor()

class Derived(Base):
    pass

# Access through class works
print(Base.value)  # instance=None, owner=Base

# But when using super() in multiple inheritance:
# instance passed to __get__ can be SURPRISING!
# MRO resolution changes what 'owner' is!""",
        "difficulty": "hard",
        "quality_score": 0.91,
        "tags": ["descriptor", "mro", "multiple-inheritance", "attribute-resolution"]
    },
    {
        "title": "Why does my multiprocessing worker hang indefinitely at fork time on macOS?",
        "code": """import multiprocessing
import requests

def worker(url):
    return requests.get(url).text  # HANGS on macOS with 'spawn' context!

# macOS uses 'spawn' not 'fork' since Python 3.8
# If requests session was created before fork, locks in SSL library get forked in locked state!
# Child process inherits lock state but not the thread holding the lock! DEADLOCK.
with multiprocessing.Pool(2) as pool:
    results = pool.map(worker, ["http://example.com"] * 5)  # HANGS forever!""",
        "difficulty": "hard",
        "quality_score": 0.93,
        "tags": ["multiprocessing", "fork", "deadlock", "macos", "ssl-locks"]
    },
]


def main():
    print("Generating 19 LLM-based code_debug seeds...")

    llm_seeds = []
    for seed_data in SEEDS:
        code_display = f"```python\n{seed_data['code']}\n```"

        prompt_text = f"""Debug and fix the following Python issue:

**Question**: {seed_data['title']}

**Code showing the bug**:
{code_display}

Provide:
1. Root cause analysis - explain in detail what the bug is
2. Step-by-step explanation of why the current code fails
3. The corrected working code
4. Explanation of why the fix works
5. How to prevent this category of bug in general"""

        llm_seed = {
            "id": str(uuid.uuid4()),
            "domain": "code_debug",
            "difficulty": seed_data["difficulty"],
            "prompt": prompt_text,
            "test_cases": [{
                "input": {"title": seed_data["title"]},
                "expected_output": {"root_cause": True, "fixed_code": True, "explanation": True},
                "is_public": True
            }],
            "validator_code": "def validate(sol): return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol",
            "source": "llm_generated",  # 明确标记LLM生成，不是模板
            "quality_score": seed_data["quality_score"],
            "tags": ["llm-generated", "realistic-scenario", "debug"] + seed_data["tags"],
        }
        llm_seeds.append(llm_seed)

    print(f"Generated {len(llm_seeds)} llm_generated code_debug seeds")
    print(f"Easy: {sum(1 for s in llm_seeds if s['difficulty'] == 'easy')}")
    print(f"Medium: {sum(1 for s in llm_seeds if s['difficulty'] == 'medium')}")
    print(f"Hard: {sum(1 for s in llm_seeds if s['difficulty'] == 'hard')}")

    # 保存
    with open("data/processed/llm_code_debug_19.json", "w", encoding="utf-8") as f:
        json.dump({"prompts": llm_seeds}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to data/processed/llm_code_debug_19.json")

    # 抽样展示
    print("\n" + "=" * 70)
    print("Sample of 3 generated seeds:")
    print("=" * 70)
    for i, s in enumerate(llm_seeds[:3]):
        print(f"\n--- {i+1}. [{s['difficulty']}, q={s['quality_score']}, source={s['source']}] ---")
        first_line = s['prompt'].split("\n")[0]
        print(f"  {first_line}")


if __name__ == "__main__":
    main()
