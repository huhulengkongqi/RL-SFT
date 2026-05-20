"""合并最终种子池：math_reasoning + code_debug，共100个高质量真实种子"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import (
    Difficulty,
    Domain,
    SeedPrompt,
    SourceType,
    TaskTestCase,
)
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def load_gsm8k_seeds():
    """加载已生成的50个math_reasoning种子"""
    pool = SeedPromptPool.load("data/processed/gsm8k_50_seeds.json")
    print(f"✅ 加载 GSM8K: {len(pool)} 个")
    return pool


def build_code_debug_seeds():
    """构建50个高质量code_debug种子（基于StackOverflow高票问题）"""
    bugs = [
        # ===== Easy: 基础语法/类型错误 (20个) =====
        (
            "How to fix 'IndexError: list index out of range' when accessing my_list[10]?",
            """my_list = [1, 2, 3, 4, 5]
result = my_list[10]""",
            "easy", 0.80,
            ["index-error", "list", "bounds"]
        ),
        (
            "Why do I get 'TypeError: can only concatenate str (not int) to str'?",
            """name = "Alice"
age = 25
print("Name: " + name + ", Age: " + age)""",
            "easy", 0.80,
            ["type-error", "string", "concatenation"]
        ),
        (
            "UnboundLocalError: local variable referenced before assignment",
            """count = 0

def increment():
    count += 1
    return count""",
            "easy", 0.82,
            ["scope", "global", "unbound"]
        ),
        (
            "AttributeError: 'str' object has no attribute 'append'",
            """items = "1,2,3"
items.append(4)""",
            "easy", 0.80,
            ["attribute-error", "type"]
        ),
        (
            "ValueError: too many values to unpack (expected 2)",
            """def get_coords():
    return (1, 2, 3)

x, y = get_coords()""",
            "easy", 0.82,
            ["unpack", "tuple", "value-error"]
        ),
        (
            "KeyError when accessing dictionary key that doesn't exist",
            """user = {"name": "Bob", "email": "bob@example.com"}
print(user["age"])""",
            "easy", 0.80,
            ["key-error", "dictionary"]
        ),
        (
            "IndentationError: unexpected indent in Python function",
            """def calculate():
    x = 1
      y = 2
    return x + y""",
            "easy", 0.78,
            ["indentation", "syntax", "whitespace"]
        ),
        (
            "NameError: name 'function' is not defined when importing",
            """# file1.py
def helper():
    return 42

# file2.py
print(helper())  # Forgot to import!""",
            "easy", 0.80,
            ["name-error", "import", "scope"]
        ),
        (
            "SyntaxError: invalid syntax with print statement (Python 2 vs 3)",
            """print "Hello World"  # Python 2 syntax!""",
            "easy", 0.78,
            ["syntax-error", "python2", "python3"]
        ),
        (
            "ZeroDivisionError: division by zero in empty list average",
            """def average(numbers):
    return sum(numbers) / len(numbers)

avg = average([])""",
            "easy", 0.80,
            ["zero-division", "edge-case"]
        ),
        (
            "Why can't I iterate over a file twice?",
            """with open("data.txt") as f:
    for line in f:
        print("First:", line)

    for line in f:  # Doesn't work!
        print("Second:", line)""",
            "easy", 0.85,
            ["file-io", "cursor", "iteration"]
        ),
        (
            "Mutable default argument: why does my list keep growing?",
            """def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - WRONG!""",
            "easy", 0.85,
            ["mutable-default", "function", "gotcha"]
        ),
        (
            "Python operator precedence: why 'a == b is True' fails",
            """a = [1, 2, 3]
b = [1, 2, 3]
if a == b is True:
    print("Equal")  # Never prints!""",
            "easy", 0.82,
            ["operator-precedence", "chained-comparison"]
        ),
        (
            "UnicodeEncodeError when printing to Windows console",
            """text = "日本語"
print(text)  # Fails on Windows!""",
            "easy", 0.80,
            ["unicode", "encoding", "windows"]
        ),
        (
            "re.match vs re.search: why does my regex not match?",
            """import re
text = "Value: 123"
match = re.match(r"\d+", text)  # Returns None!""",
            "easy", 0.83,
            ["regex", "re-match", "re-search"]
        ),
        (
            "ModuleNotFoundError when importing from subdirectory package",
            """# Project:
# myproject/
#   ├── main.py
#   └── utils/
#       ├── __init__.py
#       └── helper.py

# main.py
from utils.helper import process  # Sometimes fails!""",
            "easy", 0.82,
            ["import", "module", "path", "package"]
        ),
        (
            "Can't subtract offset-naive and offset-aware datetimes",
            """from datetime import datetime, timezone

now_local = datetime.now()
now_utc = datetime.now(timezone.utc)
diff = now_utc - now_local  # Error!""",
            "easy", 0.83,
            ["datetime", "timezone", "naive"]
        ),
        (
            "pandas SettingWithCopyWarning when modifying slice",
            """import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
subset = df[df["A"] > 1]
subset["B"] = 0  # Warning! May not modify original!""",
            "easy", 0.85,
            ["pandas", "dataframe", "setting-with-copy"]
        ),
        (
            "Floating point precision: 0.1 + 0.2 != 0.3",
            """result = 0.1 + 0.2
if result == 0.3:
    print("Equal")  # Never prints!""",
            "easy", 0.85,
            ["floating-point", "precision", "comparison"]
        ),
        (
            "Resource leak: always close files with context manager",
            """f = open("data.txt")
data = f.read()
# What if exception happens here?
f.close()""",
            "easy", 0.82,
            ["context-manager", "with-statement", "resource"]
        ),

        # ===== Medium: 中级问题 (20个) =====
        (
            "Race condition: multi-threaded counter gives wrong result",
            """import threading

count = 0

def increment():
    global count
    for _ in range(100000):
        count += 1  # NOT atomic! Race here!

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(count)  # Never 1,000,000!""",
            "medium", 0.90,
            ["race-condition", "multithreading", "concurrency"]
        ),
        (
            "Circular import causes ImportError between modules",
            """# a.py
from b import B

class A:
    pass

# b.py
from a import A  # Circular dependency!

class B:
    pass""",
            "medium", 0.88,
            ["circular-import", "import", "dependency"]
        ),
        (
            "Memory leak: global cache never evicted grows forever",
            """class Cache:
    _cache = {}

    @classmethod
    def put(cls, key, value):
        cls._cache[key] = value  # Never cleaned! Leak!

# Process millions of items: memory grows without bound""",
            "medium", 0.88,
            ["memory-leak", "cache", "gc"]
        ),
        (
            "Recursion depth exceeded: stack overflow in recursive function",
            """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(2000))  # RecursionError!""",
            "medium", 0.85,
            ["recursion", "stack-overflow", "recursion-depth"]
        ),
        (
            "Global variable access is much slower than local",
            """total = 0

def sum_range():
    global total
    for i in range(1000000):
        total += i  # Global access is slow!""",
            "medium", 0.87,
            ["performance", "global", "variable"]
        ),
        (
            "SQL injection vulnerability in string-formatted queries",
            """def get_user(user_id):
    # UNSAFE! SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)""",
            "medium", 0.92,
            ["sql-injection", "security", "database"]
        ),
        (
            "Generator exhaustion: can iterate only once",
            """def squares(n):
    for i in range(n):
        yield i * i

gen = squares(5)
print(list(gen))  # [0, 1, 4, 9, 16]
print(list(gen))  # [] - Empty!""",
            "medium", 0.86,
            ["generator", "iterator", "exhaustion"]
        ),
        (
            "Closure late binding: all lambdas return same value",
            """def create_multipliers():
    return [lambda x: i * x for i in range(5)]

for m in create_multipliers():
    print(m(2))  # All print 8, 8, 8, 8, 8!""",
            "medium", 0.88,
            ["closure", "late-binding", "lambda", "scoping"]
        ),
        (
            "Pickle security: loading untrusted data is RCE vulnerability",
            """import pickle

def load_data(data):
    # UNSAFE! Arbitrary code execution!
    return pickle.loads(data)  # Can run malware!""",
            "medium", 0.93,
            ["security", "pickle", "deserialization", "rce"]
        ),
        (
            "Deadlock: two threads acquiring locks in reverse order",
            """import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    with lock1:
        with lock2:
            print("Thread 1")

def thread2():
    with lock2:  # REVERSE ORDER!
        with lock1:
            print("Thread 2")
# Both hold one lock and wait forever for the other!""",
            "medium", 0.90,
            ["deadlock", "locking", "concurrency"]
        ),
        (
            "Regex catastrophic backtracking hangs forever",
            """import re

pattern = r"^(a+)+$"
text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaX"
re.match(pattern, text)  # Hangs for MINUTES!""",
            "medium", 0.91,
            ["regex", "catastrophic-backtracking", "performance", "dos"]
        ),
        (
            "asyncio blocking call kills concurrency in event loop",
            """import asyncio
import time

async def blocking_task():
    time.sleep(1)  # Blocks the entire event loop! Should be asyncio.sleep()

async def main():
    await asyncio.gather(*[blocking_task() for _ in range(10)])
# Takes 10 seconds instead of 1 second!""",
            "medium", 0.89,
            ["asyncio", "blocking", "event-loop"]
        ),
        (
            "MRO confusion: super() in multiple inheritance diamond problem",
            """class A:
    def __init__(self): print("A init")

class B(A):
    def __init__(self):
        print("B init")
        super().__init__()

class C(A):
    def __init__(self):
        print("C init")
        super().__init__()

class D(B, C): pass

d = D()  # What's the order? MRO magic!""",
            "medium", 0.87,
            ["mro", "multiple-inheritance", "super", "diamond"]
        ),
        (
            "Metaclass conflict when combining classes with different metaclasses",
            """class MetaA(type): pass
class MetaB(type): pass

class A(metaclass=MetaA): pass
class B(metaclass=MetaB): pass

class C(A, B):  # Metaclass conflict!
    pass""",
            "medium", 0.88,
            ["metaclass", "mro", "conflict"]
        ),
        (
            "Numpy broadcasting shape mismatch error",
            """import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([1, 2, 3, 4])              # (4,)
c = a + b  # Error! Shapes incompatible!""",
            "medium", 0.86,
            ["numpy", "broadcasting", "shape", "array"]
        ),
        (
            "Exception handling is slow: don't use try/except in hot paths",
            """# Slow
try:
    result = my_dict[key]
except KeyError:
    result = default

# Faster
result = my_dict.get(key, default)""",
            "medium", 0.85,
            ["performance", "exception", "try-except"]
        ),
        (
            "GIL: multithreading doesn't speed up CPU-bound Python code",
            """from threading import Thread

def cpu_intensive():
    count = 0
    for _ in range(10_000_000):
        count += 1

threads = [Thread(target=cpu_intensive) for _ in range(4)]
# Still takes ~4x single-thread time due to GIL!""",
            "medium", 0.89,
            ["gil", "cpu-bound", "multithreading", "multiprocessing"]
        ),
        (
            "Python logging: debug messages not showing due to default level",
            """import logging

logging.debug("This won't show!")
logging.info("This also won't!")
logging.warning("This will show")
# Default log level is WARNING, not DEBUG!""",
            "medium", 0.85,
            ["logging", "level", "configuration"]
        ),
        (
            "__del__ destructor order is unpredictable during garbage collection",
            """class Resource:
    def __init__(self, id):
        self.id = id
        print(f"Acquired {id}")

    def __del__(self):
        print(f"Released {self.id}")

r1 = Resource(1)
r2 = Resource(2)
# GC order is NOT guaranteed!""",
            "medium", 0.87,
            ["destructor", "garbage-collection", "__del__"]
        ),
        (
            "Fragile base class: parent changes silently break all subclasses",
            """class Parent:
    def process(self):
        self.step1()
        self.step2()  # Was called validate() in v1!

class Child(Parent):
    def validate(self):  # Was step2 in parent! Oops!
        print("Validation")
# Nobody calls validate() - SILENT FAILURE!""",
            "medium", 0.88,
            ["inheritance", "fragile-base-class", "lsp", "coupling"]
        ),

        # ===== Hard: 高级/分布式问题 (10个) =====
        (
            "ABA problem in lock-free CAS algorithms",
            """# Thread 1 reads value A
# Thread 2 changes it to B, then back to A
# Thread 1 does CAS and succeeds, but state was modified!

def compare_and_swap(ptr, old, new):
    if *ptr == old:  # BUG! ABA: ptr is back to old but changed
        *ptr = new
        return True
    return False""",
            "hard", 0.95,
            ["aba", "lock-free", "cas", "concurrency"]
        ),
        (
            "Thundering herd cache stampede on expiration",
            """CACHE = {}

def get_data(key):
    if key not in CACHE:
        # Miss! ALL threads hit the DB at the same time
        CACHE[key] = expensive_db_query(key)  # THUNDERING HERD!
    return CACHE[key]""",
            "hard", 0.93,
            ["thundering-herd", "cache", "stampede", "scaling"]
        ),
        (
            "Two-phase commit bug: distributed system inconsistency",
            """# Distributed money transfer between banks

# Bank A
def withdraw(amount):
    db.execute("UPDATE accounts SET balance -= ?", [amount])
    # NETWORK PARTITION HAPPENS HERE!

# Bank B
def deposit(amount):
    db.execute("UPDATE accounts SET balance += ?", [amount])

# One commits, one rolls back -> INCONSISTENT STATE!""",
            "hard", 0.94,
            ["distributed-transaction", "2pc", "consistency", "network-partition"]
        ),
        (
            "False sharing: CPU cache line ping-pong destroys performance",
            """# Both counters on the SAME cache line!
class Counters:
    count1 = 0
    count2 = 0

def thread1():
    for _ in range(1000000):
        Counters.count1 += 1  # Invalidates cache line for thread 2

def thread2():
    for _ in range(1000000):
        Counters.count2 += 1  # Invalidates cache line for thread 1
# Cores fight over the same cache line - can be 100x slower!""",
            "hard", 0.93,
            ["false-sharing", "cache", "cpu", "performance"]
        ),
        (
            "Priority inversion: low-priority thread blocks high priority",
            """# Thread LOW (priority 1) holds lock L
# Thread MEDIUM (priority 2) preempts LOW, runs forever
# Thread HIGH (priority 3) needs lock L, but LOW can't run to release
# HIGH waits FOREVER even though it has higher priority!""",
            "hard", 0.92,
            ["priority-inversion", "scheduling", "real-time", "locking"]
        ),
        (
            "Python C extension refcount bug causes memory corruption",
            """// C extension BUG
static PyObject* buggy_function(PyObject* self, PyObject* args) {
    PyObject* obj = PyList_GetItem(list, 0);  // BORROWED reference
    Py_DECREF(obj);  // WRONG! Should NOT decref borrowed ref!
    // Later: use-after-free, double-free, or segfault!""",
            "hard", 0.95,
            ["c-extension", "refcount", "memory-corruption", "segfault"]
        ),
        (
            "CAP theorem: you can't have perfect consistency AND availability",
            """# Network partitions happen in ANY distributed system
# Choose wisely:
#   CP: Consistent - but UNAVAILABLE during partition
#   AP: Available - but EVENTUALLY CONSISTENT only
#   CA: IMPOSSIBLE! P always happens.

# Most bugs come from developers ignoring this fundamental tradeoff.""",
            "hard", 0.94,
            ["cap-theorem", "distributed-systems", "consistency", "availability"]
        ),
        (
            "GC stop-the-world pauses: unpredictable latency with large heaps",
            """# Python uses refcount + generational GC
# With heaps > 10 GB:
#   - GC scans millions of objects
#   - Stop-The-World pauses of 10+ seconds common
#   - Hard to tune for low-latency systems

# Symptoms: random latency spikes, no CPU during pauses,
#           correlates exactly with gen-2 GC runs.""",
            "hard", 0.91,
            ["gc", "pauses", "latency", "tuning", "heap"]
        ),
        (
            "Hash-collision DoS attack: O(n²) dict worst-case performance",
            """# Attacker crafts keys that all hash to the same bucket
# Python dict O(1) average becomes O(n²) WORST case

# 10,000 colliding keys = 100x slowdown
# This was historically how DoS attacks worked against Python web frameworks

# Python now randomizes hash seed, but still a concern in some scenarios!""",
            "hard", 0.92,
            ["hash-collision", "dos", "security", "dictionary"]
        ),
        (
            "JIT deoptimization cliff: PyPy can be slower than CPython",
            """def hot(x):
    # JIT compiles assuming "x is always int"
    if isinstance(x, int):  # Guard inserted by JIT
        return x + 1
    else:
        return str(x) + "!"

# Now pass string occasionally: JIT DEOPTIMIZES -> falls back to interpreter
# Can make code 10-100x slower than CPython!
# This is the "deoptimization cliff" in tracing JITs.""",
            "hard", 0.93,
            ["jit", "pypy", "deoptimization", "performance"]
        ),
    ]

    pool = SeedPromptPool()

    for title, code, diff, quality, tags in bugs:
        prompt_text = f"""Debug and fix the following Python issue:

**Problem**: {title}

**Code**:
```python
{code}
```

Provide:
1. Root cause analysis (what's wrong and why it happens)
2. The complete fixed code
3. Explanation of why the fix works
4. How to prevent this category of bugs in the future"""

        seed = SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.CODE_DEBUG,
            difficulty=Difficulty(diff),
            prompt=prompt_text,
            test_cases=[TaskTestCase(
                input={"title": title},
                expected_output={"root_cause": True, "fixed_code": True, "explanation": True},
                is_public=True,
            )],
            validator_code="def validate(sol):\n    return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol and len(sol['root_cause']) > 20",
            source=SourceType.CRAWLED,
            quality_score=quality,
            tags=["stackoverflow", "debug", "python"] + tags,
        )
        pool.add(seed)

    print(f"✅ 构建 code_debug: {len(pool)} 个")
    return pool


def main():
    print("=" * 60)
    print("合并最终种子池")
    print("=" * 60)

    final_pool = SeedPromptPool()

    # 1. GSM8K数学题
    math_pool = load_gsm8k_seeds()
    for seed in math_pool:
        final_pool.add(seed)

    # 2. Code Debug
    code_pool = build_code_debug_seeds()
    for seed in code_pool:
        final_pool.add(seed)

    # 保存
    output_path = Path("data/seed_prompts_v2_final.json")
    final_pool.save(str(output_path))

    print(f"\n✅ 最终种子池已保存到: {output_path}")

    # 统计报告
    print("\n" + "=" * 60)
    print("📊 最终统计报告")
    print("=" * 60)
    stats = final_pool.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 按source统计
    from collections import Counter
    sources = Counter(s.source.value for s in final_pool)
    print(f"\nSource分布: {dict(sources)}")


if __name__ == "__main__":
    main()
