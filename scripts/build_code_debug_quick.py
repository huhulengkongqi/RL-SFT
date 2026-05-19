"""快速构建高质量code_debug种子 - 基于真实高票StackOverflow Python问题模式"""
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


def build_code_debug_seeds():
    """基于StackOverflow真实问题模式，构建50个高质量真实场景code_debug种子"""

    bugs = [
        # ===== Easy: 基础语法/类型错误 (20个) =====
        (
            "How to fix 'IndexError: list index out of range' when accessing my_list[10]?",
            """my_list = [1, 2, 3, 4, 5]
result = my_list[10]""",
            "easy", 0.80,
            ["index-error", "list", "bounds", "syntax"]
        ),
        (
            "Why do I get 'TypeError: can only concatenate str (not int) to str'?",
            """name = "Alice"
age = 25
print("Name: " + name + ", Age: " + age)""",
            "easy", 0.80,
            ["type-error", "string", "concatenation", "type-casting"]
        ),
        (
            "UnboundLocalError: local variable referenced before assignment",
            """count = 0

def increment():
    count += 1
    return count""",
            "easy", 0.82,
            ["scope", "global", "variable", "unbound"]
        ),
        (
            "AttributeError: 'str' object has no attribute 'append'",
            """items = "1,2,3"
items.append(4)""",
            "easy", 0.80,
            ["attribute-error", "type", "list-method"]
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
            ["key-error", "dictionary", "get-method"]
        ),
        (
            "IndentationError: unexpected indent - Python indentation problem",
            """def calculate():
    x = 1
      y = 2
    return x + y""",
            "easy", 0.78,
            ["indentation", "syntax", "whitespace"]
        ),
        (
            "NameError: name undefined when using function from another file",
            """# file1.py
def helper():
    return 42

# file2.py
print(helper())""",
            "easy", 0.80,
            ["name-error", "import", "scope"]
        ),
        (
            "SyntaxError: invalid syntax with print statement in Python 3",
            """print "Hello World"  # This is Python 2 syntax!""",
            "easy", 0.78,
            ["syntax-error", "python2", "python3", "print"]
        ),
        (
            "ZeroDivisionError: division by zero in average calculation",
            """def average(numbers):
    return sum(numbers) / len(numbers)

avg = average([])""",
            "easy", 0.80,
            ["zero-division", "division", "edge-case"]
        ),
        (
            "Why does my Python loop only iterate once? file.readline issue",
            """with open("data.txt") as f:
    for line in f:
        print("First loop: " + line)

    for line in f:  # This doesn't run!
        print("Second loop: " + line)""",
            "easy", 0.85,
            ["file-io", "cursor", "iteration", "seek"]
        ),
        (
            "Mutable default argument: why does my list keep growing?",
            """def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - WRONG! Should be [2]""",
            "easy", 0.85,
            ["mutable-default", "function", "argument", "gotcha"]
        ),
        (
            "Why does 'a == b is True' behave unexpectedly?",
            """a = [1, 2, 3]
b = [1, 2, 3]
if a == b is True:
    print("Equal?")  # Never prints! Why?""",
            "easy", 0.82,
            ["operator-precedence", "is-vs-equals", "chained-comparison"]
        ),
        (
            "UnicodeEncodeError when printing to console on Windows",
            """text = "日本語"
print(text)  # Fails on Windows console!""",
            "easy", 0.80,
            ["unicode", "encoding", "windows", "console"]
        ),
        (
            "Why does my regex not match anything? re.match vs re.search",
            """import re
text = "Value: 123"
match = re.match(r"\d+", text)  # Returns None!""",
            "easy", 0.83,
            ["regex", "re-match", "re-search", "pattern"]
        ),
        (
            "ModuleNotFoundError when importing module from same directory",
            """# Project structure:
# myproject/
#   ├── main.py
#   └── utils/
#       └── helper.py

# main.py
from utils.helper import process  # Fails!""",
            "easy", 0.82,
            ["import", "module", "path", "package"]
        ),
        (
            "datetime TypeError: can't subtract offset-naive and offset-aware datetimes",
            """from datetime import datetime, timezone

now_local = datetime.now()
now_utc = datetime.now(timezone.utc)
diff = now_utc - now_local  # Error!""",
            "easy", 0.83,
            ["datetime", "timezone", "naive", "offset-aware"]
        ),
        (
            "SettingWithCopyWarning in pandas when modifying DataFrame slice",
            """import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
subset = df[df["A"] > 1]
subset["B"] = 0  # Warning! And may not modify original df!""",
            "easy", 0.85,
            ["pandas", "dataframe", "setting-with-copy", "chained-indexing"]
        ),
        (
            "Why do 0.1 + 0.2 != 0.3 in Python?",
            """result = 0.1 + 0.2
if result == 0.3:
    print("Equal")  # Never prints!""",
            "easy", 0.85,
            ["floating-point", "precision", "ieee754", "comparison"]
        ),
        (
            "How to correctly close files and avoid resource leaks?",
            """f = open("data.txt")
data = f.read()
# What if an exception happens before close()?
f.close()""",
            "easy", 0.82,
            ["context-manager", "with-statement", "resource", "file"]
        ),

        # ===== Medium: 中级问题 (20个) =====
        (
            "Race condition in multi-threaded counter - threads not giving expected result",
            """import threading

count = 0

def increment():
    global count
    for _ in range(100000):
        count += 1  # Not atomic! Race condition here

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(count)  # Should be 1,000,000 but it's always less!""",
            "medium", 0.90,
            ["race-condition", "multithreading", "concurrency", "atomic"]
        ),
        (
            "Circular import: modules importing each other causes ImportError",
            """# a.py
from b import B

class A:
    pass

# b.py
from a import A  # Circular dependency!

class B:
    pass""",
            "medium", 0.88,
            ["circular-import", "import", "dependency", "architecture"]
        ),
        (
            "Memory leak: why does my Python program keep growing in memory?",
            """import gc

class Cache:
    _cache = {}

    @classmethod
    def put(cls, key, value):
        cls._cache[key] = value  # Never cleaned up!

# Process millions of items - memory keeps growing""",
            "medium", 0.88,
            ["memory-leak", "cache", "gc", "garbage-collection"]
        ),
        (
            "Python recursion depth exceeded - maximum recursion depth reached",
            """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(2000))  # RecursionError!""",
            "medium", 0.85,
            ["recursion", "stack-overflow", "recursion-depth", "iteration"]
        ),
        (
            "Why is my loop with global variables so slow?",
            """total = 0
def sum_range():
    global total
    for i in range(1000000):
        total += i  # Global variable access is slow!""",
            "medium", 0.87,
            ["performance", "global", "variable", "scope"]
        ),
        (
            "SQL injection vulnerability in string-formatted database queries",
            """def get_user(user_id):
    # UNSAFE! SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)""",
            "medium", 0.92,
            ["sql-injection", "security", "database", "prepared-statement"]
        ),
        (
            "Generator exhaustion: why can I iterate only once?",
            """def squares(n):
    for i in range(n):
        yield i * i

gen = squares(5)
print(list(gen))  # [0, 1, 4, 9, 16]
print(list(gen))  # [] - Empty! Why?""",
            "medium", 0.86,
            ["generator", "iterator", "exhaustion", "yield"]
        ),
        (
            "Closure late binding: why all lambdas return the same value?",
            """def create_multipliers():
    return [lambda x: i * x for i in range(5)]

for m in create_multipliers():
    print(m(2))  # All print 8, 8, 8, 8, 8 instead of 0, 2, 4, 6, 8!""",
            "medium", 0.88,
            ["closure", "late-binding", "lambda", "scoping"]
        ),
        (
            "Deserialization security: pickle.loads from untrusted source is dangerous",
            """import pickle

def load_data(data):
    # UNSAFE! Arbitrary code execution vulnerability
    return pickle.loads(data)  # Can run malware!""",
            "medium", 0.93,
            ["security", "pickle", "deserialization", "rce"]
        ),
        (
            "Deadlock: two threads waiting for each other's locks",
            """import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    with lock1:
        with lock2:
            print("Thread 1 got both locks")

def thread2():
    with lock2:  # Acquired in reverse order!
        with lock1:
            print("Thread 2 got both locks")
# DEADLOCK: both threads hold one lock and wait forever!""",
            "medium", 0.90,
            ["deadlock", "locking", "concurrency", "resource-order"]
        ),
        (
            "Slow regex causing catastrophic backtracking hangs forever",
            """import re

# Evil regex that can take exponential time
pattern = r"^(a+)+$"
text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaX"  # With a non-matching character
re.match(pattern, text)  # Hangs for minutes!""",
            "medium", 0.91,
            ["regex", "catastrophic-backtracking", "performance", "dos"]
        ),
        (
            "asyncio blocking call: why is my async program not concurrent?",
            """import asyncio
import time

async def blocking_task():
    time.sleep(1)  # Blocks the entire event loop! Should use asyncio.sleep()

async def main():
    await asyncio.gather(*[blocking_task() for _ in range(10)])
# Takes 10 seconds instead of 1 second!""",
            "medium", 0.89,
            ["asyncio", "blocking", "event-loop", "concurrency"]
        ),
        (
            "Python class inheritance: super() doesn't work as expected with multiple inheritance",
            """class A:
    def __init__(self):
        print("A init")

class B(A):
    def __init__(self):
        print("B init")
        super().__init__()

class C(A):
    def __init__(self):
        print("C init")
        super().__init__()

class D(B, C):
    pass

d = D()  # What's the order? MRO confusion!""",
            "medium", 0.87,
            ["mro", "inheritance", "multiple-inheritance", "super"]
        ),
        (
            "Metaclass conflict: can't create a consistent MRO",
            """class MetaA(type):
    pass

class MetaB(type):
    pass

class A(metaclass=MetaA):
    pass

class B(metaclass=MetaB):
    pass

class C(A, B):  # Metaclass conflict!
    pass""",
            "medium", 0.88,
            ["metaclass", "mro", "conflict", "inheritance"]
        ),
        (
            "numpy broadcasting error: operands could not be broadcast together",
            """import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = np.array([1, 2, 3, 4])            # shape (4,)
c = a + b  # Error! Shapes incompatible!""",
            "medium", 0.86,
            ["numpy", "broadcasting", "shape", "array"]
        ),
        (
            "Why are Python exceptions slower than if-checks?",
            """# Slow
try:
    result = my_dict[key]
except KeyError:
    result = default

# Faster
result = my_dict.get(key, default)""",
            "medium", 0.85,
            ["performance", "exception", "try-except", "flow-control"]
        ),
        (
            "How to avoid GIL for CPU-bound tasks in Python?",
            """from threading import Thread

def cpu_intensive():
    count = 0
    for _ in range(10_000_000):
        count += 1

# Multithreading doesn't speed up CPU-bound work due to GIL!
threads = [Thread(target=cpu_intensive) for _ in range(4)]
# Still takes ~4x single-thread time!""",
            "medium", 0.89,
            ["gil", "cpu-bound", "multithreading", "multiprocessing"]
        ),
        (
            "Python logging: why aren't my debug messages showing up?",
            """import logging

logging.debug("This won't show up!")
logging.info("This too won't show!")
logging.warning("This will show")
# Default level is WARNING, not DEBUG!""",
            "medium", 0.85,
            ["logging", "level", "configuration", "debug"]
        ),
        (
            "Why is __del__ called out of order with context managers?",
            """class Resource:
    def __init__(self, id):
        self.id = id
        print(f"Acquired {id}")

    def __del__(self):
        print(f"Released {self.id}")

# GC order is unpredictable!
r1 = Resource(1)
r2 = Resource(2)""",
            "medium", 0.87,
            ["destructor", "garbage-collection", "__del__", "finalizer"]
        ),
        (
            "Fragile base class problem: breaking changes in parent break all subclasses",
            """class Parent:
    def process(self):
        self.step1()
        self.step2()  # Was called validate() in v1!

class Child(Parent):
    def validate(self):  # Was step2 in parent, now renamed!
        print("Validation")
# Nobody calls validate() anymore - silent failure!""",
            "medium", 0.88,
            ["inheritance", "fragile-base-class", "coupling", "lsp"]
        ),

        # ===== Hard: 高级/分布式问题 (10个) =====
        (
            "ABA problem in lock-free data structures with compare-and-swap",
            """# Thread 1 reads value A
# Thread 2 changes it to B, then back to A
# Thread 1 does CAS and succeeds, but state was actually modified!

def compare_and_swap(ptr, old, new):
    if *ptr == old:  # Wrong! ABA: ptr is back to old but changed
        *ptr = new
        return True
    return False""",
            "hard", 0.95,
            ["aba", "lock-free", "cas", "concurrency"]
        ),
        (
            "Thundering herd: cache stampede when cache expires",
            """import time

CACHE = {}

def get_data(key):
    if key not in CACHE:
        # Cache miss! ALL threads hit the DB at the same time
        CACHE[key] = expensive_database_query(key)  # Thundering herd!
    return CACHE[key]""",
            "hard", 0.93,
            ["thundering-herd", "cache", "stampede", "scaling"]
        ),
        (
            "Two-phase commit bug: distributed system inconsistency on failure",
            """# Distributed transaction: money transfer between two banks

# Bank A
def withdraw(amount):
    db.execute("UPDATE accounts SET balance = balance - ?", [amount])
    # Network partition happens here!

# Bank B
def deposit(amount):
    db.execute("UPDATE accounts SET balance = balance + ?", [amount])

# One commits, one rolls back → inconsistent state!""",
            "hard", 0.94,
            ["distributed-transaction", "2pc", "consistency", "network-partition"]
        ),
        (
            "False sharing: CPU cache line ping-pong killing multithreaded performance",
            """class Counter:
    # Both counters on same cache line!
    count1 = 0
    count2 = 0

def thread1():
    for _ in range(1000000):
        Counter.count1 += 1  # Invalidates cache line for thread 2

def thread2():
    for _ in range(1000000):
        Counter.count2 += 1  # Invalidates cache line for thread 1
# Both cores fight over the same cache line - 100x slower!""",
            "hard", 0.93,
            ["false-sharing", "cache", "cpu", "performance"]
        ),
        (
            "Priority inversion: low-priority thread holds lock that high priority needs",
            """# Thread LOW (priority 1) holds lock L
# Thread MEDIUM (priority 2) preempts LOW, runs forever
# Thread HIGH (priority 3) needs lock L, but LOW can't run to release
# HIGH waits forever even though it has higher priority!""",
            "hard", 0.92,
            ["priority-inversion", "scheduling", "real-time", "locking"]
        ),
        (
            "Python C extension memory corruption: refcount bug causes segfault",
            """// C extension bug
static PyObject* buggy_function(PyObject* self, PyObject* args) {
    PyObject* obj = PyList_GetItem(list, 0);  // Borrowed reference
    Py_DECREF(obj);  // WRONG! Should NOT decref borrowed ref
    // Later: use-after-free, double-free, or segfault!""",
            "hard", 0.95,
            ["c-extension", "refcount", "memory-corruption", "segfault"]
        ),
        (
            "CAP theorem: why you can't have both 100% consistency and availability",
            """# Network partitions happen in ANY distributed system
# Choose:
#   CP: Consistent + Partition-tolerant (unavailable during partition)
#   AP: Available + Partition-tolerant (eventually consistent only)
#   CA: Impossible! P always happens.

# Many bugs come from developers not understanding this tradeoff is inevitable.""",
            "hard", 0.94,
            ["cap-theorem", "distributed-systems", "consistency", "availability"]
        ),
        (
            "GC pause tuning: why Python pauses unpredictably with large heaps",
            """# Python uses reference counting + generational GC
# With heaps > 10GB:
#   - GC can scan millions of objects
#   - Stop-The-World pauses of 10+ seconds are common
#   - Hard to tune for low-latency systems

# Symptoms: random latency spikes, no CPU during pauses,
#           happens exactly when GC runs in gen 2 (old generation)""",
            "hard", 0.91,
            ["gc", "pauses", "latency", "tuning", "heap"]
        ),
        (
            "Hash-collision DOS: worst-case O(n²) dict performance for attackers",
            """# Attacker crafts keys that all hash to the same bucket
# Python dict O(1) average becomes O(n) worst case

# 10,000 colliding keys = 100x slowdown
# This is how DoS attacks against Python web frameworks worked historically

# Python now randomizes hash seed, but this is still a concern!""",
            "hard", 0.92,
            ["hash-collision", "dos", "security", "dictionary"]
        ),
        (
            "JIT deoptimization: why PyPy sometimes runs slower than CPython",
            """def hot_function(x):
    # JIT compiles this as "x is always int"
    if isinstance(x, int):  # Guard check inserted by JIT
        return x + 1
    else:
        return str(x) + "!"

# Now pass string occasionally: JIT deoptimizes → falls back to interpreter
# Can make code 10-100x slower than CPython!
# This is the "deoptimization cliff" problem in tracing JITs.""",
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
1. Root cause analysis (explain what's wrong and why)
2. The fixed code
3. Explanation of why the fix works
4. If applicable: how to prevent this category of bugs in the future"""

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
            tags=["stackoverflow-pattern", "debug", "python"] + tags,
        )
        pool.add(seed)

    return pool


def main():
    print("=" * 60)
    print("构建 code_debug 高质量种子池")
    print("=" * 60)

    pool = build_code_debug_seeds()
    print(f"\n✅ 构建完成: {len(pool)} 个种子")

    # 保存
    output_path = Path("data/processed/stackoverflow_50_seeds.json")
    output_path.parent.mkdir(exist_ok=True)
    pool.save(str(output_path))
    print(f"✅ 已保存到: {output_path}")

    # 统计报告
    print("\n" + "=" * 60)
    print("最终统计报告")
    print("=" * 60)
    stats = pool.get_stats()
    print(json.dumps(stats, indent=2))

    # 抽样展示
    print("\n样例展示 (每难度1题):")
    for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        seeds = [s for s in pool if s.difficulty == diff]
        if seeds:
            seed = seeds[0]
            print(f"\n--- {diff.value.upper()}:")
            first_line = seed.prompt.split("\n")[0][:60]
            print(f"  问题: {first_line}...")
            print(f"  质量分: {seed.quality_score}")
            print(f"  Tags: {seed.tags[:5]}")


if __name__ == "__main__":
    main()
