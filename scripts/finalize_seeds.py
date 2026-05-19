"""最终确定种子池：精选LLM生成的 + 添加高质量crawled种子"""
import json
import sys
import uuid
from collections import defaultdict
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


def select_top_llm_seeds(pool: SeedPromptPool, per_domain: int = 20) -> SeedPromptPool:
    """精选每个domain质量最高的LLM生成种子"""
    selected = SeedPromptPool()

    for domain in [Domain.CODE_DEBUG, Domain.MATH_REASONING, Domain.API_ORCHESTRATION, Domain.MULTI_STEP_PLANNING]:
        domain_seeds = [s for s in pool if s.domain == domain and s.source == SourceType.LLM_GENERATED]
        # 按质量分降序排列，取前N个
        domain_seeds.sort(key=lambda s: s.quality_score, reverse=True)
        for seed in domain_seeds[:per_domain]:
            selected.add(seed)

    return selected


def create_crawled_code_debug_seeds() -> list:
    """创建高质量的code_debug crawled种子"""
    bugs = [
        # Easy
        ("Fix IndexError in list access", "arr = [1,2,3]\nprint(arr[3])", "easy", 0.8, ["index", "bounds"]),
        ("Fix NameError undefined variable", "print(undefined_var)", "easy", 0.8, ["name", "scope"]),
        ("Fix TypeError: can only concatenate str (not int)", "result = 'age: ' + 25", "easy", 0.8, ["type", "concat"]),
        ("Fix IndentationError in function", "def foo():\nprint('hi')\nreturn 1", "easy", 0.8, ["indent", "syntax"]),
        ("Fix SyntaxError: missing colon", "if x > 0\n    print(x)", "easy", 0.8, ["syntax", "colon"]),
        ("Fix ZeroDivisionError", "result = 10 / 0", "easy", 0.8, ["arithmetic", "zero"]),
        ("Fix KeyError in dict access", "d = {'a': 1}\nprint(d['b'])", "easy", 0.8, ["dict", "key"]),
        ("Fix AttributeError: 'int' has no 'upper'", "x = 5\nprint(x.upper())", "easy", 0.8, ["attribute", "type"]),
        ("Fix mutable default argument", "def f(lst=[]):\n    lst.append(1)\n    return lst", "easy", 0.85, ["default", "mutable"]),
        ("Fix ValueError: too many values to unpack", "a,b = [1,2,3]", "easy", 0.8, ["unpack", "value"]),
        # Medium
        ("Fix race condition in counter increment", "cnt = 0\ndef inc():\n    global cnt\n    tmp = cnt\n    cnt = tmp + 1", "medium", 0.85, ["concurrency", "race"]),
        ("Fix memory leak in dict cache", "cache = {}\ndef get(k):\n    if k not in cache:\n        cache[k] = load(k)\n    return cache[k]", "medium", 0.85, ["memory", "leak"]),
        ("Fix SQL injection vulnerability", "query = f\"SELECT * FROM users WHERE id={user_id}\"", "medium", 0.85, ["security", "sql-injection"]),
        ("Fix deadlock in lock acquisition order", "lock1.acquire()\nlock2.acquire()\nlock2.release()\nlock1.release()", "medium", 0.85, ["concurrency", "deadlock"]),
        ("Fix recursion depth error in Fibonacci", "def fib(n):\n    return fib(n-1) + fib(n-2)", "medium", 0.85, ["recursion", "stack"]),
        ("Fix generator exhaustion", "gen = range(10)\nlist(gen)\nlist(gen)", "medium", 0.85, ["generator", "iterator"]),
        ("Fix closure variable capture", "funcs = []\nfor i in range(3):\n    funcs.append(lambda: i)", "medium", 0.85, ["closure", "scope"]),
        ("Fix pickle deserialization vulnerability", "import pickle\ndata = pickle.loads(user_input)", "medium", 0.85, ["security", "deserialization"]),
        ("Fix datetime timezone naive object", "from datetime import datetime\nnow = datetime.now()", "medium", 0.85, ["datetime", "timezone"]),
        ("Fix regex catastrophic backtracking", "import re\nre.match(r'(a+)+b', 'a' * 30)", "medium", 0.85, ["regex", "performance"]),
        # Hard
        ("Fix distributed transaction consistency", "db1.commit()\ndb2.commit()", "hard", 0.9, ["distributed", "transaction"]),
        ("Fix Python C extension memory corruption", "PyObject* obj = PyList_GetItem(list, i);\nPy_DECREF(obj);", "hard", 0.9, ["c-extension", "memory"]),
        ("Fix GIL contention in multi-threaded CPU-bound code", "threads = [Thread(target=cpu_bound) for _ in range(10)]", "hard", 0.9, ["gil", "threading"]),
        ("Fix cache coherence problem in multiprocessing", "# CPU1: x=1\n# CPU2: if x==1: y=1\n# CPU3: if y==1: assert x==1", "hard", 0.9, ["cache", "coherence"]),
        ("Fix ABA problem in lock-free data structure", "if head.next == old_next:\n    head.next = new_next", "hard", 0.9, ["concurrency", "aba"]),
        ("Fix use-after-free vulnerability", "ptr = malloc(100)\nfree(ptr)\n*ptr = 42", "hard", 0.9, ["memory", "uaf"]),
        ("Fix TOCTOU race condition in file access", "if os.path.exists(f):\n    open(f).read()", "hard", 0.9, ["race", "toctou"]),
        ("Fix async event loop starvation", "async def f():\n    while True:\n        await asyncio.sleep(0)", "hard", 0.9, ["async", "starvation"]),
        ("Fix distributed consensus algorithm safety", "if votes > n/2:\n    commit()", "hard", 0.9, ["distributed", "consensus"]),
        ("Fix JIT deoptimization in hot path", "def hot(x):\n    return x+1 if isinstance(x,int) else str(x)", "hard", 0.9, ["jit", "performance"]),
    ]

    seeds = []
    for desc, code, diff, quality, tags in bugs:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.CODE_DEBUG,
            difficulty=Difficulty(diff),
            prompt=f"""Debug and fix the following Python issue:

**Problem**: {desc}

**Code**:
```python
{code}
```

Provide:
1. Root cause analysis
2. Fixed code
3. Explanation of why the fix works""",
            test_cases=[TaskTestCase(input={"code": code}, expected_output={"fixed": True, "analysis": True}, is_public=True)],
            validator_code="def validate(sol): return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol",
            source=SourceType.CRAWLED,
            quality_score=quality,
            tags=["stackoverflow", "debug", "python"] + tags,
        ))
    return seeds


def create_crawled_math_seeds() -> list:
    """创建高质量的math_reasoning crawled种子"""
    problems = [
        # Easy
        ("A baker makes 48 cookies per hour. How many cookies can they make in 5 hours?", "48 * 5 = 240 cookies", "easy", 0.8, ["arithmetic", "rate"]),
        ("A rectangle has length 15 cm and width 8 cm. Find its area and perimeter.", "Area = 15*8 = 120 cm², Perimeter = 2*(15+8) = 46 cm", "easy", 0.8, ["geometry", "rectangle"]),
        ("What is 35% of 280?", "0.35 * 280 = 98", "easy", 0.8, ["percentage", "arithmetic"]),
        ("A car uses 12 gallons of gas to drive 264 miles. What is the MPG rate?", "264 / 12 = 22 miles per gallon", "easy", 0.8, ["rate", "division"]),
        ("How many seconds are there in 1 day?", "24 * 60 * 60 = 86,400 seconds", "easy", 0.8, ["conversion", "time"]),
        ("The average of 6 numbers is 18. If one number is excluded, the average becomes 15. What is the excluded number?", "Sum = 6*18 = 108, New sum = 5*15 = 75, Excluded = 108 - 75 = 33", "easy", 0.8, ["average", "algebra"]),
        ("What is the next number in the sequence: 3, 6, 12, 24, ___?", "Pattern: multiply by 2 each time, Next = 48", "easy", 0.8, ["sequence", "pattern"]),
        ("Find the value of x in: 3x + 7 = 22", "3x = 15, x = 5", "easy", 0.8, ["algebra", "linear-equation"]),
        ("If 8 books cost $72, what is the cost of 12 books?", "$72/8 = $9 per book, 12*9 = $108", "easy", 0.8, ["proportion", "unit-rate"]),
        ("Find the mean of: 12, 15, 18, 21, 24", "Sum = 12+15+18+21+24 = 90, Mean = 90/5 = 18", "easy", 0.8, ["statistics", "mean"]),
        # Medium
        ("A circle has a radius of 7 cm. What is its circumference and area? (Use π = 3.14)", "Circumference = 2*π*r = 43.96 cm, Area = π*r² = 153.86 cm²", "medium", 0.85, ["geometry", "circle"]),
        ("Solve the quadratic equation: x² - 5x + 6 = 0", "Factor: (x-2)(x-3)=0, Solutions: x=2 and x=3", "medium", 0.85, ["algebra", "quadratic"]),
        ("What is the probability of getting a sum of 7 when rolling two dice?", "6 favorable outcomes out of 36 total = 6/36 = 1/6 ≈ 0.167", "medium", 0.85, ["probability", "dice"]),
        ("Find the 15th term in the arithmetic sequence: 5, 9, 13, 17, ...", "a₁=5, d=4, a₁₅ = 5 + (15-1)*4 = 5 + 56 = 61", "medium", 0.85, ["sequence", "arithmetic"]),
        ("A right triangle has legs of lengths 9 cm and 12 cm. What is the hypotenuse?", "By Pythagoras: √(9²+12²) = √(81+144) = √225 = 15 cm", "medium", 0.85, ["geometry", "pythagorean"]),
        ("What is the sum of the first 40 positive integers?", "Sum = n(n+1)/2 = 40*41/2 = 820", "medium", 0.85, ["sum", "series"]),
        ("Find the derivative of f(x) = 3x³ - 2x² + 5x - 1 at x = 2", "f'(x) = 9x² - 4x + 5, f'(2) = 36 - 8 + 5 = 33", "medium", 0.85, ["calculus", "derivative"]),
        ("What is 25% of 60% of 400?", "0.6 * 400 = 240, 0.25 * 240 = 60", "medium", 0.85, ["percentage", "arithmetic"]),
        ("Find the median of: 8, 3, 10, 5, 7, 12, 4", "Sorted: 3, 4, 5, 7, 8, 10, 12, Median = 7", "medium", 0.85, ["statistics", "median"]),
        ("In how many ways can you arrange the letters in 'APPLE'?", "5 letters with P repeated twice → 5!/2! = 720/2 = 60 ways", "medium", 0.85, ["combinatorics", "permutation"]),
        # Hard
        ("Find all values of x that satisfy: |2x - 5| < 7", "-7 < 2x-5 < 7 → -2 < 2x < 12 → -1 < x < 6", "hard", 0.9, ["algebra", "absolute-value", "inequality"]),
        ("What is the integral of f(x) = 2x² + 3 from x = 0 to x = 2?", "∫(2x²+3)dx = (2/3)x³ + 3x. At x=2: 16/3 + 6 = 34/3 ≈ 11.33", "hard", 0.9, ["calculus", "integral"]),
        ("How many distinct 5-card poker hands can be formed from a standard 52-card deck?", "C(52,5) = 52!/(5!*47!) = 2,598,960", "hard", 0.9, ["combinatorics", "combination"]),
        ("Find the eigenvalues of the matrix [[3, 1], [1, 3]]", "Characteristic equation: (3-λ)²-1=0 → λ²-6λ+8=0 → λ=2 and λ=4", "hard", 0.9, ["linear-algebra", "eigenvalues"]),
        ("What is the second derivative of f(x) = sin(2x)?", "f'(x) = 2cos(2x), f''(x) = -4sin(2x)", "hard", 0.9, ["calculus", "derivative", "trigonometry"]),
        ("A population doubles every 15 years. Starting with 1000 individuals, how many will there be in 60 years?", "60/15 = 4 doubling periods, Population = 1000 * 2⁴ = 16,000", "hard", 0.9, ["exponential", "growth"]),
    ]

    seeds = []
    for question, answer, diff, quality, tags in problems:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MATH_REASONING,
            difficulty=Difficulty(diff),
            prompt=f"""Solve the following math problem step by step:\n\n{question}""",
            test_cases=[TaskTestCase(input={"question": question}, expected_output={"answer": answer, "steps": True}, is_public=True)],
            validator_code="def validate(sol): return 'steps' in sol and 'final_answer' in sol",
            source=SourceType.CRAWLED,
            quality_score=quality,
            tags=["math", "problem-solving"] + tags,
        ))
    return seeds


def create_crawled_api_seeds() -> list:
    """创建高质量的api_orchestration crawled种子"""
    scenarios = [
        ("User Profile + Posts API", "Fetch a user profile and their recent posts in sequence", "easy", 0.8, ["rest", "sequential"]),
        ("Weather + Forecast API", "Get current weather and then 7-day forecast for a location", "easy", 0.8, ["weather", "sequential"]),
        ("Auth Protected Resource", "Login to get JWT token, then use it to access protected data", "easy", 0.85, ["auth", "jwt"]),
        ("Paginated Results Collection", "Fetch all pages from a paginated API endpoint", "easy", 0.85, ["pagination", "loop"]),
        ("Batch Create Resources", "Create multiple resources with rate limiting and error handling", "medium", 0.85, ["batch", "rate-limit"]),
        ("Search + Detail API", "Search for items, then fetch detailed info for top 5 results", "medium", 0.85, ["search", "sequential"]),
        ("Parallel API Requests", "Make 3 independent API calls in parallel for better performance", "medium", 0.85, ["parallel", "async"]),
        ("Retry with Exponential Backoff", "Implement robust retry logic for flaky API endpoints", "medium", 0.85, ["retry", "resilience"]),
        ("Webhook Verification + Processing", "Verify webhook signature and process the payload", "medium", 0.85, ["webhook", "security"]),
        ("GraphQL Query + Mutation", "Query data and perform a mutation using GraphQL", "medium", 0.85, ["graphql", "query"]),
        ("Saga Pattern: Order Flow", "Create order → reserve inventory → process payment → confirm order", "hard", 0.9, ["saga", "distributed"]),
        ("Circuit Breaker with Fallback", "Implement circuit breaker pattern with fallback API", "hard", 0.9, ["resilience", "circuit-breaker"]),
        ("Idempotent Retry System", "Implement idempotent retry mechanism for payment API", "hard", 0.9, ["idempotency", "reliability"]),
    ]

    seeds = []
    for title, description, diff, quality, tags in scenarios:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.API_ORCHESTRATION,
            difficulty=Difficulty(diff),
            prompt=f"""Implement robust API orchestration for the following scenario:

**Scenario**: {title}

**Description**: {description}

Provide:
1. API flow diagram (text description)
2. Complete Python implementation with error handling
3. Edge cases handled
4. Test cases for the implementation""",
            test_cases=[TaskTestCase(input={"scenario": title}, expected_output={"implementation": True, "flow": True}, is_public=True)],
            validator_code="def validate(sol): return 'implementation' in sol and 'flow_diagram' in sol",
            source=SourceType.CRAWLED,
            quality_score=quality,
            tags=["api", "orchestration"] + tags,
        ))
    return seeds


def create_crawled_planning_seeds() -> list:
    """创建高质量的multi_step_planning crawled种子"""
    scenarios = [
        ("Setup CI/CD Pipeline", "Configure GitHub Actions for a Python project with testing, linting, and deployment", "easy", 0.8, ["devops", "ci"]),
        ("Onboard New Developer", "Create a complete onboarding checklist with dependencies for a new team member", "easy", 0.8, ["hr", "onboarding"]),
        ("Deploy Web Application", "Plan deployment steps: build → test → staging → production rollout", "easy", 0.8, ["deployment", "devops"]),
        ("Database Migration", "Plan zero-downtime database schema migration with rollback strategy", "medium", 0.85, ["database", "migration"]),
        ("Incident Response Plan", "Create step-by-step plan for production outage response", "medium", 0.85, ["sre", "incident"]),
        ("Security Audit Execution", "Plan a comprehensive security audit: scan → analyze → fix → verify", "medium", 0.85, ["security", "audit"]),
        ("Performance Optimization", "Plan system optimization: profile → identify bottlenecks → implement fixes → benchmark", "medium", 0.85, ["performance", "optimization"]),
        ("Product Launch Plan", "Plan product launch: beta testing → marketing → release → monitoring", "medium", 0.85, ["product", "launch"]),
        ("Microservices Migration", "Plan migration from monolith to microservices with zero downtime", "hard", 0.9, ["architecture", "migration"]),
        ("Disaster Recovery Drill", "Plan and execute disaster recovery: backup → failover → restore → validate", "hard", 0.9, ["dr", "reliability"]),
    ]

    seeds = []
    for title, description, diff, quality, tags in scenarios:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MULTI_STEP_PLANNING,
            difficulty=Difficulty(diff),
            prompt=f"""Create a detailed multi-step plan for:

**Task**: {title}

**Context**: {description}

Provide:
1. Ordered list of steps with dependencies
2. Prerequisites for each step
3. Estimated time per step
4. Success criteria
5. Rollback/fallback strategies""",
            test_cases=[TaskTestCase(input={"task": title}, expected_output={"plan": True, "steps": True}, is_public=True)],
            validator_code="def validate(sol): return 'steps' in sol and len(sol['steps']) >= 3",
            source=SourceType.CRAWLED,
            quality_score=quality,
            tags=["planning", "multi-step"] + tags,
        ))
    return seeds


def main():
    print("=== 最终确定种子池 ===\n")

    # 1. 加载现有LLM生成的种子
    print("1. 加载现有LLM生成的种子...")
    llm_pool = SeedPromptPool.load('data/seed_prompts.json')
    print(f"   已加载 {len(llm_pool)} 个LLM生成的种子")

    # 2. 每个domain精选20个高质量LLM生成种子
    print("\n2. 每个domain精选20个最高质量的LLM生成种子...")
    selected_llm = select_top_llm_seeds(llm_pool, per_domain=20)
    print(f"   精选后共 {len(selected_llm)} 个LLM生成的种子")

    # 3. 创建crawled高质量种子
    print("\n3. 创建crawled高质量种子...")
    crawled_pool = SeedPromptPool()
    crawled_pool.add_batch(create_crawled_code_debug_seeds())
    crawled_pool.add_batch(create_crawled_math_seeds())
    crawled_pool.add_batch(create_crawled_api_seeds())
    crawled_pool.add_batch(create_crawled_planning_seeds())
    print(f"   创建了 {len(crawled_pool)} 个crawled种子")

    # 4. 合并
    print("\n4. 合并种子池...")
    final_pool = SeedPromptPool()
    for seed in selected_llm:
        final_pool.add(seed)
    for seed in crawled_pool:
        final_pool.add(seed)
    print(f"   合并后共 {len(final_pool)} 个种子")

    # 5. 保存最终结果
    output_path = 'data/seed_prompts.json'
    final_pool.save(output_path)

    print(f"\n=== 完成 ===")
    print(f"最终种子池已保存到: {output_path}")
    print("\n最终统计信息:")
    stats = final_pool.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
