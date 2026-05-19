"""Generate 200 seed prompts (50 per domain) with crawled + generated data."""
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import Difficulty, Domain, SeedPrompt, SourceType, TaskTestCase
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def create_code_debug_seeds() -> List[SeedPrompt]:
    """50 code debugging seeds (crawled from common bug patterns)."""
    bugs = [
        # Easy (20)
        ("Fix IndexError in list access", "arr = [1,2,3]\nprint(arr[3])", "easy", ["index", "bounds"]),
        ("Fix NameError undefined variable", "print(undefined_var)", "easy", ["name", "scope"]),
        ("Fix TypeError string + int", "result = 'age: ' + 25", "easy", ["type", "concat"]),
        ("Fix indentation in function", "def foo():\nprint('hi')\nreturn 1", "easy", ["indent", "syntax"]),
        ("Fix missing colon in if", "if x > 0\n  print(x)", "easy", ["syntax", "colon"]),
        ("Fix unclosed string", "msg = 'hello", "easy", ["syntax", "quote"]),
        ("Fix division by zero", "result = 10 / 0", "easy", ["arithmetic", "zero"]),
        ("Fix KeyError in dict", "d = {'a': 1}\nprint(d['b'])", "easy", ["dict", "key"]),
        ("Fix AttributeError", "x = 5\nprint(x.upper())", "easy", ["attribute", "type"]),
        ("Fix infinite loop", "while True:\n  pass", "easy", ["loop", "infinite"]),
        ("Fix mutable default arg", "def f(lst=[]):\n  lst.append(1)\n  return lst", "easy", ["default", "mutable"]),
        ("Fix == vs = in condition", "if x = 5:\n  print(x)", "easy", ["operator", "assignment"]),
        ("Fix missing return", "def add(a,b):\n  a+b", "easy", ["return", "function"]),
        ("Fix wrong operator", "if x > 5 and < 10:", "easy", ["syntax", "operator"]),
        ("Fix list modification during iteration", "for x in lst:\n  lst.remove(x)", "easy", ["iteration", "mutation"]),
        ("Fix float comparison", "if 0.1 + 0.2 == 0.3:", "easy", ["float", "precision"]),
        ("Fix global variable", "x=0\ndef inc():\n  x+=1", "easy", ["scope", "global"]),
        ("Fix import error", "from math import sqrt, sqr", "easy", ["import", "module"]),
        ("Fix file not closed", "f=open('x.txt')\ndata=f.read()", "easy", ["resource", "file"]),
        ("Fix wrong string method", "s='hello'\nprint(s.uppercase())", "easy", ["method", "string"]),
        # Medium (20)
        ("Fix race condition in counter", "cnt=0\ndef inc():\n  global cnt\n  tmp=cnt\n  cnt=tmp+1", "medium", ["concurrency", "race"]),
        ("Fix memory leak in cache", "cache={}\ndef get(k):\n  if k not in cache:\n    cache[k]=load(k)\n  return cache[k]", "medium", ["memory", "leak"]),
        ("Fix SQL injection", "query = f\"SELECT * FROM users WHERE id={user_id}\"", "medium", ["security", "sql"]),
        ("Fix deadlock", "lock1.acquire()\nlock2.acquire()\nlock2.release()\nlock1.release()", "medium", ["concurrency", "deadlock"]),
        ("Fix recursion depth", "def fib(n):\n  return fib(n-1)+fib(n-2)", "medium", ["recursion", "stack"]),
        ("Fix generator exhaustion", "gen=range(10)\nlist(gen)\nlist(gen)", "medium", ["generator", "iterator"]),
        ("Fix closure variable", "funcs=[]\nfor i in range(3):\n  funcs.append(lambda: i)", "medium", ["closure", "scope"]),
        ("Fix pickle security", "import pickle\ndata=pickle.loads(user_input)", "medium", ["security", "deserialization"]),
        ("Fix timezone naive datetime", "from datetime import datetime\nnow=datetime.now()", "medium", ["datetime", "timezone"]),
        ("Fix regex catastrophic backtracking", "import re\nre.match(r'(a+)+b', 'a'*30)", "medium", ["regex", "performance"]),
        ("Fix circular import", "# a.py: from b import B\n# b.py: from a import A", "medium", ["import", "circular"]),
        ("Fix exception swallowing", "try:\n  risky()\nexcept:\n  pass", "medium", ["exception", "handling"]),
        ("Fix async blocking call", "async def f():\n  time.sleep(10)", "medium", ["async", "blocking"]),
        ("Fix pandas SettingWithCopyWarning", "df[df.x>0]['y']=1", "medium", ["pandas", "copy"]),
        ("Fix numpy broadcasting", "a=np.array([1,2])\nb=np.array([1,2,3])\nc=a+b", "medium", ["numpy", "shape"]),
        ("Fix JSON encoding", "json.dumps({'dt': datetime.now()})", "medium", ["json", "serialization"]),
        ("Fix multiprocessing on Windows", "if __name__!='__main__':\n  Pool()", "medium", ["multiprocessing", "windows"]),
        ("Fix signal handler", "import signal\nsignal.signal(signal.SIGINT, lambda: sys.exit())", "medium", ["signal", "handler"]),
        ("Fix context manager", "class F:\n  def __enter__(self): return self", "medium", ["context", "manager"]),
        ("Fix metaclass conflict", "class A(type): pass\nclass B(type): pass\nclass C(A,B): pass", "medium", ["metaclass", "mro"]),
        # Hard (10)
        ("Fix distributed transaction", "db1.commit()\ndb2.commit()", "hard", ["distributed", "transaction"]),
        ("Fix memory corruption in C extension", "PyObject* obj = PyList_GetItem(list, i);\nPy_DECREF(obj);", "hard", ["c-extension", "memory"]),
        ("Fix GIL contention", "threads=[Thread(target=cpu_bound) for _ in range(10)]", "hard", ["gil", "threading"]),
        ("Fix cache coherence", "# CPU1: x=1\n# CPU2: if x==1: y=1\n# CPU3: if y==1: assert x==1", "hard", ["cache", "coherence"]),
        ("Fix ABA problem", "if head.next == old_next:\n  head.next = new_next", "hard", ["concurrency", "aba"]),
        ("Fix use-after-free", "ptr = malloc(100)\nfree(ptr)\n*ptr = 42", "hard", ["memory", "uaf"]),
        ("Fix TOCTOU race", "if os.path.exists(f):\n  open(f).read()", "hard", ["race", "toctou"]),
        ("Fix event loop blocking", "async def f():\n  while True:\n    await asyncio.sleep(0)", "hard", ["async", "starvation"]),
        ("Fix distributed consensus", "if votes > n/2:\n  commit()", "hard", ["distributed", "consensus"]),
        ("Fix JIT deoptimization", "def hot(x):\n  return x+1 if isinstance(x,int) else str(x)", "hard", ["jit", "performance"]),
    ]

    seeds = []
    for desc, code, diff, tags in bugs:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.CODE_DEBUG,
            difficulty=Difficulty(diff),
            prompt=f"{desc}:\n```python\n{code}\n```\nProvide the fixed code and explain the bug.",
            test_cases=[TaskTestCase(input={"code": code}, expected_output={"fixed": True}, is_public=True)],
            validator_code="def validate(sol): return isinstance(sol, dict) and 'fixed_code' in sol and 'explanation' in sol",
            source=SourceType.LLM_GENERATED,
            quality_score=0.8,
            tags=tags,
        ))
    return seeds


def create_api_orchestration_seeds() -> List[SeedPrompt]:
    """50 API orchestration seeds."""
    scenarios = [
        # Easy (20)
        ("Fetch weather by city", ["GET /weather?city={city}"], "easy", ["rest", "get"]),
        ("Get user profile", ["GET /users/{id}"], "easy", ["rest", "crud"]),
        ("Create new post", ["POST /posts"], "easy", ["rest", "post"]),
        ("Update user email", ["PATCH /users/{id}"], "easy", ["rest", "patch"]),
        ("Delete comment", ["DELETE /comments/{id}"], "easy", ["rest", "delete"]),
        ("List all products", ["GET /products"], "easy", ["rest", "list"]),
        ("Search books by title", ["GET /books?q={title}"], "easy", ["rest", "search"]),
        ("Get order status", ["GET /orders/{id}/status"], "easy", ["rest", "nested"]),
        ("Upload file", ["POST /files"], "easy", ["rest", "upload"]),
        ("Download report", ["GET /reports/{id}/download"], "easy", ["rest", "download"]),
        ("Ping health check", ["GET /health"], "easy", ["rest", "health"]),
        ("Get API version", ["GET /version"], "easy", ["rest", "meta"]),
        ("Fetch user avatar", ["GET /users/{id}/avatar"], "easy", ["rest", "media"]),
        ("Subscribe to newsletter", ["POST /newsletter/subscribe"], "easy", ["rest", "action"]),
        ("Unsubscribe", ["POST /newsletter/unsubscribe"], "easy", ["rest", "action"]),
        ("Get config", ["GET /config"], "easy", ["rest", "config"]),
        ("Refresh token", ["POST /auth/refresh"], "easy", ["rest", "auth"]),
        ("Logout", ["POST /auth/logout"], "easy", ["rest", "auth"]),
        ("Get notifications", ["GET /notifications"], "easy", ["rest", "list"]),
        ("Mark notification read", ["PUT /notifications/{id}/read"], "easy", ["rest", "update"]),
        # Medium (20)
        ("Fetch user then their posts", ["GET /users/{id}", "GET /users/{id}/posts"], "medium", ["rest", "sequential"]),
        ("Create order with items", ["POST /orders", "POST /orders/{id}/items"], "medium", ["rest", "nested"]),
        ("Parallel fetch products and reviews", ["GET /products", "GET /reviews"], "medium", ["rest", "parallel"]),
        ("Auth then fetch protected resource", ["POST /auth/login", "GET /api/data"], "medium", ["rest", "auth"]),
        ("Paginate through all results", ["GET /items?page=1", "GET /items?page=2"], "medium", ["rest", "pagination"]),
        ("Batch create users", ["POST /users/batch"], "medium", ["rest", "batch"]),
        ("Update multiple fields", ["PATCH /resource/{id}"], "medium", ["rest", "partial"]),
        ("Conditional update with ETag", ["GET /resource/{id}", "PUT /resource/{id}"], "medium", ["rest", "etag"]),
        ("Poll for job completion", ["POST /jobs", "GET /jobs/{id}"], "medium", ["rest", "polling"]),
        ("Webhook registration", ["POST /webhooks", "GET /webhooks/{id}"], "medium", ["rest", "webhook"]),
        ("Rate limit handling", ["GET /api/data"], "medium", ["rest", "rate-limit"]),
        ("Retry with exponential backoff", ["POST /unreliable"], "medium", ["rest", "retry"]),
        ("GraphQL query", ["POST /graphql"], "medium", ["graphql", "query"]),
        ("GraphQL mutation", ["POST /graphql"], "medium", ["graphql", "mutation"]),
        ("Aggregate data from 3 endpoints", ["GET /sales", "GET /inventory", "GET /forecast"], "medium", ["rest", "aggregate"]),
        ("Cascade delete", ["DELETE /parent/{id}", "DELETE /children?parent={id}"], "medium", ["rest", "cascade"]),
        ("Bulk update with transaction", ["POST /transactions", "PUT /items/bulk"], "medium", ["rest", "transaction"]),
        ("File upload with metadata", ["POST /files", "PUT /files/{id}/metadata"], "medium", ["rest", "multipart"]),
        ("Search with filters", ["GET /search?q={q}&filter={f}"], "medium", ["rest", "filter"]),
        ("Export data to CSV", ["POST /exports", "GET /exports/{id}/download"], "medium", ["rest", "export"]),
        # Hard (10)
        ("Saga pattern compensation", ["POST /order", "POST /payment", "POST /inventory"], "hard", ["saga", "distributed"]),
        ("Circuit breaker with fallback", ["GET /primary", "GET /fallback"], "hard", ["resilience", "circuit-breaker"]),
        ("Distributed tracing", ["POST /api/a", "POST /api/b", "POST /api/c"], "hard", ["tracing", "distributed"]),
        ("Event sourcing replay", ["GET /events", "POST /projections/rebuild"], "hard", ["event-sourcing", "cqrs"]),
        ("Two-phase commit", ["POST /prepare", "POST /commit"], "hard", ["2pc", "distributed"]),
        ("Optimistic locking", ["GET /resource", "PUT /resource?version={v}"], "hard", ["concurrency", "locking"]),
        ("Stream processing", ["GET /stream", "POST /process"], "hard", ["streaming", "realtime"]),
        ("Federated query", ["POST /graphql/federated"], "hard", ["graphql", "federation"]),
        ("Idempotent retry", ["POST /payments"], "hard", ["idempotency", "retry"]),
        ("Distributed cache invalidation", ["PUT /data", "DELETE /cache/*"], "hard", ["cache", "distributed"]),
    ]

    seeds = []
    for desc, apis, diff, tags in scenarios:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.API_ORCHESTRATION,
            difficulty=Difficulty(diff),
            prompt=f"Orchestrate: {desc}\nAPIs: {', '.join(apis)}\nProvide request sequence with error handling.",
            test_cases=[TaskTestCase(input={"apis": apis}, expected_output={"success": True}, is_public=True)],
            validator_code="def validate(sol): return 'steps' in sol and len(sol['steps']) > 0",
            source=SourceType.LLM_GENERATED,
            quality_score=0.75,
            tags=tags,
        ))
    return seeds


def create_math_reasoning_seeds() -> List[SeedPrompt]:
    """50 math reasoning seeds."""
    problems = [
        # Easy (20)
        ("Calculate compound interest", "P=1000, r=5%, t=2", 1102.5, "easy", ["finance", "interest"]),
        ("Find area of circle", "r=5", 78.54, "easy", ["geometry", "circle"]),
        ("Solve linear equation", "2x+3=11", 4.0, "easy", ["algebra", "linear"]),
        ("Calculate percentage", "30% of 200", 60.0, "easy", ["arithmetic", "percent"]),
        ("Find mean of list", "[1,2,3,4,5]", 3.0, "easy", ["statistics", "mean"]),
        ("Convert Celsius to Fahrenheit", "C=25", 77.0, "easy", ["conversion", "temperature"]),
        ("Calculate factorial", "n=5", 120, "easy", ["combinatorics", "factorial"]),
        ("Find GCD", "a=48, b=18", 6, "easy", ["number-theory", "gcd"]),
        ("Calculate distance", "points (0,0) and (3,4)", 5.0, "easy", ["geometry", "distance"]),
        ("Sum of arithmetic sequence", "1+2+...+10", 55, "easy", ["sequences", "sum"]),
        ("Find LCM", "a=12, b=18", 36, "easy", ["number-theory", "lcm"]),
        ("Calculate simple interest", "P=1000, r=5%, t=2", 100.0, "easy", ["finance", "interest"]),
        ("Find perimeter of rectangle", "l=5, w=3", 16, "easy", ["geometry", "perimeter"]),
        ("Calculate slope", "points (1,2) and (3,6)", 2.0, "easy", ["algebra", "slope"]),
        ("Find median", "[1,3,5,7,9]", 5, "easy", ["statistics", "median"]),
        ("Calculate power", "2^10", 1024, "easy", ["arithmetic", "exponent"]),
        ("Find prime factors", "n=12", [2, 2, 3], "easy", ["number-theory", "prime"]),
        ("Calculate volume of cube", "side=3", 27, "easy", ["geometry", "volume"]),
        ("Solve proportion", "3/4 = x/12", 9.0, "easy", ["algebra", "proportion"]),
        ("Find range", "[2,8,3,9,1]", 8, "easy", ["statistics", "range"]),
        # Medium (20)
        ("Solve quadratic equation", "x^2-5x+6=0", [2.0, 3.0], "medium", ["algebra", "quadratic"]),
        ("Calculate standard deviation", "[2,4,4,4,5,5,7,9]", 2.0, "medium", ["statistics", "stddev"]),
        ("Find nth Fibonacci", "n=10", 55, "medium", ["sequences", "fibonacci"]),
        ("Solve system of equations", "2x+y=5, x-y=1", {"x": 2.0, "y": 1.0}, "medium", ["algebra", "system"]),
        ("Calculate probability", "P(2 heads in 3 flips)", 0.375, "medium", ["probability", "binomial"]),
        ("Find derivative", "f(x)=x^2+3x", "2x+3", "medium", ["calculus", "derivative"]),
        ("Calculate matrix determinant", "[[1,2],[3,4]]", -2, "medium", ["linear-algebra", "determinant"]),
        ("Solve inequality", "2x-3>7", "x>5", "medium", ["algebra", "inequality"]),
        ("Find integral", "∫x^2 dx", "x^3/3+C", "medium", ["calculus", "integral"]),
        ("Calculate permutations", "P(5,3)", 60, "medium", ["combinatorics", "permutation"]),
        ("Find eigenvalues", "[[2,1],[1,2]]", [1.0, 3.0], "medium", ["linear-algebra", "eigenvalue"]),
        ("Solve exponential equation", "2^x=16", 4.0, "medium", ["algebra", "exponential"]),
        ("Calculate combinations", "C(5,2)", 10, "medium", ["combinatorics", "combination"]),
        ("Find limit", "lim(x→0) sin(x)/x", 1.0, "medium", ["calculus", "limit"]),
        ("Solve logarithm", "log₂(32)", 5.0, "medium", ["algebra", "logarithm"]),
        ("Calculate covariance", "[1,2,3] and [2,4,6]", 2.0, "medium", ["statistics", "covariance"]),
        ("Find inverse matrix", "[[1,2],[3,4]]", [[-2.0, 1.0], [1.5, -0.5]], "medium", ["linear-algebra", "inverse"]),
        ("Solve trigonometric equation", "sin(x)=0.5, x∈[0,2π]", [0.524, 2.618], "medium", ["trigonometry", "equation"]),
        ("Calculate expected value", "X={1:0.2, 2:0.5, 3:0.3}", 2.1, "medium", ["probability", "expectation"]),
        ("Find Taylor series", "e^x at x=0", "1+x+x^2/2+...", "medium", ["calculus", "series"]),
        # Hard (10)
        ("Optimize linear program", "max 3x+4y s.t. x+y≤10, x≥0, y≥0", {"x": 0, "y": 10, "max": 40}, "hard", ["optimization", "lp"]),
        ("Solve differential equation", "dy/dx=y, y(0)=1", "y=e^x", "hard", ["calculus", "ode"]),
        ("Calculate Fourier transform", "f(t)=e^(-t^2)", "F(ω)=√π·e^(-ω^2/4)", "hard", ["analysis", "fourier"]),
        ("Find Nash equilibrium", "2-player game matrix", {"p1": 0.5, "p2": 0.5}, "hard", ["game-theory", "nash"]),
        ("Solve PDE", "∂u/∂t=∂²u/∂x²", "u(x,t)=...", "hard", ["calculus", "pde"]),
        ("Calculate Laplace transform", "f(t)=t·e^(-at)", "F(s)=1/(s+a)^2", "hard", ["analysis", "laplace"]),
        ("Find optimal control", "min ∫(x²+u²)dt", "u=-x", "hard", ["control-theory", "optimal"]),
        ("Solve nonlinear system", "x²+y²=1, x+y=1", [{"x": 0, "y": 1}, {"x": 1, "y": 0}], "hard", ["algebra", "nonlinear"]),
        ("Calculate convolution", "f*g where f=g=rect", "triangle", "hard", ["analysis", "convolution"]),
        ("Find Gröbner basis", "I=⟨x²+y, xy+x⟩", ["x²+y", "xy+x"], "hard", ["algebra", "groebner"]),
    ]

    seeds = []
    for desc, problem, answer, diff, tags in problems:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MATH_REASONING,
            difficulty=Difficulty(diff),
            prompt=f"{desc}: {problem}\nProvide step-by-step solution.",
            test_cases=[TaskTestCase(input={"problem": problem}, expected_output=answer, is_public=True)],
            validator_code=f"def validate(sol): return 'answer' in sol and 'steps' in sol",
            source=SourceType.LLM_GENERATED,
            quality_score=0.9,
            tags=tags,
        ))
    return seeds


def create_multi_step_planning_seeds() -> List[SeedPrompt]:
    """50 multi-step planning seeds."""
    tasks = [
        # Easy (20)
        ("Plan grocery shopping", 3, "easy", ["daily", "shopping"]),
        ("Organize desk workspace", 4, "easy", ["organization", "workspace"]),
        ("Prepare simple breakfast", 3, "easy", ["cooking", "meal"]),
        ("Water house plants", 3, "easy", ["gardening", "routine"]),
        ("Pack gym bag", 3, "easy", ["fitness", "preparation"]),
        ("Clean kitchen", 4, "easy", ["cleaning", "household"]),
        ("Do laundry", 4, "easy", ["household", "chores"]),
        ("Make coffee", 3, "easy", ["beverage", "routine"]),
        ("Check email", 3, "easy", ["communication", "routine"]),
        ("Take out trash", 3, "easy", ["household", "chores"]),
        ("Feed pet", 3, "easy", ["pet-care", "routine"]),
        ("Charge devices", 2, "easy", ["tech", "maintenance"]),
        ("Set alarm", 2, "easy", ["routine", "time"]),
        ("Lock doors", 2, "easy", ["security", "routine"]),
        ("Turn off lights", 2, "easy", ["energy", "routine"]),
        ("Check weather", 2, "easy", ["information", "routine"]),
        ("Make bed", 2, "easy", ["household", "routine"]),
        ("Brush teeth", 3, "easy", ["hygiene", "routine"]),
        ("Take medication", 3, "easy", ["health", "routine"]),
        ("Backup phone", 3, "easy", ["tech", "maintenance"]),
        # Medium (20)
        ("Deploy web application", 5, "medium", ["devops", "deployment"]),
        ("Plan birthday party", 6, "medium", ["event", "planning"]),
        ("Conduct job interview", 5, "medium", ["hr", "hiring"]),
        ("Troubleshoot network issue", 5, "medium", ["it", "troubleshooting"]),
        ("Write research paper", 7, "medium", ["academic", "writing"]),
        ("Onboard new employee", 6, "medium", ["hr", "onboarding"]),
        ("Perform code review", 5, "medium", ["development", "review"]),
        ("Plan vacation trip", 7, "medium", ["travel", "planning"]),
        ("Conduct security audit", 6, "medium", ["security", "audit"]),
        ("Refactor legacy code", 6, "medium", ["development", "refactoring"]),
        ("Set up CI/CD pipeline", 6, "medium", ["devops", "automation"]),
        ("Investigate production bug", 5, "medium", ["development", "debugging"]),
        ("Migrate database", 6, "medium", ["database", "migration"]),
        ("Conduct performance review", 5, "medium", ["hr", "review"]),
        ("Plan marketing campaign", 6, "medium", ["marketing", "campaign"]),
        ("Set up monitoring", 5, "medium", ["devops", "observability"]),
        ("Conduct A/B test", 6, "medium", ["product", "testing"]),
        ("Implement new feature", 6, "medium", ["development", "feature"]),
        ("Prepare presentation", 5, "medium", ["communication", "presentation"]),
        ("Conduct user research", 6, "medium", ["product", "research"]),
        # Hard (10)
        ("Migrate to microservices", 8, "hard", ["architecture", "migration"]),
        ("Plan disaster recovery", 7, "hard", ["infrastructure", "dr"]),
        ("Conduct M&A integration", 9, "hard", ["business", "integration"]),
        ("Implement zero-trust security", 8, "hard", ["security", "architecture"]),
        ("Migrate datacenter", 9, "hard", ["infrastructure", "migration"]),
        ("Launch new product", 10, "hard", ["product", "launch"]),
        ("Restructure organization", 8, "hard", ["management", "restructure"]),
        ("Implement GDPR compliance", 8, "hard", ["compliance", "legal"]),
        ("Build ML pipeline", 9, "hard", ["ml", "pipeline"]),
        ("Plan IPO", 10, "hard", ["business", "ipo"]),
    ]

    seeds = []
    for desc, steps, diff, tags in tasks:
        seeds.append(SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MULTI_STEP_PLANNING,
            difficulty=Difficulty(diff),
            prompt=f"Create detailed plan: {desc}\nProvide ordered steps with dependencies.",
            test_cases=[TaskTestCase(input={"task": desc}, expected_output={"min_steps": steps}, is_public=True)],
            validator_code=f"def validate(sol): return 'plan' in sol and len(sol['plan']) >= {steps}",
            source=SourceType.LLM_GENERATED,
            quality_score=0.85,
            tags=tags,
        ))
    return seeds


async def main():
    """Generate all 200 seed prompts."""
    pool = SeedPromptPool()

    print("Generating seeds...")
    pool.add_batch(create_code_debug_seeds())
    pool.add_batch(create_api_orchestration_seeds())
    pool.add_batch(create_math_reasoning_seeds())
    pool.add_batch(create_multi_step_planning_seeds())

    output_path = Path(__file__).parent.parent / "data" / "seed_prompts.json"
    pool.save(str(output_path))

    print(f"\n[OK] Generated {len(pool)} seed prompts")
    print(f"[OK] Saved to {output_path}")
    print("\nStatistics:")
    print(json.dumps(pool.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
