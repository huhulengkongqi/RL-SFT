"""快速爬取种子prompt - 简化版"""
import json
import sys
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import (
    Difficulty,
    Domain,
    SeedPrompt,
    SourceType,
    TaskTestCase,
)
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def create_seed(domain, difficulty, prompt, test_input, expected_output, validator_code, quality, tags):
    """创建种子"""
    return SeedPrompt(
        id=str(uuid.uuid4()),
        domain=domain,
        difficulty=difficulty,
        prompt=prompt,
        test_cases=[TaskTestCase(input=test_input, expected_output=expected_output, is_public=True)],
        validator_code=validator_code,
        source=SourceType.CRAWLED,
        quality_score=quality,
        tags=tags,
    )


def crawl_stackoverflow(num_seeds: int = 30):
    """快速爬取StackOverflow Python debug问题"""
    print(f"爬取StackOverflow {num_seeds}个debug问题...")

    seeds = []
    try:
        response = requests.get(
            "https://api.stackexchange.com/2.3/questions",
            params={
                "pagesize": min(num_seeds + 10, 100),
                "order": "desc",
                "sort": "votes",
                "tagged": "python;debugging",
                "site": "stackoverflow",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            if len(seeds) >= num_seeds:
                break

            score = item.get("score", 0)
            if score < 10 or not item.get("is_answered"):
                continue

            title = item.get("title", "")
            body = item.get("body", "")

            # 根据score判断难度
            if score < 20:
                difficulty = Difficulty.EASY
                quality = 0.8
            elif score < 50:
                difficulty = Difficulty.MEDIUM
                quality = 0.85
            else:
                difficulty = Difficulty.HARD
                quality = 0.9

            # 简单提取代码
            import re
            code_match = re.search(r"<code>(.*?)</code>", body, re.DOTALL)
            code_snippet = code_match.group(1)[:300] if code_match else "See question for code details"

            prompt_text = f"""Debug and fix the following Python issue:

**Problem**: {title}

**Code Context**:
```python
{code_snippet}
```

Provide:
1. Root cause analysis
2. Fixed code
3. Explanation of why the fix works"""

            seed = create_seed(
                domain=Domain.CODE_DEBUG,
                difficulty=difficulty,
                prompt=prompt_text,
                test_input={"title": title},
                expected_output={"fixed": True, "analysis": True},
                validator_code="def validate(sol): return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol",
                quality=quality,
                tags=["stackoverflow", "debug", "python"],
            )
            seeds.append(seed)

        print(f"  ✓ 获取到 {len(seeds)} 个code_debug种子")

    except Exception as e:
        print(f"  ✗ StackOverflow错误: {e}")

    return seeds


def get_math_seeds(num_seeds: int = 30):
    """获取高质量数学题"""
    print(f"准备 {num_seeds} 个数学推理种子...")

    math_problems = [
        # Easy
        ("A baker makes 48 cookies per hour. How many cookies can they make in 5 hours?", "48 * 5 = 240 cookies", Difficulty.EASY, 0.8, ["arithmetic", "rate"]),
        ("A rectangle has length 15 cm and width 8 cm. Find its area and perimeter.", "Area=15*8=120 cm², Perimeter=2*(15+8)=46 cm", Difficulty.EASY, 0.8, ["geometry", "rectangle"]),
        ("What is 35% of 280?", "0.35 * 280 = 98", Difficulty.EASY, 0.8, ["percentage", "arithmetic"]),
        ("A car uses 12 gallons of gas to drive 264 miles. What is the miles per gallon rate?", "264 / 12 = 22 mpg", Difficulty.EASY, 0.8, ["rate", "division"]),
        ("How many seconds are there in 1 day?", "24*60*60 = 86,400 seconds", Difficulty.EASY, 0.8, ["conversion", "time"]),
        ("The average of 6 numbers is 18. If one number is excluded, the average becomes 15. What is the excluded number?", "Sum=6*18=108, New sum=5*15=75, Number=108-75=33", Difficulty.EASY, 0.8, ["average", "algebra"]),
        ("A store has apples priced at $1.20 each. If you buy 5 apples, how much do you pay?", "5 * 1.20 = $6.00", Difficulty.EASY, 0.8, ["arithmetic", "money"]),
        ("What is the next number in the sequence: 3, 6, 12, 24, ___?", "Pattern: *2 each time, Next: 48", Difficulty.EASY, 0.8, ["sequence", "pattern"]),
        ("If 8 books cost $72, what is the cost of 12 books?", "$72/8 = $9 per book, 12*9 = $108", Difficulty.EASY, 0.8, ["proportion", "unit-rate"]),
        ("Find the value of x in: 3x + 7 = 22", "3x = 15, x = 5", Difficulty.EASY, 0.8, ["algebra", "linear-equation"]),
        # Medium
        ("A circle has a radius of 7 cm. What is its circumference and area? (Use π = 3.14)", "Circumference = 2*π*r = 43.96 cm, Area = π*r² = 153.86 cm²", Difficulty.MEDIUM, 0.85, ["geometry", "circle"]),
        ("Solve the quadratic equation: x² - 5x + 6 = 0", "Factor: (x-2)(x-3)=0, Solutions: x=2 and x=3", Difficulty.MEDIUM, 0.85, ["algebra", "quadratic"]),
        ("What is the probability of getting a sum of 7 when rolling two dice?", "6 favorable outcomes out of 36 total = 6/36 = 1/6 ≈ 0.167", Difficulty.MEDIUM, 0.85, ["probability", "dice"]),
        ("Find the 15th term in the arithmetic sequence: 5, 9, 13, 17, ...", "a₁=5, d=4, a₁₅ = 5 + (15-1)*4 = 5 + 56 = 61", Difficulty.MEDIUM, 0.85, ["sequence", "arithmetic"]),
        ("A right triangle has legs of lengths 9 cm and 12 cm. What is the length of the hypotenuse?", "By Pythagoras: √(9²+12²) = √(81+144) = √225 = 15 cm", Difficulty.MEDIUM, 0.85, ["geometry", "pythagorean"]),
        ("What is the sum of the first 40 positive integers?", "Sum = n(n+1)/2 = 40*41/2 = 820", Difficulty.MEDIUM, 0.85, ["sum", "series"]),
        ("In how many different ways can you arrange the letters in the word 'BANANA'?", "6 letters with A repeated 3 times, N repeated 2 times → 6!/(3!*2!) = 720/(6*2) = 60 ways", Difficulty.MEDIUM, 0.85, ["combinatorics", "permutation"]),
        ("Find the derivative of f(x) = 3x³ - 2x² + 5x - 1 at x = 2", "f'(x) = 9x² - 4x + 5, f'(2) = 9*4 - 8 + 5 = 36 - 8 + 5 = 33", Difficulty.MEDIUM, 0.85, ["calculus", "derivative"]),
        ("A bag contains 4 red, 5 blue, and 6 green marbles. What is the probability of picking a blue marble?", "5/(4+5+6) = 5/15 = 1/3 ≈ 0.333", Difficulty.MEDIUM, 0.85, ["probability", "counting"]),
        ("What is 25% of 60% of 400?", "0.6 * 400 = 240, 0.25 * 240 = 60", Difficulty.MEDIUM, 0.85, ["percentage", "arithmetic"]),
        # Hard
        ("Find all values of x that satisfy: |2x - 5| < 7", "-7 < 2x-5 < 7 → -2 < 2x < 12 → -1 < x < 6", Difficulty.HARD, 0.9, ["algebra", "absolute-value", "inequality"]),
        ("What is the integral of f(x) = 2x² + 3 from x = 0 to x = 2?", "∫(2x²+3)dx = (2/3)x³ + 3x. At x=2: 16/3 + 6 = 34/3 ≈ 11.33", Difficulty.HARD, 0.9, ["calculus", "integral"]),
        ("How many distinct 5-card poker hands can be formed from a standard 52-card deck?", "C(52,5) = 52!/(5!*47!) = 2,598,960", Difficulty.HARD, 0.9, ["combinatorics", "combination"]),
        ("Find the eigenvalues of the matrix [[3, 1], [1, 3]]", "Characteristic equation: det[[3-λ,1],[1,3-λ]] = 0 → (3-λ)²-1=0 → λ²-6λ+8=0 → (λ-2)(λ-4)=0 → λ=2 and λ=4", Difficulty.HARD, 0.9, ["linear-algebra", "eigenvalues"]),
        ("What is the second derivative of f(x) = sin(2x)?", "f'(x) = 2cos(2x), f''(x) = -4sin(2x)", Difficulty.HARD, 0.9, ["calculus", "derivative", "trigonometry"]),
        ("A population doubles every 15 years. Starting with 1000 individuals, how many will there be in 60 years?", "60/15 = 4 doubling periods, Population = 1000 * 2⁴ = 1000 * 16 = 16,000", Difficulty.HARD, 0.9, ["exponential", "growth"]),
    ]

    seeds = []
    for question, answer, difficulty, quality, tags in math_problems[:num_seeds]:
        seed = create_seed(
            domain=Domain.MATH_REASONING,
            difficulty=difficulty,
            prompt=f"Solve the following math problem step by step:\n\n{question}",
            test_input={"question": question},
            expected_output={"answer": answer},
            validator_code="def validate(sol): return 'steps' in sol and 'final_answer' in sol",
            quality=quality,
            tags=["math"] + tags,
        )
        seeds.append(seed)

    print(f"  ✓ 准备好 {len(seeds)} 个math_reasoning种子")
    return seeds


def get_api_orchestration_seeds(num_seeds: int = 30):
    """获取API编排场景种子"""
    print(f"准备 {num_seeds} 个API编排种子...")

    scenarios = [
        ("User Profile + Posts API", "Fetch a user profile and their recent posts in sequence", Difficulty.EASY, 0.8, ["rest", "sequential"]),
        ("Weather + Forecast API", "Get current weather and then 7-day forecast for a location", Difficulty.EASY, 0.8, ["weather", "sequential"]),
        ("Auth Protected Resource", "Login to get JWT token, then use it to access protected data", Difficulty.EASY, 0.85, ["auth", "jwt"]),
        ("Paginated Results Collection", "Fetch all pages from a paginated API endpoint", Difficulty.EASY, 0.85, ["pagination", "loop"]),
        ("Batch Create Resources", "Create multiple resources with rate limiting and error handling", Difficulty.MEDIUM, 0.85, ["batch", "rate-limit"]),
        ("Search + Detail API", "Search for items, then fetch detailed info for top 5 results", Difficulty.MEDIUM, 0.85, ["search", "sequential"]),
        ("Parallel API Requests", "Make 3 independent API calls in parallel for better performance", Difficulty.MEDIUM, 0.85, ["parallel", "async"]),
        ("Retry with Exponential Backoff", "Implement robust retry logic for flaky API endpoints", Difficulty.MEDIUM, 0.85, ["retry", "resilience"]),
        ("Webhook Verification + Processing", "Verify webhook signature and process the payload", Difficulty.MEDIUM, 0.85, ["webhook", "security"]),
        ("GraphQL Query + Mutation", "Query data and perform a mutation using GraphQL", Difficulty.MEDIUM, 0.85, ["graphql", "query"]),
        ("Saga Pattern: Order Flow", "Create order → reserve inventory → process payment → confirm", Difficulty.HARD, 0.9, ["saga", "distributed"]),
        ("Circuit Breaker with Fallback", "Implement circuit breaker pattern with fallback API", Difficulty.HARD, 0.9, ["resilience", "circuit-breaker"]),
        ("Idempotent Retry System", "Implement idempotent retry mechanism for payment API", Difficulty.HARD, 0.9, ["idempotency", "reliability"]),
    ]

    seeds = []
    for title, description, difficulty, quality, tags in scenarios:
        seed = create_seed(
            domain=Domain.API_ORCHESTRATION,
            difficulty=difficulty,
            prompt=f"""Implement robust API orchestration for the following scenario:

**Scenario**: {title}

**Description**: {description}

Provide:
1. API flow diagram (text description)
2. Complete Python implementation with error handling
3. Edge cases handled
4. Test cases for the implementation""",
            test_input={"scenario": title},
            expected_output={"implementation": True, "flow": True},
            validator_code="def validate(sol): return 'implementation' in sol and 'flow_diagram' in sol",
            quality=quality,
            tags=["api", "orchestration"] + tags,
        )
        seeds.append(seed)

    print(f"  ✓ 准备好 {len(seeds)} 个api_orchestration种子")
    return seeds[:num_seeds]


def get_multi_step_planning_seeds(num_seeds: int = 30):
    """获取多步规划场景种子"""
    print(f"准备 {num_seeds} 个多步规划种子...")

    scenarios = [
        ("Setup CI/CD Pipeline", "Configure GitHub Actions for a Python project with testing, linting, and deployment", Difficulty.EASY, 0.8, ["devops", "ci"]),
        ("Onboard New Developer", "Create a complete onboarding checklist with dependencies for a new team member", Difficulty.EASY, 0.8, ["hr", "onboarding"]),
        ("Deploy Web Application", "Plan deployment steps: build → test → staging → production rollout", Difficulty.EASY, 0.8, ["deployment", "devops"]),
        ("Database Migration", "Plan zero-downtime database schema migration with rollback strategy", Difficulty.MEDIUM, 0.85, ["database", "migration"]),
        ("Incident Response Plan", "Create step-by-step plan for production outage response", Difficulty.MEDIUM, 0.85, ["sre", "incident"]),
        ("Security Audit Execution", "Plan a comprehensive security audit: scan → analyze → fix → verify", Difficulty.MEDIUM, 0.85, ["security", "audit"]),
        ("Performance Optimization", "Plan system optimization: profile → identify bottlenecks → implement fixes → benchmark", Difficulty.MEDIUM, 0.85, ["performance", "optimization"]),
        ("Product Launch Plan", "Plan product launch: beta testing → marketing → release → monitoring", Difficulty.MEDIUM, 0.85, ["product", "launch"]),
        ("Microservices Migration", "Plan migration from monolith to microservices with zero downtime", Difficulty.HARD, 0.9, ["architecture", "migration"]),
        ("Disaster Recovery Drill", "Plan and execute disaster recovery: backup → failover → restore → validate", Difficulty.HARD, 0.9, ["dr", "reliability"]),
    ]

    seeds = []
    for title, description, difficulty, quality, tags in scenarios:
        seed = create_seed(
            domain=Domain.MULTI_STEP_PLANNING,
            difficulty=difficulty,
            prompt=f"""Create a detailed multi-step plan for:

**Task**: {title}

**Context**: {description}

Provide:
1. Ordered list of steps with dependencies
2. Prerequisites for each step
3. Estimated time per step
4. Success criteria
5. Rollback/fallback strategies""",
            test_input={"task": title},
            expected_output={"plan": True, "steps": True},
            validator_code="def validate(sol): return 'steps' in sol and len(sol['steps']) >= 3",
            quality=quality,
            tags=["planning", "multi-step"] + tags,
        )
        seeds.append(seed)

    print(f"  ✓ 准备好 {len(seeds)} 个multi_step_planning种子")
    return seeds[:num_seeds]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="快速爬取高质量种子prompt")
    parser.add_argument("--output", type=str, default="data/crawled_seeds.json", help="输出文件路径")
    parser.add_argument("--per-domain", type=int, default=30, help="每个domain爬取数量")
    args = parser.parse_args()

    print("=== 快速爬取种子prompt ===\n")

    all_seeds = []

    # 1. StackOverflow: code_debug
    so_seeds = crawl_stackoverflow(args.per_domain)
    all_seeds.extend(so_seeds)

    # 2. Math reasoning
    math_seeds = get_math_seeds(args.per_domain)
    all_seeds.extend(math_seeds)

    # 3. API orchestration
    api_seeds = get_api_orchestration_seeds(args.per_domain)
    all_seeds.extend(api_seeds)

    # 4. Multi-step planning
    planning_seeds = get_multi_step_planning_seeds(args.per_domain)
    all_seeds.extend(planning_seeds)

    # 保存结果
    output_path = Path(__file__).parent.parent / args.output
    pool = SeedPromptPool()
    pool.add_batch(all_seeds)
    pool.save(str(output_path))

    print(f"\n=== 爬取完成 ===")
    print(f"总种子数: {len(all_seeds)}")
    print(f"保存到: {output_path}")
    print("\n统计信息:")
    print(json.dumps(pool.get_stats(), indent=2))


if __name__ == "__main__":
    main()
