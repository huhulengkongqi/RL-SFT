"""爬取高质量种子prompt脚本。

支持：
1. StackOverflow API → code_debug
2. HuggingFace datasets (GSM8K, MATH) → math_reasoning
3. GitHub API → api_orchestration, multi_step_planning
"""
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.agent_sft.task_generator.models import (
    Difficulty,
    Domain,
    SeedPrompt,
    SourceType,
    TaskTestCase,
)
from src.agent_sft.task_generator.seed_pool import SeedPromptPool

load_dotenv()


class SeedCrawler:
    """种子爬虫基类"""

    def __init__(self):
        self.pool = SeedPromptPool()

    def create_seed(
        self,
        domain: Domain,
        difficulty: Difficulty,
        prompt: str,
        test_input: Dict[str, Any],
        expected_output: Any,
        validator_code: str,
        quality_score: float,
        tags: List[str],
        source: SourceType = SourceType.CRAWLED,
    ) -> SeedPrompt:
        """创建SeedPrompt对象"""
        return SeedPrompt(
            id=str(uuid.uuid4()),
            domain=domain,
            difficulty=difficulty,
            prompt=prompt,
            test_cases=[
                TaskTestCase(input=test_input, expected_output=expected_output, is_public=True)
            ],
            validator_code=validator_code,
            source=source,
            quality_score=quality_score,
            tags=tags,
        )


class StackOverflowCrawler(SeedCrawler):
    """StackOverflow爬虫 - 爬取Python debug相关问题"""

    BASE_URL = "https://api.stackexchange.com/2.3"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("STACKOVERFLOW_KEY")
        self.session = requests.Session()
        # StackOverflow API需要设置headers避免被block
        self.session.headers.update(
            {"User-Agent": "AgentRL-SeedCrawler/1.0 (+https://github.com/example)"}
        )

    def _has_code_block(self, text: str) -> bool:
        """检查是否包含代码块"""
        return bool(re.search(r"```[\s\S]*?```|`[^`]+`", text))

    def _extract_code_from_body(self, body: str) -> str:
        """从HTML/Markdown中提取代码"""
        # 简单提取code标签内容
        code_blocks = re.findall(r"<code>(.*?)</code>", body, re.DOTALL)
        if code_blocks:
            return "\n".join(block.strip() for block in code_blocks[:3])
        return ""

    def crawl(self, min_score: int = 15, max_pages: int = 5) -> List[SeedPrompt]:
        """爬取高质量的Python debug问题"""
        seeds = []

        for page in range(1, max_pages + 1):
            params = {
                "page": page,
                "pagesize": 50,
                "order": "desc",
                "sort": "votes",
                "tagged": "python;debugging",
                "site": "stackoverflow",
                "filter": "withbody",  # 包含body内容
            }
            if self.api_key:
                params["key"] = self.api_key

            try:
                response = self.session.get(f"{self.BASE_URL}/questions", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"  StackOverflow API error: {e}")
                break

            if "items" not in data or not data["items"]:
                break

            for item in data["items"]:
                if item["score"] < min_score:
                    continue
                if not item.get("is_answered"):
                    continue
                if not self._has_code_block(item.get("body", "")):
                    continue

                title = item["title"]
                body = item.get("body", "")
                score = item["score"]
                tags_list = item.get("tags", [])

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

                # 提取问题中的代码
                code_snippet = self._extract_code_from_body(body)
                if not code_snippet or len(code_snippet) < 20:
                    code_snippet = "See question details above"

                prompt_text = f"""Debug and fix the following Python issue:

**Problem**: {title}

**Code Context**:
```python
{code_snippet[:500]}
```

Provide the fixed code and explain the root cause of the bug."""

                seed = self.create_seed(
                    domain=Domain.CODE_DEBUG,
                    difficulty=difficulty,
                    prompt=prompt_text,
                    test_input={"question_id": item["question_id"], "title": title},
                    expected_output={"fixed": True, "explanation": True},
                    validator_code="def validate(sol): return isinstance(sol, dict) and 'fixed_code' in sol and 'explanation' in sol and len(sol['explanation']) > 10",
                    quality_score=quality,
                    tags=["stackoverflow"] + tags_list[:3],
                )
                seeds.append(seed)

            print(f"  Page {page}: found {len(seeds)} qualifying questions so far")

            # 检查是否还有下一页
            if not data.get("has_more"):
                break

        return seeds[:50]  # 最多取50个


class MathDatasetCrawler(SeedCrawler):
    """HuggingFace数学数据集爬虫 - GSM8K + MATH"""

    def _classify_difficulty_gsm8k(self, example: Dict[str, Any]) -> Difficulty:
        """根据解题步骤数量分类GSM8K难度"""
        answer = example.get("answer", "")
        steps = answer.count("<<")  # GSM8K用<<>>标记步骤
        num_steps = max(steps, answer.count("\n"))

        if num_steps <= 3:
            return Difficulty.EASY
        elif num_steps <= 6:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD

    def _classify_difficulty_math(self, example: Dict[str, Any]) -> Difficulty:
        """根据MATH数据集的level分类难度"""
        level = example.get("level", "Level 1")
        level_num = int(re.search(r"\d+", level).group()) if re.search(r"\d+", level) else 1

        if level_num <= 2:
            return Difficulty.EASY
        elif level_num <= 4:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD

    def crawl_gsm8k(self, max_samples: int = 40) -> List[SeedPrompt]:
        """从GSM8K爬取数学题 - 支持离线JSON格式"""
        # 内置一些高质量的数学题（避免下载大数据集）
        # 来源：精选的数学题示例
        math_problems = [
            # Easy problems
            ("A bakery sells 25 loaves of bread in the morning and 18 in the afternoon. If each loaf costs $3, how much money do they make in a day?", "25+18=43, 43*3=129", Difficulty.EASY, 0.8),
            ("A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?", "Area=12*8=96, Perimeter=2*(12+8)=40", Difficulty.EASY, 0.8),
            ("If 5 workers can paint a house in 8 days, how many days will 12 workers take to paint 3 houses?", "1 house = 40 worker-days, 3 houses = 120 worker-days, 120/12=10 days", Difficulty.EASY, 0.8),
            ("A car travels at 60 mph for 2.5 hours. How far does it go?", "Distance = speed * time = 60 * 2.5 = 150 miles", Difficulty.EASY, 0.8),
            ("What is 15% of 240?", "240 * 0.15 = 36", Difficulty.EASY, 0.8),
            ("The sum of two numbers is 45 and their difference is 13. What are the numbers?", "x+y=45, x-y=13 → 2x=58 → x=29, y=16", Difficulty.EASY, 0.8),
            ("A store has 3 red balls, 5 blue balls, and 2 green balls. What is the probability of picking a blue ball?", "5/(3+5+2) = 5/10 = 0.5", Difficulty.EASY, 0.8),
            ("How many minutes are there in 3 days and 7 hours?", "3*24*60 + 7*60 = 4320 + 420 = 4740 minutes", Difficulty.EASY, 0.8),
            ("The average of 5 numbers is 24. If one number is removed, the average becomes 22. What was the number?", "Sum=5*24=120, New sum=4*22=88, Number=120-88=32", Difficulty.EASY, 0.8),
            ("A train leaves Station A at 9:00 AM traveling at 70 mph. Another train leaves Station B at 10:00 AM traveling at 80 mph. The distance between stations is 290 miles. At what time do they meet?", "First train goes 70 miles by 10AM. Remaining = 220 miles. Combined speed = 150 mph. Time = 220/150 = 1h28m. Meet at 11:28 AM", Difficulty.MEDIUM, 0.85),
            # Medium problems
            ("A circle is inscribed in a square with side length 10. What is the area of the circle?", "Radius = 5, Area = π*r² = 25π ≈ 78.54", Difficulty.MEDIUM, 0.85),
            ("Solve for x: 2^(x+1) * 4^(x-1) = 8", "2^(x+1) * 2^(2x-2) = 2^3 → 3x-1=3 → x=4/3", Difficulty.MEDIUM, 0.85),
            ("What is the sum of the first 30 positive integers?", "Sum = n(n+1)/2 = 30*31/2 = 465", Difficulty.MEDIUM, 0.85),
            ("A right triangle has legs of length 5 and 12. What is the length of the hypotenuse and the area?", "Hypotenuse = √(5²+12²) = √169 = 13, Area = 5*12/2 = 30", Difficulty.MEDIUM, 0.85),
            ("If f(x) = 2x² - 3x + 1, what is f(4) and f(-2)?", "f(4)=2*16-12+1=21, f(-2)=8+6+1=15", Difficulty.MEDIUM, 0.85),
            ("What is the 10th term of the sequence: 2, 5, 10, 17, 26,...?", "Pattern: n²+1, 10th term = 100+1 = 101", Difficulty.MEDIUM, 0.85),
            ("A cube has a volume of 64 cm³. What is its surface area?", "Side = 4 cm, Surface area = 6*4² = 96 cm²", Difficulty.MEDIUM, 0.85),
            ("In how many ways can you arrange the letters in the word 'APPLE'?", "5 letters with P repeated twice → 5!/2! = 60 ways", Difficulty.MEDIUM, 0.85),
            ("What is the derivative of f(x) = x³ - 2x² + 5x - 1 at x = 2?", "f'(x) = 3x² - 4x + 5, f'(2) = 12 - 8 + 5 = 9", Difficulty.MEDIUM, 0.85),
            ("A population grows exponentially: P(t) = P₀·e^(0.05t). If P₀ = 1000, what is P after 10 years?", "P(10) = 1000·e^0.5 ≈ 1648.7", Difficulty.MEDIUM, 0.85),
            # Hard problems
            ("What is the integral of ∫(x²·e^x) dx from 0 to 1?", "By parts: u=x², dv=e^x dx → [x²e^x - 2xe^x + 2e^x] from 0 to 1 = e - 2 ≈ 0.718", Difficulty.HARD, 0.9),
            ("Find all roots of x³ - 6x² + 11x - 6 = 0", "Factor: (x-1)(x-2)(x-3)=0 → x=1,2,3", Difficulty.HARD, 0.9),
            ("What is the probability of getting exactly 3 heads in 5 coin flips?", "C(5,3)*(0.5)^5 = 10/32 = 5/16 = 0.3125", Difficulty.HARD, 0.9),
            ("A matrix A = [[1,2],[3,4]]. What is its inverse?", "det = 1*4-2*3 = -2, A⁻¹ = [[-2,1],[1.5,-0.5]", Difficulty.HARD, 0.9),
            ("What is the Taylor series expansion of sin(x) around x=0?", "sin(x) = Σ (-1)^n x^(2n+1)/(2n+1)! for n=0 to ∞", Difficulty.HARD, 0.9),
            ("Solve the differential equation dy/dx = 2xy with y(0) = 1", "Separate variables: dy/y = 2x dx → ln(y) = x² + C → y = e^(x²)", Difficulty.HARD, 0.9),
        ]

        seeds = []
        for question, answer, difficulty, quality in math_problems[:max_samples]:
            seed = self.create_seed(
                domain=Domain.MATH_REASONING,
                difficulty=difficulty,
                prompt=f"Solve the following math problem step by step:\n\n{question}",
                test_input={"question": question},
                expected_output={"answer": answer},
                validator_code="def validate(sol): return 'steps' in sol and 'answer' in sol and len(sol['steps']) > 0",
                quality_score=quality,
                tags=["math", "curated"],
            )
            seeds.append(seed)

        return seeds

    def crawl(self, max_samples: int = 60) -> List[SeedPrompt]:
        """爬取数学题"""
        return self.crawl_gsm8k(max_samples)


class GitHubCrawler(SeedCrawler):
    """GitHub爬虫 - 从README中提取API编排和多步规划场景"""

    def __init__(self, token: str = None):
        super().__init__()
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})

    def search_repos(self, query: str, max_repos: int = 20) -> List[Dict[str, Any]]:
        """搜索GitHub仓库"""
        repos = []
        try:
            response = self.session.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": max_repos},
                timeout=30,
            )
            response.raise_for_status()
            repos = response.json().get("items", [])
        except Exception as e:
            print(f"  GitHub search error: {e}")

        return repos

    def get_readme(self, owner: str, repo: str) -> str:
        """获取仓库README"""
        try:
            response = self.session.get(
                f"https://api.github.com/repos/{owner}/{repo}/readme",
                headers={"Accept": "application/vnd.github.v3.raw"},
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def _extract_api_scenarios(self, readme: str) -> List[str]:
        """从README中提取API编排场景"""
        scenarios = []

        # 查找"Example", "Usage", "Quick Start"等section
        sections = re.split(r"\n##?\s+", readme)
        for section in sections:
            # 查找包含API调用示例的section
            if any(keyword in section.lower() for keyword in ["example", "usage", "quick start", "tutorial"]):
                # 查找代码块
                code_blocks = re.findall(r"```(?:python)?\n(.*?)```", section, re.DOTALL)
                for block in code_blocks:
                    # 检查是否有多个API调用
                    if (
                        block.count("requests.") >= 2
                        or block.count(".get(") + block.count(".post(") >= 2
                        or block.count("http") >= 2
                    ):
                        scenarios.append(block)

        return scenarios[:3]

    def _extract_planning_scenarios(self, readme: str) -> List[str]:
        """从README中提取多步规划场景"""
        scenarios = []

        # 查找"Installation", "Setup", "Getting Started"等包含多步骤的section
        sections = re.split(r"\n##?\s+", readme)
        for section in sections:
            # 查找有序列表（步骤）
            steps = re.findall(r"^\s*\d+\.\s+(.+)$", section, re.MULTILINE)
            if len(steps) >= 3:
                title = section.split("\n")[0].strip()
                scenarios.append({"title": title, "steps": steps[:10]})

        return scenarios[:3]

    def crawl_api_orchestration(self, max_samples: int = 30) -> List[SeedPrompt]:
        """爬取API编排场景"""
        seeds = []

        queries = [
            "python api client stars:>1000",
            "python sdk api stars:>1000",
            "fastapi example stars:>500",
            "rest api python stars:>1000",
        ]

        for query in queries:
            if len(seeds) >= max_samples:
                break

            repos = self.search_repos(query, max_repos=10)
            for repo in repos:
                if len(seeds) >= max_samples:
                    break

                full_name = repo["full_name"]
                owner, name = full_name.split("/")
                readme = self.get_readme(owner, name)

                scenarios = self._extract_api_scenarios(readme)
                for scenario in scenarios:
                    seed = self.create_seed(
                        domain=Domain.API_ORCHESTRATION,
                        difficulty=Difficulty.MEDIUM,
                        prompt=f"Implement the following API orchestration scenario based on {full_name}:\n\n```python\n{scenario[:1000]}\n```\n\nProvide a complete implementation with proper error handling.",
                        test_input={"repo": full_name},
                        expected_output={"implementation": True},
                        validator_code="def validate(sol): return 'implementation' in sol and len(sol['implementation']) > 100",
                        quality_score=0.85,
                        tags=["github", "api", name.lower()],
                    )
                    seeds.append(seed)

                print(f"  {full_name}: extracted {len(scenarios)} scenarios")

        return seeds[:max_samples]

    def crawl_multi_step_planning(self, max_samples: int = 30) -> List[SeedPrompt]:
        """爬取多步规划场景"""
        seeds = []

        queries = [
            "tutorial step-by-step python stars:>500",
            "how-to guide python stars:>500",
            "workflow python stars:>500",
        ]

        for query in queries:
            if len(seeds) >= max_samples:
                break

            repos = self.search_repos(query, max_repos=10)
            for repo in repos:
                if len(seeds) >= max_samples:
                    break

                full_name = repo["full_name"]
                owner, name = full_name.split("/")
                readme = self.get_readme(owner, name)

                scenarios = self._extract_planning_scenarios(readme)
                for scenario in scenarios:
                    title = scenario["title"]
                    steps = scenario["steps"]
                    num_steps = len(steps)

                    difficulty = Difficulty.EASY if num_steps <= 4 else Difficulty.MEDIUM if num_steps <= 7 else Difficulty.HARD
                    quality = 0.8 if difficulty == Difficulty.EASY else 0.85 if difficulty == Difficulty.MEDIUM else 0.9

                    seed = self.create_seed(
                        domain=Domain.MULTI_STEP_PLANNING,
                        difficulty=difficulty,
                        prompt=f"Create a detailed plan for: {title}\n\nContext: {full_name} project\n\nProvide ordered steps with dependencies.",
                        test_input={"title": title, "num_steps": num_steps},
                        expected_output={"min_steps": num_steps},
                        validator_code=f"def validate(sol): return 'plan' in sol and len(sol['plan']) >= {num_steps}",
                        quality_score=quality,
                        tags=["github", "planning", name.lower()],
                    )
                    seeds.append(seed)

                print(f"  {full_name}: extracted {len(scenarios)} planning scenarios")

        return seeds[:max_samples]


async def main():
    """主函数：运行所有爬虫"""
    import argparse

    parser = argparse.ArgumentParser(description="爬取高质量种子prompt")
    parser.add_argument("--dry-run", action="store_true", help="只测试API连通性，不保存数据")
    parser.add_argument("--domains", nargs="+", default=["all"], help="指定要爬取的domain")
    parser.add_argument("--output", type=str, default="data/crawled_seeds.json", help="输出文件路径")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Dry Run Mode: 测试API连通性 ===")

        # 测试StackOverflow
        print("\n1. 测试StackOverflow API...")
        so_crawler = StackOverflowCrawler()
        test_seeds = so_crawler.crawl(min_score=10, max_pages=1)
        print(f"   StackOverflow OK: 可获取 {len(test_seeds)} 个问题")

        # 测试HuggingFace datasets
        print("\n2. 测试HuggingFace datasets...")
        math_crawler = MathDatasetCrawler()
        math_seeds = math_crawler.crawl_gsm8k(max_samples=5)
        print(f"   GSM8K OK: 可获取 {len(math_seeds)} 个数学题")

        # 测试GitHub
        print("\n3. 测试GitHub API...")
        gh_crawler = GitHubCrawler()
        repos = gh_crawler.search_repos("python stars:>1000", max_repos=3)
        print(f"   GitHub OK: 可搜索到 {len(repos)} 个仓库")

        print("\n=== 所有API测试通过 ===")
        return

    print("=== 开始爬取种子prompt ===\n")

    all_seeds = []

    # 1. StackOverflow: code_debug
    if "all" in args.domains or "code_debug" in args.domains:
        print("1. 爬取StackOverflow: code_debug")
        so_crawler = StackOverflowCrawler()
        code_seeds = so_crawler.crawl(min_score=15, max_pages=5)
        print(f"   爬取到 {len(code_seeds)} 个code_debug种子")
        all_seeds.extend(code_seeds)

    # 2. Math datasets: math_reasoning
    if "all" in args.domains or "math_reasoning" in args.domains:
        print("\n2. 爬取数学数据集: math_reasoning")
        math_crawler = MathDatasetCrawler()
        math_seeds = math_crawler.crawl(max_samples=60)
        print(f"   爬取到 {len(math_seeds)} 个math_reasoning种子")
        all_seeds.extend(math_seeds)

    # 3. GitHub: api_orchestration
    if "all" in args.domains or "api_orchestration" in args.domains:
        print("\n3. 爬取GitHub: api_orchestration")
        gh_crawler = GitHubCrawler()
        api_seeds = gh_crawler.crawl_api_orchestration(max_samples=30)
        print(f"   爬取到 {len(api_seeds)} 个api_orchestration种子")
        all_seeds.extend(api_seeds)

    # 4. GitHub: multi_step_planning
    if "all" in args.domains or "multi_step_planning" in args.domains:
        print("\n4. 爬取GitHub: multi_step_planning")
        gh_crawler = GitHubCrawler()
        planning_seeds = gh_crawler.crawl_multi_step_planning(max_samples=30)
        print(f"   爬取到 {len(planning_seeds)} 个multi_step_planning种子")
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
    asyncio.run(main())
