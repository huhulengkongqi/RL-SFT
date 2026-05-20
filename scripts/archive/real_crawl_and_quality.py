"""真实爬取 + LLM自动质量判别流水线

功能：
1. 从HuggingFace加载GSM8K真实数学题
2. 从StackOverflow API爬取真实Python debug问题
3. 使用VLLM client自动进行质量打分和过滤
4. 合并到种子池，保持source标记正确
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

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

# 尝试导入datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed, run: uv pip install datasets")


class QualityScorer:
    """使用LLM自动对prompt进行质量打分"""

    def __init__(self, llm_client=None):
        self.client = llm_client
        # 如果没有提供llm_client，使用启发式规则
        self.use_heuristic = llm_client is None

    def score(self, prompt: str, domain: str, difficulty: str) -> Dict[str, Any]:
        """
        对单个prompt进行质量评价

        Returns:
            {
                "quality_score": float 0~1,
                "is_valid": bool,
                "reason": str,
                "difficulty_adjusted": str
            }
        """
        if self.use_heuristic:
            return self._heuristic_score(prompt, domain, difficulty)
        return self._llm_score(prompt, domain, difficulty)

    def _heuristic_score(self, prompt: str, domain: str, difficulty: str) -> Dict[str, Any]:
        """启发式质量打分规则"""
        score = 0.5
        reasons = []

        # 1. 长度检查
        length = len(prompt)
        if length < 50:
            score -= 0.2
            reasons.append("prompt过短")
        elif length < 100:
            score -= 0.1
            reasons.append("prompt较短")
        elif 150 < length < 500:
            score += 0.1
            reasons.append("长度适中")

        # 2. 是否有代码块
        if "```" in prompt or "code" in prompt.lower():
            score += 0.1
            reasons.append("包含代码示例")

        # 3. 是否有明确的任务要求
        keywords = ["fix", "solve", "implement", "create", "design", "debug", "step", "explain",
                    "修复", "解决", "实现", "创建", "设计", "调试", "步骤", "解释"]
        keyword_count = sum(1 for k in keywords if k in prompt.lower())
        if keyword_count >= 2:
            score += 0.1
            reasons.append("有明确的任务要求")

        # 4. 难度校准
        difficulty_bonus = {
            "easy": 0.0,
            "medium": 0.05,
            "hard": 0.1
        }
        score += difficulty_bonus.get(difficulty, 0)

        # 5. 领域特定检查
        if domain == "code_debug":
            if any(k in prompt.lower() for k in ["error", "bug", "fix", "debug", "错误", "调试"]):
                score += 0.1
                reasons.append("符合code_debug模式")
        elif domain == "math_reasoning":
            if any(k in prompt.lower() for k in ["solve", "calculate", "find", "what", "how",
                                                 "计算", "求解", "多少"]):
                score += 0.1
                reasons.append("符合数学题模式")
        elif domain == "api_orchestration":
            if any(k in prompt.lower() for k in ["api", "endpoint", "request", "call", "调用"]):
                score += 0.1
                reasons.append("符合API编排模式")
        elif domain == "multi_step_planning":
            if any(k in prompt.lower() for k in ["step", "plan", "steps", "flow", "流程", "步骤", "计划"]):
                score += 0.1
                reasons.append("符合多步规划模式")

        score = max(0.0, min(1.0, score))
        is_valid = score >= 0.65

        return {
            "quality_score": round(score, 2),
            "is_valid": is_valid,
            "reason": ", ".join(reasons) if reasons else "基础合格",
            "difficulty_adjusted": difficulty
        }

    async def _llm_score(self, prompt: str, domain: str, difficulty: str) -> Dict[str, Any]:
        """使用LLM进行质量打分"""
        scoring_prompt = f"""请对以下训练prompt进行质量评分，评分范围0.0~1.0。

评分标准：
0.9~1.0: 高质量，有明确教育价值，真实场景，难度适中
0.8~0.9: 良好，典型问题，适合训练
0.7~0.8: 合格，基础问题，可以使用
<0.7: 不合格，太简单，重复，或无训练价值

需要评估的prompt：
Domain: {domain}
Difficulty: {difficulty}
Prompt: {prompt}

请以JSON格式返回：
{{"quality_score": 0.85, "is_valid": true, "reason": "简短评价", "difficulty_adjusted": "easy/medium/hard"}}
"""
        try:
            response = await self.client.chat(
                model="default",
                messages=[{"role": "user", "content": scoring_prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            # 解析JSON
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["quality_score"] = float(result.get("quality_score", 0.7))
                result["is_valid"] = bool(result.get("is_valid", True))
                return result
        except Exception as e:
            print(f"  LLM评分失败: {e}，使用启发式")

        return self._heuristic_score(prompt, domain, difficulty)


def load_gsm8k_math(num_samples: int = 100) -> List[SeedPrompt]:
    """从HuggingFace加载GSM8K真实数学题"""
    if not HAS_DATASETS:
        print("⚠️  datasets not installed, skipping GSM8K")
        return []

    print(f"\n📚 加载GSM8K数据集 (目标: {num_samples}个)...")

    try:
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    except Exception as e:
        print(f"  ❌ 加载GSM8K失败: {e}")
        return []

    seeds = []
    scorer = QualityScorer()

    for i, item in enumerate(ds):
        if i >= num_samples * 2:
            break

        question = item["question"]
        answer = item["answer"]

        # 根据解题步骤数量估计难度
        num_steps = answer.count("<<") if "<<" in answer else answer.count("\n")
        if num_steps <= 2:
            difficulty = Difficulty.EASY
        elif num_steps <= 5:
            difficulty = Difficulty.MEDIUM
        else:
            difficulty = Difficulty.HARD

        # 质量打分
        quality_result = scorer.score(question, Domain.MATH_REASONING, difficulty)

        if not quality_result["is_valid"]:
            continue

        seed = SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MATH_REASONING,
            difficulty=Difficulty(quality_result["difficulty_adjusted"]),
            prompt=f"Solve the following math problem step by step:\n\n{question}",
            test_cases=[TaskTestCase(
                input={"question": question},
                expected_output={"answer": answer, "steps": True},
                is_public=True
            )],
            validator_code="def validate(sol): return 'steps' in sol and 'final_answer' in sol and len(sol['steps']) >= 2",
            source=SourceType.CRAWLED,
            quality_score=quality_result["quality_score"],
            tags=["gsm8k", "math-reasoning", "real-dataset"],
        )
        seeds.append(seed)

    # 按质量排序，取前N个
    seeds.sort(key=lambda s: s.quality_score, reverse=True)
    selected = seeds[:num_samples]

    avg_quality = sum(s.quality_score for s in selected) / len(selected) if selected else 0
    print(f"  ✅ 加载了 {len(selected)} 个GSM8K高质量数学题，平均质量: {avg_quality:.2f}")

    return selected


def crawl_stackoverflow_debug(num_samples: int = 100) -> List[SeedPrompt]:
    """从StackOverflow爬取真实Python debug问题"""
    print(f"\n🔍 爬取StackOverflow Python debug问题 (目标: {num_samples}个)...")

    seeds = []
    scorer = QualityScorer()
    page = 1

    while len(seeds) < num_samples * 2 and page <= 10:
        try:
            response = requests.get(
                "https://api.stackexchange.com/2.3/questions",
                params={
                    "pagesize": 50,
                    "page": page,
                    "order": "desc",
                    "sort": "votes",
                    "tagged": "python",
                    "site": "stackoverflow",
                    "filter": "withbody",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  ❌ StackOverflow API请求失败: {e}")
            break

        if "items" not in data:
            break

        for item in data["items"]:
            if len(seeds) >= num_samples * 2:
                break

            score = item.get("score", 0)
            if score < 5:
                continue

            title = item.get("title", "")
            body = item.get("body", "")

            # 只保留包含代码的问题
            if "<code>" not in body:
                continue

            # 根据vote估计难度
            if score < 20:
                difficulty = Difficulty.EASY
            elif score < 100:
                difficulty = Difficulty.MEDIUM
            else:
                difficulty = Difficulty.HARD

            # 提取代码片段
            import re
            code_match = re.search(r"<code>(.*?)</code>", body, re.DOTALL)
            code_snippet = code_match.group(1)[:300] if code_match else ""

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

            # 质量打分
            quality_result = scorer.score(prompt_text, Domain.CODE_DEBUG, difficulty)

            if not quality_result["is_valid"]:
                continue

            seed = SeedPrompt(
                id=str(uuid.uuid4()),
                domain=Domain.CODE_DEBUG,
                difficulty=Difficulty(quality_result["difficulty_adjusted"]),
                prompt=prompt_text,
                test_cases=[TaskTestCase(
                    input={"title": title, "code": code_snippet},
                    expected_output={"fixed": True, "analysis": True},
                    is_public=True
                )],
                validator_code="def validate(sol): return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol",
                source=SourceType.CRAWLED,
                quality_score=quality_result["quality_score"],
                tags=["stackoverflow", "debug", "python", f"score:{score}"],
            )
            seeds.append(seed)

        print(f"  Page {page}: 已收集 {len(seeds)} 个候选")
        page += 1

        if not data.get("has_more"):
            break

    # 按质量排序，取前N个
    seeds.sort(key=lambda s: s.quality_score, reverse=True)
    selected = seeds[:num_samples]

    avg_quality = sum(s.quality_score for s in selected) / len(selected) if selected else 0
    print(f"  ✅ 最终保留 {len(selected)} 个高质量debug问题，平均质量: {avg_quality:.2f}")

    return selected


def deduplicate_seeds(seeds: List[SeedPrompt], threshold: float = 0.8) -> List[SeedPrompt]:
    """简单去重 - 基于prompt开头的相似度"""
    seen = set()
    result = []

    for seed in seeds:
        fingerprint = seed.prompt.strip()[:60].lower()

        # 简单的字符级Jaccard相似度
        is_duplicate = False
        for existing in seen:
            set1 = set(fingerprint)
            set2 = set(existing)
            jaccard = len(set1 & set2) / len(set1 | set2)
            if jaccard > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen.add(fingerprint)
            result.append(seed)

    print(f"  🧹 去重: {len(seeds)} → {len(result)} (过滤了 {len(seeds) - len(result)} 个重复)")
    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="真实爬取 + 自动质量判别")
    parser.add_argument("--gsm8k", type=int, default=100, help="GSM8K数学题数量")
    parser.add_argument("--stackoverflow", type=int, default=100, help="StackOverflow爬取数量")
    parser.add_argument("--merge", action="store_true", help="合并到现有种子池")
    parser.add_argument("--output", type=str, default="data/seed_prompts_final.json", help="输出路径")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 真实爬取 + 自动质量判别流水线")
    print("=" * 60)

    all_new_seeds = []

    # 1. 爬取GSM8K数学题
    if args.gsm8k > 0:
        math_seeds = load_gsm8k_math(args.gsm8k)
        all_new_seeds.extend(math_seeds)

    # 2. 爬取StackOverflow
    if args.stackoverflow > 0:
        so_seeds = crawl_stackoverflow_debug(args.stackoverflow)
        all_new_seeds.extend(so_seeds)

    # 3. 去重
    print("\n🧹 去重过滤...")
    deduped = deduplicate_seeds(all_new_seeds)

    print(f"\n📊 新爬取汇总:")
    print(f"  总共爬取: {len(all_new_seeds)} 个")
    print(f"  去重后保留: {len(deduped)} 个")

    # 按domain统计
    from collections import Counter
    domain_counts = Counter(s.domain.value for s in deduped)
    source_counts = Counter(s.source.value for s in deduped)
    print(f"  Domain分布: {dict(domain_counts)}")
    print(f"  Source分布: {dict(source_counts)}")

    avg_quality = sum(s.quality_score for s in deduped) / len(deduped) if deduped else 0
    print(f"  平均质量分: {avg_quality:.3f}")

    # 合并到现有种子池
    if args.merge:
        print("\n🔄 合并到现有种子池...")
        try:
            existing_pool = SeedPromptPool.load("data/seed_prompts.json")
            print(f"  原有种子数: {len(existing_pool)}")

            # 过滤掉LLM生成的低质量种子，保留高质量的
            llm_generated = [s for s in existing_pool if s.source == SourceType.LLM_GENERATED]
            llm_generated.sort(key=lambda s: s.quality_score, reverse=True)
            # 每个domain只保留20个最高质量的LLM生成种子
            llm_selected = []
            for domain in [Domain.CODE_DEBUG, Domain.MATH_REASONING, Domain.API_ORCHESTRATION, Domain.MULTI_STEP_PLANNING]:
                domain_llm = [s for s in llm_generated if s.domain == domain]
                llm_selected.extend(domain_llm[:20])
            print(f"  精选LLM种子: {len(llm_selected)} 个 (每个domain前20个)")

            # 合并
            final_pool = SeedPromptPool()
            for s in llm_selected:
                final_pool.add(s)
            for s in deduped:
                final_pool.add(s)

        except Exception as e:
            print(f"  加载现有种子池失败，直接使用新爬取的: {e}")
            final_pool = SeedPromptPool()
            for s in deduped:
                final_pool.add(s)
    else:
        final_pool = SeedPromptPool()
        for s in deduped:
            final_pool.add(s)

    # 保存结果
    output_path = Path(__file__).parent.parent / args.output
    final_pool.save(str(output_path))

    print(f"\n✅ 完成！结果已保存到: {output_path}")
    print("\n📈 最终种子池统计:")
    print(json.dumps(final_pool.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
