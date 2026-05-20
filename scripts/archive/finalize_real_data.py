"""从真实爬取的500个StackOverflow候选中精选50个code_debug种子
与GSM8K的50个合并，组成最终100个100%真实来源的种子池
"""
import json
import sys
import uuid
from collections import Counter
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


def classify_difficulty(score: int) -> Difficulty:
    """根据StackOverflow的投票数分级难度"""
    if score < 100:
        return Difficulty.EASY      # 基础常见问题
    elif score < 500:
        return Difficulty.MEDIUM   # 需要一定理解的问题
    else:
        return Difficulty.HARD     # 高级概念，很多人都困惑的问题


def quality_score(question: dict) -> float:
    """质量打分"""
    score = 0.7  # 基础分

    # 高票加分
    if question["Score"] >= 5000:
        score += 0.2
    elif question["Score"] >= 1000:
        score += 0.15
    elif question["Score"] >= 500:
        score += 0.1
    elif question["Score"] >= 100:
        score += 0.05

    # 有采纳答案加分
    if question["_has_accepted_answer"]:
        score += 0.05

    # 代码块数量加分（说明问题描述详细）
    if question["_code_count"] >= 2:
        score += 0.05

    return min(0.95, round(score, 2))


def select_diverse_questions(candidates: list, target: int = 50) -> list:
    """多样性精选：按难度和标签分布选最好的"""
    by_difficulty = {
        Difficulty.EASY: [],
        Difficulty.MEDIUM: [],
        Difficulty.HARD: [],
    }

    for q in candidates:
        diff = classify_difficulty(q["Score"])
        by_difficulty[diff].append(q)

    # 每个难度按质量排序
    for diff in by_difficulty:
        by_difficulty[diff].sort(key=quality_score, reverse=True)

    selected = []
    # 均衡分布：20 easy, 20 medium, 10 hard
    selected.extend(by_difficulty[Difficulty.EASY][:20])
    selected.extend(by_difficulty[Difficulty.MEDIUM][:20])
    selected.extend(by_difficulty[Difficulty.HARD][:10])

    return selected


def convert_to_seed_prompt(question: dict) -> SeedPrompt:
    """把StackOverflow问题转为标准SeedPrompt格式"""
    difficulty = classify_difficulty(question["Score"])
    q_score = quality_score(question)

    # 提取标签，过滤掉python本身，保留更具体的
    tags = [t for t in question["Tags"] if t != "python"]

    prompt_text = f"""Debug and understand the following Python question:

**Question Title**: {question['Title']}

**This is a real StackOverflow question** with Score: {question['Score']}, ViewCount: {question.get('ViewCount', 'N/A')}

Provide:
1. Clear explanation of what the question is asking
2. Root cause analysis - why is this a common source of bugs/confusion?
3. Correct solution with code example
4. Common pitfalls and how to avoid them in future
5. If applicable: alternative approaches with trade-offs"""

    return SeedPrompt(
        id=str(uuid.uuid4()),
        domain=Domain.CODE_DEBUG,
        difficulty=difficulty,
        prompt=prompt_text,
        test_cases=[TaskTestCase(
            input={"so_id": question["Id"], "title": question["Title"]},
            expected_output={"explanation": True, "root_cause": True, "solution": True},
            is_public=True,
        )],
        validator_code="def validate(sol):\n    return isinstance(sol, dict) and 'root_cause' in sol and 'solution_code' in sol and len(sol['root_cause']) > 30",
        source=SourceType.CRAWLED,
        quality_score=q_score,
        tags=["stackoverflow", "real-world", "debug"] + tags[:5],
    )


def main():
    print("=" * 70)
    print("🎯 从真实爬取的StackOverflow数据中精选code_debug种子")
    print("=" * 70)

    final_pool = SeedPromptPool()

    # ========== 1. 加载GSM8K的50个数学题 ==========
    print("\n1. 加载 GSM8K math_reasoning 种子...")
    gsm_pool = SeedPromptPool.load("data/processed/gsm8k_50_seeds.json")
    for seed in gsm_pool:
        final_pool.add(seed)
    print(f"   ✅ 已加载 {len(gsm_pool)} 个真实数学题")

    # ========== 2. 处理StackOverflow的code_debug ==========
    print("\n2. 加载 StackOverflow 候选...")
    with open("data/processed/so_python_candidates.json", "r", encoding="utf-8") as f:
        so_candidates = json.load(f)
    print(f"   已加载 {len(so_candidates)} 个StackOverflow Python候选")

    # 多样性精选50个
    selected_so = select_diverse_questions(so_candidates, 50)
    print(f"   精选出 {len(selected_so)} 个高质量code_debug问题")

    # 转为SeedPrompt格式
    print("\n3. 转换为标准SeedPrompt格式...")
    for q in selected_so:
        seed = convert_to_seed_prompt(q)
        final_pool.add(seed)

    # ========== 3. 保存最终结果 ==========
    output_path = Path("data/seed_prompts_100_real_final.json")
    final_pool.save(str(output_path))

    print(f"\n✅ 最终种子池已保存到: {output_path}")

    # ========== 4. 统计报告 ==========
    print("\n" + "=" * 70)
    print("📊 最终种子池统计报告")
    print("=" * 70)
    stats = final_pool.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Source分布（应该全部是 crawled）
    sources = Counter(s.source.value for s in final_pool)
    print(f"\n✅ Source 100% 真实爬取: {dict(sources)}")

    # 质量分统计
    qualities = [s.quality_score for s in final_pool]
    print(f"\n质量分: min={min(qualities):.2f}, avg={sum(qualities)/len(qualities):.2f}, max={max(qualities):.2f}")

    # 抽样展示
    print("\n" + "=" * 70)
    print("🔍 抽样展示")
    print("=" * 70)

    # 每个domain每个难度各抽1个
    for domain in [Domain.MATH_REASONING, Domain.CODE_DEBUG]:
        for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            seeds = [s for s in final_pool if s.domain == domain and s.difficulty == diff]
            if seeds:
                s = seeds[0]
                print(f"\n{domain.value.upper()} / {diff.value.upper()} (quality: {s.quality_score})")
                print(f"  {s.prompt[:100]}...")


if __name__ == "__main__":
    main()
