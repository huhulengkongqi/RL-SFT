"""GSM8K真实数学题处理脚本 - 精选50道高质量题"""
import json
import re
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import (
    Difficulty,
    Domain,
    SeedPrompt,
    SourceType,
    TaskTestCase,
)
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def count_solution_steps(answer: str) -> int:
    """统计解题步骤数，用于难度分级"""
    # GSM8K用<<>>标记中间计算
    step_markers = re.findall(r"<<.*?>>", answer)
    if step_markers:
        return len(step_markers)
    # 退回到按换行统计
    return max(1, answer.count("\n"))


def classify_difficulty_by_steps(steps: int) -> Difficulty:
    """按步骤数分级难度"""
    if steps <= 3:
        return Difficulty.EASY
    elif steps <= 6:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


def classify_question_type(question: str) -> str:
    """简单分类数学题型"""
    q_lower = question.lower()
    if any(word in q_lower for word in ["percent", "%", "rate", "interest"]):
        return "percentage_rate"
    elif any(word in q_lower for word in ["time", "hour", "minute", "speed", "mph", "km/h"]):
        return "time_speed"
    elif any(word in q_lower for word in ["ratio", "proportion", "per"]):
        return "ratio_proportion"
    elif any(word in q_lower for word in ["probability", "chance", "likely", "marble", "card"]):
        return "probability"
    elif any(word in q_lower for word in ["area", "volume", "rectangle", "circle", "triangle", "length", "width"]):
        return "geometry"
    elif any(word in q_lower for word in ["average", "mean", "sum", "total"]):
        return "arithmetic"
    else:
        return "algebra_word_problem"


def extract_final_answer(answer: str) -> str:
    """提取GSM8K的最终答案（####后面的部分）"""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def select_diverse_problems(df: pd.DataFrame, target_per_difficulty: dict) -> list:
    """保证题型多样性的精选算法"""
    selected = []

    for difficulty, target_count in target_per_difficulty.items():
        mask = df["difficulty"] == difficulty.value
        diff_df = df[mask].copy()

        if len(diff_df) <= target_count:
            selected.extend(diff_df.to_dict("records"))
            continue

        # 按题型分组，均匀抽取每种题型
        type_groups = {}
        for _, row in diff_df.iterrows():
            q_type = row["question_type"]
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(row.to_dict())

        # 每个组至少抽1个，剩余按数量分配
        types_list = list(type_groups.values())
        selected_in_diff = []

        # 第一轮：每个类型至少1个
        for group in types_list:
            if group:
                selected_in_diff.append(group.pop(0))

        # 第二轮：剩下的按比例分配
        remaining = target_count - len(selected_in_diff)
        all_remaining = []
        for group in types_list:
            all_remaining.extend(group)

        # 优先选步骤更多、质量更好的（更长的问题和答案）
        all_remaining.sort(key=lambda x: len(x["question"]) + len(x["answer"]), reverse=True)
        selected_in_diff.extend(all_remaining[:remaining])

        selected.extend(selected_in_diff)

    return selected


def main():
    print("=" * 60)
    print("GSM8K 数学题处理")
    print("=" * 60)

    # 1. 读取数据
    input_path = Path("data/raw/train-00000-of-00001.parquet")
    print(f"\n1. 读取数据: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   总题数: {len(df)}")

    # 2. 预处理
    print("\n2. 预处理和特征提取...")
    df["steps"] = df["answer"].apply(count_solution_steps)
    df["difficulty"] = df["steps"].apply(classify_difficulty_by_steps)
    df["question_type"] = df["question"].apply(classify_question_type)
    df["final_answer"] = df["answer"].apply(extract_final_answer)
    df["quality_score"] = df.apply(
        lambda row: min(0.95, 0.7 + row["steps"] * 0.04),
        axis=1,
    )

    print("   难度分布:")
    for diff, count in df["difficulty"].value_counts().items():
        print(f"     {diff}: {count}")

    print("\n   题型分布:")
    for q_type, count in df["question_type"].value_counts().items():
        print(f"     {q_type}: {count}")

    # 3. 多样性精选
    print("\n3. 多样性精选50道题...")
    target_dist = {
        Difficulty.EASY: 20,
        Difficulty.MEDIUM: 20,
        Difficulty.HARD: 10,
    }
    selected_records = select_diverse_problems(df, target_dist)
    print(f"   精选完成: {len(selected_records)} 道题")

    # 4. 转为SeedPrompt格式
    print("\n4. 转换为标准SeedPrompt格式...")
    pool = SeedPromptPool()

    for record in selected_records:
        seed = SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.MATH_REASONING,
            difficulty=record["difficulty"],
            prompt=f"""Solve the following math problem step by step. Show your work clearly and provide the final answer.\n\n{record["question"]}""",
            test_cases=[TaskTestCase(
                input={"question": record["question"]},
                expected_output={"final_answer": record["final_answer"], "steps_required": True},
                is_public=True,
            )],
            validator_code="def validate(sol):\n    return isinstance(sol, dict) and 'steps' in sol and len(sol['steps']) >= 2 and 'final_answer' in sol",
            source=SourceType.CRAWLED,
            quality_score=round(record["quality_score"], 2),
            tags=["gsm8k", "math_reasoning", record["question_type"], f"steps:{record['steps']}"],
        )
        pool.add(seed)

    # 5. 保存结果
    output_path = Path("data/processed/gsm8k_50_seeds.json")
    output_path.parent.mkdir(exist_ok=True)
    pool.save(str(output_path))
    print(f"   已保存到: {output_path}")

    # 6. 统计报告
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
            print(f"  问题: {seed.prompt[:100]}...")
            print(f"  质量分: {seed.quality_score}")
            print(f"  Tags: {seed.tags}")


if __name__ == "__main__":
    main()
