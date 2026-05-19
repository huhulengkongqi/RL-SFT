"""StackOverflow 104GB XML流式处理脚本 - 精选50个高质量Python Debug问题"""
import html
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import (
    Difficulty,
    Domain,
    SeedPrompt,
    SourceType,
    TaskTestCase,
)
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def clean_html(body: str) -> str:
    """清理HTML标签，提取纯文本和代码"""
    # 提取所有code标签内容
    code_blocks = re.findall(r"<code>(.*?)</code>", body, re.DOTALL)
    # 提取所有p标签内容
    text_blocks = re.findall(r"<p>(.*?)</p>", body, re.DOTALL)

    # 清理HTML实体
    clean_codes = [html.unescape(code).strip() for code in code_blocks]
    clean_texts = [html.unescape(re.sub(r"<[^>]+>", "", text)).strip() for text in text_blocks]

    return {
        "text": "\n".join(clean_texts),
        "code_blocks": clean_codes,
        "all_code": "\n".join(clean_codes),
    }


def classify_difficulty_by_score(score: int) -> Difficulty:
    """按投票分数分级难度"""
    if score < 50:
        return Difficulty.EASY
    elif score < 200:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


def calculate_quality_score(row: dict) -> float:
    """计算质量分 0.7~0.95"""
    score = 0.7

    # 高票数加分
    if row["Score"] >= 500:
        score += 0.15
    elif row["Score"] >= 200:
        score += 0.1
    elif row["Score"] >= 100:
        score += 0.05

    # 代码块数量加分
    code_len = len(row["_all_code"])
    if code_len >= 200:
        score += 0.05
    if code_len >= 500:
        score += 0.05

    # 问题描述长度加分
    if len(row["_text"]) >= 200:
        score += 0.05

    return min(0.95, score)


def extract_bug_type(tags: str, title: str) -> list:
    """简单提取bug类型作为tags"""
    tag_list = tags.strip("|").split("|") if tags else []

    bug_keywords = [
        ("syntax-error", ["syntax", "parse", "indent"]),
        ("type-error", ["type", "typing", "string", "int"]),
        ("index-error", ["index", "out of range", "list"]),
        ("key-error", ["key", "dict", "dictionary"]),
        ("name-error", ["name", "undefined", "variable"]),
        ("attribute-error", ["attribute", "object has no"]),
        ("concurrency", ["thread", "concurrency", "race", "deadlock"]),
        ("performance", ["performance", "speed", "slow", "memory"]),
        ("django", ["django"]),
        ("pandas", ["pandas", "dataframe"]),
        ("numpy", ["numpy"]),
        ("regex", ["regex", "regular expression"]),
    ]

    found = ["python", "debug"]
    title_lower = title.lower()

    for bug_type, keywords in bug_keywords:
        for kw in keywords:
            if kw in title_lower or kw in " ".join(tag_list).lower():
                found.append(bug_type)
                break

    return list(set(found))


def main():
    print("=" * 60)
    print("StackOverflow 流式处理 - Python Debug问题精选")
    print("=" * 60)

    input_path = Path("data/raw/Posts.xml")
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return

    file_size_gb = input_path.stat().st_size / (1024 ** 3)
    print(f"\n文件大小: {file_size_gb:.1f} GB")

    # 第一阶段：筛选候选
    print("\n" + "=" * 60)
    print("阶段1: 流式筛选候选问题...")
    print("=" * 60)

    candidates = []
    count = 0
    python_count = 0

    context = iterparse(str(input_path), events=("start", "end"))
    _, root = next(context)  # 获取根节点

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            continue

        count += 1

        if count % 1000000 == 0:
            print(f"  已处理 {count // 1000000}M 行, 找到 {python_count} 个Python候选")

        attrs = elem.attrib

        # 只处理问题，不是答案
        if attrs.get("PostTypeId") != "1":
            root.clear()
            continue

        # 必须有Python标签
        tags = attrs.get("Tags", "")
        if "python" not in tags:
            root.clear()
            continue

        # 必须有接受的答案
        if not attrs.get("AcceptedAnswerId"):
            root.clear()
            continue

        score = int(attrs.get("Score", 0))
        if score < 20:
            root.clear()
            continue

        body = attrs.get("Body", "")
        if "<code>" not in body:
            root.clear()
            continue

        # 清理HTML
        cleaned = clean_html(body)
        if len(cleaned["all_code"]) < 50:
            root.clear()
            continue

        python_count += 1

        # 保存候选
        candidate = {
            "Id": int(attrs["Id"]),
            "Score": score,
            "Title": html.unescape(attrs.get("Title", "")),
            "Tags": tags,
            "AcceptedAnswerId": int(attrs.get("AcceptedAnswerId", 0)),
            "_text": cleaned["text"],
            "_all_code": cleaned["all_code"],
            "_code_blocks": cleaned["code_blocks"],
        }

        # 质量分
        candidate["quality_score"] = calculate_quality_score(candidate)
        candidate["difficulty"] = classify_difficulty_by_score(score).value
        candidate["tags"] = extract_bug_type(tags, candidate["Title"])

        candidates.append(candidate)

        # 清理内存
        root.clear()

    print(f"\n✅ 第一阶段完成: 处理 {count} 行, 找到 {len(candidates)} 个Python候选问题")

    if not candidates:
        print("❌ 没有找到符合条件的问题")
        return

    # 第二阶段：精选
    print("\n" + "=" * 60)
    print("阶段2: 多样性精选50道题...")
    print("=" * 60)

    # 按难度分组
    by_difficulty = defaultdict(list)
    for c in candidates:
        by_difficulty[c["difficulty"]].append(c)

    print(f"  难度分布:")
    for diff, items in by_difficulty.items():
        print(f"    {diff}: {len(items)}")

    # 每个难度选最佳的
    target_dist = {
        "easy": 20,
        "medium": 20,
        "hard": 10,
    }

    selected_candidates = []
    for diff, target in target_dist.items():
        candidates_for_diff = sorted(by_difficulty.get(diff, []), key=lambda x: x["quality_score"], reverse=True)
        if len(candidates_for_diff) > target:
            selected = candidates_for_diff[:target]
        else:
            selected = candidates_for_diff
            # 不够的从别的级别补
            remaining = target - len(selected)
            for other_diff, other_candidates in by_difficulty.items():
                if other_diff != diff and remaining > 0:
                    others_sorted = sorted(other_candidates, key=lambda x: x["quality_score"], reverse=True)
                    selected.extend(others_sorted[:remaining])
                    remaining = target - len(selected)
        selected_candidates.extend(selected)

    print(f"  精选完成: {len(selected_candidates)} 个问题")

    # 第三阶段：转为SeedPrompt
    print("\n" + "=" * 60)
    print("阶段3: 转换为标准SeedPrompt格式...")
    print("=" * 60)

    pool = SeedPromptPool()

    for candidate in selected_candidates:
        code_preview = candidate["_all_code"][:500]
        if len(candidate["_all_code"]) > 500:
            code_preview += "\n... (更多代码)"

        prompt_text = f"""Debug and fix the following Python issue:

**Problem**: {candidate['Title']}

**Code Context**:
```python
{code_preview}
```

Provide:
1. Root cause analysis
2. The fixed code
3. Explanation of why the fix works"""

        seed = SeedPrompt(
            id=str(uuid.uuid4()),
            domain=Domain.CODE_DEBUG,
            difficulty=Difficulty(candidate["difficulty"]),
            prompt=prompt_text,
            test_cases=[TaskTestCase(
                input={"question_id": candidate["Id"], "title": candidate["Title"]},
                expected_output={"root_cause": True, "fixed_code": True},
                is_public=True,
            )],
            validator_code="def validate(sol):\n    return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol and len(sol['root_cause']) > 10",
            source=SourceType.CRAWLED,
            quality_score=round(candidate["quality_score"], 2),
            tags=candidate["tags"],
        )
        pool.add(seed)

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
            print(f"\n--- {diff.value.upper()} (Score ~{25 if diff == 'easy' else 100 if diff == 'medium' else 500}):")
            print(f"  标题: {seed.prompt.split('**Problem**: ')[1].split('**Code')[0][:80]}...")
            print(f"  质量分: {seed.quality_score}")
            print(f"  Tags: {seed.tags[:5]}")


if __name__ == "__main__":
    main()
