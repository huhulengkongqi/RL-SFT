"""从StackOverflow Posts.xml中提取带代码的真实问题"""
import json
import html
import re
import sys
import uuid
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def extract_code_blocks(body: str) -> list:
    """从HTML Body中提取<code>标签里的内容"""
    # 提取所有<code>...</code>之间的内容
    code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    matches = code_pattern.findall(body)

    # 解码HTML实体，过滤短代码
    results = []
    for m in matches:
        decoded = html.unescape(m)
        if len(decoded.strip()) >= 20:  # 只保留有意义的代码片段
            results.append(decoded)
    return results


def stream_and_extract(xml_path: str, target_count: int = 50):
    """流式处理，提取带代码的问题"""
    import xml.etree.ElementTree as ET

    print(f"开始流式处理: {xml_path}")
    print(f"目标: 提取 {target_count} 个带代码的Python问题")

    results = []
    count = 0

    context = ET.iterparse(xml_path, events=("end",))

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            elem.clear()
            continue

        count += 1

        if count % 500_000 == 0:
            print(f"  已扫描 {count//1000}K 行，已找到 {len(results)} 个符合条件的...")

        attrs = elem.attrib

        # 筛选条件
        if attrs.get("PostTypeId") != "1":  # 只处理问题
            elem.clear()
            continue

        if "python" not in attrs.get("Tags", ""):
            elem.clear()
            continue

        score = int(attrs.get("Score", 0))
        if score < 50:  # 提高门槛，只要高质量问题
            elem.clear()
            continue

        body = attrs.get("Body", "")
        code_blocks = extract_code_blocks(body)

        if not code_blocks:  # 必须有代码片段
            elem.clear()
            continue

        # 保存结果
        results.append({
            "Id": int(attrs["Id"]),
            "Score": score,
            "Title": html.unescape(attrs.get("Title", "")),
            "Tags": html.unescape(attrs.get("Tags", "")).strip("|").split("|"),
            "CodeBlocks": code_blocks[:3],  # 最多保留3个代码片段
            "ViewCount": int(attrs.get("ViewCount", 0)),
            "AcceptedAnswerId": int(attrs.get("AcceptedAnswerId", 0)) if attrs.get("AcceptedAnswerId") else None,
        })

        elem.clear()

        if len(results) >= target_count * 2:  # 多提取一些供后续筛选
            print(f"  已收集足够候选，停止处理")
            break

    print(f"\n扫描完成！共扫描 {count} 行")
    print(f"找到 {len(results)} 个带代码的Python高票问题")

    # 按Score降序排序，精选50个
    results.sort(key=lambda x: x["Score"], reverse=True)
    selected = results[:target_count]

    # 按难度分层
    easy = [r for r in selected if r["Score"] < 200][:20]
    medium = [r for r in selected if 200 <= r["Score"] < 1000][:20]
    hard = [r for r in selected if r["Score"] >= 1000][:10]
    final = easy + medium + hard

    print(f"精选 {len(final)} 个：")
    print(f"  Easy (Score<200): {len(easy)}")
    print(f"  Medium (200<=Score<1000): {len(medium)}")
    print(f"  Hard (Score>=1000): {len(hard)}")

    # 转为SeedPrompt格式
    seeds = []
    for q in final:
        score = q["Score"]
        if score < 200:
            diff = "easy"
            q_score = min(0.9, 0.75 + score / 1000)
        elif score < 1000:
            diff = "medium"
            q_score = min(0.92, 0.8 + score / 3000)
        else:
            diff = "hard"
            q_score = min(0.95, 0.85 + score / 10000)

        tags = [t for t in q["Tags"] if t != "python"][:5]

        # 格式化代码显示
        code_display = "\n\n".join([f"```python\n{cb[:300]}\n```" for cb in q["CodeBlocks"][:2]])

        prompt_text = f"""Debug and fix the following Python issue:

**Question**: {q['Title']}

**Code Context from the question**:
{code_display}

Provide:
1. Root cause analysis - explain the bug
2. The corrected working code
3. Explanation of why the fix works
4. Common pitfalls to avoid with this pattern"""

        seeds.append({
            "id": str(uuid.uuid4()),
            "domain": "code_debug",
            "difficulty": diff,
            "prompt": prompt_text,
            "test_cases": [{
                "input": {"so_id": q["Id"], "title": q["Title"]},
                "expected_output": {"root_cause": True, "fixed_code": True, "explanation": True},
                "is_public": True
            }],
            "validator_code": "def validate(sol): return isinstance(sol, dict) and 'root_cause' in sol and 'fixed_code' in sol",
            "source": "crawled",
            "quality_score": round(q_score, 2),
            "tags": ["stackoverflow", "real-world", "code", "debug"] + tags,
        })

    print(f"\n代码长度统计:")
    code_lengths = [sum(len(c) for c in q["CodeBlocks"]) for q in final]
    print(f"  平均代码长度: {sum(code_lengths)/len(code_lengths):.0f} 字符")
    print(f"  最长: {max(code_lengths)}, 最短: {min(code_lengths)}")

    # 保存
    output_path = Path("data/processed/so_code_debug_50_real.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"prompts": seeds}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已保存到: {output_path}")

    # 抽样展示
    print("\n" + "=" * 60)
    print("抽样展示（前2个）：")
    print("=" * 60)
    for i, s in enumerate(seeds[:2]):
        print(f"\n--- {i+1}. [{s['difficulty']}, q={s['quality_score']}] ---")
        first_lines = s['prompt'].split("\n")[:5]
        for line in first_lines:
            print(f"  {line}")

    return seeds


if __name__ == "__main__":
    stream_and_extract("data/raw/Posts.xml", 50)
