"""流式处理96GB StackOverflow Posts.xml，筛选Python相关高票问题
Windows兼容版本 - 不依赖WSL
"""
import json
import html
import sys
from pathlib import Path

# Windows编码兼容
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def stream_filter_python_questions(xml_path: str, min_score: int = 10, max_results: int = 500):
    """
    流式处理XML，不加载整个文件到内存
    只处理PostTypeId=1的问题，Tags包含python，Score>=min_score
    """
    import xml.etree.ElementTree as ET

    candidates = []
    count = 0

    print(f"开始流式处理: {xml_path}")
    print(f"筛选条件: Tags包含'python', Score>={min_score}")

    context = ET.iterparse(xml_path, events=("end",))

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            elem.clear()
            continue

        count += 1

        if count % 1_000_000 == 0:
            print(f"  已处理 {count//1_000_000}M 行, 找到 {len(candidates)} 个候选...")

        attrs = elem.attrib

        # 只处理问题PostTypeId=1，不是答案
        if attrs.get("PostTypeId") != "1":
            elem.clear()
            continue

        # 必须包含python标签
        tags = attrs.get("Tags", "")
        if "python" not in tags:
            elem.clear()
            continue

        score = int(attrs.get("Score", 0))
        if score < min_score:
            elem.clear()
            continue

        # 必须有代码块标记
        body = attrs.get("Body", "")
        if "<code>" not in body:
            elem.clear()
            continue

        # 保存候选
        candidate = {
            "Id": int(attrs["Id"]),
            "Score": score,
            "Title": html.unescape(attrs.get("Title", "")),
            "Tags": tags.strip("|").split("|"),
            "AcceptedAnswerId": int(attrs.get("AcceptedAnswerId", 0)) if attrs.get("AcceptedAnswerId") else None,
            "ViewCount": int(attrs.get("ViewCount", 0)),
            "_has_accepted_answer": bool(attrs.get("AcceptedAnswerId")),
            "_code_count": body.count("<code>"),
        }
        candidates.append(candidate)

        if len(candidates) >= max_results:
            print(f"已收集 {max_results} 个候选，停止处理")
            break

        elem.clear()

    print(f"\n处理完成: 总共扫描 {count} 行")
    print(f"找到 {len(candidates)} 个Python相关高质量问题")

    # 按Score降序排序
    candidates.sort(key=lambda x: x["Score"], reverse=True)

    # 保存候选
    output_path = Path("data/processed/so_python_candidates.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"\n候选已保存到: {output_path}")

    # 打印Top 10
    print("\nTop 10最高票Python问题:")
    for i, c in enumerate(candidates[:10]):
        print(f"  {i+1}. [Score: {c['Score']}] {c['Title'][:80]}...")

    return candidates


if __name__ == "__main__":
    stream_filter_python_questions(
        "data/raw/Posts.xml",
        min_score=20,
        max_results=500,
    )
