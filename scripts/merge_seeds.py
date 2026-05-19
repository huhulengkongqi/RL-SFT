"""合并LLM生成和爬取的种子prompt，进行质量过滤和去重"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.models import Difficulty, Domain, SeedPrompt
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def filter_by_quality(pool: SeedPromptPool, min_quality: float = 0.75) -> SeedPromptPool:
    """按质量分过滤"""
    filtered = SeedPromptPool()
    for seed in pool:
        if seed.quality_score >= min_quality:
            filtered.add(seed)
    return filtered


def balance_domains(pool: SeedPromptPool, target_per_domain: int = 50) -> SeedPromptPool:
    """平衡每个domain的数量"""
    domain_to_seeds = defaultdict(list)
    for seed in pool:
        domain_to_seeds[seed.domain].append(seed)

    balanced = SeedPromptPool()
    for domain, seeds in domain_to_seeds.items():
        # 按质量分降序排列，取前N个
        sorted_seeds = sorted(seeds, key=lambda s: s.quality_score, reverse=True)
        for seed in sorted_seeds[:target_per_domain]:
            balanced.add(seed)

    return balanced


def balance_difficulty(pool: SeedPromptPool) -> SeedPromptPool:
    """平衡难度分布，目标 easy:medium:hard = 4:4:2"""
    balanced = SeedPromptPool()

    for domain in [Domain.CODE_DEBUG, Domain.MATH_REASONING, Domain.API_ORCHESTRATION, Domain.MULTI_STEP_PLANNING]:
        domain_seeds = [s for s in pool if s.domain == domain]
        if not domain_seeds:
            continue

        by_difficulty = defaultdict(list)
        for seed in domain_seeds:
            by_difficulty[seed.difficulty].append(seed)

        # 按质量分降序排列
        easy = sorted(by_difficulty.get(Difficulty.EASY, []), key=lambda s: s.quality_score, reverse=True)[:20]
        medium = sorted(by_difficulty.get(Difficulty.MEDIUM, []), key=lambda s: s.quality_score, reverse=True)[:20]
        hard = sorted(by_difficulty.get(Difficulty.HARD, []), key=lambda s: s.quality_score, reverse=True)[:10]

        for seed in easy + medium + hard:
            balanced.add(seed)

    return balanced


def deduplicate_simple(pool: SeedPromptPool) -> SeedPromptPool:
    """简单去重 - 基于prompt开头的相似性"""
    seen = set()
    deduped = SeedPromptPool()

    for seed in pool:
        # 取前50个字符作为fingerprint
        fingerprint = seed.prompt.strip()[:50].lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.add(seed)

    return deduped


def balance_source_distribution(pool: SeedPromptPool) -> SeedPromptPool:
    """平衡source分布：约50% crawled，50% llm_generated"""
    crawled = [s for s in pool if s.source == "crawled"]
    llm_generated = [s for s in pool if s.source == "llm_generated"]

    # 每个domain取一半crawled，一半llm_generated
    balanced = SeedPromptPool()

    for domain in [Domain.CODE_DEBUG, Domain.MATH_REASONING, Domain.API_ORCHESTRATION, Domain.MULTI_STEP_PLANNING]:
        domain_crawled = sorted([s for s in crawled if s.domain == domain], key=lambda s: s.quality_score, reverse=True)[:25]
        domain_llm = sorted([s for s in llm_generated if s.domain == domain], key=lambda s: s.quality_score, reverse=True)[:25]

        for seed in domain_crawled + domain_llm:
            balanced.add(seed)

    return balanced


def main():
    import argparse

    parser = argparse.ArgumentParser(description="合并并过滤种子prompt")
    parser.add_argument("--llm-seeds", type=str, default="data/seed_prompts.json", help="LLM生成的种子文件")
    parser.add_argument("--crawled-seeds", type=str, default="data/crawled_seeds.json", help="爬取的种子文件")
    parser.add_argument("--output", type=str, default="data/final_seeds.json", help="输出文件路径")
    args = parser.parse_args()

    print("=== 合并种子prompt ===\n")

    # 加载LLM生成的种子
    print(f"1. 加载LLM生成的种子: {args.llm_seeds}")
    llm_pool = SeedPromptPool.load(args.llm_seeds)
    print(f"   已加载 {len(llm_pool)} 个LLM生成的种子")

    # 加载爬取的种子（如果不存在，就用当前目录下的，如果还是不存在，就创建一个）
    print(f"\n2. 加载爬取的种子: {args.crawled_seeds}")
    try:
        crawled_pool = SeedPromptPool.load(args.crawled_seeds)
        print(f"   已加载 {len(crawled_pool)} 个爬取的种子")
    except Exception:
        print("   爬取的种子文件不存在，将使用LLM生成的种子进行精选")
        crawled_pool = SeedPromptPool()

    # 合并
    print("\n3. 合并种子池...")
    combined_pool = SeedPromptPool()
    for seed in llm_pool:
        combined_pool.add(seed)
    for seed in crawled_pool:
        combined_pool.add(seed)
    print(f"   合并后共 {len(combined_pool)} 个种子")

    # 去重
    print("\n4. 简单去重...")
    deduped_pool = deduplicate_simple(combined_pool)
    print(f"   去重后共 {len(deduped_pool)} 个种子")

    # 质量过滤
    print("\n5. 质量过滤（保留 quality >= 0.75）...")
    filtered_pool = filter_by_quality(deduped_pool, 0.75)
    print(f"   过滤后共 {len(filtered_pool)} 个种子")

    # 平衡source分布
    print("\n6. 平衡source分布（约50% crawled，50% llm_generated）...")
    source_balanced = balance_source_distribution(filtered_pool)
    print(f"   平衡后共 {len(source_balanced)} 个种子")

    # 平衡每个domain的难度
    print("\n7. 平衡难度分布（easy:medium:hard ≈ 2:2:1 per domain）...")
    final_pool = balance_difficulty(source_balanced)
    print(f"   最终共 {len(final_pool)} 个种子")

    # 保存结果
    output_path = Path(__file__).parent.parent / args.output
    final_pool.save(str(output_path))

    print(f"\n=== 合并完成 ===")
    print(f"保存到: {output_path}")
    print("\n最终统计信息:")
    stats = final_pool.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
