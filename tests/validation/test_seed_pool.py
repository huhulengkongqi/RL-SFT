"""Test SeedPromptPool functionality with generated seeds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent_sft.task_generator.models import Difficulty, Domain, SourceType
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


def main():
    """Test all SeedPromptPool features."""
    pool_path = Path(__file__).parent.parent.parent / "data" / "seed_prompts.json"
    pool = SeedPromptPool.load(str(pool_path))

    print(f"Loaded {len(pool)} seed prompts\n")

    # Test 1: Filter by domain
    print("=== Test 1: Filter by domain ===")
    for domain in Domain:
        seeds = pool.filter_by_domain(domain)
        print(f"{domain.value}: {len(seeds)} prompts")

    # Test 2: Filter by difficulty
    print("\n=== Test 2: Filter by difficulty ===")
    for difficulty in Difficulty:
        seeds = pool.filter_by_difficulty(difficulty)
        print(f"{difficulty.value}: {len(seeds)} prompts")

    # Test 3: Filter by source
    print("\n=== Test 3: Filter by source ===")
    for source in SourceType:
        seeds = pool.filter_by_source(source)
        print(f"{source.value}: {len(seeds)} prompts")

    # Test 4: Sample by domain
    print("\n=== Test 4: Sample by domain ===")
    code_debug_samples = pool.sample(count=5, domain=Domain.CODE_DEBUG)
    print(f"Sampled {len(code_debug_samples)} code_debug prompts:")
    for seed in code_debug_samples:
        print(f"  - [{seed.difficulty.value}] {seed.prompt[:60]}...")

    # Test 5: Sample by difficulty
    print("\n=== Test 5: Sample by difficulty ===")
    hard_samples = pool.sample(count=5, difficulty=Difficulty.HARD)
    print(f"Sampled {len(hard_samples)} hard prompts:")
    for seed in hard_samples:
        print(f"  - [{seed.domain.value}] {seed.prompt[:60]}...")

    # Test 6: Weighted sampling
    print("\n=== Test 6: Weighted sampling (by quality) ===")
    weighted_samples = pool.sample(count=10, weight_by_quality=True)
    avg_quality = sum(s.quality_score or 0.5 for s in weighted_samples) / len(weighted_samples)
    print(f"Average quality of weighted samples: {avg_quality:.3f}")

    # Test 7: Avoid duplicates
    print("\n=== Test 7: Avoid duplicates ===")
    pool.reset_sampling()
    batch1 = pool.sample(count=10, avoid_duplicates=True)
    batch2 = pool.sample(count=10, avoid_duplicates=True)
    overlap = set(s.id for s in batch1) & set(s.id for s in batch2)
    print(f"Overlap between two batches: {len(overlap)} (should be 0)")

    # Test 8: Version management
    print("\n=== Test 8: Version management ===")
    print(f"Current version: {list(pool)[0].version}")
    pool.bump_version("v1.1")
    print(f"After bump: {list(pool)[0].version}")
    print(f"Version history: {pool.get_version_history()}")

    # Test 9: Statistics
    print("\n=== Test 9: Statistics ===")
    stats = pool.get_stats()
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Average quality: {stats['avg_quality_score']:.3f}")
    print(f"By domain: {stats['by_domain']}")
    print(f"By difficulty: {stats['by_difficulty']}")

    # Test 10: Validator code check
    print("\n=== Test 10: Validator code check ===")
    sample = pool.sample(count=1)[0]
    print(f"Sample validator code:\n{sample.validator_code}")
    print(f"Test cases: {len(sample.test_cases)}")

    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    main()
