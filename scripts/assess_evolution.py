"""
Evolution data quality assessment
For evaluating evolved generation outputs
"""
import json
from collections import Counter
from pathlib import Path
import sys


def load_evolved(file_path: str):
    """Load evolved prompts file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list format and dict {"prompts": [...]} format
    return data.get("prompts", data) if isinstance(data, dict) else data


def assess_evolution(prompts):
    """Assess evolved generation quality"""
    print("=" * 70)
    print("EVOLUTION GENERATION ASSESSMENT")
    print("=" * 70)

    # Basic stats
    total = len(prompts)
    print(f"\nTotal prompts: {total}")

    # Generation number
    gen_numbers = Counter(p["evolution_metadata"]["generation"] for p in prompts if "evolution_metadata" in p)
    print(f"\n[Generation Distribution]")
    for gen, count in sorted(gen_numbers.items()):
        print(f"  Generation {gen}: {count} prompts")

    # Strategy distribution
    strategies = Counter(p["evolution_metadata"]["strategy"] for p in prompts if "evolution_metadata" in p)
    print(f"\n[Evolution Strategy Distribution]")
    for strategy, count in strategies.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {strategy:20s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Domain distribution
    domains = Counter(p["domain"] for p in prompts)
    print(f"\n[Domain Distribution]")
    for domain, count in domains.most_common():
        pct = count / total * 100
        print(f"  {domain:20s}: {count:3d} ({pct:5.1f}%)")

    # Difficulty distribution
    difficulties = Counter(p["difficulty"] for p in prompts)
    print(f"\n[Difficulty Distribution]")
    for diff, count in difficulties.most_common():
        pct = count / total * 100
        print(f"  {diff:20s}: {count:3d} ({pct:5.1f}%)")

    # Quality score statistics
    scores = [p["quality_score"] for p in prompts if "quality_score" in p]
    if scores:
        print(f"\n[Quality Score Statistics]")
        print(f"  Min:  {min(scores):.2f}")
        print(f"  Max:  {max(scores):.2f}")
        print(f"  Avg:  {sum(scores)/len(scores):.2f}")

        # Quality score distribution
        buckets = [(0.7, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.96)]
        print(f"\n[Quality Score Distribution]")
        for low, high in buckets:
            count = sum(1 for s in scores if low <= s < high)
            pct = count / len(scores) * 100
            bar = "█" * int(pct / 2)
            print(f"  {low:.2f} - {high:.2f}: {count:3d} ({pct:5.1f}%) {bar}")

    # Prompt length statistics
    prompt_lengths = [len(p["prompt"]) for p in prompts]
    print(f"\n[Prompt Length Statistics]")
    print(f"  Min:  {min(prompt_lengths)} chars")
    print(f"  Max:  {max(prompt_lengths)} chars")
    print(f"  Avg:  {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")

    # Test case coverage
    with_test_cases = sum(1 for p in prompts if p.get("test_cases") and len(p["test_cases"]) > 0)
    print(f"\n[Test Case Coverage]")
    print(f"  With test cases: {with_test_cases}/{total} ({with_test_cases/total*100:.1f}%)")

    # Validator code coverage
    with_validator = sum(1 for p in prompts if p.get("validator_code") is not None and len(p["validator_code"]) > 0)
    print(f"  With validator code: {with_validator}/{total} ({with_validator/total*100:.1f}%)")

    # Discriminator score
    discriminator_scores = [
        p["evolution_metadata"]["discriminator_score"]
        for p in prompts
        if p.get("evolution_metadata", {}).get("discriminator_score") is not None
    ]
    if discriminator_scores:
        print(f"\n[LM Discriminator Score]")
        print(f"  Scored prompts: {len(discriminator_scores)}")
        print(f"  Min:  {min(discriminator_scores):.2f}")
        print(f"  Max:  {max(discriminator_scores):.2f}")
        print(f"  Avg:  {sum(discriminator_scores)/len(discriminator_scores):.2f}")

    # Uniqueness
    unique_prompts = len(set(p["prompt"] for p in prompts))
    print(f"\n[Uniqueness]")
    print(f"  Unique prompts: {unique_prompts}/{total} ({unique_prompts/total*100:.1f}%)")

    # Parent seed diversity
    parent_ids = Counter(p["evolution_metadata"]["parent_id"] for p in prompts if "evolution_metadata" in p)
    print(f"\n[Parent Seed Diversity]")
    print(f"  Unique parent seeds: {len(parent_ids)}")
    print(f"  Max variants per parent: {max(parent_ids.values())}")
    print(f"  Avg variants per parent: {total/len(parent_ids):.1f}")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/assess_evolution.py <evolution_file.json>")
        sys.exit(1)

    file_path = sys.argv[1]
    prompts = load_evolved(file_path)
    print(f"Loaded {len(prompts)} prompts from: {file_path}")

    assess_evolution(prompts)


if __name__ == "__main__":
    main()
