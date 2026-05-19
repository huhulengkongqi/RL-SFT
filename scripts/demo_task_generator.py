"""Demo: Using seed prompts with TaskGenerator."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_sft.task_generator.generator import TaskGenerator, TaskGeneratorConfig
from src.agent_sft.task_generator.models import Difficulty, Domain
from src.agent_sft.task_generator.seed_pool import SeedPromptPool


class MockLLMClient:
    """Mock LLM client for demo purposes."""

    async def achat(self, model: str, messages: list, temperature: float = 0.7) -> str:
        """Mock async chat completion."""
        return messages[0]["content"]  # Echo back for demo


async def main():
    """Demonstrate TaskGenerator with seed prompts."""
    print("=== TaskGenerator Demo ===\n")

    # Load seed pool
    pool_path = Path(__file__).parent.parent / "data" / "seed_prompts.json"
    pool = SeedPromptPool.load(str(pool_path))
    print(f"[1] Loaded {len(pool)} seed prompts")

    # Initialize TaskGenerator
    llm_client = MockLLMClient()
    config = TaskGeneratorConfig(
        max_concurrent_requests=3,
        temperature=0.7,
        enable_mutation=False,  # Disable for demo
        min_quality_score=0.7,
    )
    generator = TaskGenerator(seed_pool=pool, llm_client=llm_client, config=config)
    print(f"[2] Initialized TaskGenerator with config: {config}")

    # Generate tasks in seed_only mode
    print("\n[3] Generating 10 code_debug tasks (seed_only mode)...")
    tasks = await generator.generate_batch(
        batch_size=10,
        domains=[Domain.CODE_DEBUG],
        difficulty=Difficulty.EASY,
        mode="seed_only",
    )
    print(f"    Generated {len(tasks)} valid tasks")

    # Show sample task
    if tasks:
        sample = tasks[0]
        print(f"\n[4] Sample task:")
        print(f"    ID: {sample.id}")
        print(f"    Domain: {sample.domain.value}")
        print(f"    Difficulty: {sample.difficulty.value}")
        print(f"    Prompt: {sample.prompt[:100]}...")
        print(f"    Test cases: {len(sample.test_cases)}")
        print(f"    Validation passed: {sample.validation_passed}")
        print(f"    Tags: {sample.tags}")

    # Generate tasks from multiple domains
    print("\n[5] Generating 20 tasks from all domains...")
    mixed_tasks = await generator.generate_batch(
        batch_size=20,
        domains=None,  # All domains
        difficulty=None,  # All difficulties
        mode="seed_only",
    )
    print(f"    Generated {len(mixed_tasks)} valid tasks")

    # Show distribution
    domain_counts = {}
    difficulty_counts = {}
    for task in mixed_tasks:
        domain_counts[task.domain.value] = domain_counts.get(task.domain.value, 0) + 1
        difficulty_counts[task.difficulty.value] = difficulty_counts.get(task.difficulty.value, 0) + 1

    print(f"\n[6] Task distribution:")
    print(f"    By domain: {domain_counts}")
    print(f"    By difficulty: {difficulty_counts}")

    # Test with hard tasks only
    print("\n[7] Generating 5 hard tasks...")
    hard_tasks = await generator.generate_batch(
        batch_size=5,
        domains=None,
        difficulty=Difficulty.HARD,
        mode="seed_only",
    )
    print(f"    Generated {len(hard_tasks)} hard tasks")
    for task in hard_tasks:
        print(f"    - [{task.domain.value}] {task.prompt[:60]}...")

    print("\n[OK] Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
