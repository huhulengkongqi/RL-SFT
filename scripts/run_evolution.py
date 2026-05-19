"""Run EvolInstruct pipeline to expand seed prompts."""

import argparse
import asyncio
import logging
import os
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_sft import EvolutionPipeline

# Add random sleep wrapper for rate limiting
original_achat = None


def wrap_achat_with_random_sleep(client, min_sleep: float = 10.0, max_sleep: float = 20.0):
    """Wrap client.achat with random sleep between requests."""
    original_achat = client.achat

    async def achat_with_sleep(model: str, messages: list, **kwargs):
        sleep_time = random.uniform(min_sleep, max_sleep)
        logging.info(f"⏱️  Sleeping {sleep_time:.1f}s before next request...")
        await asyncio.sleep(sleep_time)
        return await original_achat(model, messages, **kwargs)

    client.achat = achat_with_sleep
    return client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MockLLMClient:
    """Mock LLM client for testing - generates deterministic evolved prompts."""

    model = "mock-model"
    sleep_before_request = 0.05

    async def achat(self, model: str, messages: list, **kwargs) -> str:
        """Mock async chat completion."""
        await asyncio.sleep(self.sleep_before_request)

        # Extract original prompt from the evolution prompt
        original_prompt = messages[0]["content"].split("原始任务：\n")[-1].split("请直接输出")[0]

        # Generate a mock evolved prompt based on the strategy
        strategy_hint = ""
        if "增加约束条件" in messages[0]["content"]:
            strategy_hint = "【增加约束】"
        elif "多步推理" in messages[0]["content"]:
            strategy_hint = "【加深推理】"
        elif "具体化" in messages[0]["content"]:
            strategy_hint = "【具体化】"
        elif "复杂输入" in messages[0]["content"]:
            strategy_hint = "【复杂输入】"
        elif "Chain of Thought" in messages[0]["content"]:
            strategy_hint = "【CoT】"
        elif "全新但相关" in messages[0]["content"]:
            strategy_hint = "【广度变异】"

        return (
            f"{strategy_hint} "
            f"这是进化后的任务：{original_prompt.strip()[:50]}... "
            f"请注意，你需要按照以下额外要求来完成任务："
            f"1. 考虑所有边界情况，包括输入为空、超出范围等；"
            f"2. 输出需要包含详细的分析过程和最终的结论；"
            f"3. 你的解决方案需要考虑性能优化要求；"
            f"4. 代码需要符合Python的PEP 8规范。"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run EvolInstruct prompt evolution")
    parser.add_argument(
        "--seed-file",
        type=str,
        default="data/seed_prompts.json",
        help="Path to seed prompts JSON file",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=4,
        help="Number of evolution generations",
    )
    parser.add_argument(
        "--evolutions-per-seed",
        type=int,
        default=3,
        help="Number of evolved variants per seed per generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evolved",
        help="Output directory for evolved prompts",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock LLM client (for testing without real LLM)",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLLM server base URL",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="VLLM model name",
    )
    parser.add_argument(
        "--discriminator-min-score",
        type=float,
        default=0.5,
        help="Minimum quality score from discriminator to keep prompt (0=disable)",
    )
    parser.add_argument(
        "--disable-discriminator",
        action="store_true",
        help="Disable LLM quality discriminator for speed",
    )
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Volcano Claude API instead of vLLM (requires ANTHROPIC_AUTH_TOKEN env var)",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default="ark-code-latest",
        help="Claude model name (default: ark-code-latest)",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=12.0,
        help="Minimum sleep seconds between Claude requests (default: 12.0)",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=18.0,
        help="Maximum sleep seconds between Claude requests (default: 18.0)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    evolver_config = None

    if args.use_mock:
        print("Using Mock LLM client for testing...")
        llm_client = MockLLMClient()
    elif args.use_claude:
        from infra.vllm_client.client import VLLMClient
        from agent_sft.evol_instruct.evolver import EvolInstructConfig

        # Volcano Claude config from test_volcano_claude.py (OpenAI compatible)
        base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        api_key = "***REMOVED***"

        print(f"[Claude Mode] Using Volcano Claude API (OpenAI compatible)")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {args.claude_model}")
        print(f"   Rate limiting: random sleep {args.min_sleep:.1f}s - {args.max_sleep:.1f}s between requests")
        print(f"   Max concurrent: 1 (Claude rate-limit safe)")

        llm_client = VLLMClient(
            base_url=base_url,
            api_key=api_key,
            timeout=120,
            model=args.claude_model,
        )

        # Wrap with random sleep for rate limiting
        llm_client = wrap_achat_with_random_sleep(
            llm_client,
            min_sleep=args.min_sleep,
            max_sleep=args.max_sleep,
        )

        # Low concurrency for Claude API (rate limiting)
        evolver_config = EvolInstructConfig(
            max_concurrent_requests=1,  # Serial execution for Claude
            temperature=0.8,
            max_tokens=1024,
        )
    else:
        # Try to use vLLM client
        try:
            from infra.vllm_client.client import VLLMClient
            from agent_sft.evol_instruct.evolver import EvolInstructConfig

            print(f"Using VLLM client: {args.vllm_base_url}, model: {args.vllm_model}")
            llm_client = VLLMClient(base_url=args.vllm_base_url, model=args.vllm_model)
            # High-concurrency config for vLLM throughput
            evolver_config = EvolInstructConfig(
                max_concurrent_requests=5,
                temperature=0.8,
                max_tokens=1024,
            )
            print(f"Configured max concurrent requests: {evolver_config.max_concurrent_requests}")
        except Exception as e:
            print(f"Failed to initialize VLLM client: {e}")
            print("Falling back to mock client. Run with --use-mock to suppress this warning.")
            llm_client = MockLLMClient()

    # Determine discriminator score
    discriminator_score = 0 if args.disable_discriminator else args.discriminator_min_score

    print(f"LLM Quality Discriminator: {'DISABLED' if discriminator_score == 0 else f'ENABLED (min_score={discriminator_score})'}")

    # Run evolution pipeline
    await EvolutionPipeline(
        llm_client,
        evolver_config=evolver_config,
        discriminator_min_score=discriminator_score,
    ).run(
        seed_file=args.seed_file,
        num_generations=args.generations,
        evolutions_per_seed=args.evolutions_per_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
