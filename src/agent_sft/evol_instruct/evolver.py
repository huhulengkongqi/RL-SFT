"""EvolInstruct instruction evolution engine."""

import asyncio
import random
from typing import Any, Dict, List, Optional

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .models import EvolvedPrompt, EvolutionMetadata, EvolutionResult, EvolutionStrategy


class EvolInstructConfig:
    """Configuration for EvolInstruct evolution."""

    def __init__(
        self,
        max_concurrent_requests: int = 1,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        retry_attempts: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        deep_evolution_prob: float = 0.7,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.deep_evolution_prob = deep_evolution_prob


class EvolInstruct:
    """Evolves prompts using deep and broad evolution strategies."""

    def __init__(self, llm_client: Any, config: Optional[EvolInstructConfig] = None):
        """
        Initialize EvolInstruct evolution engine.

        Args:
            llm_client: LLM client with `achat()` method (VLLMClient, AnthropicClient, etc.)
            config: Evolution configuration
        """
        self.llm_client = llm_client
        self.config = config or EvolInstructConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._retry_strategy = AsyncRetrying(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
        )

        self._deep_strategies = [
            EvolutionStrategy.ADD_CONSTRAINTS,
            EvolutionStrategy.DEEPEN_REASONING,
            EvolutionStrategy.CONCRETIZE,
            EvolutionStrategy.COMPLEX_INPUT,
            EvolutionStrategy.ADD_COT,
        ]

        self._strategy_prompts: Dict[EvolutionStrategy, str] = {
            EvolutionStrategy.ADD_CONSTRAINTS: self._add_constraints_prompt,
            EvolutionStrategy.DEEPEN_REASONING: self._deepen_reasoning_prompt,
            EvolutionStrategy.CONCRETIZE: self._concretize_prompt,
            EvolutionStrategy.COMPLEX_INPUT: self._complex_input_prompt,
            EvolutionStrategy.ADD_COT: self._add_cot_prompt,
            EvolutionStrategy.BROAD_MUTATION: self._broad_mutation_prompt,
        }

    def _select_strategy(self) -> EvolutionStrategy:
        """Select evolution strategy: weighted towards deep evolution."""
        if random.random() < self.config.deep_evolution_prob:
            return random.choice(self._deep_strategies)
        return EvolutionStrategy.BROAD_MUTATION

    @staticmethod
    def _add_constraints_prompt(prompt: str) -> str:
        return f"""请为以下任务增加2-3个具体的约束条件，使任务变得更有挑战性且更具体。
约束条件应该是实质性的，比如：
- 性能要求（时间/空间复杂度）
- 禁止使用某些方法
- 边界条件处理
- 额外的输出格式要求

原始任务：
{prompt}

请直接输出增加了约束条件后的完整任务描述，不要输出其他解释。"""

    @staticmethod
    def _deepen_reasoning_prompt(prompt: str) -> str:
        return f"""请将以下任务转变为需要多步推理才能解决的任务。
增加中间推理步骤要求，使任务从"直接给出答案"变成"逐步推导"。
可以要求：
- 先分析问题
- 列出可能的解法
- 比较不同方法的优劣
- 最终给出答案

原始任务：
{prompt}

请直接输出需要多步推理的完整任务描述，不要输出其他解释。"""

    @staticmethod
    def _concretize_prompt(prompt: str) -> str:
        return f"""请将以下泛化的任务具体化到一个真实的应用场景中。
用具体的上下文、真实的数据格式、明确的业务目标来替换抽象描述。

原始任务：
{prompt}

请直接输出具体化后的完整任务描述，不要输出其他解释。"""

    @staticmethod
    def _complex_input_prompt(prompt: str) -> str:
        return f"""请修改以下任务，使其输入变得更加复杂和有挑战性。
可以增加：
- 多条件组合输入
- 带噪声的数据
- 嵌套的数据结构
- 多源数据融合要求

原始任务：
{prompt}

请直接输出包含复杂输入的完整任务描述，不要输出其他解释。"""

    @staticmethod
    def _add_cot_prompt(prompt: str) -> str:
        return f"""请修改以下任务，要求解答者必须输出完整的思考过程（Chain of Thought）。
要求：
1. 先分析问题要求
2. 列出思考步骤
3. 逐步推导
4. 最后给出结论

原始任务：
{prompt}

请直接输出要求Chain of Thought的完整任务描述，不要输出其他解释。"""

    @staticmethod
    def _broad_mutation_prompt(prompt: str) -> str:
        return f"""基于以下任务的核心概念，创造一个全新但相关的任务。
新任务应该：
- 保留原任务的领域和核心能力要求
- 改变具体的问题场景和目标
- 变成一个不同的、但难度相当的任务
- 不能只是原任务的简单重述

原始任务：
{prompt}

请直接输出全新的任务描述，不要输出其他解释。"""

    async def _evolve_single(
        self,
        seed_prompt: Dict[str, Any],
        generation: int,
        strategy: Optional[EvolutionStrategy] = None,
    ) -> EvolutionResult:
        """
        Evolve a single prompt using the specified strategy.

        Args:
            seed_prompt: Seed prompt dict with 'id', 'prompt', 'domain', 'difficulty', etc.
            generation: Current generation number
            strategy: Optional strategy (random if None)

        Returns:
            EvolutionResult with evolved prompt or error
        """
        strategy = strategy or self._select_strategy()
        evolution_prompt = self._strategy_prompts[strategy](seed_prompt["prompt"])

        try:
            async with self._semaphore:
                async for attempt in self._retry_strategy:
                    with attempt:
                        model_name = getattr(self.llm_client, "model", "default")
                        evolved_text = await self.llm_client.achat(
                            model=model_name,
                            messages=[{"role": "user", "content": evolution_prompt}],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )

                        evolved_text = evolved_text.strip()

                        if len(evolved_text) < 20:
                            raise ValueError("Evolved prompt too short")

                        # Skip if too similar (exact copy or trivial change)
                        original_text = seed_prompt["prompt"].strip()
                        if evolved_text == original_text:
                            raise ValueError("Evolved prompt is exact copy of original")

                        evolved = EvolvedPrompt(
                            prompt=evolved_text,
                            domain=seed_prompt["domain"],
                            difficulty=seed_prompt["difficulty"],
                            quality_score=seed_prompt.get("quality_score", 0.7),
                            evolution_metadata=EvolutionMetadata(
                                generation=generation,
                                parent_id=seed_prompt["id"],
                                strategy=strategy,
                                evolution_prompt=evolution_prompt,
                                original_quality_score=seed_prompt.get("quality_score", 0.7),
                            ),
                            test_cases=seed_prompt.get("test_cases", []),
                            tags=seed_prompt.get("tags", []),
                        )

                        return EvolutionResult(
                            original_id=seed_prompt["id"],
                            evolved_prompt=evolved,
                            success=True,
                            strategy=strategy,
                        )
        except Exception as e:
            return EvolutionResult(
                original_id=seed_prompt["id"],
                success=False,
                error=str(e),
                strategy=strategy,
            )

    async def evolve_batch(
        self,
        seed_prompts: List[Dict[str, Any]],
        generation: int,
        strategies: Optional[List[EvolutionStrategy]] = None,
    ) -> List[EvolutionResult]:
        """
        Evolve a batch of prompts.

        Args:
            seed_prompts: List of seed prompt dicts
            generation: Current generation number
            strategies: Optional list of strategies to cycle through (random if None)

        Returns:
            List of evolution results
        """
        tasks = []
        for i, seed in enumerate(seed_prompts):
            strategy = None
            if strategies:
                strategy = strategies[i % len(strategies)]
            tasks.append(self._evolve_single(seed, generation, strategy))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in results:
            if isinstance(result, EvolutionResult):
                processed.append(result)
            else:
                processed.append(
                    EvolutionResult(
                        original_id="unknown",
                        success=False,
                        error=f"Exception: {str(result)}",
                        strategy=EvolutionStrategy.BROAD_MUTATION,
                    )
                )

        return processed

    def get_strategy_stats(self, results: List[EvolutionResult]) -> Dict[EvolutionStrategy, int]:
        """Count successful evolutions by strategy."""
        stats: Dict[EvolutionStrategy, int] = {}
        for result in results:
            if result.success and result.evolved_prompt:
                stats[result.strategy] = stats.get(result.strategy, 0) + 1
        return stats
