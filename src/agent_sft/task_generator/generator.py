"""Task generator for SFT data synthesis with batch generation support."""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .models import Difficulty, Domain, SeedPrompt, Task
from .seed_pool import SeedPromptPool
from .validator import TaskValidator

logger = logging.getLogger(__name__)


@dataclass
class TaskGeneratorConfig:
    """Configuration for TaskGenerator."""

    max_concurrent_requests: int = 5
    temperature: float = 0.7
    enable_mutation: bool = True
    mutation_rate: float = 0.3
    min_quality_score: float = 0.8
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0


ValidatorType = Callable[[Dict[str, Any]], bool]


class TaskGenerator:
    """Generate SFT training tasks from seed prompts with LLM augmentation."""

    def __init__(
        self,
        seed_pool: SeedPromptPool,
        llm_client: Any,
        config: Optional[TaskGeneratorConfig] = None,
        validators: Optional[Dict[Domain, ValidatorType]] = None,
        task_validator: Optional[TaskValidator] = None,
    ) -> None:
        """
        Initialize the TaskGenerator.

        Args:
            seed_pool: SeedPromptPool instance for sampling base prompts
            llm_client: LLM client instance for generation (VLLMClient-compatible)
            config: TaskGeneratorConfig instance with generation parameters
            validators: Dictionary mapping domains to validation functions
            task_validator: Optional TaskValidator for comprehensive task validation
        """
        self.seed_pool = seed_pool
        self.llm_client = llm_client
        self.config = config or TaskGeneratorConfig()
        self.validators = validators or {}
        self.task_validator = task_validator
        self._retry_strategy = AsyncRetrying(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(min=self.config.retry_min_wait, max=self.config.retry_max_wait),
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    def register_validator(self, domain: Domain, validator: ValidatorType) -> None:
        """Register a validator function for a domain."""
        self.validators[domain] = validator

    async def generate_batch(
        self,
        batch_size: int,
        domains: Optional[List[Domain]] = None,
        difficulty: Optional[Difficulty] = None,
        mode: str = "seed_based",
        enable_validation: bool = True,
    ) -> List[Task]:
        """
        Generate a batch of training tasks.

        Args:
            batch_size: Number of tasks to generate
            domains: Optional list of domains to restrict generation to
            difficulty: Optional difficulty level filter
            mode: Generation mode - 'seed_only' (use as-is), 'seed_based' (mutate seeds),
                  or 'full_generation' (create from scratch)
            enable_validation: Whether to enable comprehensive task validation

        Returns:
            List of validated Task objects
        """
        if mode not in {"seed_only", "seed_based", "full_generation"}:
            raise ValueError(f"Invalid mode: {mode}. Use 'seed_only', 'seed_based', or 'full_generation'")

        logger.info(f"Generating batch of {batch_size} tasks in {mode} mode")

        sample_count = max(batch_size, int(batch_size * 1.2))
        seed_prompts = self._sample_seeds(sample_count, domains, difficulty)

        if not seed_prompts:
            logger.warning("No seed prompts available for sampling")
            return []

        tasks = await self._generate_tasks(seed_prompts[:batch_size], mode)

        # Apply comprehensive validation if enabled and validator is available
        if enable_validation and self.task_validator:
            logger.info("Applying comprehensive task validation...")
            validation_reports = await self.task_validator.validate_batch(tasks)

            for task, report in zip(tasks, validation_reports):
                task.validation_report = report
                task.validation_passed = (
                    report.function_signature_valid and
                    report.sandbox_execution_passed
                )
        else:
            # Use existing validation
            for task in tasks:
                task.validation_passed = await self._validate_task(task)

        valid_tasks = [task for task in tasks if task.validation_passed]

        logger.info(f"Generated {len(valid_tasks)} valid tasks out of {len(tasks)}")
        return valid_tasks

    def _sample_seeds(
        self,
        count: int,
        domains: Optional[List[Domain]] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> List[SeedPrompt]:
        """Sample seed prompts with optional filtering."""
        if domains is None and difficulty is None:
            return self.seed_pool.sample(count=count)

        all_seeds: List[SeedPrompt] = []
        if domains:
            for domain in domains:
                seeds = self.seed_pool.filter_by_domain(domain)
                all_seeds.extend(seeds)
        else:
            all_seeds = list(self.seed_pool)

        if difficulty:
            all_seeds = [s for s in all_seeds if s.difficulty == difficulty]

        if len(all_seeds) <= count:
            return all_seeds

        weights = [s.quality_score or 0.5 for s in all_seeds]
        import random

        return random.choices(all_seeds, weights=weights, k=count)

    async def _generate_tasks(self, seeds: List[SeedPrompt], mode: str) -> List[Task]:
        """Generate tasks from seeds based on mode."""
        coroutines = [self._generate_single_task(seed, mode) for seed in seeds]
        tasks = await asyncio.gather(*coroutines, return_exceptions=True)

        valid_tasks: List[Task] = []
        for task in tasks:
            if isinstance(task, Task):
                valid_tasks.append(task)
            elif isinstance(task, Exception):
                logger.error(f"Task generation failed: {task}")

        return valid_tasks

    async def _generate_single_task(self, seed: SeedPrompt, mode: str) -> Task:
        """Generate a single task with rate limiting and retries."""
        async with self._semaphore:
            async for attempt in self._retry_strategy:
                with attempt:
                    return await self._generate_one(seed, mode)
        raise RuntimeError("Failed to generate task after retries")

    async def _generate_one(self, seed: SeedPrompt, mode: str) -> Task:
        """Generate one task with optional mutation."""
        if mode == "seed_only":
            prompt_text = seed.prompt
            generation_trace = {"mode": "seed_only", "unchanged": True}
        elif mode == "seed_based" and self.config.enable_mutation:
            import random

            if random.random() < self.config.mutation_rate:
                prompt_text = await self._mutate_prompt(seed.prompt, seed.domain)
                generation_trace = {"mode": "seed_based", "mutated": True}
            else:
                prompt_text = seed.prompt
                generation_trace = {"mode": "seed_based", "mutated": False}
        else:
            prompt_text = seed.prompt
            generation_trace = {"mode": mode}

        task = Task(
            id=seed.id,
            domain=seed.domain,
            difficulty=seed.difficulty,
            prompt=prompt_text,
            test_cases=seed.test_cases,
            validator_code=seed.validator_code,
            source=seed.source,
            version=seed.version,
            quality_score=seed.quality_score,
            tags=seed.tags,
            parent_seed_id=seed.id,
            generation_mode=mode,
            generation_trace=generation_trace,
        )

        return task

    async def _mutate_prompt(self, prompt: str, domain: Domain) -> str:
        """Mutate a prompt using LLM to create variations."""
        mutation_prompt = f"""Rewrite the following task description to create a new but similar task.
        Keep the same domain ({domain.value}) and difficulty level. Make meaningful but not drastic changes.

        Original task:
        {prompt}

        Rewritten task:
        """

        if hasattr(self.llm_client, "achat"):
            return await self.llm_client.achat(
                model=getattr(self.llm_client, "model", "default"),
                messages=[{"role": "user", "content": mutation_prompt}],
                temperature=self.config.temperature,
            )

        return prompt

    async def _validate_task(self, task: Task) -> bool:
        """Validate a task using the domain-specific validator or executing validator code."""
        if task.domain in self.validators:
            return self.validators[task.domain](task.model_dump())

        try:
            namespace: Dict[str, Any] = {}
            exec(task.validator_code, namespace)
            return True
        except Exception as e:
            logger.warning(f"Validator code execution failed: {e}")
            return False
