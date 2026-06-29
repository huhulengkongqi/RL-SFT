import asyncio
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .. import quality_filter as qf_module
from ..dataset_builder import DatasetBuilder
from ..evol_instruct.evolver import EvolInstruct
from ..trajectory_sampler.trajectory_sample import sample_one_trajectory
from .checkpoint import CheckpointManager
from .queue import BaseQueue
from .worker_base import BaseWorker

import logging

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Get API key from environment."""
    return os.environ.get("ANTHROPIC_AUTH_TOKEN", "")


class SeedWorker(BaseWorker):
    """Worker that validates and enriches seed prompts."""

    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        task = {
            "task_id": item.get("task_id"),
            "id": item.get("id", item.get("task_id")),
            "domain": item.get("domain", "code_debug"),
            "difficulty": item.get("difficulty", "medium"),
            "prompt": item.get("prompt", ""),
            "test_cases": item.get("test_cases", []),
            "validator_code": item.get("validator_code"),
            "source": item.get("source", "seed_pool"),
            "quality_score": item.get("quality_score", 0.8),
            "tags": item.get("tags", []),
            "generation": 0,
            "evolution_history": [],
        }
        return task


class EvolutionWorker(BaseWorker):
    """Worker that evolves tasks using LLM."""

    def __init__(
        self,
        queue: BaseQueue,
        checkpoint: CheckpointManager,
        stage: str,
        input_stream: str,
        output_stream: Optional[str] = None,
        worker_id: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(queue, checkpoint, stage, input_stream, output_stream, worker_id)
        self.config = config or type('obj', (object,), {})()
        self._evolver = None

    def _get_evolver(self):
        if self._evolver is None:
            from infra.vllm_client.client import VLLMClient
            from agent_sft.evol_instruct import EvolInstructConfig

            base_url = os.environ.get("VOLCANO_CLAUDE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
            model = os.environ.get("VLLM_MODEL", "ark-code-latest")
            client = VLLMClient(base_url=base_url, api_key=_get_api_key())
            client.model = model  # 设置供 EvolInstruct 调用使用
            config = EvolInstructConfig(
                max_concurrent_requests=1,
            )
            self._evolver = EvolInstruct(llm_client=client, config=config)
        return self._evolver

    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        min_sleep = getattr(self.config, 'min_sleep', 10.0)
        max_sleep = getattr(self.config, 'max_sleep', 18.0)
        generations = getattr(self.config, 'generations', 1)

        await asyncio.sleep(random.uniform(min_sleep, max_sleep))

        evolver = self._get_evolver()
        current_prompt = item.get("prompt", "")

        evolution_history = item.get("evolution_history", [])

        for gen in range(generations):
            try:
                seed_prompt = {
                    "id": item.get("task_id", item.get("id")),
                    "prompt": current_prompt,
                    "domain": item.get("domain", "general"),
                    "difficulty": item.get("difficulty", "medium"),
                }
                result = await evolver._evolve_single(
                    seed_prompt=seed_prompt,
                    generation=gen + 1,
                )
                if result.success and result.evolved_prompt:
                    current_prompt = result.evolved_prompt.prompt
                    evolution_history.append({
                        "generation": gen + 1,
                        "strategy": str(result.strategy),
                        "original": seed_prompt["prompt"],
                    })
            except Exception as e:
                logger.warning(f"Evolution failed for task {item.get('task_id')}: {e}")
                if gen == 0:
                    raise

        result = dict(item)
        result["prompt"] = current_prompt
        result["evolution_history"] = evolution_history
        result["generation"] = len(evolution_history)
        result["evolved"] = True

        return result


class TrajectoryWorker(BaseWorker):
    """Worker that generates trajectories."""

    def __init__(
        self,
        queue: BaseQueue,
        checkpoint: CheckpointManager,
        stage: str,
        input_stream: str,
        output_stream: Optional[str] = None,
        worker_id: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(queue, checkpoint, stage, input_stream, output_stream, worker_id)
        self.config = config or type('obj', (object,), {})()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from infra.vllm_client.client import VLLMClient

            base_url = os.environ.get("VOLCANO_CLAUDE_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
            model = os.environ.get("VLLM_MODEL", "ark-code-latest")
            self._client = VLLMClient(base_url=base_url, api_key=_get_api_key())
            self._client.model = model
        return self._client

    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        sleep_min = getattr(self.config, 'sleep_min', 10.0)
        sleep_max = getattr(self.config, 'sleep_max', 18.0)
        max_steps = getattr(self.config, 'max_steps', 20)
        sandbox_pool_size = getattr(self.config, 'sandbox_pool_size', 4)

        await asyncio.sleep(random.uniform(sleep_min, sleep_max))

        client = self._get_client()

        task = {
            "id": item.get("task_id", item.get("id")),
            "domain": item.get("domain", "code_debug"),
            "prompt": item.get("prompt", ""),
            "test_cases": item.get("test_cases", []),
            "reference_solution": item.get("reference_solution"),
            "validator_code": item.get("validator_code"),
        }

        from agent_sft.trajectory_sampler.trajectory_sample import SamplingConfig

        config = SamplingConfig(
            n=1,
            max_steps=max_steps,
            sandbox_pool_size=sandbox_pool_size,
        )

        result = await sample_one_trajectory(
            task=task,
            sample_id=0,
            llm_client=client,
            config=config,
        )

        trajectory = result.trajectory

        # Handle None trajectory (generation failure)
        if trajectory is None:
            error_msg = result.error or "Trajectory generation returned None"
            raise RuntimeError(f"Trajectory generation failed: {error_msg}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            raw_path = f.name
            # Use Pydantic's JSON mode to handle enum serialization
            json.dump(trajectory.model_dump(mode='json'), f, ensure_ascii=False)

        formatted = trajectory.to_sft_format()

        # Get token count from final state if available
        final_state = trajectory.final_state or {}
        total_tokens = final_state.get('total_tokens', 0)
        if isinstance(total_tokens, dict):
            total_tokens = total_tokens.get('total', 0)

        return {
            "task_id": item.get("task_id"),
            "raw_path": raw_path,
            "raw_data": trajectory.model_dump(mode='json'),
            "sft": formatted.model_dump(mode='json') if hasattr(formatted, 'model_dump') else dict(formatted),
            "success": bool(trajectory.success),
            "score": trajectory.final_score or 0.0,
            "step_count": len(trajectory.steps),
            "total_tokens": int(total_tokens),
            "elapsed_time": result.elapsed_seconds,
            "termination_reason": trajectory.termination_reason,
            "domain": item.get("domain"),
        }


class QualityFilterWorker(BaseWorker):
    """Worker that filters trajectories (Level 1 and Level 2)."""

    def __init__(
        self,
        queue: BaseQueue,
        checkpoint: CheckpointManager,
        stage: str,
        input_stream: str,
        output_stream: Optional[str] = None,
        worker_id: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(queue, checkpoint, stage, input_stream, output_stream, worker_id)
        self.config = config or type('obj', (object,), {})()

    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        enable_level2 = getattr(self.config, 'level2_prm', True)

        # Try to get raw_data from different sources
        raw_data = item.get("raw_data", {})
        raw_path = item.get("raw_path")

        if not raw_data and raw_path and Path(raw_path).exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

        result = dict(item)
        result["quality_checks"] = {}

        success = item.get("success", False)
        step_count = item.get("step_count", 0)
        score = item.get("score", 0.0)

        level1_passed = success and step_count > 0
        result["quality_checks"]["level1"] = {
            "passed": level1_passed,
            "success": success,
            "step_count": step_count,
        }

        if not level1_passed:
            result["quality_passed"] = False
            result["quality_score"] = 0.0
            return result

        level2_score = self._compute_prm_score(raw_data, item) if enable_level2 else score
        level2_passed = level2_score >= 0.5

        result["quality_checks"]["level2"] = {
            "passed": level2_passed,
            "score": level2_score,
        }

        result["quality_passed"] = level1_passed and level2_passed
        result["quality_score"] = level2_score

        return result

    def _compute_prm_score(self, raw_data: Dict[str, Any], item: Dict[str, Any]) -> float:
        """Compute process reward model score."""
        base_score = item.get("score", 0.0)
        step_count = item.get("step_count", 0)

        if step_count == 0:
            return 0.0

        penalty = min(1.0, 20.0 / step_count)

        diversity_bonus = 0.0
        steps = raw_data.get("steps", [])
        if steps:
            # Convert action dict to hashable type
            action_types = []
            for s in steps:
                action = s.get("action")
                if isinstance(action, dict):
                    # Use action type or JSON string for uniqueness
                    action_str = action.get("type", action.get("name", str(action)))
                    action_types.append(action_str)
                else:
                    action_types.append(str(action))
            unique_actions = len(set(action_types))
            diversity_bonus = min(0.2, unique_actions / step_count * 0.2)

        final_score = min(1.0, base_score * penalty + diversity_bonus)
        return final_score


class DatasetBuilderWorker(BaseWorker):
    """Worker that builds final dataset."""

    def __init__(
        self,
        queue: BaseQueue,
        checkpoint: CheckpointManager,
        stage: str,
        input_stream: str,
        output_stream: Optional[str] = None,
        worker_id: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(queue, checkpoint, stage, input_stream, output_stream, worker_id)
        self.config = config or type('obj', (object,), {})()
        self._collected: list = []

    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("quality_passed", False):
            self._collected.append(item)

        return {
            "task_id": item.get("task_id"),
            "collected_count": len(self._collected),
            "quality_score": item.get("quality_score", 0.0),
        }


def create_seed_worker(
    queue: BaseQueue,
    checkpoint: CheckpointManager,
    config: Any,
    worker_id: str,
    input_stream: str,
    output_stream: str,
) -> SeedWorker:
    return SeedWorker(
        queue=queue,
        checkpoint=checkpoint,
        stage="seed_generation",
        input_stream=input_stream,
        output_stream=output_stream,
        worker_id=worker_id,
    )


def create_evolution_worker(
    queue: BaseQueue,
    checkpoint: CheckpointManager,
    config: Any,
    worker_id: str,
    input_stream: str,
    output_stream: str,
) -> EvolutionWorker:
    return EvolutionWorker(
        queue=queue,
        checkpoint=checkpoint,
        stage="evolution",
        input_stream=input_stream,
        output_stream=output_stream,
        worker_id=worker_id,
        config=config,
    )


def create_trajectory_worker(
    queue: BaseQueue,
    checkpoint: CheckpointManager,
    config: Any,
    worker_id: str,
    input_stream: str,
    output_stream: str,
) -> TrajectoryWorker:
    return TrajectoryWorker(
        queue=queue,
        checkpoint=checkpoint,
        stage="trajectory_generation",
        input_stream=input_stream,
        output_stream=output_stream,
        worker_id=worker_id,
        config=config,
    )


def create_quality_filter_worker(
    queue: BaseQueue,
    checkpoint: CheckpointManager,
    config: Any,
    worker_id: str,
    input_stream: str,
    output_stream: str,
) -> QualityFilterWorker:
    return QualityFilterWorker(
        queue=queue,
        checkpoint=checkpoint,
        stage="quality_filter",
        input_stream=input_stream,
        output_stream=output_stream,
        worker_id=worker_id,
        config=config,
    )


def create_dataset_builder(
    queue: BaseQueue,
    checkpoint: CheckpointManager,
    config: Any,
    worker_id: str,
    input_stream: str,
    output_stream: Optional[str] = None,
) -> DatasetBuilderWorker:
    return DatasetBuilderWorker(
        queue=queue,
        checkpoint=checkpoint,
        stage="dataset_build",
        input_stream=input_stream,
        output_stream=output_stream,
        worker_id=worker_id,
        config=config,
    )
