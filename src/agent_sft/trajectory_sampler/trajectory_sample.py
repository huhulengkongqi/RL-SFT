"""Best-of-N concurrent trajectory sampling utilities."""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from infra.environment.environment import Environment
from infra.environment.sandbox_pool import SandboxPool

from .agent_loop import AgentLoop, LayeredTemperatureConfig, Trajectory, TrajectoryFormat


@dataclass
class SamplingConfig:
    n: int = 16
    max_steps: int = 10
    token_budget: int = 20000
    trajectory_format: str = TrajectoryFormat.REACT.value
    save_failed_steps: bool = True
    sandbox_pool_size: int = 1
    sandbox_pre_warm_count: int = 1
    sandbox_max_container_reuse: int = 50
    thought_temperature: float = 0.8
    action_temperature: float = 0.2
    thought_max_tokens: int = 2048
    action_max_tokens: int = 4096
    layered_temperature: bool = True


class TrajectorySampleResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: int
    trajectory: Optional[Trajectory] = None
    success: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


def default_env_factory(llm_client: Any, config: SamplingConfig) -> Environment:
    sandbox_pool = SandboxPool(
        pool_size=config.sandbox_pool_size,
        max_container_reuse=config.sandbox_max_container_reuse,
        pre_warm_count=config.sandbox_pre_warm_count,
    )
    return Environment(
        sandbox_pool=sandbox_pool,
        max_steps=config.max_steps,
        judge_client=llm_client,
        judge_model=getattr(llm_client, "model", None),
    )


async def sample_one_trajectory(
    task: Dict[str, Any],
    sample_id: int,
    llm_client: Any,
    config: SamplingConfig,
    env_factory: Optional[Callable[[], Any]] = None,
) -> TrajectorySampleResult:
    started = time.perf_counter()
    try:
        env = env_factory() if env_factory else default_env_factory(llm_client, config)
        temperature_config = LayeredTemperatureConfig(
            enabled=config.layered_temperature,
            thought_temperature=config.thought_temperature,
            action_temperature=config.action_temperature,
            thought_max_tokens=config.thought_max_tokens,
            action_max_tokens=config.action_max_tokens,
        )
        async with env:
            loop = AgentLoop(
                env=env,
                llm_client=llm_client,
                max_steps=config.max_steps,
                token_budget=config.token_budget,
                save_failed_steps=config.save_failed_steps,
                trajectory_format=config.trajectory_format,
                temperature_config=temperature_config,
            )
            trajectory = await loop.run(task)
        return TrajectorySampleResult(
            sample_id=sample_id,
            trajectory=trajectory,
            success=trajectory.success,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return TrajectorySampleResult(
            sample_id=sample_id,
            success=False,
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - started,
        )


async def sample_trajectories(
    task: Dict[str, Any],
    n: int = 16,
    llm_client: Any = None,
    config: Optional[SamplingConfig] = None,
    env_factory: Optional[Callable[[], Any]] = None,
) -> List[TrajectorySampleResult]:
    if llm_client is None:
        raise ValueError("llm_client is required")
    effective_config = config or SamplingConfig(n=n)
    effective_config.n = n
    coroutines = [
        sample_one_trajectory(task, sample_id, llm_client, effective_config, env_factory=env_factory)
        for sample_id in range(n)
    ]
    return await asyncio.gather(*coroutines)


def _trajectory_total_tokens(trajectory: Trajectory) -> int:
    total = trajectory.final_state.get("total_tokens")
    if isinstance(total, int):
        return total
    return sum((step.token_usage or {}).get("total_tokens", 0) for step in trajectory.steps)


def select_best_trajectory(results: List[TrajectorySampleResult]) -> Optional[Trajectory]:
    candidates = [result for result in results if result.trajectory is not None]
    if not candidates:
        return None

    def rank(result: TrajectorySampleResult) -> tuple:
        trajectory = result.trajectory
        assert trajectory is not None
        score = trajectory.final_score if trajectory.final_score is not None else 0.0
        return (
            0 if trajectory.success else 1,
            -score,
            len(trajectory.steps),
            _trajectory_total_tokens(trajectory),
            result.elapsed_seconds,
        )

    return min(candidates, key=rank).trajectory


def failure_reason(result: TrajectorySampleResult) -> str:
    if result.error:
        return "sample_exception"
    if not result.trajectory:
        return "missing_trajectory"
    if result.trajectory.success:
        return "success"
    return result.trajectory.termination_reason or "unknown"


def is_sandbox_failure(result: TrajectorySampleResult) -> bool:
    text_parts: List[str] = []
    if result.error:
        text_parts.append(result.error)
    if result.trajectory:
        text_parts.append(result.trajectory.termination_reason)
        text_parts.append(str(result.trajectory.termination_details))
        for step in result.trajectory.steps:
            text_parts.append(step.observation.error or "")
            text_parts.append(str(step.observation.metadata or {}))
    text = " ".join(text_parts).lower()
    return bool(re.search(r"sandbox|docker|container|timeout|permission|denied", text))


def summarize_sample_results(results: List[TrajectorySampleResult]) -> Dict[str, Any]:
    total = len(results)
    successful = sum(1 for result in results if result.success)
    trajectories = [result.trajectory for result in results if result.trajectory is not None]
    failure_reasons: Dict[str, int] = {}
    for result in results:
        reason = failure_reason(result)
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    total_steps = sum(len(trajectory.steps) for trajectory in trajectories)
    total_tokens = sum(_trajectory_total_tokens(trajectory) for trajectory in trajectories)
    sandbox_failures = sum(1 for result in results if is_sandbox_failure(result))
    return {
        "total_samples": total,
        "completed_samples": len(trajectories),
        "successful_samples": successful,
        "success_rate": successful / total if total else 0.0,
        "avg_steps": total_steps / len(trajectories) if trajectories else 0.0,
        "avg_tokens": total_tokens / len(trajectories) if trajectories else 0.0,
        "failure_reasons": failure_reasons,
        "sandbox_failures": sandbox_failures,
        "sandbox_failure_rate": sandbox_failures / total if total else 0.0,
    }
