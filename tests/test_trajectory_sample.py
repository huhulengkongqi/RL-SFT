"""Tests for Best-of-N trajectory sampling."""

import asyncio
import json
import time
from typing import Any, Dict, List, Tuple

import pytest

from agent_sft.trajectory_sampler import (
    AgentLoop,
    LayeredTemperatureConfig,
    SamplingConfig,
    TrajectoryRecorder,
    sample_trajectories,
    select_best_trajectory,
)
from agent_sft.trajectory_sampler.agent_loop import _parse_llm_response
from infra.environment.models import Action, FinalAnswerAction, Observation, StepResult, ToolCallAction


class RecordingLLM:
    model = "mock"

    def __init__(self, responses: List[str], delay: float = 0.0):
        self.responses = responses
        self.delay = delay
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []

    async def achat(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Tuple[str, Dict[str, int]]:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append(kwargs)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response, {"total_tokens": 10}


class MockEnv:
    instances: List["MockEnv"] = []
    fail_sample_id: int | None = None

    def __init__(self, sample_id: int):
        self.sample_id = sample_id
        self.step_count = 0
        MockEnv.instances.append(self)

    async def __aenter__(self) -> "MockEnv":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if self.sample_id == MockEnv.fail_sample_id:
            raise RuntimeError("forced env failure")
        self.step_count = 0
        return {"task_id": task.get("id"), "domain": task.get("domain"), "prompt": task.get("prompt"), "test_cases_count": 1}

    async def step(self, action: Action) -> StepResult:
        self.step_count += 1
        return StepResult(
            action=action,
            observation=Observation(
                success=True,
                content="ok",
                metadata={"score": 1.0, "test_cases_passed": 1, "test_cases_total": 1},
            ),
            done=True,
            step=self.step_count,
        )


def task() -> Dict[str, Any]:
    return {"id": "task-1", "domain": "code_debug", "difficulty": "easy", "prompt": "Fix it", "test_cases": [{}]}


def test_parse_json_tool_call() -> None:
    action = _parse_llm_response('{"thought":"test","action":"tool_call","tool":"exec","arguments":{"code":"print(1)"}}', "function_json")
    assert isinstance(action, ToolCallAction)
    assert action.name == "exec"
    assert action.kwargs["code"] == "print(1)"
    assert action.thought == "test"


def test_parse_json_final_answer() -> None:
    action = _parse_llm_response('{"thought":"done","action":"final_answer","answer":"42"}', "function_json")
    assert isinstance(action, FinalAnswerAction)
    assert action.answer == "42"
    assert action.thought == "done"


def test_parse_react_action_input() -> None:
    action = _parse_llm_response("Thought: run it\nAction: eval\nAction Input: 2 + 2")
    assert isinstance(action, ToolCallAction)
    assert action.name == "eval"
    assert action.kwargs["expr"] == "2 + 2"
    assert action.thought == "run it"


def test_sft_format_selection() -> None:
    recorder = TrajectoryRecorder()
    from agent_sft.trajectory_sampler import AgentState

    state = AgentState(task_id="task-1", domain="code_debug", difficulty="easy", metadata={"task_prompt": "Fix it"})
    action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"}, thought="check")
    observation = Observation(success=True, content="1")
    recorder.record_step(state, action, observation, {"total_tokens": 1}, thought="explicit thought")
    trajectory = recorder.finalize(state, "max_steps", {})

    react = trajectory.to_sft_format("react")
    assert react["messages"][2]["content"] == "Thought: explicit thought"
    assert "Action: exec" in react["messages"][3]["content"]
    function_json = trajectory.to_sft_format("function_json")
    thought = json.loads(function_json["messages"][2]["content"])
    parsed = json.loads(function_json["messages"][3]["content"])
    assert thought["type"] == "thought"
    assert thought["content"] == "explicit thought"
    assert parsed["type"] == "action"
    assert parsed["action"] == "tool_call"
    assert parsed["tool"] == "exec"


@pytest.mark.asyncio
async def test_layered_temperature_routes_high_then_low() -> None:
    llm = RecordingLLM(["reason", "Thought: done\nFinal Answer:\n42"])
    env = MockEnv(0)
    loop = AgentLoop(
        env,
        llm,
        max_steps=1,
        temperature_config=LayeredTemperatureConfig(
            enabled=True,
            thought_temperature=0.9,
            action_temperature=0.2,
        ),
    )
    trajectory = await loop.run({**task(), "domain": "multi_step_planning"})
    assert trajectory.success is True
    assert llm.call_count == 2
    assert llm.calls[0]["temperature"] == 0.9
    assert llm.calls[1]["temperature"] == 0.2
    assert trajectory.steps[0].thought == "reason"
    assert trajectory.to_sft_format("react")["messages"][2]["content"] == "Thought: reason"
    assert trajectory.to_sft_format("react")["messages"][3]["content"].startswith("Final Answer:")


@pytest.mark.asyncio
async def test_sample_trajectories_runs_concurrently_with_unique_envs() -> None:
    MockEnv.instances = []
    MockEnv.fail_sample_id = None
    llm = RecordingLLM(["Thought: done\nFinal Answer:\n42"], delay=0.1)
    next_id = 0

    def env_factory() -> MockEnv:
        nonlocal next_id
        env = MockEnv(next_id)
        next_id += 1
        return env

    started = time.perf_counter()
    results = await sample_trajectories(
        task(),
        n=4,
        llm_client=llm,
        config=SamplingConfig(n=4, max_steps=1, layered_temperature=False),
        env_factory=env_factory,
    )
    elapsed = time.perf_counter() - started
    assert len(results) == 4
    assert len(MockEnv.instances) == 4
    assert all(result.trajectory is not None for result in results)
    assert elapsed < 0.35


@pytest.mark.asyncio
async def test_sample_failure_does_not_cancel_siblings() -> None:
    MockEnv.instances = []
    MockEnv.fail_sample_id = 2
    llm = RecordingLLM(["Thought: done\nFinal Answer:\n42"])
    next_id = 0

    def env_factory() -> MockEnv:
        nonlocal next_id
        env = MockEnv(next_id)
        next_id += 1
        return env

    results = await sample_trajectories(
        task(),
        n=4,
        llm_client=llm,
        config=SamplingConfig(n=4, max_steps=1, layered_temperature=False),
        env_factory=env_factory,
    )
    assert sum(1 for result in results if result.error) == 1
    assert sum(1 for result in results if result.trajectory is not None) == 3


def test_select_best_trajectory_prefers_success_and_score() -> None:
    from agent_sft.trajectory_sampler.trajectory_sample import TrajectorySampleResult

    recorder = TrajectoryRecorder()
    from agent_sft.trajectory_sampler import AgentState

    state = AgentState(task_id="task-1", domain="code_debug", difficulty="easy")
    high = recorder.finalize(state, "success", {})
    high.success = True
    high.final_score = 0.9
    low = recorder.finalize(state, "success", {})
    low.success = True
    low.final_score = 0.5
    best = select_best_trajectory([
        TrajectorySampleResult(sample_id=0, trajectory=low, success=True, elapsed_seconds=1),
        TrajectorySampleResult(sample_id=1, trajectory=high, success=True, elapsed_seconds=2),
    ])
    assert best is high
