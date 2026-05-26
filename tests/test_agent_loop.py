"""Tests for AgentLoop and trajectory sampling components.

Covers:
- AgentState immutability and deep copy
- All 5 termination conditions
- Trajectory recording and SFT format export
- Full AgentLoop integration with mock clients
"""

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytest

from agent_sft.trajectory_sampler import (
    AgentLoop,
    AgentState,
    TerminationDecision,
    TerminationDetector,
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStep,
)
from infra.environment.models import (
    Action,
    FinalAnswerAction,
    Observation,
    StepResult,
    ToolCallAction,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def base_state() -> AgentState:
    """Create base agent state for testing."""
    return AgentState(
        task_id="test-001",
        domain="code_debug",
        difficulty="medium",
        max_steps=10,
        token_budget=100000,
    )


@pytest.fixture
def sample_task() -> Dict[str, Any]:
    """Load a sample task from evolved dataset."""
    with open("data/claude_evolved_4gen/final_evolved_v1.0_complete.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Return the first task
    return data[0]


class MockLLMClient:
    """Mock LLM client for testing that returns predefined responses."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Return next response in the list."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


class MockEnvironment:
    """Mock Environment that doesn't require Docker."""

    def __init__(self, always_succeed: bool = True):
        self.always_succeed = always_succeed
        self.step_count = 0

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Reset environment for new task."""
        self.step_count = 0
        return {
            "task_id": task.get("id", "unknown"),
            "domain": task.get("domain", "unknown"),
            "prompt": task.get("prompt", ""),
            "max_steps": 10,
            "test_cases_count": len(task.get("test_cases", [])),
        }

    async def step(self, action: Action) -> StepResult:
        """Execute action and return observation."""
        self.step_count += 1

        if isinstance(action, FinalAnswerAction):
            # Final answer - terminate
            return StepResult(
                action=action,
                observation=Observation(
                    success=self.always_succeed,
                    content="Solution verified successfully.",
                    metadata={"test_cases_passed": 5, "test_cases_total": 5, "score": 1.0},
                ),
                done=True,
                step=self.step_count,
            )
        else:
            # Tool call - return mock output
            return StepResult(
                action=action,
                observation=Observation(
                    success=True,
                    content=f"Executed {action.name} successfully",
                    execution_time=0.1,
                ),
                done=False,
                step=self.step_count,
            )


# ============================================================================
# AgentState Tests
# ============================================================================


class TestAgentState:
    """Test AgentState immutability and update behavior."""

    def test_initial_state(self, base_state: AgentState):
        """Initial state should have correct default values."""
        assert base_state.step == 0
        assert base_state.done is False
        assert base_state.total_tokens == 0
        assert base_state.action_history == []
        assert base_state.observation_history == []

    def test_immutability_frozen(self, base_state: AgentState):
        """AgentState should be frozen - cannot modify directly."""
        with pytest.raises(Exception):  # Pydantic frozen validation
            base_state.step = 5  # type: ignore

    def test_update_returns_new_instance(self, base_state: AgentState):
        """update() should return NEW state instance, not modify original."""
        action = ToolCallAction(
            action_type="tool_call",
            name="exec",
            kwargs={"code": "print(1)"},
        )
        obs = Observation(success=True, content="1", execution_time=0.1)
        step_result = StepResult(action=action, observation=obs, done=False, step=1)

        new_state = base_state.update(action, obs, step_result)

        # Original should be UNCHANGED
        assert base_state.step == 0
        assert len(base_state.action_history) == 0

        # New should be updated
        assert new_state.step == 1
        assert len(new_state.action_history) == 1
        assert new_state is not base_state  # Different objects!

    def test_update_deep_copy_history(self, base_state: AgentState):
        """History lists should be deep copied, not shared references."""
        action1 = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs1 = Observation(success=True, content="1")
        step_result1 = StepResult(action=action1, observation=obs1, done=False, step=1)

        state1 = base_state.update(action1, obs1, step_result1)

        # Modify the original action (should NOT affect state1's history)
        action1.kwargs["code"] = "MODIFIED!"

        # History should have original value
        assert state1.action_history[0].kwargs["code"] == "print(1)"

    def test_snapshot_is_deep_copy(self, base_state: AgentState):
        """snapshot() should return completely independent copy."""
        action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=action, observation=obs, done=False, step=1)
        state = base_state.update(action, obs, step_result)

        snapshot = state.snapshot()

        # Modify original state's data (should NOT affect snapshot)
        state.action_history[0].kwargs["code"] = "MODIFIED!"

        # Snapshot should remain unchanged
        assert snapshot["action_history"][0]["kwargs"]["code"] == "print(1)"

    def test_action_hash_consistent(self, base_state: AgentState):
        """Same action content should produce same hash."""
        action1 = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        action2 = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})

        # Same content = same hash
        assert base_state.get_action_hash(action1) == base_state.get_action_hash(action2)

        # Different content = different hash
        action3 = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(2)"})
        assert base_state.get_action_hash(action1) != base_state.get_action_hash(action3)


# ============================================================================
# TerminationDetector Tests
# ============================================================================


class TestTerminationDetector:
    """Test all 5 termination conditions with correct priority."""

    def test_continue_when_no_termination(self, base_state: AgentState):
        """Should continue when none of the conditions are met."""
        detector = TerminationDetector()
        decision = detector.check(base_state)

        assert decision.should_terminate is False
        assert decision.reason == "continue"

    def test_max_steps_termination(self):
        """Should terminate when step >= max_steps."""
        state = AgentState(
            task_id="test-001",
            domain="code_debug",
            step=10,
            max_steps=10,
            token_budget=100000,
        )
        detector = TerminationDetector()
        decision = detector.check(state)

        assert decision.should_terminate is True
        assert decision.reason == "max_steps"
        assert "steps_taken" in decision.details

    def test_token_overflow_termination(self):
        """Should terminate when token budget exceeded."""
        state = AgentState(
            task_id="test-001",
            domain="code_debug",
            step=3,
            max_steps=10,
            total_tokens=200000,
            token_budget=100000,
        )
        detector = TerminationDetector()
        decision = detector.check(state)

        assert decision.should_terminate is True
        assert decision.reason == "token_overflow"

    def test_success_termination(self, base_state: AgentState):
        """Should terminate when FinalAnswerAction with passing verification."""
        action = FinalAnswerAction(action_type="final_answer", answer="def solution(): pass")
        obs = Observation(success=True, content="Verified", metadata={"score": 1.0})
        step_result = StepResult(action=action, observation=obs, done=True, step=3)

        state = base_state.update(action, obs, step_result)
        detector = TerminationDetector()
        decision = detector.check(state)

        assert decision.should_terminate is True
        assert decision.reason == "success"
        assert decision.score == 1.0

    def test_loop_detection_termination(self, base_state: AgentState):
        """Should terminate when same action repeats threshold times."""
        detector = TerminationDetector(loop_detection_threshold=3)
        same_action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=same_action, observation=obs, done=False, step=1)

        state = base_state
        for i in range(3):
            state = state.update(same_action, obs, step_result)
            decision = detector.check(state)

        # After 3rd repetition, should detect loop
        assert decision.should_terminate is True
        assert decision.reason == "loop_detected"
        assert "repeat_count" in decision.details

    def test_fatal_error_termination(self):
        """Should terminate immediately on fatal errors (highest priority)."""
        fatal_obs = Observation(
            success=False,
            content="",
            error="Docker container timeout - sandbox crashed completely",
        )
        state = AgentState(
            task_id="test-001",
            domain="code_debug",
            step=2,
            max_steps=10,
            token_budget=100000,
            last_observation=fatal_obs,
        )
        detector = TerminationDetector()
        decision = detector.check(state)

        assert decision.should_terminate is True
        assert decision.reason == "fatal_error"

    def test_fatal_error_higher_priority_than_success(self):
        """Fatal error should override even if answer would be correct."""
        # Even though step=3 < max_steps and we might have an answer...
        # Fatal error wins
        fatal_obs = Observation(success=False, content="", error="Permission denied: cannot write to /")
        state = AgentState(
            task_id="test-001",
            domain="code_debug",
            step=3,
            max_steps=10,
            total_tokens=1000,
            token_budget=100000,
            last_observation=fatal_obs,
        )
        detector = TerminationDetector()
        decision = detector.check(state)

        # Fatal error wins - it's highest priority
        assert decision.reason == "fatal_error"

    def test_reset_clears_loop_history(self, base_state: AgentState):
        """reset() should clear loop detection history for new episode."""
        detector = TerminationDetector(loop_detection_threshold=3)
        same_action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=same_action, observation=obs, done=False, step=1)

        # First episode - 2 repetitions (no termination)
        state = base_state
        for _ in range(2):
            state = state.update(same_action, obs, step_result)
            detector.check(state)

        # Reset for new episode
        detector.reset()

        # New episode - 2 repetitions should NOT trigger (history was cleared)
        state2 = base_state
        for _ in range(2):
            state2 = state2.update(same_action, obs, step_result)
            decision = detector.check(state2)

        assert decision.should_terminate is False


# ============================================================================
# TrajectoryRecorder Tests
# ============================================================================


class TestTrajectoryRecorder:
    """Test trajectory recording and SFT format export."""

    def test_record_single_step(self, base_state: AgentState):
        """Should correctly record a single step."""
        recorder = TrajectoryRecorder()

        action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=action, observation=obs, done=False, step=1)
        state = base_state.update(action, obs, step_result)

        recorder.record_step(state, action, obs)

        assert len(recorder.steps) == 1
        assert recorder.steps[0].step == 1

    def test_record_multiple_steps(self, base_state: AgentState):
        """Should correctly record multiple steps in order."""
        recorder = TrajectoryRecorder()
        state = base_state

        for i in range(5):
            action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": f"print({i})"})
            obs = Observation(success=True, content=str(i))
            step_result = StepResult(action=action, observation=obs, done=False, step=i + 1)
            state = state.update(action, obs, step_result)
            recorder.record_step(state, action, obs)

        assert len(recorder.steps) == 5
        assert recorder.steps[0].step == 1
        assert recorder.steps[4].step == 5

    def test_failed_steps_preserved_by_default(self, base_state: AgentState):
        """Failed steps should be preserved for 'what not to do' training."""
        recorder = TrajectoryRecorder(save_failed_steps=True)

        # Record a failure
        action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1 / 0)"})
        obs = Observation(success=False, content="", error="ZeroDivisionError")
        step_result = StepResult(action=action, observation=obs, done=False, step=1)
        state = base_state.update(action, obs, step_result)

        recorder.record_step(state, action, obs)

        assert len(recorder.steps) == 1  # Failure was preserved

    def test_finalize_creates_valid_trajectory(self, base_state: AgentState):
        """finalize() should return complete valid Trajectory object."""
        recorder = TrajectoryRecorder()

        # Record some steps
        action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=action, observation=obs, done=False, step=1)
        state = base_state.update(action, obs, step_result)
        recorder.record_step(state, action, obs)

        # Finalize
        trajectory = recorder.finalize(state, "success", {"steps": 1})

        assert isinstance(trajectory, Trajectory)
        assert trajectory.task_id == base_state.task_id
        assert trajectory.domain == base_state.domain
        assert trajectory.termination_reason == "success"
        assert trajectory.success is True
        assert len(trajectory.steps) == 1

    def test_sft_export_format(self, base_state: AgentState):
        """to_sft_format() should produce standard training data format."""
        recorder = TrajectoryRecorder()

        # Record one tool call and one final answer step
        action1 = ToolCallAction(
            action_type="tool_call",
            name="exec",
            kwargs={"code": "print(1)"},
            thought="Let me test this code.",
        )
        obs1 = Observation(success=True, content="1")
        step_result1 = StepResult(action=action1, observation=obs1, done=False, step=1)
        state = base_state.update(action1, obs1, step_result1)
        recorder.record_step(state, action1, obs1)

        action2 = FinalAnswerAction(
            action_type="final_answer",
            answer="def solution(): return 42",
            thought="I have the solution.",
        )
        obs2 = Observation(success=True, content="Verified", metadata={"score": 1.0})
        step_result2 = StepResult(action=action2, observation=obs2, done=True, step=2)
        state = state.update(action2, obs2, step_result2)
        recorder.record_step(state, action2, obs2)

        trajectory = recorder.finalize(state, "success", {})
        sft_data = trajectory.to_sft_format()

        # Check SFT format structure
        assert "messages" in sft_data
        assert sft_data["termination_reason"] == "success"
        assert sft_data["success"] is True

        # Messages should alternate between user/assistant
        messages = sft_data["messages"]
        assert len(messages) >= 4  # system + step pairs

    def test_trajectory_json_serializable(self, base_state: AgentState):
        """Trajectory should be fully JSON serializable for storage."""
        recorder = TrajectoryRecorder()

        action = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs = Observation(success=True, content="1")
        step_result = StepResult(action=action, observation=obs, done=False, step=1)
        state = base_state.update(action, obs, step_result)
        recorder.record_step(state, action, obs)

        trajectory = recorder.finalize(state, "success", {})

        # This should not raise any serialization errors
        json_str = json.dumps(trajectory.to_sft_format(), indent=2, default=str)
        assert "task_id" in json_str
        assert "messages" in json_str


# ============================================================================
# AgentLoop Integration Tests
# ============================================================================


class TestAgentLoopIntegration:
    """Full end-to-end integration tests for AgentLoop."""

    @pytest.mark.asyncio
    async def test_full_loop_success_case(self):
        """Test complete successful run with mock LLM and mock env."""
        # Mock LLM: tool call then final answer
        llm = MockLLMClient(
            [
                "Let me write some test code.\n\nTool: exec\nCode: print('testing')",
                "Now I have the solution.\n\nFinal Answer:\ndef solution(x):\n    return x * 2",
            ]
        )

        # Mock Environment
        env = MockEnvironment(always_succeed=True)

        # Create loop
        loop = AgentLoop(env, llm, max_steps=5)

        # Run with sample task
        task = {
            "id": "test-task-001",
            "domain": "code_debug",
            "difficulty": "easy",
            "prompt": "Write a function that doubles a number.",
            "test_cases": [{"input": (2,), "expected": 4}],
        }

        trajectory = await loop.run(task)

        # Verify trajectory
        assert isinstance(trajectory, Trajectory)
        assert trajectory.termination_reason == "success"
        assert trajectory.success is True
        assert len(trajectory.steps) == 2  # 1 tool call + 1 final answer
        assert llm.call_count == 2

        # Verify SFT export works
        sft_data = trajectory.to_sft_format()
        assert sft_data["success"] is True
        assert len(sft_data["messages"]) > 0

    @pytest.mark.asyncio
    async def test_full_loop_max_steps(self):
        """Test loop terminates correctly at max_steps limit."""
        # Mock LLM that NEVER produces final answer (loops forever)
        llm = MockLLMClient(
            [
                "Tool: exec\nCode: print(1)",
                "Tool: exec\nCode: print(2)",
                "Tool: exec\nCode: print(3)",
                "Tool: exec\nCode: print(4)",
                "Tool: exec\nCode: print(5)",
                "Tool: exec\nCode: print(6)",  # Should never reach this
            ]
        )

        env = MockEnvironment()
        loop = AgentLoop(env, llm, max_steps=3)  # Only 3 steps allowed

        task = {
            "id": "test-task-002",
            "domain": "code_debug",
            "difficulty": "easy",
            "prompt": "Do something forever.",
            "test_cases": [],
        }

        trajectory = await loop.run(task)

        # Should terminate due to max_steps, not success
        assert trajectory.termination_reason == "max_steps"
        assert trajectory.success is False
        assert len(trajectory.steps) == 3  # Exactly max_steps

    @pytest.mark.asyncio
    async def test_full_loop_detection(self):
        """Test loop detection catches repetitive actions."""
        # Mock LLM that produces EXACT same action every time
        llm = MockLLMClient(
            [
                "Tool: exec\nCode: print(1)",
                "Tool: exec\nCode: print(1)",  # Same
                "Tool: exec\nCode: print(1)",  # Same - should trigger loop detection
                "Tool: exec\nCode: print(1)",  # Should never reach
            ]
        )

        env = MockEnvironment()
        loop = AgentLoop(env, llm, max_steps=10, loop_detection_threshold=3)

        task = {
            "id": "test-task-003",
            "domain": "code_debug",
            "difficulty": "easy",
            "prompt": "This agent is stuck in a loop.",
            "test_cases": [],
        }

        trajectory = await loop.run(task)

        # Should terminate due to loop detection
        assert trajectory.termination_reason == "loop_detected"
        assert trajectory.success is False

    @pytest.mark.asyncio
    async def test_prompt_building_includes_history(self):
        """Test that conversation prompt includes full history."""
        from agent_sft.trajectory_sampler.agent_loop import _build_conversation_prompt

        state = AgentState(
            task_id="test",
            domain="code_debug",
            step=2,
            max_steps=10,
        )

        # Add some history
        action1 = ToolCallAction(action_type="tool_call", name="exec", kwargs={"code": "print(1)"})
        obs1 = Observation(success=True, content="1")
        step_result1 = StepResult(action=action1, observation=obs1, done=False, step=1)
        state = state.update(action1, obs1, step_result1)

        action2 = ToolCallAction(action_type="tool_call", name="eval", kwargs={"code": "1 + 1"})
        obs2 = Observation(success=True, content="2")
        step_result2 = StepResult(action=action2, observation=obs2, done=False, step=2)
        state = state.update(action2, obs2, step_result2)

        messages = _build_conversation_prompt(state, "Solve this problem.")

        # System + 2 steps = 5 messages minimum
        assert len(messages) >= 5
        assert messages[0]["role"] == "system"
        assert "code_debug" in messages[0]["content"]


# ============================================================================
# Action Parsing Tests
# ============================================================================


class TestActionParsing:
    """Test LLM response parsing into typed Actions."""

    def test_parse_final_answer(self):
        """Should parse Final Answer format correctly."""
        from agent_sft.trajectory_sampler.agent_loop import _parse_llm_response

        response = """Let me think about this...

Final Answer:
def solution(x, y):
    return x + y
"""
        action = _parse_llm_response(response)

        assert isinstance(action, FinalAnswerAction)
        assert action.action_type == "final_answer"
        assert "def solution(x, y)" in action.answer
        assert "Let me think" in action.thought

    def test_parse_tool_call_exec(self):
        """Should parse Tool call format correctly."""
        from agent_sft.trajectory_sampler.agent_loop import _parse_llm_response

        response = """I should test this code.

Tool: exec
Code: print("Hello World")
"""
        action = _parse_llm_response(response)

        assert isinstance(action, ToolCallAction)
        assert action.name == "exec"
        assert "Hello World" in action.kwargs["code"]
        assert "test this code" in action.thought

    def test_parse_tool_call_eval(self):
        """Should parse eval tool call."""
        from agent_sft.trajectory_sampler.agent_loop import _parse_llm_response

        response = "Tool: eval\nInput: 2 + 2"
        action = _parse_llm_response(response)

        assert isinstance(action, ToolCallAction)
        assert action.name == "eval"

    def test_parse_fallback_to_exec(self):
        """Unrecognized format should default to exec tool call."""
        from agent_sft.trajectory_sampler.agent_loop import _parse_llm_response

        response = "Just some raw code: def f(): pass"
        action = _parse_llm_response(response)

        assert isinstance(action, ToolCallAction)
        assert action.name == "exec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
