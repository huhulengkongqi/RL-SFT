"""Tests for SFT data formatter module."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_sft.dataset_builder.data_formatter import (
    ChatMessage,
    DataFormatter,
    DatasetExporter,
    ExportConfig,
    FormatConfig,
    LossMaskBuilder,
    LossMaskConfig,
    SFTDataPipeline,
    SFTFormatterConfig,
    TokenCounter,
    TrajectoryTruncator,
    TruncationConfig,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


def make_raw_trajectory(
    task_id="test_task_001",
    domain="code_debug",
    difficulty="medium",
    num_steps=3,
    success=True,
    final_score=1.0,
):
    """Create a raw trajectory dict for testing."""
    steps = []
    for i in range(num_steps):
        is_final = i == num_steps - 1
        step = {
            "state_snapshot": {
                "metadata": {"task_prompt": "Fix the bug in this Python code"} if i == 0 else {},
            },
            "thought": f"Let me think about step {i}. I need to analyze this carefully.",
            "action": {
                "action_type": "final_answer" if is_final else "tool_call",
                "thought": f"Let me think about step {i}.",
                "name": "python" if not is_final else None,
                "kwargs": {"code": f"print('step {i}')"} if not is_final else {},
                "answer": "The fix is to add a null check" if is_final else None,
            },
            "observation": {
                "success": True,
                "content": f"Executed successfully, output: {i}" if not is_final else "",
                "error": None,
            },
        }
        steps.append(step)

    return {
        "task_id": task_id,
        "domain": domain,
        "difficulty": difficulty,
        "steps": steps,
        "final_state": {"done": True},
        "termination_reason": "success",
        "termination_details": {},
        "final_score": final_score,
        "total_time": 10.5,
        "success": success,
    }


# -----------------------------------------------------------------------------
# TokenCounter Tests
# -----------------------------------------------------------------------------


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_init(self):
        """Test token counter initialization."""
        counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
        assert counter.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert counter.tokenizer is not None

    def test_count_text(self):
        """Test counting tokens in plain text."""
        counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_count_empty_text(self):
        """Test counting empty text."""
        counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
        count = counter.count_text("")
        assert count == 0

    def test_count_messages(self):
        """Test counting tokens in chat messages."""
        counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
        messages: list[ChatMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0
        assert isinstance(count, int)

    def test_count_messages_with_tool(self):
        """Test counting messages with tool role."""
        counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
        messages: list[ChatMessage] = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "Let me use the tool."},
            {"role": "tool", "name": "python", "content": "Execution result"},
        ]
        count = counter.count_messages(messages)
        assert count > 0


# -----------------------------------------------------------------------------
# DataFormatter Tests
# -----------------------------------------------------------------------------


class TestDataFormatter:
    """Tests for DataFormatter class."""

    def test_init_default(self):
        """Test default initialization."""
        formatter = DataFormatter()
        assert formatter.config.trajectory_format == "react"
        assert formatter.config.merge_thought_action is True
        assert formatter.config.use_tool_role is True

    def test_format_basic(self):
        """Test basic trajectory formatting."""
        formatter = DataFormatter()
        raw = make_raw_trajectory(num_steps=2)
        result = formatter.format(raw)

        assert result["task_id"] == "test_task_001"
        assert result["domain"] == "code_debug"
        assert result["difficulty"] == "medium"
        assert result["success"] is True
        assert result["final_score"] == 1.0
        assert len(result["messages"]) > 0

    def test_format_has_four_roles(self):
        """Test output contains 4 roles (system, user, assistant, tool)."""
        formatter = DataFormatter()
        raw = make_raw_trajectory(num_steps=3)
        result = formatter.format(raw)

        roles = {m["role"] for m in result["messages"]}
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_format_tool_role_has_name(self):
        """Test tool messages have name field."""
        formatter = DataFormatter()
        raw = make_raw_trajectory(num_steps=2)
        result = formatter.format(raw)

        tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
        for msg in tool_messages:
            assert "name" in msg
            assert msg["name"] is not None

    def test_format_function_json(self):
        """Test formatting with function_json format."""
        config = FormatConfig(trajectory_format="function_json")
        formatter = DataFormatter(config)
        raw = make_raw_trajectory(num_steps=2)
        result = formatter.format(raw)

        assistant_messages = [m for m in result["messages"] if m["role"] == "assistant"]
        for msg in assistant_messages:
            content = msg["content"]
            if content and content.strip().startswith("{"):
                # Should be valid JSON
                parsed = json.loads(content.split("\n")[0])
                assert "type" in parsed

    def test_format_without_tool_role(self):
        """Test formatting without tool role (uses user for observations)."""
        config = FormatConfig(use_tool_role=False)
        formatter = DataFormatter(config)
        raw = make_raw_trajectory(num_steps=2)
        result = formatter.format(raw)

        roles = {m["role"] for m in result["messages"]}
        assert "tool" not in roles


# -----------------------------------------------------------------------------
# TrajectoryTruncator Tests
# -----------------------------------------------------------------------------


class TestTrajectoryTruncator:
    """Tests for TrajectoryTruncator class."""

    def test_init_default(self):
        """Test default initialization."""
        truncator = TrajectoryTruncator()
        assert truncator.config.strategy == "middle"
        assert truncator.config.max_tokens > 0

    def test_no_truncation_needed(self):
        """Test short trajectories are not truncated."""
        formatter = DataFormatter()
        truncator = TrajectoryTruncator(TruncationConfig(max_tokens=10000))
        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)

        result = truncator.truncate(formatted)

        # Should have same number of messages
        assert len(result["messages"]) == len(formatted["messages"])

    def test_truncation_strategy_middle(self):
        """Test middle truncation strategy."""
        formatter = DataFormatter()
        config = TruncationConfig(strategy="middle", max_tokens=200, auto_scale=False)
        truncator = TrajectoryTruncator(config)

        # Create long trajectory
        raw = make_raw_trajectory(num_steps=10)
        formatted = formatter.format(raw)

        result = truncator.truncate(formatted)
        assert result["token_count"] <= config.max_tokens

    def test_truncation_strategy_head(self):
        """Test head truncation strategy."""
        formatter = DataFormatter()
        config = TruncationConfig(strategy="head", max_tokens=200, auto_scale=False)
        truncator = TrajectoryTruncator(config)

        raw = make_raw_trajectory(num_steps=10)
        formatted = formatter.format(raw)

        result = truncator.truncate(formatted)
        assert result["token_count"] <= config.max_tokens

    def test_truncation_strategy_tail(self):
        """Test tail truncation strategy."""
        formatter = DataFormatter()
        config = TruncationConfig(strategy="tail", max_tokens=200, auto_scale=False)
        truncator = TrajectoryTruncator(config)

        raw = make_raw_trajectory(num_steps=10)
        formatted = formatter.format(raw)

        result = truncator.truncate(formatted)
        assert result["token_count"] <= config.max_tokens

    def test_preserves_system_and_task_prompt(self):
        """Test truncation preserves system prompt and initial task."""
        formatter = DataFormatter()
        config = TruncationConfig(max_tokens=200, auto_scale=False)
        truncator = TrajectoryTruncator(config)

        raw = make_raw_trajectory(num_steps=10)
        formatted = formatter.format(raw)

        result = truncator.truncate(formatted)

        # System prompt should be first
        assert result["messages"][0]["role"] == "system"
        # First user message should be the task prompt
        user_messages = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert "Task:" in user_messages[0]["content"]


# -----------------------------------------------------------------------------
# LossMaskBuilder Tests
# -----------------------------------------------------------------------------


class TestLossMaskBuilder:
    """Tests for LossMaskBuilder class."""

    def test_init_default(self):
        """Test default initialization."""
        builder = LossMaskBuilder()
        assert builder.config.mask_system is True
        assert builder.config.mask_user is True
        assert builder.config.mask_tool is True

    def test_build_message_mask(self):
        """Test message-level loss mask generation."""
        formatter = DataFormatter()
        builder = LossMaskBuilder()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)
        result = builder.build_message_mask(formatted)

        assert "message_loss_mask" in result
        assert len(result["message_loss_mask"]) == len(result["messages"])

    def test_mask_system_user_tool(self):
        """Test system, user, tool messages are masked by default."""
        formatter = DataFormatter()
        builder = LossMaskBuilder()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)
        result = builder.build_message_mask(formatted)

        for msg, mask in zip(result["messages"], result["message_loss_mask"]):
            if msg["role"] in ("system", "user", "tool"):
                # Mask should be False for these roles (don't calculate loss)
                assert mask["loss_mask"] is False, f"{msg['role']} should be masked"

    def test_assistant_not_masked(self):
        """Test assistant messages are not masked (unless thought is too long)."""
        formatter = DataFormatter()
        builder = LossMaskBuilder()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)
        result = builder.build_message_mask(formatted)

        assistant_masks = [
            m for m in result["message_loss_mask"] if m["role"] == "assistant"
        ]
        for mask in assistant_masks:
            # Short assistant messages should not be masked
            if mask["mask_reason"] is None:
                assert mask["loss_mask"] is True

    def test_thought_action_splitting(self):
        """Test splitting thought and action token counts."""
        builder = LossMaskBuilder()

        # ReAct format
        react_content = """Thought: Let me think about this problem
Action: python
Action Input: print('hello')"""

        thought_len, action_len = builder._split_thought_action(react_content)
        assert thought_len > 0
        assert action_len > 0

        # Final answer only
        answer_content = "Final Answer:\nThe answer is 42"
        thought_len, action_len = builder._split_thought_action(answer_content)
        assert action_len > 0


# -----------------------------------------------------------------------------
# DatasetExporter Tests
# -----------------------------------------------------------------------------


class TestDatasetExporter:
    """Tests for DatasetExporter class."""

    def test_init_default(self):
        """Test default initialization."""
        exporter = DatasetExporter()
        assert exporter.config.format == "both"

    @pytest.mark.skip(reason="Requires datasets library, test with optional deps")
    def test_to_hf_dataset(self):
        """Test conversion to HuggingFace Dataset."""
        formatter = DataFormatter()
        exporter = DatasetExporter()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)

        dataset = exporter.to_hf_dataset([formatted])
        assert dataset is not None
        assert len(dataset) == 1
        assert "messages" in dataset.column_names

    @pytest.mark.skip(reason="Requires datasets library")
    def test_save_jsonl(self):
        """Test saving to JSONL format."""
        formatter = DataFormatter()
        exporter = DatasetExporter()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)
        dataset = exporter.to_hf_dataset([formatted])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            exporter.save_jsonl(dataset, output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    @pytest.mark.skip(reason="Requires datasets and pyarrow")
    def test_save_parquet(self):
        """Test saving to Parquet format."""
        formatter = DataFormatter()
        exporter = DatasetExporter()

        raw = make_raw_trajectory(num_steps=2)
        formatted = formatter.format(raw)
        dataset = exporter.to_hf_dataset([formatted])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.parquet"
            exporter.save_parquet(dataset, output_path)
            assert output_path.exists()


# -----------------------------------------------------------------------------
# SFTDataPipeline Tests
# -----------------------------------------------------------------------------


class TestSFTDataPipeline:
    """Tests for end-to-end SFTDataPipeline."""

    def test_init_default(self):
        """Test default pipeline initialization."""
        pipeline = SFTDataPipeline()
        assert pipeline.formatter is not None
        assert pipeline.token_counter is not None
        assert pipeline.truncator is not None
        assert pipeline.mask_builder is not None
        assert pipeline.exporter is not None

    def test_process_trajectory(self):
        """Test processing a single trajectory."""
        pipeline = SFTDataPipeline()
        raw = make_raw_trajectory(num_steps=3)

        result = pipeline.process_trajectory(raw)

        assert result["task_id"] == "test_task_001"
        assert len(result["messages"]) > 0
        assert result["token_count"] > 0
        assert result["num_turns"] > 0
        assert "message_loss_mask" in result
        assert len(result["message_loss_mask"]) == len(result["messages"])

    def test_process_batch(self):
        """Test processing a batch of trajectories."""
        pipeline = SFTDataPipeline()
        trajectories = [
            make_raw_trajectory(num_steps=2),
            make_raw_trajectory(num_steps=3),
            make_raw_trajectory(num_steps=1, task_id="task_2"),
        ]

        records, export_paths = pipeline.process_batch(trajectories)

        assert len(records) == 3
        assert all("token_count" in r for r in records)
        assert export_paths == {}  # No output_dir specified

    @pytest.mark.skip(reason="Requires datasets library")
    def test_process_batch_with_export(self):
        """Test processing batch with export."""
        pipeline = SFTDataPipeline(SFTFormatterConfig())
        trajectories = [make_raw_trajectory(num_steps=2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            records, export_paths = pipeline.process_batch(
                trajectories, output_dir=tmpdir, dataset_name="test"
            )

            assert len(records) == 1
            assert "jsonl" in export_paths
            assert "parquet" in export_paths
            assert Path(export_paths["jsonl"]).exists()
            assert Path(export_paths["parquet"]).exists()

    def test_token_distribution(self):
        """Test token counts are within reasonable range."""
        pipeline = SFTDataPipeline()
        trajectories = [make_raw_trajectory(num_steps=i + 1) for i in range(5)]
        records, _ = pipeline.process_batch(trajectories)

        token_counts = [r["token_count"] for r in records]
        # All should have positive token counts
        assert all(c > 0 for c in token_counts)
        # Longer trajectories should have more tokens (generally)
        assert token_counts[-1] >= token_counts[0]


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


def test_full_pipeline_integration():
    """Integration test: format -> count -> truncate -> mask -> export."""
    # Setup
    formatter = DataFormatter()
    token_counter = TokenCounter("Qwen/Qwen2.5-0.5B-Instruct")
    truncator = TrajectoryTruncator(
        TruncationConfig(max_tokens=5000, auto_scale=False),
        token_counter,
    )
    mask_builder = LossMaskBuilder(token_counter=token_counter)

    # Create raw trajectory
    raw = make_raw_trajectory(num_steps=4)

    # Format
    formatted = formatter.format(raw)
    assert "messages" in formatted
    assert len(formatted["messages"]) > 0

    # Count tokens
    token_count = token_counter.count_messages(formatted["messages"])
    assert token_count > 0
    formatted["token_count"] = token_count

    # Truncate
    truncated = truncator.truncate(formatted)
    assert truncated["token_count"] <= 5000

    # Build mask
    masked = mask_builder.build_message_mask(truncated)
    assert "message_loss_mask" in masked
    assert len(masked["message_loss_mask"]) == len(masked["messages"])

    # Verify roles
    roles = {m["role"] for m in masked["messages"]}
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


def test_json_serializable():
    """Test formatted trajectories are JSON serializable."""
    formatter = DataFormatter()
    raw = make_raw_trajectory(num_steps=3)
    formatted = formatter.format(raw)

    # Should serialize without error
    serialized = json.dumps(formatted, ensure_ascii=False)
    assert serialized is not None
    assert len(serialized) > 0

    # Should deserialize back correctly
    deserialized = json.loads(serialized)
    assert deserialized["task_id"] == formatted["task_id"]
    assert len(deserialized["messages"]) == len(formatted["messages"])
