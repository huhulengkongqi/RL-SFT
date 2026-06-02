"""Standard SFT training data formatter with chat template support.

Converts raw agent trajectories to HuggingFace/trl-compatible format with:
- 4-role chat template (system/user/assistant/tool)
- Token counting and truncation strategies
- Loss mask generation for scratchpad masking
- Parquet/JSONL export for HuggingFace Dataset
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------


class ChatMessage(TypedDict, total=False):
    """Standard chat message format compatible with OpenAI/HuggingFace."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str]  # For tool role: the name of the tool


class FormattedTrajectory(TypedDict, total=False):
    """Formatted trajectory ready for SFT training."""

    task_id: str
    domain: str
    difficulty: str
    messages: List[ChatMessage]
    token_count: int
    num_turns: int
    success: bool
    final_score: Optional[float]
    message_loss_mask: List[Dict[str, Any]]


RawTrajectory = Union["Trajectory", Dict[str, Any]]  # type: ignore[name-defined]


# -----------------------------------------------------------------------------
# Token Counter
# -----------------------------------------------------------------------------


class TokenCounter:
    """Count tokens using HuggingFace tokenizers with chat template support.

    Uses AutoTokenizer for accurate token counting compatible with most models.
    Supports caching tokenizer instances for repeated use.
    """

    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code: bool = True,
    ):
        """Initialize token counter.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code for custom tokenizers
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.tokenizer = self._get_tokenizer(model_name, trust_remote_code)

    @classmethod
    def _get_tokenizer(cls, model_name: str, trust_remote_code: bool) -> Any:
        """Get or create cached tokenizer."""
        cache_key = f"{model_name}:{trust_remote_code}"
        if cache_key not in cls._tokenizer_cache:
            try:
                from transformers import AutoTokenizer

                cls._tokenizer_cache[cache_key] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                )
            except ImportError:
                raise ImportError(
                    "transformers is required for TokenCounter. "
                    "Install it with: uv pip install transformers"
                )
        return cls._tokenizer_cache[cache_key]

    def count_messages(self, messages: List[ChatMessage]) -> int:
        """Count tokens in a list of chat messages using chat template.

        Args:
            messages: List of chat messages with role and content

        Returns:
            Total token count
        """
        # Try chat template first, but fall back gracefully
        # Note: Some tokenizers don't support the 'tool' role in chat template
        try:
            # Convert to standard roles for tokenizer compatibility
            safe_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                # Qwen tokenizer may not support 'tool' role - convert to user
                if role == "tool":
                    role = "user"
                    if not content.startswith("Observation:"):
                        content = f"Observation: {content}"
                safe_messages.append({"role": role, "content": content})

            token_ids = self.tokenizer.apply_chat_template(
                safe_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if token_ids is not None and len(token_ids) > 2:
                return len(token_ids)
        except Exception as e:
            logger.debug(f"Chat template token counting failed: {e}")

        # Fallback: count each message's content manually
        total = 0
        for msg in messages:
            total += self.count_text(msg.get("content", ""))
            # Add overhead for role tokens
            total += 3  # Approximate overhead for role markers and separators
        return total

    def count_text(self, text: str) -> int:
        """Count tokens in plain text.

        Args:
            text: Plain text to count

        Returns:
            Token count
        """
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    def encode_messages(self, messages: List[ChatMessage]) -> List[int]:
        """Encode messages to token IDs using chat template.

        Args:
            messages: List of chat messages

        Returns:
            List of token IDs
        """
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        return list(token_ids) if token_ids is not None else []


# -----------------------------------------------------------------------------
# Data Formatter
# -----------------------------------------------------------------------------


@dataclass
class FormatConfig:
    """Configuration for trajectory formatting."""

    trajectory_format: Literal["react", "function_json"] = "react"
    merge_thought_action: bool = True  # Merge thought and action into one assistant message
    include_system_prompt: bool = True
    system_prompt_template: str = "You are an agent solving a {domain} task. Follow the tool protocol exactly."
    use_tool_role: bool = True  # Use 'tool' role for observations, otherwise 'user'


class DataFormatter:
    """Convert raw agent trajectories to standard SFT chat format.

    Supports two trajectory formats:
    - react: Human-readable Thought/Action/Observation format
    - function_json: Structured JSON format for tool calls

    Implements 4-role chat template: system -> user -> assistant -> tool -> ...
    """

    def __init__(self, config: Optional[FormatConfig] = None):
        self.config = config or FormatConfig()

    def format(self, trajectory: RawTrajectory) -> FormattedTrajectory:
        """Format a raw trajectory to standard SFT chat format.

        Args:
            trajectory: Trajectory object or raw dict from JSON

        Returns:
            Formatted trajectory with chat messages
        """
        # Extract basic info (works for both Trajectory objects and dicts)
        if hasattr(trajectory, "model_dump"):
            # It's a Pydantic model
            traj_dict = trajectory.model_dump()  # type: ignore[attr-defined]
        else:
            traj_dict = cast(Dict[str, Any], trajectory)

        task_id = str(traj_dict.get("task_id", ""))
        domain = str(traj_dict.get("domain", ""))
        difficulty = str(traj_dict.get("difficulty", ""))
        success = bool(traj_dict.get("success", False))
        final_score = traj_dict.get("final_score")

        messages: List[ChatMessage] = []

        # System message
        if self.config.include_system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt_template.format(domain=domain),
            })

        # Task prompt (first user message)
        steps = traj_dict.get("steps", [])
        if steps:
            first_step = steps[0]
            state_snapshot = first_step.get("state_snapshot", {}) or {}
            task_prompt = state_snapshot.get("metadata", {}).get("task_prompt")
            if task_prompt:
                messages.append({
                    "role": "user",
                    "content": f"Task:\n{task_prompt}",
                })

        # Process each step
        for step in steps:
            step_messages = self._format_step(step)
            messages.extend(step_messages)

        num_turns = sum(1 for m in messages if m["role"] == "assistant")

        return {
            "task_id": task_id,
            "domain": domain,
            "difficulty": difficulty,
            "messages": messages,
            "token_count": 0,  # Populated later by TokenCounter
            "num_turns": num_turns,
            "success": success,
            "final_score": final_score,
        }

    def _format_step(self, step: Dict[str, Any]) -> List[ChatMessage]:
        """Format a single trajectory step.

        Returns:
            [assistant_thought_action, tool_observation] (or subset thereof)
        """
        messages: List[ChatMessage] = []
        action = step.get("action", {}) or {}
        thought = step.get("thought") or action.get("thought") or ""
        observation = step.get("observation", {}) or {}

        # Format thought + action
        if self.config.trajectory_format == "function_json":
            content_parts = []
            if thought and self.config.merge_thought_action:
                content_parts.append(json.dumps(
                    {"type": "thought", "content": thought},
                    ensure_ascii=False,
                ))

            # Action content
            action_type = action.get("action_type")
            if action_type == "tool_call":
                action_content = json.dumps({
                    "type": "action",
                    "action": "tool_call",
                    "tool": action.get("name", ""),
                    "arguments": action.get("kwargs", {}),
                }, ensure_ascii=False)
                content_parts.append(action_content)
            elif action_type == "final_answer":
                answer_content = json.dumps({
                    "type": "action",
                    "action": "final_answer",
                    "answer": action.get("answer", ""),
                }, ensure_ascii=False)
                content_parts.append(answer_content)

            if content_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts),
                })

        else:  # react format
            content_parts = []
            if thought and self.config.merge_thought_action:
                content_parts.append(f"Thought: {thought}")

            action_type = action.get("action_type")
            if action_type == "tool_call":
                tool_input = (
                    action.get("kwargs", {}).get("code")
                    or action.get("kwargs", {}).get("expr")
                    or action.get("kwargs", {}).get("input")
                    or str(action.get("args", ""))
                )
                content_parts.append(f"Action: {action.get('name', '')}")
                content_parts.append(f"Action Input: {tool_input}")
            elif action_type == "final_answer":
                content_parts.append(f"Final Answer:\n{action.get('answer', '')}")

            if content_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts).strip(),
                })

        # Observation (tool role)
        obs_content = observation.get("content") or observation.get("error")
        if obs_content is not None:
            obs_success = observation.get("success", True)
            obs_prefix = "" if obs_success else "Error: "
            tool_name = action.get("name", "unknown") if action.get("action_type") == "tool_call" else "observation"

            if self.config.use_tool_role:
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"{obs_prefix}{obs_content}",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{obs_prefix}{obs_content}",
                })

        return messages


# -----------------------------------------------------------------------------
# Truncation Strategy
# -----------------------------------------------------------------------------


@dataclass
class TruncationConfig:
    """Configuration for trajectory truncation."""

    strategy: Literal["middle", "head", "tail"] = "middle"
    max_tokens: int = 3277  # 80% of 4096 context window
    context_window: int = 4096
    auto_scale: bool = True
    head_keep_steps: int = 2  # For middle strategy: keep N steps from start
    tail_keep_steps: int = 3  # For middle strategy: keep N steps from end

    def __post_init__(self) -> None:
        if self.auto_scale:
            self.max_tokens = int(self.context_window * 0.8)


TruncationStrategy = Literal["middle", "head", "tail"]


class TrajectoryTruncator:
    """Truncate long trajectories to fit within token budget.

    Supports three strategies:
    - middle: Keep head and tail steps, remove middle steps (recommended)
    - head: Keep first N steps, remove tail
    - tail: Keep last N steps, remove head
    """

    def __init__(
        self,
        config: Optional[TruncationConfig] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.config = config or TruncationConfig()
        self.token_counter = token_counter or TokenCounter()

    def truncate(self, formatted: FormattedTrajectory) -> FormattedTrajectory:
        """Truncate trajectory if it exceeds token limit.

        Args:
            formatted: Formatted trajectory with messages

        Returns:
            Potentially truncated trajectory
        """
        messages = formatted["messages"]
        current_tokens = self.token_counter.count_messages(messages)

        if current_tokens <= self.config.max_tokens:
            formatted["token_count"] = current_tokens
            return formatted

        logger.info(
            f"Truncating trajectory: {current_tokens} > {self.config.max_tokens} tokens "
            f"(strategy: {self.config.strategy})"
        )

        truncated_messages = self._truncate_messages(messages, self.config.strategy)
        truncated_tokens = self.token_counter.count_messages(truncated_messages)

        result = formatted.copy()
        result["messages"] = truncated_messages
        result["token_count"] = truncated_tokens
        result["num_turns"] = sum(1 for m in truncated_messages if m["role"] == "assistant")

        return result

    def _truncate_messages(
        self,
        messages: List[ChatMessage],
        strategy: TruncationStrategy,
    ) -> List[ChatMessage]:
        """Truncate message list dynamically based on token count.

        Instead of fixed steps, we iteratively remove steps until token count
        falls below max_tokens. This handles variable-length steps better.
        """
        # Identify message groups (turns) - preserve system prompt always
        system_messages = [m for m in messages if m["role"] == "system"]
        first_user = [m for m in messages if m["role"] == "user" and m == messages[len(system_messages)]]
        rest_start = len(system_messages) + len(first_user)
        conversation_messages = messages[rest_start:]

        # Group into turns: [assistant + tool/user] pairs
        turn_groups: List[List[ChatMessage]] = []
        current_group: List[ChatMessage] = []

        for msg in conversation_messages:
            if msg["role"] == "assistant" and current_group:
                turn_groups.append(current_group)
                current_group = [msg]
            else:
                current_group.append(msg)

        if current_group:
            turn_groups.append(current_group)

        if not turn_groups:
            return system_messages + first_user

        # Apply strategy and dynamically adjust based on token count
        kept = self._apply_truncation_strategy(turn_groups, strategy)

        # Iteratively reduce until under token limit or min steps reached
        min_kept = 1  # Minimum steps: can keep just the final answer step if needed
        while len(kept) > min_kept:
            result = system_messages + first_user + [m for g in kept for m in g]
            token_count = self.token_counter.count_messages(result)

            if token_count <= self.config.max_tokens:
                return result

            # Still too long: remove one more step from middle
            mid_idx = len(kept) // 2
            kept.pop(mid_idx)

        # If even 1 step is too long, we just keep it as-is
        # We will NOT truncate thought content - loss masking handles long thoughts
        # by not training the model to generate them
        result = system_messages + first_user + [m for g in kept for m in g]
        final_tokens = self.token_counter.count_messages(result)

        if final_tokens > self.config.max_tokens:
            logger.debug(
                f"Single step still exceeds token limit: {final_tokens} > {self.config.max_tokens}. "
                "Keeping as-is - loss masking will handle long thoughts during training."
            )

        return result

    def _apply_truncation_strategy(
        self,
        turn_groups: List[List[ChatMessage]],
        strategy: TruncationStrategy,
    ) -> List[List[ChatMessage]]:
        """Apply initial truncation strategy."""
        if strategy == "middle":
            # Start with ALL steps, then we'll iteratively remove from middle
            # This way we preserve as much context as possible
            return turn_groups.copy()
        elif strategy == "head":
            # Keep first half (will reduce further if needed)
            keep = max(2, (len(turn_groups) + 1) // 2)
            return turn_groups[:keep]
        elif strategy == "tail":
            # Keep last half (will reduce further if needed)
            keep = max(2, (len(turn_groups) + 1) // 2)
            return turn_groups[-keep:]
        else:
            return turn_groups.copy()

    def _truncate_head(self, turn_groups: List[List[ChatMessage]]) -> List[List[ChatMessage]]:
        """Keep first N steps."""
        keep = max(1, len(turn_groups) // 2)
        return turn_groups[:keep]

    def _truncate_tail(self, turn_groups: List[List[ChatMessage]]) -> List[List[ChatMessage]]:
        """Keep last N steps."""
        keep = max(1, len(turn_groups) // 2)
        return turn_groups[-keep:]


# -----------------------------------------------------------------------------
# Loss Mask Builder
# -----------------------------------------------------------------------------


@dataclass
class LossMaskConfig:
    """Configuration for loss mask generation."""

    mask_system: bool = True
    mask_user: bool = True
    mask_tool: bool = True
    mask_thought: bool = True
    thought_max_tokens: int = 500  # Mask thought longer than this
    thought_prefix: str = "Thought: "  # ReAct format thought prefix


class LossMaskBuilder:
    """Generate loss masks for SFT training.

    Masks out:
    - System prompts
    - User inputs
    - Tool outputs
    - Long thought sections (scratchpad)

    Only actual action/output generation contributes to loss.
    """

    def __init__(
        self,
        config: Optional[LossMaskConfig] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.config = config or LossMaskConfig()
        self.token_counter = token_counter or TokenCounter()

    def build_message_mask(self, formatted: FormattedTrajectory) -> FormattedTrajectory:
        """Build message-level loss mask (coarse-grained).

        Adds 'message_loss_mask' field indicating which messages contribute to loss.
        """
        messages = formatted["messages"]
        message_mask: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            should_mask = False
            mask_reason = None
            thought_tokens = 0
            action_tokens = 0

            if role == "system" and self.config.mask_system:
                should_mask = True
                mask_reason = "system_prompt"
            elif role == "user" and self.config.mask_user:
                should_mask = True
                mask_reason = "user_input"
            elif role == "tool" and self.config.mask_tool:
                should_mask = True
                mask_reason = "tool_output"
            elif role == "assistant":
                # Check thought content length
                thought_len, action_len = self._split_thought_action(content)
                thought_tokens = thought_len
                action_tokens = action_len

                if thought_len > self.config.thought_max_tokens and self.config.mask_thought:
                    should_mask = True
                    mask_reason = f"long_thought_{thought_len}_tokens"

            message_mask.append({
                "role": role,
                "loss_mask": not should_mask,  # True = calculate loss, False = mask
                "mask_reason": mask_reason,
                "thought_tokens": thought_tokens,
                "action_tokens": action_tokens,
            })

        result = formatted.copy()
        result["message_loss_mask"] = message_mask
        return result

    def _split_thought_action(self, content: str) -> tuple[int, int]:
        """Split assistant message content into thought and action parts, count tokens."""
        if not content:
            return (0, 0)

        # Handle JSON format
        if content.strip().startswith("{"):
            try:
                lines = content.strip().split("\n")
                thought_tokens = 0
                action_tokens = 0

                for line in lines:
                    if '"type": "thought"' in line:
                        thought_tokens += self.token_counter.count_text(line)
                    elif '"type": "action"' in line:
                        action_tokens += self.token_counter.count_text(line)

                return (thought_tokens, action_tokens) if thought_tokens or action_tokens else (0, self.token_counter.count_text(content))
            except Exception:
                pass

        # Handle ReAct format
        thought_end = content.find("\nAction:")
        if thought_end == -1:
            thought_end = content.find("\nFinal Answer:")

        if thought_end > 0:
            thought_content = content[:thought_end]
            action_content = content[thought_end:]
            return (
                self.token_counter.count_text(thought_content),
                self.token_counter.count_text(action_content),
            )

        # Couldn't split, assume all action (or all thought if looks like thought)
        if content.startswith(self.config.thought_prefix):
            return (self.token_counter.count_text(content), 0)
        return (0, self.token_counter.count_text(content))


# -----------------------------------------------------------------------------
# Dataset Exporter
# -----------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    format: Literal["parquet", "jsonl", "both"] = "both"
    split: str = "train"
    max_records_per_file: Optional[int] = None
    compression: Optional[str] = None  # For parquet: 'gzip', 'snappy', etc.


class DatasetExporter:
    """Export formatted trajectories to HuggingFace-compatible formats.

    Supports:
    - HuggingFace Dataset object (in-memory)
    - JSONL (line-delimited JSON, human-readable)
    - Parquet (columnar storage, efficient for large datasets)
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def to_hf_dataset(self, records: List[FormattedTrajectory]) -> Any:
        """Convert records to HuggingFace Dataset object.

        Args:
            records: List of formatted trajectories

        Returns:
            HuggingFace Dataset object
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets is required for HuggingFace export. "
                "Install it with: uv pip install datasets"
            )

        # Convert to plain dicts for Dataset compatibility
        plain_records = []
        for rec in records:
            plain_rec = dict(rec)
            # Ensure messages are JSON-serializable
            if "messages" in plain_rec:
                plain_rec["messages"] = [dict(m) for m in plain_rec["messages"]]
            if "message_loss_mask" in plain_rec:
                plain_rec["message_loss_mask"] = [dict(m) for m in plain_rec["message_loss_mask"]]
            plain_records.append(plain_rec)

        return Dataset.from_list(plain_records)

    def save_jsonl(
        self,
        dataset: Any,
        output_path: Union[str, Path],
    ) -> Path:
        """Save dataset to JSONL format.

        Args:
            dataset: HuggingFace Dataset object
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
        except AttributeError:
            # Fallback: manual JSONL writing
            with open(output_path, "w", encoding="utf-8") as f:
                for record in dataset:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved JSONL to {output_path}")
        return output_path

    def save_parquet(
        self,
        dataset: Any,
        output_path: Union[str, Path],
    ) -> Path:
        """Save dataset to Parquet format.

        Args:
            dataset: HuggingFace Dataset object
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            kwargs = {}
            if self.config.compression is not None:
                kwargs["compression"] = self.config.compression
            dataset.to_parquet(str(output_path), **kwargs)
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install it with: uv pip install pyarrow"
            )

        logger.info(f"Saved Parquet to {output_path}")
        return output_path

    def export(
        self,
        records: List[FormattedTrajectory],
        output_dir: Union[str, Path],
        name: str = "sft_dataset",
    ) -> Dict[str, Path]:
        """Export records to specified formats.

        Args:
            records: List of formatted trajectories
            output_dir: Output directory
            name: Base name for output files

        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        dataset = self.to_hf_dataset(records)
        outputs: Dict[str, Path] = {}

        if self.config.format in ("jsonl", "both"):
            outputs["jsonl"] = self.save_jsonl(dataset, output_dir / f"{name}.jsonl")

        if self.config.format in ("parquet", "both"):
            outputs["parquet"] = self.save_parquet(dataset, output_dir / f"{name}.parquet")

        return outputs

    @staticmethod
    def validate_with_trl(dataset_path: Union[str, Path], model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> bool:
        """Validate that exported dataset can be loaded by trl.SFTTrainer.

        Note: This is a lightweight validation that doesn't actually train.

        Args:
            dataset_path: Path to Parquet or JSONL dataset
            model_name: Model name for tokenizer

        Returns:
            True if dataset is compatible
        """
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer

            # Load dataset
            path_str = str(dataset_path)
            if path_str.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=path_str, split="train")
            else:
                dataset = load_dataset("json", data_files=path_str, split="train")

            # Check for required field
            if "messages" not in dataset.column_names:
                logger.error("Dataset missing 'messages' field")
                return False

            # Verify messages can be tokenized with chat template
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            sample = dataset[0]["messages"]

            tokenizer.apply_chat_template(sample, tokenize=True)

            logger.info(f"Dataset validated: {len(dataset)} records, compatible with trl.SFTTrainer")
            return True

        except ImportError as e:
            logger.warning(f"Cannot validate with trl: {e}")
            return False
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False


# -----------------------------------------------------------------------------
# Convenience Pipeline
# -----------------------------------------------------------------------------


@dataclass
class SFTFormatterConfig:
    """Full pipeline configuration."""

    format: FormatConfig = field(default_factory=FormatConfig)
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    loss_mask: LossMaskConfig = field(default_factory=LossMaskConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct"


class SFTDataPipeline:
    """End-to-end pipeline for converting raw trajectories to training-ready SFT data.

    Combines formatting, token counting, truncation, loss masking, and export.
    """

    def __init__(self, config: Optional[SFTFormatterConfig] = None):
        self.config = config or SFTFormatterConfig()
        self.token_counter = TokenCounter(self.config.tokenizer_model)
        self.formatter = DataFormatter(self.config.format)
        self.truncator = TrajectoryTruncator(self.config.truncation, self.token_counter)
        self.mask_builder = LossMaskBuilder(self.config.loss_mask, self.token_counter)
        self.exporter = DatasetExporter(self.config.export)

    def process_trajectory(self, trajectory: RawTrajectory) -> FormattedTrajectory:
        """Process a single trajectory through the full pipeline.

        Args:
            trajectory: Raw trajectory object or dict

        Returns:
            Fully processed SFT-ready trajectory
        """
        # Format
        formatted = self.formatter.format(trajectory)

        # Count tokens
        formatted["token_count"] = self.token_counter.count_messages(formatted["messages"])

        # Truncate if needed
        truncated = self.truncator.truncate(formatted)

        # Build loss mask
        with_mask = self.mask_builder.build_message_mask(truncated)

        return with_mask

    def process_batch(
        self,
        trajectories: List[RawTrajectory],
        output_dir: Optional[Union[str, Path]] = None,
        dataset_name: str = "sft_dataset",
    ) -> tuple[List[FormattedTrajectory], Dict[str, Path]]:
        """Process a batch of trajectories and optionally export.

        Args:
            trajectories: List of raw trajectories
            output_dir: Optional output directory for exported files
            dataset_name: Base name for output files

        Returns:
            Tuple of (processed_records, export_paths)
        """
        records = [self.process_trajectory(t) for t in trajectories]

        # Log stats
        token_counts = [r["token_count"] for r in records]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            logger.info(
                f"Processed {len(records)} trajectories: "
                f"avg_tokens={avg_tokens:.1f}, max_tokens={max_tokens}"
            )

        export_paths: Dict[str, Path] = {}
        if output_dir is not None:
            export_paths = self.exporter.export(records, output_dir, dataset_name)

        return records, export_paths
