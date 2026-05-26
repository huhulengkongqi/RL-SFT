"""Agent Harness main loop for SFT data synthesis.

Core components:
- AgentState: Immutable state container with deep copy snapshots
- TerminationDetector: 5 termination conditions
- TrajectoryRecorder: Step recording with failure preservation
- AgentLoop: Main orchestration loop
"""

import copy
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

from infra.environment.models import Action, FinalAnswerAction, Observation, StepResult, ToolCallAction


class AgentState(BaseModel):
    """Immutable agent state container with deep copy snapshots.

    All updates return NEW instances - original instance is never modified.
    Ensures trajectory reproducibility and prevents reference pollution.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Task metadata
    task_id: str = Field(..., description="Unique task identifier")
    domain: str = Field(..., description="Task domain (code_debug, math_reasoning, etc.)")
    difficulty: str = Field(default="medium", description="Task difficulty level")

    # Step tracking
    step: int = Field(default=0, ge=0, description="Current step number")
    max_steps: int = Field(default=10, ge=1, description="Maximum allowed steps")

    # Termination state
    done: bool = Field(default=False, description="Whether episode is complete")
    termination_reason: Optional[str] = Field(None, description="Why episode terminated")

    # Last action/observation
    last_action: Optional[Action] = Field(None, description="Most recent action")
    last_observation: Optional[Observation] = Field(None, description="Most recent observation")

    # Full history
    observation_history: List[Observation] = Field(default_factory=list, description="Complete observation history")
    action_history: List[Action] = Field(default_factory=list, description="Complete action history")
    step_results: List[StepResult] = Field(default_factory=list, description="Complete step result history")

    # Token budget tracking
    token_usage: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Token usage per step: {step_key: {prompt, completion, total}}")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    token_budget: int = Field(default=100000, ge=0, description="Maximum allowed tokens")

    # Verification metrics
    test_cases_passed: int = Field(default=0, ge=0, description="Number of test cases passed")
    test_cases_total: int = Field(default=0, ge=0, description="Total test cases")
    final_score: Optional[float] = Field(None, description="Final quality score 0-1")

    # Extension point
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Timestamp
    created_at: datetime = Field(default_factory=datetime.now, description="State creation timestamp")

    def update(
        self,
        action: Action,
        observation: Observation,
        step_result: StepResult,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> "AgentState":
        """Create NEW updated state instance with deep copy.

        Original state remains unchanged. All history is deep copied
        to prevent reference pollution from downstream code.

        Args:
            action: Action that was executed
            observation: Result observation
            step_result: Complete step result
            token_usage: Optional token usage dict from LLM

        Returns:
            New AgentState instance
        """
        # Deep copy all mutable collections
        new_obs_history = copy.deepcopy(self.observation_history)
        new_obs_history.append(copy.deepcopy(observation))

        new_action_history = copy.deepcopy(self.action_history)
        new_action_history.append(copy.deepcopy(action))

        new_step_results = copy.deepcopy(self.step_results)
        new_step_results.append(copy.deepcopy(step_result))

        new_token_usage = copy.deepcopy(self.token_usage)
        if token_usage:
            step_key = f"step_{self.step + 1}"
            new_token_usage[step_key] = copy.deepcopy(token_usage)

        # Calculate new total tokens
        additional_tokens = token_usage.get("total_tokens", 0) if token_usage else 0
        new_total_tokens = self.total_tokens + additional_tokens

        # Extract verification info from observation metadata
        test_passed = observation.metadata.get("test_cases_passed", 0)
        test_total = observation.metadata.get("test_cases_total", self.test_cases_total)
        score = observation.metadata.get("score", self.final_score)

        # Check if done from step_result
        is_done = step_result.done or self.done
        term_reason = self.termination_reason
        if is_done and not term_reason and isinstance(action, FinalAnswerAction):
            term_reason = "success"

        return AgentState(
            # Task metadata (unchanged)
            task_id=self.task_id,
            domain=self.domain,
            difficulty=self.difficulty,
            # Step tracking (incremented)
            step=self.step + 1,
            max_steps=self.max_steps,
            # Termination state
            done=is_done,
            termination_reason=term_reason,
            # Last values
            last_action=copy.deepcopy(action),
            last_observation=copy.deepcopy(observation),
            # History (deep copied)
            observation_history=new_obs_history,
            action_history=new_action_history,
            step_results=new_step_results,
            # Token tracking
            token_usage=new_token_usage,
            total_tokens=new_total_tokens,
            token_budget=self.token_budget,
            # Verification metrics
            test_cases_passed=test_passed if test_passed > 0 else self.test_cases_passed,
            test_cases_total=test_total,
            final_score=score,
            # Metadata (deep copied)
            metadata=copy.deepcopy(self.metadata),
            # New timestamp
            created_at=datetime.now(),
        )

    def snapshot(self) -> Dict[str, Any]:
        """Create complete deep copy snapshot for trajectory recording.

        Returns:
            Dictionary with full state, safe for persistence
        """
        return copy.deepcopy(self.model_dump())

    def get_action_hash(self, action: Action) -> str:
        """Generate hash of action for loop detection.

        Args:
            action: Action to hash

        Returns:
            MD5 hash string of action content
        """
        action_dict = action.model_dump()
        # Exclude thought and id from hash for loop detection
        action_dict.pop("thought", None)
        action_dict.pop("id", None)
        action_str = str(sorted(action_dict.items()))
        return hashlib.md5(action_str.encode()).hexdigest()


class TerminationDecision(BaseModel):
    """Result of termination check."""

    should_terminate: bool = Field(..., description="Whether episode should end")
    reason: str = Field(..., description="Termination reason: success, max_steps, token_overflow, loop_detected, fatal_error")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    score: Optional[float] = Field(None, description="Final score if available")


class TerminationDetector:
    """Detects 5 termination conditions for agent loop.

    Conditions checked in priority order:
    1. fatal_error - Sandbox crash, permission denied, unrecoverable exception
    2. success - FinalAnswerAction submitted
    3. max_steps - Step limit reached
    4. token_overflow - Token budget exceeded
    5. loop_detected - Same action repeated N times
    """

    def __init__(self, loop_detection_threshold: int = 3):
        self.loop_detection_threshold = loop_detection_threshold
        self.action_hash_counts: Dict[str, int] = {}

    def check(self, state: AgentState) -> TerminationDecision:
        """Check all termination conditions in priority order.

        Args:
            state: Current agent state

        Returns:
            TerminationDecision with reason and details
        """
        # Priority 1: Fatal error check
        if state.last_observation and not state.last_observation.success:
            error = state.last_observation.error or ""
            is_fatal = any(
                keyword in error.lower()
                for keyword in ["fatal", "crash", "permission", "denied", "container", "timeout"]
            )
            if is_fatal:
                return TerminationDecision(
                    should_terminate=True,
                    reason="fatal_error",
                    details={"error": error, "step": state.step},
                )

        # Priority 2: Success (final answer submitted)
        if state.done and state.termination_reason == "success":
            return TerminationDecision(
                should_terminate=True,
                reason="success",
                details={
                    "steps": state.step,
                    "test_cases_passed": state.test_cases_passed,
                    "test_cases_total": state.test_cases_total,
                },
                score=state.final_score,
            )

        # Priority 3: Max steps reached
        if state.step >= state.max_steps:
            return TerminationDecision(
                should_terminate=True,
                reason="max_steps",
                details={"steps_taken": state.step, "max_allowed": state.max_steps},
            )

        # Priority 4: Token overflow
        if state.total_tokens >= state.token_budget:
            return TerminationDecision(
                should_terminate=True,
                reason="token_overflow",
                details={"tokens_used": state.total_tokens, "budget": state.token_budget},
            )

        # Priority 5: Loop detection
        if state.last_action:
            action_hash = state.get_action_hash(state.last_action)
            self.action_hash_counts[action_hash] = self.action_hash_counts.get(action_hash, 0) + 1
            count = self.action_hash_counts[action_hash]
            if count >= self.loop_detection_threshold:
                return TerminationDecision(
                    should_terminate=True,
                    reason="loop_detected",
                    details={
                        "action_hash": action_hash,
                        "repeat_count": count,
                        "threshold": self.loop_detection_threshold,
                    },
                )

        # No termination
        return TerminationDecision(should_terminate=False, reason="continue", details={})

    def reset(self) -> None:
        """Reset loop detection state for new episode."""
        self.action_hash_counts = {}


class TrajectoryStep(BaseModel):
    """Single step recorded in trajectory."""

    step: int = Field(..., description="Step number")
    state_snapshot: Dict[str, Any] = Field(..., description="Deep copied state snapshot")
    action: Action = Field(..., description="Action taken")
    observation: Observation = Field(..., description="Observation received")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage for this step")
    timestamp: datetime = Field(default_factory=datetime.now, description="Recording timestamp")


class Trajectory(BaseModel):
    """Complete agent interaction trajectory for SFT training."""

    task_id: str = Field(..., description="Task identifier")
    domain: str = Field(..., description="Task domain")
    difficulty: str = Field(..., description="Task difficulty")
    steps: List[TrajectoryStep] = Field(default_factory=list, description="All trajectory steps")
    final_state: Dict[str, Any] = Field(..., description="Final state snapshot")
    termination_reason: str = Field(..., description="Why trajectory ended")
    termination_details: Dict[str, Any] = Field(default_factory=dict, description="Termination details")
    final_score: Optional[float] = Field(None, description="Final quality score")
    total_time: float = Field(0.0, description="Total execution time in seconds")
    success: bool = Field(..., description="Whether task was solved successfully")

    def to_sft_format(self) -> Dict[str, Any]:
        """Convert trajectory to standard SFT training format.

        Returns:
            Dictionary with conversation-style SFT data
        """
        messages = []

        # System message with task info
        messages.append({
            "role": "system",
            "content": f"You are an agent solving a {self.domain} task. Follow the tool protocol exactly.",
        })

        # Initial task message
        first_snapshot = self.steps[0].state_snapshot if self.steps else {}
        task_prompt = first_snapshot.get("metadata", {}).get("task_prompt")
        if task_prompt:
            messages.append({"role": "user", "content": f"Task:\n{task_prompt}"})

        # Conversation: assistant action -> user observation
        for step in self.steps:
            action = step.action
            thought = action.thought or ""
            if action.action_type == "tool_call":
                tool_input = action.kwargs.get("code") or action.kwargs.get("expr") or action.kwargs.get("input") or str(action.args)
                content = f"{thought}\n\nTool: {action.name}\nInput: {tool_input}"
            else:
                content = f"{thought}\n\nFinal Answer:\n{action.answer}"
            messages.append({"role": "assistant", "content": content.strip()})

            obs = step.observation
            if obs.success:
                user_content = f"Observation:\n{obs.content}"
            else:
                user_content = f"Observation Error:\n{obs.error or obs.content or 'Unknown error'}"
            messages.append({"role": "user", "content": user_content})

        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "messages": messages,
            "termination_reason": self.termination_reason,
            "final_score": self.final_score,
            "success": self.success,
        }


class TrajectoryRecorder:
    """Records complete agent trajectory with deep copy snapshots.

    Preserves ALL steps including failures for "what not to do" training.
    """

    def __init__(self, save_failed_steps: bool = True):
        self.save_failed_steps = save_failed_steps
        self.steps: List[TrajectoryStep] = []
        self.start_time = datetime.now()

    def record_step(
        self,
        state: AgentState,
        action: Action,
        observation: Observation,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record single step with deep copy snapshot.

        Args:
            state: Current agent state
            action: Action executed
            observation: Result observation
            token_usage: Optional token usage from LLM
        """
        # Always record successful steps
        should_record = observation.success or self.save_failed_steps

        if should_record:
            step = TrajectoryStep(
                step=state.step,
                state_snapshot=state.snapshot(),
                action=copy.deepcopy(action),
                observation=copy.deepcopy(observation),
                token_usage=copy.deepcopy(token_usage) if token_usage else None,
            )
            self.steps.append(step)

    def finalize(
        self,
        final_state: AgentState,
        termination_reason: str,
        termination_details: Dict[str, Any],
    ) -> Trajectory:
        """Finalize and return complete trajectory.

        Args:
            final_state: Final agent state
            termination_reason: Why episode ended
            termination_details: Additional termination metadata

        Returns:
            Complete Trajectory object
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        success = termination_reason == "success" and bool(final_state.last_observation and final_state.last_observation.success)

        return Trajectory(
            task_id=final_state.task_id,
            domain=final_state.domain,
            difficulty=final_state.difficulty,
            steps=copy.deepcopy(self.steps),
            final_state=final_state.snapshot(),
            termination_reason=termination_reason,
            termination_details=copy.deepcopy(termination_details),
            final_score=final_state.final_score,
            total_time=total_time,
            success=success,
        )

    def reset(self) -> None:
        """Reset recorder for new episode."""
        self.steps = []
        self.start_time = datetime.now()


def _parse_llm_response(response: str) -> Action:
    """Parse LLM response text into typed Action with improved format handling.

    Supports formats (priority order):
    1. Tool call: "Tool: exec\nCode: ..." or "Tool: eval\nInput: ..."
    2. Final answer: "Final Answer:\n..."
    3. Fallback: Markdown detection (LLM ignored format) -> wrap appropriately

    Args:
        response: Raw LLM response text

    Returns:
        Parsed Action (ToolCallAction or FinalAnswerAction)
    """
    cleaned_response = re.sub(r"</think_never_used_[^>]+>", "", response)
    cleaned_response_stripped = cleaned_response.strip()

    # 1. Check for tool call FIRST.
    # If the LLM mixes Tool and Final Answer in one response, execute the first tool only.
    # The model will see the real observation and can provide Final Answer on the next turn.

    # Check for tool call - supports both formats:
    #    Format A: "Tool: eval\nInput: 1+1"
    #    Format B: "Tool: eval Input: 1+1" (same line)
    tool_match = re.search(r"tool\s*:\s*(\w+)", cleaned_response, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).lower().strip()
        # Try matching "Code:" or "Input:" after tool match
        after_tool = cleaned_response[tool_match.end() :]
        code_match = re.search(r"(?:code|input)\s*:\s*(.*)", after_tool, re.IGNORECASE | re.DOTALL)
        if code_match:
            code_raw = code_match.group(1).strip()
            next_marker = re.search(
                r"(Tool\s*:|Final\s+Answer\s*:|Output\s*:|Observation\s*:|User\s*:|Assistant\s*:|Step\s+\d+\s+result\s*:)",
                code_raw,
                re.IGNORECASE,
            )
            code_content = code_raw[: next_marker.start()].strip() if next_marker else code_raw
            code_content = re.sub(r"</think_never_used_[^>]+>", "", code_content).strip()
        else:
            code_content = after_tool.strip().split("\n")[0]
            code_content = re.sub(r"</think_never_used_[^>]+>", "", code_content).strip()

        if tool_name == "eval":
            first_line = next((line.strip() for line in code_content.splitlines() if line.strip()), "")
            arithmetic = re.match(r"[\d\s+\-*/().,%]+", first_line)
            code_content = arithmetic.group(0).strip() if arithmetic and arithmetic.group(0).strip() else first_line
        elif tool_name == "exec":
            fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", code_content, flags=re.IGNORECASE | re.DOTALL)
            if fenced:
                code_content = fenced[0].strip()
            else:
                code_content = code_content.replace("```python", "").replace("```", "").strip()
        thought_end = tool_match.start()
        thought = cleaned_response[:thought_end].strip()
        return ToolCallAction(
            action_type="tool_call",
            name=tool_name,
            kwargs={"expr" if tool_name == "eval" else "code": code_content},
            thought=thought if thought else None,
        )

    # 2. Check for final answer only when no tool call is present
    final_answer_match = re.search(r"final\s*answer\s*:\s*(.*)", cleaned_response, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        answer_content = final_answer_match.group(1).strip()
        thought_end = final_answer_match.start()
        thought = cleaned_response[:thought_end].strip()
        return FinalAnswerAction(
            action_type="final_answer",
            answer=answer_content,
            thought=thought if thought else None,
        )

    # 3. Check for markdown patterns - LLM ignored format instructions!
    # Common patterns: "###", "1.", bullet points, etc.
    looks_like_answer_directly = (
        cleaned_response_stripped.startswith("###")
        or cleaned_response_stripped.startswith("#")
        or cleaned_response_stripped.startswith("1.")
        or cleaned_response_stripped.startswith("- ")
        or "###" in cleaned_response_stripped[:100]
        or (len(cleaned_response_stripped) > 200 and "\n" in cleaned_response_stripped)  # Multi-line content = probably answer
    )

    if looks_like_answer_directly:
        # LLM output full answer directly without format marker
        return FinalAnswerAction(
            action_type="final_answer",
            answer=cleaned_response_stripped,
            thought=None,
        )

    # 4. Final fallback: arithmetic expression -> eval, short code-like content -> exec, long content -> final answer
    if re.fullmatch(r"[\d\s+\-*/().,%]+", cleaned_response_stripped) and len(cleaned_response_stripped) < 150:
        return ToolCallAction(
            action_type="tool_call",
            name="eval",
            kwargs={"expr": cleaned_response_stripped},
            thought=None,
        )
    if len(cleaned_response_stripped) < 150:
        return ToolCallAction(
            action_type="tool_call",
            name="exec",
            kwargs={"code": cleaned_response_stripped},
            thought=None,
        )
    return FinalAnswerAction(
        action_type="final_answer",
        answer=cleaned_response_stripped,
        thought=None,
    )


def _build_conversation_prompt(state: AgentState, task_prompt: str) -> List[Dict[str, str]]:
    """Build conversation history prompt from agent state.

    Args:
        state: Current agent state
        task_prompt: Original task description

    Returns:
        List of messages in OpenAI format
    """
    messages = []

    # System message - domain-specific strategies with strict format enforcement
    if state.domain == "math_reasoning":
        system_content = (
            f"=== MATH REASONING AGENT - STRICTEST PROTOCOL ===\n"
            "\n"
            "YOU MUST FOLLOW THIS EXACTLY - NO DEVIATIONS:\n"
            "\n"
            "1. ONE ACTION PER MESSAGE. ONE TOOL CALL PER RESPONSE.\n"
            "2. DO NOT CALL MULTIPLE TOOLS IN ONE MESSAGE.\n"
            "3. DO NOT SIMULATE OR GUESS TOOL OUTPUT. NEVER WRITE 'Output:'.\n"
            "   The environment will give you the output after you call the tool.\n"
            "4. AFTER SEEING AT LEAST ONE ENVIRONMENT RESPONSE, you SHOULD output Final Answer when you have enough information.\n"
            "\n"
            "=== TOOL CALL FORMAT ===\n"
            "Tool: eval\n"
            "Input: <your single arithmetic expression>\n"
            "\n"
            "=== FINAL ANSWER FORMAT ===\n"
            "Final Answer:\n"
            "<your solution>\n"
            "\n"
            "=== CORRECT DIALOGUE EXAMPLE ===\n"
            "Assistant: Tool: eval\n"
            "           Input: 250 / 10\n"
            "\n"
            "User: Step result: 25\n"
            "\n"
            "Assistant: Tool: eval\n"
            "           Input: 200 + 25\n"
            "\n"
            "User: Step result: 225\n"
            "\n"
            "Assistant: Final Answer:\n"
            "           The answer is 225.\n"
            "\n"
            "=== WHAT NOT TO DO ===\n"
            "- DO NOT write multiple Tool: calls in one message\n"
            "- DO NOT write 'Output:' - that is the environment's job\n"
            "- DO NOT write 'Final Answer' on the first step before any tool result\n"
            "- DO NOT write markdown headers\n"
            "\n"
            "- DO NOT write Python functions\n"
            "\n"
            "Solve step by step. Use Tool: eval for simple arithmetic computations.\n"
            "\n"
            f"Task: {task_prompt}"
        )
    elif state.domain == "multi_step_planning":
        system_content = (
            f"You are a planning expert solving a {state.domain} task.\n"
            "\n"
            "Do NOT use any tools. This is a reasoning task, not a coding task.\n"
            "Think through the requirements carefully.\n"
            "Create a structured, detailed plan with:\n"
            "- Ordered steps with dependencies\n"
            "- Timeline estimates\n"
            "- Risk mitigations\n"
            "\n"
            "Format instructions:\n"
            "Submit your complete plan as 'Final Answer:\\n<your plan>'\n"
            "\n"
            f"Task: {task_prompt}"
        )
    elif state.domain == "api_orchestration":
        system_content = (
            f"You are an API integration expert solving a {state.domain} task.\n"
            "\n"
            "Map out the API endpoint sequence, handling authentication and errors.\n"
            "Use exec ONLY to validate JSON structures or request payloads.\n"
            "\n"
            "Available tools:\n"
            "- exec: Execute Python code to validate JSON structures\n"
            "- eval: Evaluate simple expressions\n"
            "\n"
            "Format instructions:\n"
            "For tool calls: 'Tool: exec\\nCode: <your code>'\n"
            "For final answer: 'Final Answer:\\n<your complete API sequence>'\n"
            "\n"
            f"Task: {task_prompt}"
        )
    else:
        # Default: code_debug, coding tasks - use full toolset
        system_content = (
            f"You are a debugging and coding expert solving a {state.domain} task.\n"
            "\n"
            "=== STRICT ACTION PROTOCOL ===\n"
            "You must output exactly ONE action per message.\n"
            "Do NOT combine Tool calls with Final Answer in the same response.\n"
            "Do NOT simulate tool output. Never write 'Output:' yourself.\n"
            "The environment will execute your code and return the observation.\n"
            "\n"
            "Available tools:\n"
            "- exec: Execute Python code.\n"
            "- eval: Evaluate a Python expression.\n"
            "- check_solution: Verify your solution without ending the task.\n"
            "\n"
            "Tool call format:\n"
            "Tool: exec\n"
            "Code: <one Python script to run>\n"
            "\n"
            "Final answer format, only after enough observations:\n"
            "Final Answer:\n"
            "<your final fixed code and explanation>\n"
            "\n"
            f"Task: {task_prompt}"
        )
    messages.append({"role": "system", "content": system_content})

    # Build conversation history from state
    for i, (action, obs) in enumerate(zip(state.action_history, state.observation_history)):
        # Assistant message
        if action.action_type == "tool_call":
            tool_input = action.kwargs.get("code") or action.kwargs.get("expr") or action.kwargs.get("input") or ""
            input_label = "Input" if action.name == "eval" else "Code"
            content = f"Tool: {action.name}\n{input_label}: {tool_input}"
        else:
            content = f"Final Answer: {action.answer}"
        if action.thought:
            content = f"{action.thought}\n\n{content}"
        messages.append({"role": "assistant", "content": content})

        # User message (observation)
        if obs.success:
            user_content = f"Step {i + 1} result: {obs.content}"
        else:
            user_content = f"Step {i + 1} error: {obs.error or 'Unknown error'}"
        messages.append({"role": "user", "content": user_content})

    return messages


class AgentLoop:
    """Main agent harness orchestration loop.

    Drives complete agent-environment interaction:
    Task -> Observation -> LLM -> Action -> Sandbox -> Observation -> ... -> Recorded Trajectory
    """

    def __init__(
        self,
        env: Any,  # Duck typed Environment
        llm_client: Any,  # Duck typed LLM client
        max_steps: int = 10,
        token_budget: int = 100000,
        loop_detection_threshold: int = 3,
        save_failed_steps: bool = True,
    ):
        self.env = env
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.token_budget = token_budget

        # Core components
        self.termination_detector = TerminationDetector(loop_detection_threshold)
        self.recorder = TrajectoryRecorder(save_failed_steps)

    async def run(
        self,
        task: Dict[str, Any],
    ) -> Trajectory:
        """Run complete agent-environment interaction loop.

        Flow:
        1. Reset environment and initialize AgentState
        2. While not terminated:
           a. Build prompt from state history
           b. Generate action via LLM
           c. Execute action via env.step()
           d. Update state (immutable)
           e. Check termination
           f. Record step
        3. Finalize trajectory and return

        Args:
            task: Task to solve (SeedPrompt, Task, or dict)

        Returns:
            Complete trajectory with all steps and metadata
        """
        # Reset components
        self.termination_detector.reset()
        self.recorder.reset()

        # Extract task data
        if isinstance(task, dict):
            task_id = task.get("id", str(hash(task.get("prompt", ""))))
            domain = task.get("domain", "unknown")
            difficulty = task.get("difficulty", "medium")
            task_prompt = task.get("prompt", "")
        else:
            task_id = task.id
            domain = task.domain
            difficulty = task.difficulty
            task_prompt = task.prompt

        # Reset environment
        initial_obs = self.env.reset(task if isinstance(task, dict) else task.model_dump())

        # Initialize state
        state = AgentState(
            task_id=task_id,
            domain=domain,
            difficulty=difficulty,
            step=0,
            max_steps=self.max_steps,
            done=False,
            token_budget=self.token_budget,
            test_cases_total=initial_obs.get("test_cases_count", 0),
            metadata={"task_prompt": task_prompt},
        )

        # Initial observation
        observation = Observation(
            success=True,
            content=f"Task started. {initial_obs.get('prompt', '')}",
            metadata={"initial_observation": initial_obs},
        )

        # Main loop
        termination = TerminationDecision(should_terminate=False, reason="continue", details={})

        while not termination.should_terminate:
            # 1. Build conversation prompt
            messages = _build_conversation_prompt(state, task_prompt)

            # 2. Generate action via LLM
            llm_response, token_usage = await self._call_llm(messages)

            # 3. Parse LLM cleaned_response into action
            action = _parse_llm_response(llm_response)

            # 4. Execute action in environment
            step_result = await self.env.step(action)
            observation = step_result.observation

            # 5. Update state (immutable - creates new instance)
            state = state.update(action, observation, step_result, token_usage)

            # 6. Check termination
            termination = self.termination_detector.check(state)

            # 7. Record step
            self.recorder.record_step(state, action, observation, token_usage)

            # Update done flag in state if terminating
            if termination.should_terminate and not state.done:
                # Create final state with termination info
                state = AgentState(
                    task_id=state.task_id,
                    domain=state.domain,
                    difficulty=state.difficulty,
                    step=state.step,
                    max_steps=state.max_steps,
                    done=True,
                    termination_reason=termination.reason,
                    last_action=state.last_action,
                    last_observation=state.last_observation,
                    observation_history=state.observation_history,
                    action_history=state.action_history,
                    step_results=state.step_results,
                    token_usage=state.token_usage,
                    total_tokens=state.total_tokens,
                    token_budget=state.token_budget,
                    test_cases_passed=state.test_cases_passed,
                    test_cases_total=state.test_cases_total,
                    final_score=termination.score if termination.score is not None else state.final_score,
                    metadata={**state.metadata, "termination_details": termination.details},
                    created_at=datetime.now(),
                )

        # Finalize trajectory
        return self.recorder.finalize(
            final_state=state,
            termination_reason=termination.reason,
            termination_details=termination.details,
        )

    async def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, int]]]:
        """Call LLM client with graceful fallback.

        Supports both:
        - VLLMClient.achat() -> returns str
        - AnthropicClient.chat() -> returns str or tuple

        Args:
            messages: Conversation messages

        Returns:
            Tuple of (response text, token usage dict)
        """
        try:
            # Try async method first (achat), fallback to sync chat
            if hasattr(self.llm_client, "achat"):
                # VLLMClient style - returns str only
                response = await self.llm_client.achat(
                    model=getattr(self.llm_client, "model", "default"),
                    messages=messages,
                )
            else:
                # Other clients
                response = await self.llm_client.chat(messages=messages)

            # Handle different response formats
            if isinstance(response, tuple):
                return response[0], response[1] if len(response) > 1 else None
            if hasattr(response, "content"):
                content = response.content
                usage = getattr(response, "usage", None)
                return content, usage.model_dump() if usage else None
            if isinstance(response, str):
                return response, {"total_tokens": len(response) // 4}  # Rough estimate

            return str(response), None

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e
