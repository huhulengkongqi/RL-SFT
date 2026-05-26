"""Trajectory sampling / Agent Harness module.

Core components:
- AgentState: Immutable state container with deep copy snapshots
- TerminationDetector: 5 termination conditions detection
- TrajectoryRecorder: Step recording with failure preservation
- Trajectory: Complete interaction trajectory (SFT ready)
- AgentLoop: Main orchestration harness
"""

from agent_sft.trajectory_sampler.agent_loop import (
    AgentLoop,
    AgentState,
    TerminationDecision,
    TerminationDetector,
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStep,
)

__all__ = [
    "AgentLoop",
    "AgentState",
    "TerminationDecision",
    "TerminationDetector",
    "Trajectory",
    "TrajectoryRecorder",
    "TrajectoryStep",
]

