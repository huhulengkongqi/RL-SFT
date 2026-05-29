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
    LayeredTemperatureConfig,
    TerminationDecision,
    TerminationDetector,
    Trajectory,
    TrajectoryFormat,
    TrajectoryRecorder,
    TrajectoryStep,
)
from agent_sft.trajectory_sampler.trajectory_sample import (
    SamplingConfig,
    TrajectorySampleResult,
    sample_one_trajectory,
    sample_trajectories,
    select_best_trajectory,
    summarize_sample_results,
)

__all__ = [
    "AgentLoop",
    "AgentState",
    "LayeredTemperatureConfig",
    "SamplingConfig",
    "TerminationDecision",
    "TerminationDetector",
    "Trajectory",
    "TrajectoryFormat",
    "TrajectoryRecorder",
    "TrajectorySampleResult",
    "TrajectoryStep",
    "sample_one_trajectory",
    "sample_trajectories",
    "select_best_trajectory",
    "summarize_sample_results",
]

