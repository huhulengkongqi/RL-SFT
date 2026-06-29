from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class TaskState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DAGStage(str, Enum):
    SEED_GENERATION = "seed_generation"
    EVOLUTION = "evolution"
    TRAJECTORY_GENERATION = "trajectory_generation"
    QUALITY_FILTER = "quality_filter"
    DATASET_BUILD = "dataset_build"


@dataclass
class DAGNode:
    stage: DAGStage
    state: TaskState = TaskState.IDLE
    dependencies: Set[DAGStage] = field(default_factory=set)
    handler: Optional[Callable[..., Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def mark_ready(self) -> None:
        self.state = TaskState.READY

    def mark_running(self) -> None:
        self.state = TaskState.RUNNING
        self.started_at = datetime.now().isoformat()

    def mark_completed(self, stats: Optional[Dict[str, Any]] = None) -> None:
        self.state = TaskState.COMPLETED
        self.completed_at = datetime.now().isoformat()
        if stats:
            self.stats.update(stats)

    def mark_failed(self, error: str) -> None:
        self.state = TaskState.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error = error

    def can_run(self, completed_stages: Set[DAGStage]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_stages)


class DAG:
    def __init__(self):
        self.nodes: Dict[DAGStage, DAGNode] = {}
        self.completed_stages: Set[DAGStage] = set()

    def add_node(
        self,
        stage: DAGStage,
        dependencies: Optional[List[DAGStage]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ) -> DAGNode:
        """Add a node to the DAG."""
        node = DAGNode(
            stage=stage,
            dependencies=set(dependencies) if dependencies else set(),
            handler=handler,
        )
        self.nodes[stage] = node
        return node

    def get_ready_nodes(self) -> List[DAGNode]:
        """Get all nodes that are ready to run (dependencies satisfied)."""
        ready = []
        for node in self.nodes.values():
            if node.state == TaskState.IDLE and node.can_run(self.completed_stages):
                node.mark_ready()
                ready.append(node)
        return ready

    def get_running_nodes(self) -> List[DAGNode]:
        """Get all currently running nodes."""
        return [n for n in self.nodes.values() if n.state == TaskState.RUNNING]

    def mark_stage_completed(self, stage: DAGStage, stats: Optional[Dict[str, Any]] = None) -> None:
        """Mark a stage as completed."""
        if stage in self.nodes:
            self.nodes[stage].mark_completed(stats)
            self.completed_stages.add(stage)

    def mark_stage_failed(self, stage: DAGStage, error: str) -> None:
        """Mark a stage as failed."""
        if stage in self.nodes:
            self.nodes[stage].mark_failed(error)

    def is_complete(self) -> bool:
        """Check if all enabled stages are completed."""
        for node in self.nodes.values():
            if node.state not in (TaskState.COMPLETED, TaskState.FAILED):
                return False
        return True

    def has_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(n.state == TaskState.FAILED for n in self.nodes.values())

    def get_status_summary(self) -> Dict[str, str]:
        """Get a summary of all stages' statuses."""
        return {stage.value: node.state.value for stage, node in self.nodes.items()}

    @classmethod
    def build_default(cls) -> "DAG":
        """Build the default pipeline DAG.

        Flow:
        seed_generation → evolution → trajectory_generation → quality_filter → dataset_build
        """
        dag = cls()
        dag.add_node(DAGStage.SEED_GENERATION, dependencies=[])
        dag.add_node(DAGStage.EVOLUTION, dependencies=[DAGStage.SEED_GENERATION])
        dag.add_node(DAGStage.TRAJECTORY_GENERATION, dependencies=[DAGStage.EVOLUTION])
        dag.add_node(DAGStage.QUALITY_FILTER, dependencies=[DAGStage.TRAJECTORY_GENERATION])
        dag.add_node(DAGStage.DATASET_BUILD, dependencies=[DAGStage.QUALITY_FILTER])
        return dag
