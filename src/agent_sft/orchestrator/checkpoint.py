import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskCheckpoint:
    task_id: str
    stage: str
    status: TaskStatus
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskCheckpoint":
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


def _make_key(stage: str, task_id: str) -> str:
    """Make a unique key for checkpoint storage.

    Ensures that the same task_id in different stages doesn't overwrite each other.
    """
    return f"{stage}:{task_id}"


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, run_id: str, resume: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_id = run_id
        self.resume = resume
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{run_id}.jsonl"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: Dict[str, TaskCheckpoint] = {}
        self._completed_tasks: Dict[str, Set[str]] = {}

        if self.resume and self.checkpoint_file.exists():
            self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Load existing checkpoints from file."""
        with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cp = TaskCheckpoint.from_dict(data)
                    # Use stage:task_id as key to avoid overwriting
                    self._checkpoints[_make_key(cp.stage, cp.task_id)] = cp

                    if cp.stage not in self._completed_tasks:
                        self._completed_tasks[cp.stage] = set()

                    if cp.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED):
                        self._completed_tasks[cp.stage].add(cp.task_id)
                except json.JSONDecodeError:
                    continue

    def _write_checkpoint(self, cp: TaskCheckpoint) -> None:
        """Append checkpoint to file."""
        with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cp.to_dict(), ensure_ascii=False) + '\n')

    def get(self, stage: str, task_id: str) -> Optional[TaskCheckpoint]:
        """Get checkpoint for a task in a specific stage."""
        return self._checkpoints.get(_make_key(stage, task_id))

    def is_completed(self, stage: str, task_id: str) -> bool:
        """Check if a task in a stage is already completed."""
        if stage not in self._completed_tasks:
            return False
        return task_id in self._completed_tasks[stage]

    def get_completed_tasks(self, stage: str) -> Set[str]:
        """Get all completed task IDs for a stage."""
        return self._completed_tasks.get(stage, set())

    def mark_running(self, stage: str, task_id: str, input_data: Optional[Dict[str, Any]] = None) -> TaskCheckpoint:
        """Mark a task as running."""
        key = _make_key(stage, task_id)
        cp = TaskCheckpoint(
            task_id=task_id,
            stage=stage,
            status=TaskStatus.RUNNING,
            input_data=input_data,
            started_at=datetime.now().isoformat(),
        )
        self._checkpoints[key] = cp
        self._write_checkpoint(cp)
        return cp

    def mark_completed(self, stage: str, task_id: str, output_data: Optional[Dict[str, Any]] = None) -> TaskCheckpoint:
        """Mark a task as completed."""
        key = _make_key(stage, task_id)
        cp = self._checkpoints.get(key)
        if cp is None:
            cp = TaskCheckpoint(
                task_id=task_id,
                stage=stage,
                status=TaskStatus.COMPLETED,
                started_at=datetime.now().isoformat(),
            )

        cp.status = TaskStatus.COMPLETED
        cp.output_data = output_data
        cp.completed_at = datetime.now().isoformat()

        self._checkpoints[key] = cp
        self._write_checkpoint(cp)

        if stage not in self._completed_tasks:
            self._completed_tasks[stage] = set()
        self._completed_tasks[stage].add(task_id)

        return cp

    def mark_failed(self, stage: str, task_id: str, error: str, retries: int = 0) -> TaskCheckpoint:
        """Mark a task as failed."""
        key = _make_key(stage, task_id)
        cp = self._checkpoints.get(key)
        if cp is None:
            cp = TaskCheckpoint(
                task_id=task_id,
                stage=stage,
                status=TaskStatus.FAILED,
                started_at=datetime.now().isoformat(),
            )

        cp.status = TaskStatus.FAILED
        cp.error = error
        cp.retries = retries
        cp.completed_at = datetime.now().isoformat()

        self._checkpoints[key] = cp
        self._write_checkpoint(cp)

        return cp

    def mark_skipped(self, stage: str, task_id: str, reason: str = "duplicate") -> TaskCheckpoint:
        """Mark a task as skipped."""
        key = _make_key(stage, task_id)
        cp = TaskCheckpoint(
            task_id=task_id,
            stage=stage,
            status=TaskStatus.SKIPPED,
            error=reason,
            completed_at=datetime.now().isoformat(),
        )
        self._checkpoints[key] = cp
        self._write_checkpoint(cp)

        if stage not in self._completed_tasks:
            self._completed_tasks[stage] = set()
        self._completed_tasks[stage].add(task_id)

        return cp

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics per stage."""
        stats: Dict[str, Dict[str, int]] = {}
        for cp in self._checkpoints.values():
            if cp.stage not in stats:
                stats[cp.stage] = {
                    TaskStatus.PENDING.value: 0,
                    TaskStatus.RUNNING.value: 0,
                    TaskStatus.COMPLETED.value: 0,
                    TaskStatus.FAILED.value: 0,
                    TaskStatus.SKIPPED.value: 0,
                    "total": 0,
                }
            stats[cp.stage][cp.status.value] += 1
            stats[cp.stage]["total"] += 1
        return stats

    def get_failed_tasks(self) -> List[TaskCheckpoint]:
        """Get all failed tasks."""
        return [cp for cp in self._checkpoints.values() if cp.status == TaskStatus.FAILED]
