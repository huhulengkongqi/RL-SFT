import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .checkpoint import CheckpointManager, TaskStatus
from .queue import BaseQueue, QueueMessage

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    total_time: float = 0.0

    @property
    def avg_time(self) -> float:
        if self.processed == 0:
            return 0.0
        return self.total_time / self.processed


class BaseWorker(ABC):
    """Abstract base class for pipeline workers."""

    def __init__(
        self,
        queue: BaseQueue,
        checkpoint: CheckpointManager,
        stage: str,
        input_stream: str,
        output_stream: Optional[str] = None,
        worker_id: Optional[str] = None,
        max_retries: int = 3,
        idle_timeout: Optional[float] = 5.0,
    ):
        self.queue = queue
        self.checkpoint = checkpoint
        self.stage = stage
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.worker_id = worker_id or f"worker_{stage}_{uuid.uuid4().hex[:8]}"
        self.max_retries = max_retries
        self.idle_timeout = idle_timeout
        self.stats = WorkerStats()
        self._running = False
        self._group = f"{stage}_workers"
        self._idle_start: Optional[float] = None

    async def start(self) -> None:
        """Start the worker loop."""
        self._running = True
        logger.info(f"Worker {self.worker_id} starting on stream {self.input_stream}")
        await self.queue.ensure_group(self.input_stream, self._group)
        await self.run_forever()

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        logger.info(f"Worker {self.worker_id} stopping")

    async def run_forever(self) -> None:
        """Run the worker main loop."""
        import time
        while self._running:
            try:
                messages = await self.queue.pull(
                    stream=self.input_stream,
                    group=self._group,
                    consumer=self.worker_id,
                    count=1,
                    timeout=1000,
                )

                if messages:
                    self._idle_start = None
                    for msg in messages:
                        await self._process_message(msg)
                else:
                    if self._idle_start is None:
                        self._idle_start = time.time()
                    elif self.idle_timeout and (time.time() - self._idle_start) > self.idle_timeout:
                        logger.info(f"Worker {self.worker_id} idle for {self.idle_timeout}s, exiting")
                        break

            except asyncio.CancelledError:
                logger.info(f"Worker {self.worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_message(self, msg: QueueMessage) -> None:
        """Process a single message."""
        import time

        task_id = msg.data.get("task_id", msg.id)
        start_time = time.time()

        logger.debug(f"Worker {self.worker_id} processing task {task_id}")

        if self.checkpoint.is_completed(self.stage, task_id):
            logger.debug(f"Task {task_id} already completed, skipping")
            await self.queue.ack(msg.stream, msg.group or self._group, msg.id)
            self.checkpoint.mark_skipped(self.stage, task_id, "already_completed")
            return

        self.checkpoint.mark_running(self.stage, task_id, input_data=msg.data)

        try:
            retries = msg.data.get("_retries", 0)

            result = await self.process_one(msg.data)

            self.checkpoint.mark_completed(self.stage, task_id, output_data=result)

            if self.output_stream:
                output_msg = {
                    "task_id": task_id,
                    "input_task_id": task_id,
                    **result,
                }
                await self.queue.push(self.output_stream, output_msg)

            await self.queue.ack(msg.stream, msg.group or self._group, msg.id)

            self.stats.succeeded += 1
            logger.debug(f"Worker {self.worker_id} completed task {task_id}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Worker {self.worker_id} failed task {task_id}: {error_msg}", exc_info=True)

            retries = msg.data.get("_retries", 0)
            if retries < self.max_retries:
                msg.data["_retries"] = retries + 1
                await self.queue.nack(msg.stream, msg.group or self._group, msg.id)
                logger.info(f"Task {task_id} will be retried ({retries + 1}/{self.max_retries})")
            else:
                self.checkpoint.mark_failed(self.stage, task_id, error_msg, retries)
                await self.queue.ack(msg.stream, msg.group or self._group, msg.id)
                self.stats.failed += 1

        finally:
            self.stats.processed += 1
            self.stats.total_time += time.time() - start_time

    @abstractmethod
    async def process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single work item.

        Args:
            item: Input data from queue.

        Returns:
            Result dict to be passed to next stage.

        Raises:
            Exception: If processing fails.
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "stage": self.stage,
            "processed": self.stats.processed,
            "succeeded": self.stats.succeeded,
            "failed": self.stats.failed,
            "avg_time_seconds": self.stats.avg_time,
        }
