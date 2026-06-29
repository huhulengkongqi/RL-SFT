import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .checkpoint import CheckpointManager
from .config import PipelineConfig, save_config
from .dag import DAG, DAGStage, TaskState
from .queue import BaseQueue, create_queue

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Main pipeline orchestrator.

    Manages DAG execution, worker coordination, and pipeline state.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            run_id=config.run_id,
            resume=config.resume,
        )
        self.queue: BaseQueue = create_queue(
            use_redis=config.queue.use_redis,
            redis_url=config.queue.redis_url,
        )
        self.dag = DAG.build_default()
        self.workers: List[asyncio.Task] = []
        self._running = False
        self._worker_tasks: Dict[str, List[asyncio.Task]] = {}
        self._stage_tasks: Set[asyncio.Task] = set()

        self.output_dir = Path(config.output_dir) / config.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        save_config(config, self.output_dir / "pipeline_config.yaml")

        logger.info(f"Orchestrator initialized with run_id: {config.run_id}")

    async def run(self) -> bool:
        """Run the full pipeline."""
        self._running = True
        start_time = time.time()

        logger.info("Starting pipeline execution")
        logger.info(f"Pipeline config: {self.config.name} v{self.config.version}")

        try:
            while self._running and not self.dag.is_complete():
                ready_nodes = self.dag.get_ready_nodes()

                for node in ready_nodes:
                    if self._is_stage_enabled(node.stage):
                        await self._start_stage(node)
                    else:
                        logger.info(f"Skipping disabled stage: {node.stage}")
                        self.dag.mark_stage_completed(node.stage, {"skipped": True})

                await self._check_completed_stages()
                await asyncio.sleep(1.0)

                if self.dag.has_failed():
                    logger.error("Pipeline failed due to stage failure")
                    return False

            duration = time.time() - start_time
            logger.info(f"Pipeline completed in {duration:.2f} seconds")

            await self._generate_summary()
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False
        finally:
            await self._cleanup()

    def _is_stage_enabled(self, stage: DAGStage) -> bool:
        """Check if a stage is enabled in config."""
        stage_config = getattr(self.config, stage.value, None)
        if stage_config is None:
            return False
        return getattr(stage_config, "enabled", True)

    async def _start_stage(self, node: Any) -> None:
        """Start a pipeline stage."""
        logger.info(f"Starting stage: {node.stage}")
        node.mark_running()

        stage_config = getattr(self.config, node.stage.value)
        num_workers = getattr(stage_config, "workers", 4)

        stage_handler = self._get_stage_handler(node.stage)
        if stage_handler:
            task = asyncio.create_task(
                self._run_stage_with_workers(
                    stage=node.stage,
                    handler=stage_handler,
                    num_workers=num_workers,
                )
            )
            self._stage_tasks.add(task)
            task.add_done_callback(self._stage_tasks.discard)

    async def _run_stage_with_workers(
        self,
        stage: DAGStage,
        handler: Any,
        num_workers: int,
    ) -> None:
        """Run a stage with multiple worker coroutines."""
        logger.info(f"Starting {num_workers} workers for stage {stage}")

        worker_tasks = []
        for i in range(num_workers):
            worker = handler(worker_id=f"{stage.value}_worker_{i}", stage=stage)
            worker_task = asyncio.create_task(worker.start())
            worker_tasks.append(worker_task)

        try:
            await self._wait_for_stage_completion(stage, worker_tasks)
        finally:
            for t in worker_tasks:
                t.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _wait_for_stage_completion(
        self,
        stage: DAGStage,
        worker_tasks: List[asyncio.Task],
    ) -> None:
        """Wait for stage completion by monitoring queue lengths."""
        stage_config = getattr(self.config, stage.value)
        input_stream = self._get_input_stream(stage)
        output_stream = self._get_output_stream(stage)

        while True:
            pending = await self.queue.len(input_stream)
            processing = sum(
                1 for t in worker_tasks if not t.done() and not t.cancelled()
            )

            if pending == 0 and processing == 0:
                await asyncio.sleep(2.0)
                pending = await self.queue.len(input_stream)
                if pending == 0:
                    break

            await asyncio.sleep(1.0)

        stats = self.checkpoint.get_stats().get(stage.value, {})
        logger.info(f"Stage {stage} completed: {stats}")
        self.dag.mark_stage_completed(stage, stats)

    def _get_input_stream(self, stage: DAGStage) -> str:
        """Get the input stream name for a stage."""
        stream_map = {
            DAGStage.SEED_GENERATION: "tasks:new",
            DAGStage.EVOLUTION: "tasks:seeded",
            DAGStage.TRAJECTORY_GENERATION: "tasks:evolved",
            DAGStage.QUALITY_FILTER: "trajectories:new",
            DAGStage.DATASET_BUILD: "trajectories:filtered",
        }
        return stream_map.get(stage, f"{stage.value}:input")

    def _get_output_stream(self, stage: DAGStage) -> Optional[str]:
        """Get the output stream name for a stage."""
        stream_map = {
            DAGStage.SEED_GENERATION: "tasks:seeded",
            DAGStage.EVOLUTION: "tasks:evolved",
            DAGStage.TRAJECTORY_GENERATION: "trajectories:new",
            DAGStage.QUALITY_FILTER: "trajectories:filtered",
            DAGStage.DATASET_BUILD: None,
        }
        return stream_map.get(stage)

    def _get_stage_handler(self, stage: DAGStage) -> Optional[Any]:
        """Get the handler function for a stage."""
        from .workers import (
            create_evolution_worker,
            create_quality_filter_worker,
            create_seed_worker,
            create_trajectory_worker,
            create_dataset_builder,
        )

        handlers = {
            DAGStage.SEED_GENERATION: self._wrap_worker_creator(
                create_seed_worker, self.config.seed_generation
            ),
            DAGStage.EVOLUTION: self._wrap_worker_creator(
                create_evolution_worker, self.config.evolution
            ),
            DAGStage.TRAJECTORY_GENERATION: self._wrap_worker_creator(
                create_trajectory_worker, self.config.trajectory_generation
            ),
            DAGStage.QUALITY_FILTER: self._wrap_worker_creator(
                create_quality_filter_worker, self.config.quality_filter
            ),
            DAGStage.DATASET_BUILD: self._wrap_worker_creator(
                create_dataset_builder, self.config.dataset_build
            ),
        }
        return handlers.get(stage)

    def _wrap_worker_creator(self, creator: Any, stage_config: Any) -> Any:
        """Wrap worker creator to inject common dependencies."""
        def wrapped(worker_id: str, stage: DAGStage):
            return creator(
                queue=self.queue,
                checkpoint=self.checkpoint,
                config=stage_config,
                worker_id=worker_id,
                input_stream=self._get_input_stream(stage),
                output_stream=self._get_output_stream(stage),
            )
        return wrapped

    async def _check_completed_stages(self) -> None:
        """Check for and handle completed stages."""
        pass

    async def seed_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Seed initial tasks into the pipeline."""
        stream = self._get_input_stream(DAGStage.SEED_GENERATION)
        for task in tasks:
            if "task_id" not in task:
                task["task_id"] = task.get("id", f"task_{int(time.time() * 1000000)}")
            await self.queue.push(stream, task)
        logger.info(f"Seeded {len(tasks)} tasks into {stream}")

    async def load_seed_file(self, seed_file: str, limit: Optional[int] = None) -> int:
        """Load tasks from seed file.

        Args:
            seed_file: Path to seed JSON file
            limit: If set, limit to first N tasks
        """
        seed_path = Path(seed_file)
        if not seed_path.exists():
            logger.warning(f"Seed file not found: {seed_file}")
            return 0

        with open(seed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict):
            if "tasks" in data:
                tasks = data["tasks"]
            elif "prompts" in data:
                tasks = data["prompts"]
            else:
                tasks = [data]
        else:
            tasks = [data]

        if limit is not None and len(tasks) > limit:
            logger.info(f"Limiting seed tasks from {len(tasks)} to {limit}")
            tasks = tasks[:limit]

        await self.seed_tasks(tasks)
        return len(tasks)

    async def _generate_summary(self) -> None:
        """Generate pipeline execution summary."""
        summary = {
            "run_id": self.config.run_id,
            "pipeline_name": self.config.name,
            "pipeline_version": self.config.version,
            "completed_at": datetime.now().isoformat(),
            "stage_status": self.dag.get_status_summary(),
            "checkpoint_stats": self.checkpoint.get_stats(),
            "failed_tasks": [cp.to_dict() for cp in self.checkpoint.get_failed_tasks()],
        }

        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Pipeline summary written to {summary_path}")

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Run ID: {self.config.run_id}")
        print(f"Pipeline: {self.config.name} v{self.config.version}")
        print("\nStage Status:")
        for stage, status in self.dag.get_status_summary().items():
            stats = self.checkpoint.get_stats().get(stage, {})
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            print(f"  {stage:25s} {status:10s} completed={completed} failed={failed}")
        print("=" * 60 + "\n")

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False

        for task in self._stage_tasks:
            task.cancel()

        if self._stage_tasks:
            await asyncio.gather(*self._stage_tasks, return_exceptions=True)

        await self.queue.close()
