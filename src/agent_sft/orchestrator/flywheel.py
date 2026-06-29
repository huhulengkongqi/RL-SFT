import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class FlywheelIteration:
    iteration: int
    teacher_model: str
    dataset_path: str
    metrics: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None


class DataFlywheel:
    """Incremental data flywheel for continuous improvement.

    After each iteration:
    1. The generated SFT data trains a new version of the model
    2. The new model becomes the teacher for next iteration
    3. Quality metrics are compared to decide whether to continue
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.iterations: List[FlywheelIteration] = []
        self.output_dir = Path(config.output_dir) / "flywheel"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> bool:
        """Run the flywheel loop for configured iterations."""
        if not self.config.flywheel.enabled:
            logger.info("Flywheel is disabled in config")
            return True

        logger.info(f"Starting data flywheel with max {self.config.flywheel.max_iterations} iterations")

        current_model = self.config.flywheel.teacher_model_version
        best_metrics: Optional[Dict[str, Any]] = None

        for iteration in range(1, self.config.flywheel.max_iterations + 1):
            logger.info(f"Starting flywheel iteration {iteration} with teacher model {current_model}")

            iteration_config = self._create_iteration_config(iteration, current_model)
            orchestrator = PipelineOrchestrator(iteration_config)

            seed_file = self.config.seed_generation.seed_file
            seed_count = await orchestrator.load_seed_file(seed_file)
            logger.info(f"Seeded {seed_count} tasks for iteration {iteration}")

            iteration_obj = FlywheelIteration(
                iteration=iteration,
                teacher_model=current_model,
                dataset_path=str(orchestrator.output_dir),
                metrics={},
                started_at=datetime.now().isoformat(),
            )
            self.iterations.append(iteration_obj)

            success = await orchestrator.run()
            iteration_obj.completed_at = datetime.now().isoformat()

            if not success:
                logger.error(f"Iteration {iteration} failed")
                return False

            iteration_metrics = self._extract_metrics(orchestrator)
            iteration_obj.metrics = iteration_metrics

            improvement = self._calculate_improvement(iteration_metrics, best_metrics)
            logger.info(f"Iteration {iteration} improvement: {improvement:.4f}")

            if best_metrics is None or improvement > self.config.flywheel.min_improvement:
                best_metrics = iteration_metrics
                current_model = f"v{iteration + 1}"
                logger.info(f"New best model: {current_model}")
            else:
                logger.info(f"No significant improvement ({improvement:.4f} < {self.config.flywheel.min_improvement}), stopping")
                break

            self._save_state()

        logger.info(f"Flywheel completed after {len(self.iterations)} iterations")
        return True

    def _create_iteration_config(self, iteration: int, teacher_model: str) -> PipelineConfig:
        """Create config for a flywheel iteration."""
        import copy
        config = copy.deepcopy(self.config)
        config.run_id = f"flywheel_{iteration}_{config.run_id.split('_')[-1]}"
        config.flywheel.teacher_model_version = teacher_model
        return config

    def _extract_metrics(self, orchestrator: PipelineOrchestrator) -> Dict[str, Any]:
        """Extract quality metrics from completed pipeline."""
        summary_path = orchestrator.output_dir / "pipeline_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            checkpoint_stats = summary.get("checkpoint_stats", {})
            qf_stats = checkpoint_stats.get("quality_filter", {})
            total = qf_stats.get("total", 0)
            passed = qf_stats.get("completed", 0)

            return {
                "total_tasks": total,
                "quality_pass_rate": passed / total if total > 0 else 0.0,
            }

        return {"total_tasks": 0, "quality_pass_rate": 0.0}

    def _calculate_improvement(
        self,
        current_metrics: Dict[str, Any],
        best_metrics: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate improvement over best metrics."""
        if best_metrics is None:
            return 1.0

        current_pass_rate = current_metrics.get("quality_pass_rate", 0.0)
        best_pass_rate = best_metrics.get("quality_pass_rate", 0.0)

        if best_pass_rate == 0:
            return 1.0 if current_pass_rate > 0 else 0.0

        return (current_pass_rate - best_pass_rate) / best_pass_rate

    def _save_state(self) -> None:
        """Save flywheel state to file."""
        state = {
            "iterations": [
                {
                    "iteration": it.iteration,
                    "teacher_model": it.teacher_model,
                    "dataset_path": it.dataset_path,
                    "metrics": it.metrics,
                    "started_at": it.started_at,
                    "completed_at": it.completed_at,
                }
                for it in self.iterations
            ],
        }

        state_path = self.output_dir / "flywheel_state.json"
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.debug(f"Flywheel state saved to {state_path}")
