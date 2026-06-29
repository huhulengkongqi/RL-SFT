import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PipelineStageConfig:
    enabled: bool = True
    workers: int = 4


@dataclass
class SeedGenerationConfig(PipelineStageConfig):
    seed_file: str = "data/final_seed_pool_181_real.json"


@dataclass
class EvolutionConfig(PipelineStageConfig):
    generations: int = 4
    evolutions_per_seed: int = 3
    use_claude: bool = False
    claude_model: str = "ark-code-latest"
    min_sleep: float = 10.0
    max_sleep: float = 18.0


@dataclass
class TrajectoryGenerationConfig(PipelineStageConfig):
    max_steps: int = 20
    sleep_min: float = 10.0
    sleep_max: float = 18.0
    model: str = "ark-code-latest"
    sandbox_pool_size: int = 8


@dataclass
class QualityFilterConfig(PipelineStageConfig):
    level2_prm: bool = True
    level2_judge: bool = True
    level3_dedup: bool = True
    level4_sampling: bool = True
    her_relabeling: bool = True
    target_count: int = 1500


@dataclass
class DatasetBuildConfig(PipelineStageConfig):
    target_size: int = 1500
    token_shaping: bool = True
    token_min: int = 512
    token_max: int = 3000
    export_format: str = "both"
    validate: bool = True


@dataclass
class FlywheelConfig:
    enabled: bool = False
    max_iterations: int = 3
    teacher_model_version: str = "v1"
    min_improvement: float = 0.05


@dataclass
class QueueConfig:
    use_redis: bool = True
    redis_url: str = "redis://localhost:6379/0"
    stream_max_len: int = 10000
    poll_interval: float = 0.5


@dataclass
class PipelineConfig:
    name: str = "sft_v1"
    version: str = "1.0.0"
    resume: bool = True
    checkpoint_dir: str = "data/checkpoints/"
    output_dir: str = "data/pipeline_output/"

    queue: QueueConfig = field(default_factory=QueueConfig)
    flywheel: FlywheelConfig = field(default_factory=FlywheelConfig)

    seed_generation: SeedGenerationConfig = field(default_factory=SeedGenerationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    trajectory_generation: TrajectoryGenerationConfig = field(default_factory=TrajectoryGenerationConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    dataset_build: DatasetBuildConfig = field(default_factory=DatasetBuildConfig)

    run_id: str = field(default_factory=lambda: f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "version": self.version,
            "resume": self.resume,
            "checkpoint_dir": self.checkpoint_dir,
            "output_dir": self.output_dir,
            "run_id": self.run_id,
            "queue": {
                "use_redis": self.queue.use_redis,
                "redis_url": self.queue.redis_url,
                "stream_max_len": self.queue.stream_max_len,
                "poll_interval": self.queue.poll_interval,
            },
            "flywheel": {
                "enabled": self.flywheel.enabled,
                "max_iterations": self.flywheel.max_iterations,
                "teacher_model_version": self.flywheel.teacher_model_version,
                "min_improvement": self.flywheel.min_improvement,
            },
            "stages": {
                "seed_generation": {
                    "enabled": self.seed_generation.enabled,
                    "workers": self.seed_generation.workers,
                    "seed_file": self.seed_generation.seed_file,
                },
                "evolution": {
                    "enabled": self.evolution.enabled,
                    "workers": self.evolution.workers,
                    "generations": self.evolution.generations,
                    "evolutions_per_seed": self.evolution.evolutions_per_seed,
                    "use_claude": self.evolution.use_claude,
                    "claude_model": self.evolution.claude_model,
                    "min_sleep": self.evolution.min_sleep,
                    "max_sleep": self.evolution.max_sleep,
                },
                "trajectory_generation": {
                    "enabled": self.trajectory_generation.enabled,
                    "workers": self.trajectory_generation.workers,
                    "max_steps": self.trajectory_generation.max_steps,
                    "sleep_min": self.trajectory_generation.sleep_min,
                    "sleep_max": self.trajectory_generation.sleep_max,
                    "model": self.trajectory_generation.model,
                    "sandbox_pool_size": self.trajectory_generation.sandbox_pool_size,
                },
                "quality_filter": {
                    "enabled": self.quality_filter.enabled,
                    "workers": self.quality_filter.workers,
                    "level2_prm": self.quality_filter.level2_prm,
                    "level2_judge": self.quality_filter.level2_judge,
                    "level3_dedup": self.quality_filter.level3_dedup,
                    "level4_sampling": self.quality_filter.level4_sampling,
                    "her_relabeling": self.quality_filter.her_relabeling,
                    "target_count": self.quality_filter.target_count,
                },
                "dataset_build": {
                    "enabled": self.dataset_build.enabled,
                    "workers": self.dataset_build.workers,
                    "target_size": self.dataset_build.target_size,
                    "token_shaping": self.dataset_build.token_shaping,
                    "token_min": self.dataset_build.token_min,
                    "token_max": self.dataset_build.token_max,
                    "export_format": self.dataset_build.export_format,
                    "validate": self.dataset_build.validate,
                },
            },
        }
        return result


def _dict_to_dataclass(data: Dict[str, Any], dc: Any) -> Any:
    """Recursively convert dict to dataclass."""
    if not isinstance(data, dict):
        return data

    kwargs = {}
    for f in getattr(dc, '__dataclass_fields__', {}).keys():
        if f in data:
            field_type = dc.__dataclass_fields__[f].type
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[f] = _dict_to_dataclass(data[f], field_type)
            else:
                kwargs[f] = data[f]

    return dc(**kwargs)


def load_config(config_path: str, override_run_id: Optional[str] = None) -> PipelineConfig:
    """Load YAML config file and return PipelineConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    pipeline = data.get('pipeline', {})
    queue = data.get('queue', {})
    flywheel = data.get('flywheel', {})
    stages = data.get('stages', {})

    config = PipelineConfig(
        name=pipeline.get('name', 'sft_v1'),
        version=pipeline.get('version', '1.0.0'),
        resume=pipeline.get('resume', True),
        checkpoint_dir=pipeline.get('checkpoint_dir', 'data/checkpoints/'),
        output_dir=pipeline.get('output_dir', 'data/pipeline_output/'),
    )

    if override_run_id:
        config.run_id = override_run_id

    for key, value in queue.items():
        if hasattr(config.queue, key):
            setattr(config.queue, key, value)

    for key, value in flywheel.items():
        if hasattr(config.flywheel, key):
            setattr(config.flywheel, key, value)

    stage_configs = {
        'seed_generation': config.seed_generation,
        'evolution': config.evolution,
        'trajectory_generation': config.trajectory_generation,
        'quality_filter': config.quality_filter,
        'dataset_build': config.dataset_build,
    }

    for stage_name, stage_data in stages.items():
        if stage_name in stage_configs:
            sc = stage_configs[stage_name]
            for key, value in stage_data.items():
                if hasattr(sc, key):
                    setattr(sc, key, value)

    return config


def save_config(config: PipelineConfig, output_path: str) -> None:
    """Save config to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
