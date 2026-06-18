"""Dataset mixing, versioning, and manifest generation."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .data_formatter import ExportConfig, FormattedTrajectory, SFTDataPipeline, SFTFormatterConfig

SourceKind = Literal["raw_dir", "quality_report", "sft_json", "jsonl"]

DEFAULT_DOMAIN_CATEGORY_MAP = {
    "code_debug": "code",
    "math_reasoning": "reasoning",
    "api_orchestration": "tool_calling",
    "multi_step_planning": "general_instruction",
}


@dataclass
class DatasetSourceConfig:
    path: Path | str
    kind: SourceKind
    label: str | None = None
    limit: int | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.label is None:
            self.label = self.path.stem

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class DatasetMixConfig:
    target_size: int | None = None
    seed: int = 0
    general_share_min: float = 0.30
    general_share_max: float = 0.50
    category_ratio: dict[str, int] = field(default_factory=lambda: {"code": 3, "reasoning": 3, "tool_calling": 4})
    domain_category_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DOMAIN_CATEGORY_MAP))
    min_difficulty_share: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    dataset_name: str
    version: str
    created_at: str
    source_summaries: list[dict[str, Any]]
    selected_counts: dict[str, Any]
    ratio_gaps: list[dict[str, Any]]
    output_files: dict[str, str]
    config_hash: str
    git_sha: str | None = None
    metrics_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetBuildResult:
    raw_records: list[dict[str, Any]]
    formatted_records: list[FormattedTrajectory]
    export_paths: dict[str, Path]
    manifest: DatasetManifest
    manifest_path: Path
    ratio_gaps: list[dict[str, Any]]


class DatasetBuilder:
    def __init__(
        self,
        mix_config: DatasetMixConfig | None = None,
        pipeline: SFTDataPipeline | None = None,
        formatter_config: SFTFormatterConfig | None = None,
    ):
        self.mix_config = mix_config or DatasetMixConfig()
        self.pipeline = pipeline or SFTDataPipeline(formatter_config)

    @staticmethod
    def latest_quality_report(report_dir: Path | str = Path("data/quality_filter")) -> Path | None:
        reports = sorted(Path(report_dir).glob("quality_filter_report_*.json"), reverse=True)
        for report_path in reports:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if report.get("kept"):
                return report_path
        return None

    def load_sources(self, sources: list[DatasetSourceConfig]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for source in sorted(sources, key=lambda item: item.priority):
            loaded = self._load_source(source)
            records.extend(loaded)
            summaries.append({
                "label": source.label,
                "kind": source.kind,
                "path": str(source.path),
                "count": len(loaded),
                "limit": source.limit,
                "priority": source.priority,
            })
        return self._deduplicate(records), summaries

    def select_records(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        if not records:
            return [], {"total": 0, "by_category": {}, "by_domain": {}, "by_difficulty": {}}, []

        rng = random.Random(self.mix_config.seed)
        pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            category = self.category_for_domain(str(record.get("domain", "")))
            pools[category].append(record)
        for pool in pools.values():
            rng.shuffle(pool)

        target_size = min(self.mix_config.target_size or len(records), len(records))
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        gaps: list[dict[str, Any]] = []

        general_target = self._bounded_target(
            target_size,
            len(pools.get("general_instruction", [])),
            self.mix_config.general_share_min,
            self.mix_config.general_share_max,
        )
        self._take_from_pool("general_instruction", general_target, pools, selected, selected_ids, gaps)

        remaining = target_size - len(selected)
        ratio_targets = self._ratio_targets(remaining)
        for category, target in ratio_targets.items():
            self._take_from_pool(category, target, pools, selected, selected_ids, gaps)

        if len(selected) < target_size:
            leftovers = [record for pool in pools.values() for record in pool if self._record_key(record) not in selected_ids]
            rng.shuffle(leftovers)
            selected.extend(leftovers[: target_size - len(selected)])

        counts = self._count_selected(selected)
        return selected, counts, gaps

    def build(
        self,
        sources: list[DatasetSourceConfig],
        output_dir: Path | str,
        dataset_name: str = "sft_dataset",
        version: str | None = None,
        metrics_snapshot: dict[str, Any] | None = None,
    ) -> DatasetBuildResult:
        raw_records, source_summaries = self.load_sources(sources)
        selected, selected_counts, ratio_gaps = self.select_records(raw_records)
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{dataset_name}_{version}"
        formatted_records, export_paths = self.pipeline.process_batch(selected, output_dir=output_dir, dataset_name=export_name)
        manifest = DatasetManifest(
            dataset_name=dataset_name,
            version=version,
            created_at=datetime.now().isoformat(timespec="seconds"),
            source_summaries=source_summaries,
            selected_counts=selected_counts,
            ratio_gaps=ratio_gaps,
            output_files={key: str(path) for key, path in export_paths.items()},
            config_hash=self._config_hash(sources),
            git_sha=self._git_sha(),
            metrics_snapshot=metrics_snapshot,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / f"manifest_{version}.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return DatasetBuildResult(selected, formatted_records, export_paths, manifest, manifest_path, ratio_gaps)

    def category_for_domain(self, domain: str) -> str:
        return self.mix_config.domain_category_map.get(domain, "unknown")

    def _load_source(self, source: DatasetSourceConfig) -> list[dict[str, Any]]:
        if source.kind == "raw_dir":
            records = self._load_raw_dir(source.path)
        elif source.kind == "quality_report":
            records = self._load_quality_report(source.path)
        elif source.kind == "sft_json":
            records = self._load_json_array(source.path)
        elif source.kind == "jsonl":
            records = self._load_jsonl(source.path)
        else:
            raise ValueError(f"Unsupported source kind: {source.kind}")
        if source.limit is not None:
            return records[: source.limit]
        return records

    def _load_raw_dir(self, path: Path) -> list[dict[str, Any]]:
        records = []
        for raw_path in sorted(path.glob("*_raw.json")):
            record = self._load_raw_file(raw_path)
            if record is not None:
                records.append(record)
        return records

    def _load_quality_report(self, path: Path) -> list[dict[str, Any]]:
        report = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for entry in report.get("kept", []):
            raw_path_value = entry.get("raw_path")
            if not raw_path_value:
                continue
            raw_path = Path(raw_path_value)
            if not raw_path.is_absolute() and not raw_path.exists():
                candidate = path.parent / raw_path
                if candidate.exists():
                    raw_path = candidate
            record = self._load_raw_file(raw_path)
            if record is not None:
                record["_quality_metadata"] = {
                    "quality_score": entry.get("quality_score"),
                    "quality_score_source": entry.get("quality_score_source"),
                    "difficulty_bucket": entry.get("difficulty"),
                }
                records.append(record)
        return records

    def _load_raw_file(self, path: Path) -> dict[str, Any] | None:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if "steps" not in record:
            return None
        record.setdefault("_source_path", str(path))
        return record

    def _load_json_array(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return [item for item in data if isinstance(item, dict)]

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
        return records

    def _deduplicate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for record in records:
            key = self._record_key(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _record_key(record: dict[str, Any]) -> str:
        if record.get("_source_path"):
            return str(record["_source_path"])
        task_id = str(record.get("task_id", ""))
        total_time = str(record.get("total_time", ""))
        steps = record.get("steps") or []
        return f"{task_id}:{len(steps)}:{total_time}:{record.get('final_score')}"

    def _bounded_target(self, total: int, available: int, min_share: float, max_share: float) -> int:
        min_target = round(total * min_share)
        max_target = round(total * max_share)
        if available >= min_target:
            return min(max_target, available)
        return available

    def _ratio_targets(self, total: int) -> dict[str, int]:
        ratios = self.mix_config.category_ratio
        ratio_sum = sum(ratios.values()) or 1
        targets = {category: total * weight // ratio_sum for category, weight in ratios.items()}
        remainder = total - sum(targets.values())
        ordered = sorted(ratios, key=lambda category: (-ratios[category], category))
        for category in ordered[:remainder]:
            targets[category] += 1
        return targets

    def _take_from_pool(
        self,
        category: str,
        target: int,
        pools: dict[str, list[dict[str, Any]]],
        selected: list[dict[str, Any]],
        selected_ids: set[str],
        gaps: list[dict[str, Any]],
    ) -> None:
        pool = pools.get(category, [])
        available = [record for record in pool if self._record_key(record) not in selected_ids]
        take = min(target, len(available))
        for record in available[:take]:
            selected.append(record)
            selected_ids.add(self._record_key(record))
        if take < target:
            gaps.append({"category": category, "target": target, "available": len(available), "shortfall": target - take})

    def _count_selected(self, selected: list[dict[str, Any]]) -> dict[str, Any]:
        by_domain = Counter(str(record.get("domain", "unknown")) for record in selected)
        by_category = Counter(self.category_for_domain(str(record.get("domain", ""))) for record in selected)
        by_difficulty = Counter(str(record.get("difficulty", "unknown")) for record in selected)
        total = len(selected)
        return {
            "total": total,
            "by_domain": dict(by_domain),
            "by_category": dict(by_category),
            "by_difficulty": dict(by_difficulty),
            "category_share": {key: value / total for key, value in by_category.items()} if total else {},
        }

    def _config_hash(self, sources: list[DatasetSourceConfig]) -> str:
        payload = {
            "sources": [source.to_dict() for source in sources],
            "mix_config": self.mix_config.to_dict(),
            "export_format": self.pipeline.config.export.format if hasattr(self.pipeline, "config") else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _git_sha() -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
        except Exception:
            return None
        return result.stdout.strip() or None


def make_default_pipeline(export_format: Literal["parquet", "jsonl", "both"] = "both") -> SFTDataPipeline:
    config = SFTFormatterConfig(export=ExportConfig(format=export_format))
    return SFTDataPipeline(config)
