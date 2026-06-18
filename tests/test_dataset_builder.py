import json
from pathlib import Path

from agent_sft.dataset_builder.dataset_builder import DatasetBuilder, DatasetMixConfig, DatasetSourceConfig


class StubPipeline:
    def __init__(self):
        self.config = type("Config", (), {"export": type("Export", (), {"format": "jsonl"})()})()

    def process_batch(self, trajectories, output_dir=None, dataset_name="sft_dataset"):
        records = [
            {
                "task_id": item["task_id"],
                "domain": item["domain"],
                "difficulty": item.get("difficulty", "medium"),
                "messages": [],
                "token_count": 0,
                "num_turns": len(item.get("steps", [])),
                "success": bool(item.get("success")),
                "final_score": item.get("final_score"),
            }
            for item in trajectories
        ]
        output_path = Path(output_dir) / f"{dataset_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return records, {"jsonl": output_path}


def make_traj(task_id, domain):
    return {
        "task_id": task_id,
        "domain": domain,
        "difficulty": "medium",
        "steps": [{"thought": "x"}],
        "success": True,
        "final_score": 1.0,
    }


def test_ratio_sampling_reports_insufficient_bucket():
    builder = DatasetBuilder(DatasetMixConfig(target_size=10, seed=1), pipeline=StubPipeline())
    records = [
        make_traj("general_1", "multi_step_planning"),
        make_traj("code_1", "code_debug"),
        make_traj("reason_1", "math_reasoning"),
        make_traj("tool_1", "api_orchestration"),
    ]

    selected, counts, gaps = builder.select_records(records)

    assert len(selected) == 4
    assert counts["by_category"]["general_instruction"] == 1
    assert any(gap["shortfall"] > 0 for gap in gaps)


def test_quality_report_missing_raw_loads_zero(tmp_path):
    report_path = tmp_path / "quality_filter_report_20260101_000000.json"
    report_path.write_text(
        json.dumps({"kept": [{"raw_path": str(tmp_path / "missing_raw.json"), "quality_score": 1.0}]}),
        encoding="utf-8",
    )
    builder = DatasetBuilder(DatasetMixConfig(target_size=2, seed=1), pipeline=StubPipeline())

    records, summaries = builder.load_sources([DatasetSourceConfig(path=report_path, kind="quality_report")])

    assert records == []
    assert summaries[0]["count"] == 0


def test_build_writes_manifest(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "a_raw.json").write_text(json.dumps(make_traj("a", "code_debug")), encoding="utf-8")
    (data_dir / "b_raw.json").write_text(json.dumps(make_traj("b", "multi_step_planning")), encoding="utf-8")
    builder = DatasetBuilder(DatasetMixConfig(target_size=2, seed=1), pipeline=StubPipeline())

    result = builder.build(
        [DatasetSourceConfig(path=data_dir, kind="raw_dir", label="test_raw")],
        output_dir=tmp_path / "out",
        dataset_name="unit_dataset",
        version="vtest",
    )

    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "unit_dataset"
    assert manifest["version"] == "vtest"
    assert manifest["selected_counts"]["total"] == 2
    assert Path(manifest["output_files"]["jsonl"]).exists()
