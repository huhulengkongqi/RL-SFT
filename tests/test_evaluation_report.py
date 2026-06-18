from agent_sft.dataset_builder.report_generator import EvaluationReportGenerator


def test_report_contains_matrix_charts_and_weak_areas():
    metrics = {
        "dataset_name": "unit_dataset",
        "record_count": 10,
        "task_count": 2,
        "metrics": {
            "pass_at_1": {"value": 0.7, "threshold": ">= 0.60", "passed": True, "note": ""},
            "difficulty_coverage": {
                "value": 0.0,
                "threshold": "each >= 0.10",
                "passed": False,
                "note": "extreme=0.0%",
            },
        },
        "distribution": {
            "category_share": {"code": 0.3, "general_instruction": 0.4, "tool_calling": 0.3},
            "domain_share": {"code_debug": 0.3, "multi_step_planning": 0.4, "api_orchestration": 0.3},
        },
        "observed_difficulty": {"share": {"easy": 0.5, "medium": 0.5, "hard": 0.0, "extreme": 0.0}},
        "pass_at": {"pass_at_1": 0.7, "pass_at_8": 0.9, "under_sampled_tasks_lt_8": 1},
        "trajectory_length": {"avg_steps": 4, "median_steps": 4, "min_steps": 3, "max_steps": 5},
        "diversity": {"self_bleu": 0.2, "diversity_score": 0.8},
        "weak_areas": [
            {"value": 0.0, "threshold": "each >= 0.10", "passed": False, "note": "extreme=0.0%"}
        ],
    }

    report = EvaluationReportGenerator().render_markdown(metrics)

    assert "指标矩阵" in report
    assert "Category Share" in report
    assert "████" in report
    assert "Weak Areas" in report
    assert "extreme=0.0%" in report
