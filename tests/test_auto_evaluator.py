import pytest

from agent_sft.dataset_builder.auto_evaluator import AutoEvaluator


class StubDiversityMetrics:
    def self_bleu(self, texts, sample_size=500):
        return 0.25


def make_traj(task_id, success, domain="code_debug", difficulty="medium", steps=4):
    return {
        "task_id": task_id,
        "domain": domain,
        "difficulty": difficulty,
        "steps": [{"thought": f"step {i}"} for i in range(steps)],
        "success": success,
        "final_score": 1.0 if success else 0.0,
    }


def test_pass_at_metrics_and_diversity_score():
    records = [make_traj("a", True), make_traj("a", False), make_traj("b", False), make_traj("b", False)]
    evaluator = AutoEvaluator(diversity_metrics=StubDiversityMetrics())

    metrics = evaluator.evaluate(records)

    assert metrics["pass_at"]["pass_at_1"] == pytest.approx(0.25)
    assert metrics["pass_at"]["pass_at_8"] == pytest.approx(0.5)
    assert metrics["diversity"]["diversity_score"] == pytest.approx(0.75)
    assert metrics["metrics"]["diversity_score"]["passed"] is True


def test_missing_extreme_difficulty_is_flagged():
    records = [
        make_traj("easy_task", True, domain="multi_step_planning"),
        make_traj("reasoning_task", True, domain="math_reasoning"),
        make_traj("tool_task", True, domain="api_orchestration"),
    ]
    evaluator = AutoEvaluator(diversity_metrics=StubDiversityMetrics())

    metrics = evaluator.evaluate(records)

    assert metrics["metrics"]["difficulty_coverage"]["passed"] is False
    assert "extreme=0.0%" in metrics["metrics"]["difficulty_coverage"]["note"]
    assert metrics["distribution"]["category_share"]["general_instruction"] == pytest.approx(1 / 3)
