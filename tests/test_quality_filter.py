import json
from pathlib import Path

import pytest

from agent_sft.quality_filter.quality_filter import (
    DifficultyAwareSampler,
    DifficultyClassifier,
    DeduplicatorMinhash,
    DedupTextConfig,
    DiversityMonitor,
    FailureDetector,
    HERRelabeler,
    LLMJudgeConsensus,
    LLMMCEstimator,
    OfflineMCEstimator,
    OutcomeExtractor,
    ProcessRewardScorer,
    QualityFilter,
    QualityFilterConfig,
    ResultVerifier,
    TrajectoryRecord,
    build_dedup_text,
    cosine_similarity_matrix,
    encode_texts_with_transformers,
    extract_final_answer,
    load_task_map,
)
from infra.environment import VerificationMode, VerificationResult


class FakeAnswerVerifier:
    def __init__(self):
        self.calls = []

    async def verify(self, answer, mode, **kwargs):
        self.calls.append({"answer": answer, "mode": mode, "kwargs": kwargs})
        return VerificationResult(mode=mode, passed=True, score=1.0, details={"ok": True})


def make_raw(task_id="task-1", domain="math_reasoning", answer="answer is 4", success=True):
    return {
        "task_id": task_id,
        "domain": domain,
        "difficulty": "medium",
        "success": success,
        "termination_reason": "success" if success else "final_answer_failed",
        "final_score": 1.0 if success else 0.0,
        "steps": [
            {
                "state_snapshot": {"metadata": {"task_prompt": "Solve 2 + 2"}},
                "action": {"action_type": "final_answer", "answer": answer},
                "observation": {"success": success, "metadata": {"verification_score": 1.0 if success else 0.0}},
            }
        ],
    }


def make_sft():
    return {"messages": [{"role": "user", "content": "Solve 2 + 2"}, {"role": "assistant", "content": "4"}]}


def make_record(raw=None, sft=None, path=None):
    raw = raw or make_raw()
    path = path or Path("bestofn_task_sample0_raw.json")
    record = TrajectoryRecord(
        id=path.stem.replace("_raw", ""),
        task_id=raw["task_id"],
        domain=raw["domain"],
        difficulty=raw["difficulty"],
        raw_path=path,
        sft_path=Path(str(path).replace("_raw.json", "_sft.json")),
        raw=raw,
        sft=make_sft() if sft is None else sft,
        final_answer=extract_final_answer(raw),
    )
    record.dedup_text = build_dedup_text(record)
    return record


def test_extract_final_answer_from_steps():
    assert extract_final_answer(make_raw(answer="final value")) == "final value"


def test_load_task_map_supports_wrapped_data(tmp_path):
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"tasks": [{"id": "task-1", "prompt": "x"}]}), encoding="utf-8")
    assert load_task_map(task_file) == {"task-1": {"id": "task-1", "prompt": "x"}}


def test_build_dedup_text_uses_full_sft_messages():
    sft = {
        "messages": [
            {"role": "user", "content": "Solve 2 + 2"},
            {"role": "assistant", "content": "Thought: add two numbers"},
            {"role": "tool", "content": "Observation: calculator returned 4"},
            {"role": "assistant", "content": "Final Answer: 4"},
        ]
    }
    record = make_record(sft=sft)

    text = build_dedup_text(record)

    assert "prompt: Solve 2 + 2" in text
    assert "assistant: Thought: add two numbers" in text
    assert "tool: Observation: calculator returned 4" in text
    assert "assistant: Final Answer: 4" in text


def test_build_dedup_text_falls_back_to_raw_steps():
    raw = make_raw()
    raw["steps"].insert(
        0,
        {
            "thought": "I should calculate directly",
            "action": {"action_type": "tool_call", "name": "eval", "kwargs": {"expr": "2+2"}},
            "observation": {"success": True, "content": "4"},
            "state_snapshot": {"metadata": {"task_prompt": "Solve 2 + 2"}},
        },
    )
    record = make_record(raw=raw, sft={})

    text = build_dedup_text(record)

    assert "Solve 2 + 2" in text
    assert "I should calculate directly" in text
    assert "tool_call" in text
    assert "observation" in text


def test_build_dedup_text_modes_select_fields():
    sft = {
        "messages": [
            {"role": "user", "content": "Prompt text"},
            {"role": "assistant", "content": "Thought: explore method"},
            {"role": "tool", "content": "Observation: tool output"},
            {"role": "assistant", "content": "Final Answer: result"},
        ]
    }
    raw = make_raw(answer="result")
    record = make_record(raw=raw, sft=sft)

    answer_text = build_dedup_text(record, DedupTextConfig(mode="answer"))
    process_text = build_dedup_text(record, DedupTextConfig(mode="process"))
    custom_text = build_dedup_text(
        record,
        DedupTextConfig(mode="answer", include_prompt=True, include_final_answer=False),
    )

    assert "final_answer: result" in answer_text
    assert "Prompt text" not in answer_text
    assert "Thought: explore method" in process_text
    assert "Observation: tool output" in process_text
    assert "Final Answer: result" not in process_text
    assert "prompt: Prompt text" in custom_text
    assert "final_answer" not in custom_text


@pytest.mark.asyncio
async def test_result_verifier_dispatches_math_mode():
    fake = FakeAnswerVerifier()
    verifier = ResultVerifier(answer_verifier=fake)
    record = make_record()
    task = {"id": "task-1", "domain": "math_reasoning", "test_cases": [{"expected_output": "4"}]}

    verified = await verifier.verify_record(record, task)

    assert verified.metadata["level1"]["passed"] is True
    assert fake.calls[0]["mode"] == VerificationMode.MATH_EQUATION
    assert fake.calls[0]["kwargs"]["ground_truth"] == "4"


@pytest.mark.asyncio
async def test_result_verifier_dispatches_code_mode():
    fake = FakeAnswerVerifier()
    verifier = ResultVerifier(answer_verifier=fake)
    raw = make_raw(domain="code_debug", answer="```python\ndef add(a, b):\n    return a + b\n```")
    record = make_record(raw=raw)
    task = {
        "id": "task-1",
        "domain": "code_debug",
        "test_cases": [{"input": {"args": [1, 2]}, "expected_output": 3}],
    }

    await verifier.verify_record(record, task)

    assert fake.calls[0]["mode"] == VerificationMode.CODE_EXECUTION
    assert fake.calls[0]["kwargs"]["function_name"] == "add"


class FailingCodeExecVerifier:
    """Returns a failed CODE_EXECUTION result so the report fallback path is exercised."""

    async def verify(self, answer, mode, **kwargs):
        return VerificationResult(
            mode=mode,
            passed=False,
            score=0.0,
            details={"tests_passed": 0, "tests_total": 1, "error": None},
        )


def _code_debug_report_raw(with_exec_evidence: bool):
    answer = (
        "## Root Cause\nThe bug is a missing return.\n\n"
        "## Fixed Code\n```python\ndef merge_dicts(a, b):\n    return {**a, **b}\n```\n\n"
        "## Explanation\nWe fix it by merging dicts."
    )
    steps = []
    if with_exec_evidence:
        steps.append({
            "state_snapshot": {"metadata": {"task_prompt": "Fix the bug"}},
            "action": {"action_type": "tool_call", "name": "exec",
                       "kwargs": {"code": "def merge_dicts(a, b):\n    return {**a, **b}\nprint(merge_dicts({1:1},{2:2}))"}},
            "observation": {"success": True, "content": "{1: 1, 2: 2}"},
        })
    steps.append({
        "state_snapshot": {"metadata": {"task_prompt": "Fix the bug"}},
        "action": {"action_type": "final_answer", "answer": answer},
        "observation": {"success": True, "metadata": {}},
    })
    return {
        "task_id": "task-cd",
        "domain": "code_debug",
        "difficulty": "medium",
        "success": True,
        "termination_reason": "success",
        "final_score": None,
        "steps": steps,
    }


@pytest.mark.asyncio
async def test_code_debug_report_fallback_passes_with_exec_evidence():
    verifier = ResultVerifier(answer_verifier=FailingCodeExecVerifier())
    record = make_record(raw=_code_debug_report_raw(with_exec_evidence=True))
    task = {
        "id": "task-cd",
        "domain": "code_debug",
        "test_cases": [{"expected_output": {"root_cause": True, "fixed_code": True, "explanation": True}}],
    }

    verified = await verifier.verify_record(record, task)

    level1 = verified.metadata["level1"]
    assert level1["passed"] is True
    assert level1["mode"] == VerificationMode.FORMAT_VALIDATION.value
    assert level1["details"]["report_fallback"] is True
    assert level1["details"]["successful_exec_evidence"]["matched"] is True


@pytest.mark.asyncio
async def test_code_debug_report_fallback_fails_without_exec_evidence():
    verifier = ResultVerifier(answer_verifier=FailingCodeExecVerifier())
    record = make_record(raw=_code_debug_report_raw(with_exec_evidence=False))
    task = {
        "id": "task-cd",
        "domain": "code_debug",
        "test_cases": [{"expected_output": {"root_cause": True, "fixed_code": True, "explanation": True}}],
    }

    verified = await verifier.verify_record(record, task)

    level1 = verified.metadata["level1"]
    assert level1["passed"] is False
    assert level1["details"]["report_fallback"] is True
    assert level1["details"]["successful_exec_evidence"] is None


@pytest.mark.asyncio
async def test_missing_task_fails_level1_by_default():
    verifier = ResultVerifier(answer_verifier=FakeAnswerVerifier())
    record = make_record()

    verified = await verifier.verify_record(record, None)

    assert verified.metadata["level1"]["passed"] is False
    assert verified.metadata["level1"]["error"] == "missing_task_reference"


def test_minhash_signature_is_deterministic():
    dedup = DeduplicatorMinhash(enable_embedding_dedup=False)
    assert dedup.signature("same text") == dedup.signature("same text")


def test_deduplicator_keeps_highest_quality_duplicate():
    first = make_record(path=Path("first_raw.json"))
    second = make_record(path=Path("second_raw.json"))
    first.id = "first"
    second.id = "second"
    first.dedup_text = "alpha beta gamma delta epsilon"
    second.dedup_text = "alpha beta gamma delta epsilon"
    first.quality_score = 0.2
    second.quality_score = 0.9
    dedup = DeduplicatorMinhash(num_perm=16, bands=4, rows=4, ngram_size=2, enable_embedding_dedup=False)

    kept, metadata = dedup.deduplicate([first, second])

    assert [record.id for record in kept] == ["second"]
    assert metadata["filtered_duplicates"] == 1
    assert first.metadata["level3"]["duplicate_of"] == "second"


def test_diversity_entropy_repeated_vs_varied():
    repeated = DiversityMonitor.ngram_entropy(["same same same", "same same same"], 1)[0]
    varied = DiversityMonitor.ngram_entropy(["alpha beta gamma", "delta epsilon zeta"], 1)[0]
    assert varied > repeated


def test_transformers_embedding_encoder_smoke():
    embeddings = encode_texts_with_transformers(
        ["fix python bug", "repair python issue", "solve math equation"],
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    similarities = cosine_similarity_matrix(embeddings)

    assert len(embeddings) == 3
    assert len(embeddings[0]) > 0
    assert similarities[0][0] == pytest.approx(1.0, abs=1e-5)
    assert similarities[0][1] > similarities[0][2]


def test_embedding_dedup_enabled_smoke():
    first = make_record(path=Path("first_raw.json"))
    second = make_record(path=Path("second_raw.json"))
    first.id = "first"
    second.id = "second"
    first.dedup_text = "Fix the Python function and explain the bug."
    second.dedup_text = "Fix the Python function and explain the bug."
    first.quality_score = 0.2
    second.quality_score = 0.9
    dedup = DeduplicatorMinhash(
        num_perm=16,
        bands=4,
        rows=4,
        ngram_size=2,
        enable_embedding_dedup=True,
        embedding_similarity_threshold=0.9,
    )

    kept, metadata = dedup.deduplicate([first, second])

    assert metadata["embedding_enabled"] is True
    assert metadata["embedding_error"] is None
    assert metadata["embedding_pairs_above_threshold"] >= 1
    assert [record.id for record in kept] == ["second"]


def test_offline_mc_estimator_scores_steps_from_observations():
    raw = make_raw()
    raw["steps"].insert(
        0,
        {
            "action": {"action_type": "tool_call", "name": "eval"},
            "observation": {"success": False, "error": "bad call", "metadata": {}},
        },
    )
    record = make_record(raw=raw, sft={})
    scores, metadata = OfflineMCEstimator().estimate(record, None)

    assert len(scores) == 2
    assert scores[0].mc_score < scores[1].mc_score
    assert metadata["step_count"] == 2


@pytest.mark.asyncio
async def test_llm_mc_estimator_scores_steps_with_fake_llm():
    async def fake_mc(_record, _step_index, rollout_index):
        return "correct" if rollout_index < 2 else "wrong"

    raw = make_raw()
    raw["steps"].insert(0, {"action": {"action_type": "tool_call"}, "observation": {"success": True, "content": "x"}})
    record = make_record(raw=raw, sft={})

    scores, metadata = await LLMMCEstimator(
        fake_mc,
        verifier=ResultVerifier(answer_verifier=FakeAnswerVerifier()),
        rollouts=3,
    ).estimate_async(
        record,
        None,
        {"id": "task-1", "domain": "math_reasoning", "test_cases": [{"expected_output": "4"}]},
    )

    assert metadata["replay_mc_enabled"] is True
    assert [score.mc_score for score in scores] == pytest.approx([1.0, 1.0])


@pytest.mark.asyncio
async def test_process_reward_scorer_fake_judge_consensus():
    async def fake_judge(_record):
        return '{"passed": true, "score": 0.8}'

    record = make_record()
    scorer = ProcessRewardScorer(
        judge_consensus=LLMJudgeConsensus(judge_client=fake_judge, consensus_k=2),
        min_score=0.5,
    )
    result = await scorer.score_record(record, None, None)

    assert result.passed is True
    assert result.judge_consensus["score"] == pytest.approx(0.8)
    assert result.step_scores[0].judge_score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_process_reward_scorer_bad_judge_json_fail_closed():
    async def bad_judge(_record):
        return "not json"

    record = make_record()
    scorer = ProcessRewardScorer(
        judge_consensus=LLMJudgeConsensus(judge_client=bad_judge, consensus_k=1, fail_closed=True),
        min_score=0.5,
    )
    result = await scorer.score_record(record, None, None)

    assert result.judge_consensus["passed"] is False
    assert result.judge_consensus["errors"]


def test_difficulty_classifier_assigns_pass32_buckets():
    records = []
    for task_id, successes in [("easy", 32), ("medium", 16), ("hard", 4), ("extreme", 0)]:
        for i in range(32):
            records.append(
                make_record(
                    raw=make_raw(task_id=task_id, success=i < successes),
                    path=Path(f"{task_id}_{i}_raw.json"),
                )
            )

    stats, summary = DifficultyClassifier().classify(records)

    assert stats["easy"].bucket == "easy"
    assert stats["medium"].bucket == "medium"
    assert stats["hard"].bucket == "hard"
    assert stats["extreme"].bucket == "extreme"
    assert summary["bucket_counts_by_task"] == {"easy": 1, "medium": 1, "hard": 1, "extreme": 1}


def test_difficulty_sampler_applies_ratio():
    records = []
    for bucket in ["easy", "medium", "hard", "extreme"]:
        for i in range(5):
            record = make_record(path=Path(f"{bucket}_{i}_raw.json"))
            record.id = f"{bucket}_{i}"
            record.difficulty = bucket
            record.quality_score = 1.0 - i * 0.01
            record.metadata["difficulty"] = {"bucket": bucket}
            records.append(record)

    selected, metadata = DifficultyAwareSampler(seed=0).sample(records, target_count=10)

    assert len(selected) == 10
    assert metadata["selected_by_bucket"] == {"hard": 4, "medium": 3, "extreme": 2, "easy": 1}


@pytest.mark.asyncio
async def test_llm_her_relabeler_rewrites_and_validates():
    responses = iter(
        [
            json.dumps({"achievements": [{"description": "found partial result", "step_index": 0, "confidence": 0.9}]}),
            json.dumps(
                {
                    "system_prompt": "HER system",
                    "user_prompt": "Do the achieved subgoal",
                    "final_answer": "found partial result",
                }
            ),
            json.dumps({"valid": True, "score": 0.9, "reason": "supported"}),
        ]
    )

    async def fake_llm(_system_prompt, _payload):
        return next(responses)

    raw = make_raw(success=False)
    raw["steps"].insert(
        0,
        {"action": {"action_type": "tool_call"}, "observation": {"success": True, "content": "partial result"}},
    )
    record = make_record(raw=raw, sft={})
    record.metadata["level1"] = {"passed": False, "error": "verification_failed"}

    her_sft, report = await HERRelabeler(mode="hybrid", llm_client=fake_llm).relabel([record])

    assert report["output_her_count"] == 1
    assert her_sft[0]["messages"][0]["content"] == "HER system"
    assert her_sft[0]["messages"][1]["content"] == "Do the achieved subgoal"
    assert "found partial result" in her_sft[0]["messages"][2]["content"]
    assert her_sft[0]["metadata"]["llm_validation"]["valid"] is True


@pytest.mark.asyncio
async def test_her_relabeler_generates_rewritten_sft():
    raw = make_raw(success=False)
    raw["steps"].insert(
        0,
        {
            "action": {"action_type": "tool_call", "name": "eval"},
            "observation": {"success": True, "content": "partial result"},
            "state_snapshot": {"metadata": {"task_prompt": "Solve all"}},
        },
    )
    record = make_record(raw=raw, sft={})
    record.metadata["level1"] = {"passed": False, "error": "verification_failed"}

    her_sft, report = await HERRelabeler(mode="heuristic").relabel([record])

    assert report["output_her_count"] == 1
    assert her_sft[0]["metadata"]["her"] is True
    assert "hindsight-relabelled" in her_sft[0]["messages"][0]["content"]
    assert "partial result" in her_sft[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_quality_filter_report_has_four_stages(tmp_path):
    raw_path = tmp_path / "sample_raw.json"
    sft_path = tmp_path / "sample_sft.json"
    task_path = tmp_path / "tasks.json"
    raw_path.write_text(json.dumps(make_raw()), encoding="utf-8")
    sft_path.write_text(json.dumps(make_sft()), encoding="utf-8")
    task_path.write_text(
        json.dumps([{"id": "task-1", "domain": "math_reasoning", "test_cases": [{"expected_output": "4"}]}]),
        encoding="utf-8",
    )
    config = QualityFilterConfig(
        input_dir=tmp_path,
        task_file=task_path,
        enable_embedding_dedup=False,
        minhash_num_perm=16,
        lsh_bands=4,
        lsh_rows=4,
    )
    quality_filter = QualityFilter(config=config, result_verifier=ResultVerifier(answer_verifier=FakeAnswerVerifier()))

    report = await quality_filter.run()

    assert [stage["stage"] for stage in report["funnel"]] == [
        "level1_result_verifier",
        "level2_prm_mc_llm_judge",
        "level3_deduplication_diversity",
        "level4_difficulty_aware_sampling",
    ]
    assert report["summary"]["input_count"] == 1
    assert report["summary"]["final_count"] == 1
    assert report["funnel"][1]["implemented"] is True
    assert report["funnel"][3]["implemented"] is True
    assert "her_relabeling" in report
    assert "difficulty_diagnostics" in report
