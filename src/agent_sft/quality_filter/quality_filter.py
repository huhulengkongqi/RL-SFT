"""Trajectory quality filtering funnel."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from infra.environment import AnswerVerifier, Environment, SandboxPool, VerificationMode, VerificationResult

from .metrics import DiversityMetrics


DEDUP_TEXT_PRESETS: dict[str, dict[str, bool]] = {
    "answer": {
        "include_prompt": False,
        "include_thought": False,
        "include_action": False,
        "include_observation": False,
        "include_error": False,
        "include_final_answer": True,
    },
    "prompt_answer": {
        "include_prompt": True,
        "include_thought": False,
        "include_action": False,
        "include_observation": False,
        "include_error": False,
        "include_final_answer": True,
    },
    "process": {
        "include_prompt": False,
        "include_thought": True,
        "include_action": True,
        "include_observation": True,
        "include_error": True,
        "include_final_answer": False,
    },
    "trajectory": {
        "include_prompt": True,
        "include_thought": True,
        "include_action": True,
        "include_observation": True,
        "include_error": True,
        "include_final_answer": True,
    },
    "limo_reasoning": {
        "include_prompt": False,
        "include_thought": True,
        "include_action": False,
        "include_observation": False,
        "include_error": False,
        "include_final_answer": True,
    },
    "quagmires_exploration": {
        "include_prompt": False,
        "include_thought": True,
        "include_action": True,
        "include_observation": True,
        "include_error": True,
        "include_final_answer": True,
    },
}


@dataclass
class DedupTextConfig:
    mode: str = "trajectory"
    include_prompt: bool | None = None
    include_thought: bool | None = None
    include_action: bool | None = None
    include_observation: bool | None = None
    include_error: bool | None = None
    include_final_answer: bool | None = None

    def resolved(self) -> dict[str, bool]:
        if self.mode not in DEDUP_TEXT_PRESETS:
            raise ValueError(f"Unknown dedup text mode: {self.mode}")
        values = dict(DEDUP_TEXT_PRESETS[self.mode])
        for field_name in (
            "include_prompt",
            "include_thought",
            "include_action",
            "include_observation",
            "include_error",
            "include_final_answer",
        ):
            value = getattr(self, field_name)
            if value is not None:
                values[field_name] = value
        return values

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, **self.resolved()}


@dataclass
class QualityFilterConfig:
    input_dir: Path = Path("data/sft_trajectories")
    raw_glob: str = "*_raw.json"
    task_file: Path | None = None
    limit: int | None = None
    domain: str | None = None
    level1_concurrency: int = 2
    dedup_text_config: DedupTextConfig = field(default_factory=DedupTextConfig)
    minhash_num_perm: int = 128
    minhash_ngram: int = 5
    lsh_bands: int = 32
    lsh_rows: int = 4
    minhash_jaccard_threshold: float = 0.8
    embedding_similarity_threshold: float = 0.9
    enable_embedding_dedup: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_diagnostics_top_k: int = 50
    diversity_sample_size: int = 500
    enable_level2_prm: bool = True
    level2_min_score: float = 0.5
    level2_mc_weight: float = 0.6
    level2_judge_weight: float = 0.4
    level2_judge_consensus_k: int = 3
    level2_fail_closed_on_judge_error: bool = False
    level2_offline_mc_mode: str = "hybrid"
    enable_level4_sampling: bool = True
    difficulty_sampling_ratio: dict[str, int] = field(
        default_factory=lambda: {"easy": 1, "medium": 3, "hard": 4, "extreme": 2}
    )
    difficulty_pass32_buckets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "easy": (0.75, 1.01),
            "medium": (0.35, 0.75),
            "hard": (0.05, 0.35),
            "extreme": (0.0, 0.05),
        }
    )
    level4_target_count: int | None = None
    level4_seed: int = 0
    enable_her_relabeling: bool = True
    her_include_in_filtered_sft: bool = False
    her_mode: str = "hybrid"
    her_max_per_failed_task: int = 2
    her_min_partial_score: float = 0.2
    fail_open_missing_task: bool = False

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        if self.task_file is not None:
            self.task_file = Path(self.task_file)
        if self.lsh_bands * self.lsh_rows != self.minhash_num_perm:
            raise ValueError("lsh_bands * lsh_rows must equal minhash_num_perm")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["task_file"] = str(self.task_file) if self.task_file else None
        data["dedup_text_config"] = self.dedup_text_config.to_dict()
        return data


@dataclass
class TrajectoryRecord:
    id: str
    task_id: str
    domain: str
    difficulty: str
    raw_path: Path
    sft_path: Path | None
    raw: dict[str, Any]
    sft: dict[str, Any] | None
    final_answer: Any = None
    dedup_text: str = ""
    quality_score: float = 0.0
    quality_score_source: str = "unset"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "raw_path": str(self.raw_path),
            "sft_path": str(self.sft_path) if self.sft_path else None,
            "quality_score": self.quality_score,
            "quality_score_source": self.quality_score_source,
            "metadata": self.metadata,
        }


@dataclass
class StageResult:
    stage: str
    implemented: bool
    input_count: int
    output_count: int
    skipped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.input_count:
            return 1.0 if self.skipped else 0.0
        return self.output_count / self.input_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "implemented": self.implemented,
            "skipped": self.skipped,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "pass_rate": self.pass_rate,
            "metadata": self.metadata,
            "failures": self.failures,
        }


class ResultVerifier:
    def __init__(self, answer_verifier: AnswerVerifier | None = None, sandbox_pool: SandboxPool | None = None):
        self.answer_verifier = answer_verifier or AnswerVerifier(sandbox_pool or SandboxPool())

    async def verify_record(
        self,
        record: TrajectoryRecord,
        task: dict[str, Any] | None,
        fail_open_missing_task: bool = False,
    ) -> TrajectoryRecord:
        if record.final_answer is None:
            return self._mark_failed(record, "missing_final_answer")
        if task is None:
            if fail_open_missing_task and bool(record.raw.get("success")):
                record.quality_score = self._recorded_score(record)
                record.quality_score_source = "recorded_success_missing_task"
                record.metadata["level1"] = {
                    "implemented": True,
                    "passed": True,
                    "mode": "recorded_success",
                    "score": record.quality_score,
                    "details": {"warning": "missing_task_reference"},
                    "error": None,
                }
                return record
            return self._mark_failed(record, "missing_task_reference")

        domain = str(task.get("domain") or record.domain)
        test_cases = task.get("test_cases") or []
        try:
            result = await self._verify_by_domain(domain, record.final_answer, test_cases, record=record)
        except Exception as exc:
            return self._mark_failed(record, "verification_exception", str(exc))

        record.quality_score = float(result.score)
        record.quality_score_source = "level1_verification_score"
        record.metadata["level1"] = {
            "implemented": True,
            "passed": bool(result.passed),
            "mode": result.mode.value if hasattr(result.mode, "value") else str(result.mode),
            "score": float(result.score),
            "details": result.details,
            "error": result.error,
        }
        return record

    async def _verify_by_domain(
        self,
        domain: str,
        answer: Any,
        test_cases: list[dict[str, Any]],
        record: TrajectoryRecord | None = None,
    ) -> VerificationResult:
        if domain in {"math", "math_reasoning", "arithmetic", "algebra"}:
            ground_truth = test_cases[0].get("expected_output", "") if test_cases else ""
            if isinstance(ground_truth, dict):
                ground_truth = ground_truth.get("final_answer", "")
            return await self.answer_verifier.verify(
                Environment._extract_math_answer(answer),
                mode=VerificationMode.MATH_EQUATION,
                ground_truth=ground_truth,
                expression=True,
            )

        if domain in {"code", "coding", "code_debug", "programming"}:
            extracted_code = Environment._extract_python_code(answer)
            function_name = Environment._infer_function_name(extracted_code, answer)
            result = await self.answer_verifier.verify(
                extracted_code,
                mode=VerificationMode.CODE_EXECUTION,
                test_cases=test_cases,
                function_name=function_name,
            )
            if not result.passed:
                error_text = str(result.details.get("error", "")) + str(result.details.get("test_details", ""))
                if "not defined" in error_text or function_name == "solution":
                    wrapped_result = await self.answer_verifier.verify(
                        Environment._wrap_script_as_solution(extracted_code),
                        mode=VerificationMode.CODE_EXECUTION,
                        test_cases=test_cases,
                        function_name="solution",
                    )
                    if wrapped_result.score >= result.score:
                        result = wrapped_result
            if result.passed:
                return result

            # Report-style fallback (parity with Environment._handle_final_answer, offline):
            # code_debug tasks whose expected_output is a debugging report cannot be
            # validated by code execution. Fall back to format validation plus evidence
            # of a successful exec observation recorded in the trajectory.
            expected_output = test_cases[0].get("expected_output", {}) if test_cases else {}
            report_fields = {"root_cause", "fixed_code", "explanation"}
            expected_keys = set(expected_output.keys()) if isinstance(expected_output, dict) else set()
            if expected_keys & report_fields:
                fmt_passed, fmt_score, fmt_details = Environment._verify_code_debug_answer(answer, expected_output)
                evidence = self._exec_evidence_from_record(record, answer, extracted_code)
                fallback_passed = fmt_passed and evidence is not None
                details = {
                    "report_fallback": True,
                    "format_checks": fmt_details,
                    "format_score": fmt_score,
                    "successful_exec_evidence": evidence,
                    "code_execution": result.details,
                }
                return VerificationResult(
                    mode=VerificationMode.FORMAT_VALIDATION,
                    passed=fallback_passed,
                    score=fmt_score if fallback_passed else min(fmt_score, 0.99),
                    details=details,
                    error=None if fallback_passed else "code execution failed and report fallback (format+exec evidence) did not pass",
                )
            return result

        kwargs: dict[str, Any] = {}
        if test_cases and isinstance(test_cases[0].get("expected_output"), dict):
            kwargs["required_fields"] = list(test_cases[0]["expected_output"].keys())
        return await self.answer_verifier.verify(answer, mode=VerificationMode.FORMAT_VALIDATION, **kwargs)

    @staticmethod
    def _exec_evidence_from_record(
        record: TrajectoryRecord | None, answer: Any, extracted_code: Any
    ) -> dict[str, Any] | None:
        """Find a recorded successful exec step that shares a function with the answer.

        Offline analogue of Environment._find_successful_exec_evidence: scans the
        recorded trajectory steps (instead of a live Environment history).
        """
        if record is None:
            return None
        answer_text = str(answer)
        extracted_text = str(extracted_code or "")
        steps = record.raw.get("steps") or []
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            action = step.get("action") or {}
            observation = step.get("observation") or {}
            if action.get("name") != "exec" or not observation.get("success"):
                continue
            kwargs = action.get("kwargs") or {}
            args = action.get("args") or []
            executed_code = str(kwargs.get("code") or (args[0] if args else ""))
            if not executed_code.strip():
                continue
            executed_functions = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", executed_code)
            shared_function = any(
                f"def {name}" in answer_text or f"def {name}" in extracted_text for name in executed_functions
            )
            substantial_overlap = (
                len(set(executed_code.split()) & set(extracted_text.split())) >= 20 if extracted_text else False
            )
            if shared_function or substantial_overlap:
                return {
                    "matched": True,
                    "step": step.get("step"),
                    "executed_functions": executed_functions,
                }
        return None

    def _mark_failed(self, record: TrajectoryRecord, reason: str, error: str | None = None) -> TrajectoryRecord:
        record.quality_score = self._recorded_score(record)
        record.quality_score_source = "recorded_score_after_level1_failure"
        record.metadata["level1"] = {
            "implemented": True,
            "passed": False,
            "mode": None,
            "score": 0.0,
            "details": {"reason": reason},
            "error": error or reason,
        }
        return record

    @staticmethod
    def _recorded_score(record: TrajectoryRecord) -> float:
        if record.raw.get("final_score") is not None:
            return float(record.raw.get("final_score") or 0.0)
        for step in reversed(record.raw.get("steps") or []):
            metadata = ((step.get("observation") or {}).get("metadata") or {}) if isinstance(step, dict) else {}
            if metadata.get("verification_score") is not None:
                return float(metadata.get("verification_score") or 0.0)
        return 1.0 if record.raw.get("success") else 0.0


@dataclass
class TaskPassStats:
    task_id: str
    attempts: int
    successes: int
    pass32_rate: float
    bucket: str
    estimated_from_observed_n: bool


@dataclass
class StepPRMScore:
    step_index: int
    action_type: str | None
    mc_score: float
    judge_score: float | None
    prm_score: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessRewardResult:
    passed: bool
    score: float
    step_scores: list[StepPRMScore]
    judge_consensus: dict[str, Any]
    mc_metadata: dict[str, Any]


class DifficultyClassifier:
    def __init__(self, buckets: dict[str, tuple[float, float]] | None = None):
        self.buckets = buckets or {
            "easy": (0.75, 1.01),
            "medium": (0.35, 0.75),
            "hard": (0.05, 0.35),
            "extreme": (0.0, 0.05),
        }

    def classify(self, records: list[TrajectoryRecord]) -> tuple[dict[str, TaskPassStats], dict[str, Any]]:
        grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.task_id].append(record)

        stats: dict[str, TaskPassStats] = {}
        for task_id, task_records in grouped.items():
            attempts = min(32, len(task_records))
            successes = sum(1 for record in task_records[:32] if self._is_success(record))
            pass32_rate = successes / attempts if attempts else 0.0
            bucket = self.bucket_for_rate(pass32_rate)
            task_stats = TaskPassStats(
                task_id=task_id,
                attempts=attempts,
                successes=successes,
                pass32_rate=pass32_rate,
                bucket=bucket,
                estimated_from_observed_n=attempts < 32,
            )
            stats[task_id] = task_stats
            for record in task_records:
                record.difficulty = bucket
                record.metadata["difficulty"] = asdict(task_stats)

        return stats, self.summary(stats)

    def bucket_for_rate(self, pass32_rate: float) -> str:
        for bucket, (lower, upper) in self.buckets.items():
            if lower <= pass32_rate < upper:
                return bucket
        return "medium"

    @staticmethod
    def _is_success(record: TrajectoryRecord) -> bool:
        return bool(record.raw.get("success")) or float(record.raw.get("final_score") or 0.0) >= 1.0

    def summary(self, stats: dict[str, TaskPassStats]) -> dict[str, Any]:
        by_task = Counter(stat.bucket for stat in stats.values())
        return {
            "task_count": len(stats),
            "bucket_counts_by_task": dict(by_task),
            "insufficient_attempt_tasks": sum(1 for stat in stats.values() if stat.estimated_from_observed_n),
            "bucket_thresholds": {bucket: list(bounds) for bucket, bounds in self.buckets.items()},
        }


class OfflineMCEstimator:
    def estimate(
        self,
        record: TrajectoryRecord,
        task_stats: TaskPassStats | None,
    ) -> tuple[list[StepPRMScore], dict[str, Any]]:
        terminal_score = self._terminal_score(record)
        pass32_prior = task_stats.pass32_rate if task_stats else terminal_score
        step_scores = []
        for index, step in enumerate(record.raw.get("steps") or []):
            observation = step.get("observation") or {}
            immediate_score = self._observation_score(observation)
            mc_score = clamp(0.55 * terminal_score + 0.25 * immediate_score + 0.20 * pass32_prior)
            action = step.get("action") or {}
            step_scores.append(
                StepPRMScore(
                    step_index=index,
                    action_type=action.get("action_type"),
                    mc_score=mc_score,
                    judge_score=None,
                    prm_score=mc_score,
                    evidence={
                        "terminal_score": terminal_score,
                        "immediate_step_score": immediate_score,
                        "task_pass32_prior": pass32_prior,
                        "observation_success": observation.get("success"),
                        "observation_error": observation.get("error"),
                    },
                )
            )
        if not step_scores:
            step_scores.append(
                StepPRMScore(0, None, terminal_score, None, terminal_score, {"terminal_score": terminal_score})
            )
        return step_scores, {
            "terminal_score": terminal_score,
            "task_pass32_prior": pass32_prior,
            "step_count": len(step_scores),
        }

    @staticmethod
    def _terminal_score(record: TrajectoryRecord) -> float:
        if record.metadata.get("level1", {}).get("passed"):
            return float(record.metadata["level1"].get("score") or 1.0)
        if record.raw.get("final_score") is not None:
            return clamp(float(record.raw.get("final_score") or 0.0))
        return 1.0 if record.raw.get("success") else 0.0

    @staticmethod
    def _observation_score(observation: dict[str, Any]) -> float:
        metadata = observation.get("metadata") or {}
        for key in ("verification_score", "score"):
            if metadata.get(key) is not None:
                return clamp(float(metadata.get(key) or 0.0))
        if observation.get("error"):
            return 0.0
        return 1.0 if observation.get("success") else 0.25


class ReplayMCEstimator(OfflineMCEstimator):
    def __init__(
        self,
        llm_client: Any,
        verifier: ResultVerifier | None = None,
        rollouts: int = 3,
        fail_closed: bool = False,
    ):
        self.llm_client = llm_client
        self.verifier = verifier or ResultVerifier()
        self.rollouts = rollouts
        self.fail_closed = fail_closed

    async def estimate_async(
        self,
        record: TrajectoryRecord,
        task_stats: TaskPassStats | None,
        task: dict[str, Any] | None = None,
    ) -> tuple[list[StepPRMScore], dict[str, Any]]:
        fallback_scores, fallback_metadata = super().estimate(record, task_stats)
        step_scores = []
        errors = []
        for fallback in fallback_scores:
            rollout_results = []
            for rollout_index in range(self.rollouts):
                try:
                    answer = await self._sample_continuation(record, fallback.step_index, rollout_index)
                    passed = await self._verify_continuation(record, task, answer)
                    rollout_results.append({"answer": answer, "passed": passed})
                except Exception as exc:
                    errors.append({"step_index": fallback.step_index, "error": str(exc)})
                    rollout_results.append({"answer": None, "passed": False})
            successes = sum(1 for result in rollout_results if result["passed"])
            score = successes / self.rollouts if self.rollouts else 0.0
            if errors and self.fail_closed and not rollout_results:
                score = 0.0
            step_scores.append(
                StepPRMScore(
                    step_index=fallback.step_index,
                    action_type=fallback.action_type,
                    mc_score=score,
                    judge_score=None,
                    prm_score=score,
                    evidence={
                        **fallback.evidence,
                        "replay_rollouts": rollout_results,
                        "successes": successes,
                        "rollouts": self.rollouts,
                    },
                )
            )
        return step_scores, {
            **fallback_metadata,
            "mc_mode": "replay_rollout",
            "replay_mc_enabled": True,
            "replay_mc_rollouts": self.rollouts,
            "replay_mc_errors": errors,
        }

    async def _sample_continuation(self, record: TrajectoryRecord, step_index: int, rollout_index: int) -> str:
        prefix = record.raw.get("steps", [])[: step_index + 1]
        prompt = {
            "task_id": record.task_id,
            "domain": record.domain,
            "difficulty": record.difficulty,
            "rollout_index": rollout_index,
            "trajectory_prefix": prefix,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Continue the agent trajectory from the provided prefix and produce only the final answer. "
                    "Do not repeat the prefix. Output the final answer text directly."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        if hasattr(self.llm_client, "achat"):
            return await self.llm_client.achat(
                model=getattr(self.llm_client, "model", "default"),
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
        result = self.llm_client(record, step_index, rollout_index)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    async def _verify_continuation(self, record: TrajectoryRecord, task: dict[str, Any] | None, answer: str) -> bool:
        continuation = TrajectoryRecord(
            id=f"{record.id}_replay",
            task_id=record.task_id,
            domain=record.domain,
            difficulty=record.difficulty,
            raw_path=record.raw_path,
            sft_path=record.sft_path,
            raw=record.raw,
            sft=record.sft,
            final_answer=answer,
            dedup_text=record.dedup_text,
        )
        verified = await self.verifier.verify_record(continuation, task, fail_open_missing_task=False)
        return bool(verified.metadata.get("level1", {}).get("passed"))


LLMMCEstimator = ReplayMCEstimator


class LLMJudgeConsensus:
    def __init__(self, judge_client: Any | None = None, consensus_k: int = 3, fail_closed: bool = False):
        self.judge_client = judge_client
        self.consensus_k = consensus_k
        self.fail_closed = fail_closed

    async def judge(self, record: TrajectoryRecord) -> dict[str, Any]:
        if self.judge_client is None:
            return {"enabled": False, "skipped": True, "reason": "no_judge_client", "score": None, "passed": True}

        votes = []
        errors = []
        for _ in range(self.consensus_k):
            try:
                raw = await self._call_judge(record)
                parsed = parse_judge_json(raw)
                score = clamp(float(parsed.get("score", 0.0)))
                votes.append({"passed": bool(parsed.get("passed", False)), "score": score, "raw": parsed})
            except Exception as exc:
                errors.append(str(exc))
                if self.fail_closed:
                    votes.append({"passed": False, "score": 0.0, "raw": {"error": str(exc)}})

        if not votes:
            return {"enabled": True, "skipped": False, "score": None, "passed": not self.fail_closed, "errors": errors}

        mean_score = sum(vote["score"] for vote in votes) / len(votes)
        passed_votes = sum(1 for vote in votes if vote["passed"])
        return {
            "enabled": True,
            "skipped": False,
            "score": mean_score,
            "passed": passed_votes > len(votes) / 2 and mean_score >= 0.5,
            "passed_votes": passed_votes,
            "failed_votes": len(votes) - passed_votes,
            "errors": errors,
            "votes": votes,
        }

    async def _call_judge(self, record: TrajectoryRecord) -> str:
        prompt = {
            "task_id": record.task_id,
            "domain": record.domain,
            "final_answer": record.final_answer,
            "dedup_text_preview": record.dedup_text[:2000],
        }
        if hasattr(self.judge_client, "achat"):
            return await self.judge_client.achat(
                model=getattr(self.judge_client, "model", "default"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict process-reward judge for agent trajectories. Evaluate whether the "
                            "trajectory demonstrates a reliable path toward the correct answer. Return ONLY JSON "
                            "with fields: passed (bool), score (0-1), reason (short string)."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=512,
            )
        result = self.judge_client(record)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)


class ProcessRewardScorer:
    def __init__(
        self,
        mc_estimator: OfflineMCEstimator | None = None,
        judge_consensus: LLMJudgeConsensus | None = None,
        min_score: float = 0.5,
        mc_weight: float = 0.6,
        judge_weight: float = 0.4,
    ):
        self.mc_estimator = mc_estimator or OfflineMCEstimator()
        self.judge_consensus = judge_consensus or LLMJudgeConsensus()
        self.min_score = min_score
        self.mc_weight = mc_weight
        self.judge_weight = judge_weight

    async def score_record(
        self,
        record: TrajectoryRecord,
        task: dict[str, Any] | None,
        task_stats: TaskPassStats | None,
    ) -> ProcessRewardResult:
        if hasattr(self.mc_estimator, "estimate_async"):
            step_scores, mc_metadata = await self.mc_estimator.estimate_async(record, task_stats, task)
        else:
            step_scores, mc_metadata = self.mc_estimator.estimate(record, task_stats)
        mc_score = sum(step.prm_score for step in step_scores) / len(step_scores)
        judge_consensus = await self.judge_consensus.judge(record)
        judge_score = judge_consensus.get("score")
        if judge_score is None:
            score = mc_score
            applied_weights = {"mc": 1.0, "judge": 0.0}
        else:
            score = self.mc_weight * mc_score + self.judge_weight * float(judge_score)
            applied_weights = {"mc": self.mc_weight, "judge": self.judge_weight}
            for step in step_scores:
                step.judge_score = float(judge_score)
                step.prm_score = self.mc_weight * step.mc_score + self.judge_weight * float(judge_score)
        return ProcessRewardResult(
            passed=score >= self.min_score and bool(judge_consensus.get("passed", True)),
            score=clamp(score),
            step_scores=step_scores,
            judge_consensus={**judge_consensus, "applied_weights": applied_weights},
            mc_metadata=mc_metadata,
        )


class DifficultyAwareSampler:
    def __init__(self, ratio: dict[str, int] | None = None, seed: int = 0):
        self.ratio = ratio or {"easy": 1, "medium": 3, "hard": 4, "extreme": 2}
        self.seed = seed

    def sample(
        self,
        records: list[TrajectoryRecord],
        target_count: int | None = None,
    ) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
        if not records:
            return [], {"target_count": 0, "available_by_bucket": {}, "selected_by_bucket": {}}
        target_count = min(target_count or len(records), len(records))
        grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for record in records:
            bucket = record.metadata.get("difficulty", {}).get("bucket", record.difficulty or "medium")
            grouped[bucket].append(record)

        available_by_bucket = {bucket: len(grouped.get(bucket, [])) for bucket in self.ratio}
        quotas = self._quotas(target_count)
        selected: list[TrajectoryRecord] = []
        deficit = 0
        selected_by_bucket: dict[str, int] = {}

        for bucket, quota in quotas.items():
            bucket_records = self._rank(grouped.get(bucket, []))
            chosen = bucket_records[:quota]
            selected.extend(chosen)
            selected_by_bucket[bucket] = len(chosen)
            deficit += max(0, quota - len(chosen))

        if deficit:
            already = {record.id for record in selected}
            leftovers = [record for record in self._rank(records) if record.id not in already]
            selected.extend(leftovers[:deficit])

        selected = selected[:target_count]
        selected_ids = {record.id for record in selected}
        for record in records:
            bucket = record.metadata.get("difficulty", {}).get("bucket", record.difficulty or "medium")
            record.metadata["level4"] = {
                "implemented": True,
                "bucket": bucket,
                "selected": record.id in selected_ids,
                "sampling_ratio": self.ratio,
            }
        selected_by_bucket = Counter(record.metadata["level4"]["bucket"] for record in selected)
        return selected, {
            "target_count": target_count,
            "planned_sampling_ratio": self.ratio,
            "available_by_bucket": available_by_bucket,
            "selected_by_bucket": dict(selected_by_bucket),
            "deficit": deficit,
            "seed": self.seed,
        }

    def _quotas(self, target_count: int) -> dict[str, int]:
        total = sum(self.ratio.values())
        quotas = {bucket: int(target_count * weight / total) for bucket, weight in self.ratio.items()}
        remaining = target_count - sum(quotas.values())
        priority = sorted(self.ratio, key=lambda bucket: self.ratio[bucket], reverse=True)
        for bucket in priority[:remaining]:
            quotas[bucket] += 1
        return quotas

    def _rank(self, records: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
        rng = random.Random(self.seed)
        decorated = [(rng.random(), record) for record in records]
        decorated.sort(
            key=lambda item: (
                record_composite_score(item[1]),
                item[1].quality_score,
                -len(item[1].raw.get("steps") or []),
                -item[0],
            ),
            reverse=True,
        )
        return [record for _, record in decorated]


@dataclass
class PartialOutcome:
    description: str
    evidence: dict[str, Any]
    step_index: int
    confidence: float


class FailureDetector:
    def classify(self, record: TrajectoryRecord) -> dict[str, Any]:
        level1 = record.metadata.get("level1", {})
        reason = level1.get("error") or record.raw.get("termination_reason") or "unknown_failure"
        if level1 and not level1.get("passed") and str(reason) == "success":
            reason = "verification_failed"
        text = json.dumps(record.raw.get("steps") or [], ensure_ascii=False).lower()
        if any(token in text for token in ["timeout", "container", "permission", "fatal"]):
            failure_type = "tool_error"
            recoverable = False
            severity = 0.3
        elif reason in {"max_steps", "loop_detected"}:
            failure_type = "incomplete"
            recoverable = True
            severity = 0.7
        elif reason in {"verification_failed", "final_answer_failed"} or "failed" in str(reason):
            failure_type = "verification_failed"
            recoverable = True
            severity = 0.8
        else:
            failure_type = str(reason)
            recoverable = True
            severity = 0.5
        return {"failure_type": failure_type, "recoverable": recoverable, "severity_weight": severity, "reason": reason}


class OutcomeExtractor:
    def extract(self, record: TrajectoryRecord, min_confidence: float = 0.2) -> list[PartialOutcome]:
        outcomes = []
        for index, step in enumerate(record.raw.get("steps") or []):
            observation = step.get("observation") or {}
            content = observation.get("content")
            if observation.get("success") and content not in (None, ""):
                confidence = 0.8 if not observation.get("error") else 0.4
                outcomes.append(
                    PartialOutcome(
                        description=f"Successfully produced intermediate result: {str(content)[:500]}",
                        evidence={"observation": observation, "action": step.get("action") or {}},
                        step_index=index,
                        confidence=confidence,
                    )
                )
        return [outcome for outcome in outcomes if outcome.confidence >= min_confidence]


class HERRelabeler:
    def __init__(
        self,
        failure_detector: FailureDetector | None = None,
        outcome_extractor: OutcomeExtractor | None = None,
        max_per_failed_task: int = 2,
        min_partial_score: float = 0.2,
        mode: str = "hybrid",
        llm_client: Any | None = None,
        progress: Any | None = None,
    ):
        self.failure_detector = failure_detector or FailureDetector()
        self.outcome_extractor = outcome_extractor or OutcomeExtractor()
        self.max_per_failed_task = max_per_failed_task
        self.min_partial_score = min_partial_score
        self.mode = mode
        self.llm_client = llm_client
        self.progress = progress

    def _progress(self, event: str, **payload: Any) -> None:
        if self.progress is not None:
            self.progress(event, **payload)

    async def relabel(self, failed_records: list[TrajectoryRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        her_sft = []
        skipped = Counter()
        failure_modes = Counter()
        per_task_counts = Counter()
        relabelable = 0
        for index, record in enumerate(failed_records, start=1):
            self._progress("her:record:start", index=index, total=len(failed_records), record_id=record.id)
            failure = self.failure_detector.classify(record)
            failure_modes[failure["failure_type"]] += 1
            if not failure["recoverable"]:
                skipped["unrecoverable"] += 1
                self._progress("her:record:skip", record_id=record.id, reason="unrecoverable")
                continue
            if per_task_counts[record.task_id] >= self.max_per_failed_task:
                skipped["max_per_failed_task"] += 1
                self._progress("her:record:skip", record_id=record.id, reason="max_per_failed_task")
                continue
            outcomes = self.outcome_extractor.extract(record, self.min_partial_score)
            self._progress("her:heuristic:done", record_id=record.id, candidates=len(outcomes))
            if self.mode in {"llm", "hybrid"} and self.llm_client is not None:
                self._progress("her:llm_extract:start", record_id=record.id)
                llm_outcomes = await self._llm_extract_outcomes(record, failure, outcomes)
                self._progress("her:llm_extract:done", record_id=record.id, candidates=len(llm_outcomes))
                if llm_outcomes:
                    outcomes = llm_outcomes
            if not outcomes:
                skipped["no_partial_outcome"] += 1
                self._progress("her:record:skip", record_id=record.id, reason="no_partial_outcome")
                continue
            outcome = max(outcomes, key=lambda item: item.confidence)
            relabeled = self._to_sft(record, outcome, failure)
            if self.mode in {"llm", "hybrid"} and self.llm_client is not None:
                self._progress("her:llm_rewrite_validate:start", record_id=record.id)
                relabeled = await self._llm_rewrite_and_validate(record, outcome, failure, relabeled)
                if relabeled is None:
                    skipped["llm_validation_failed"] += 1
                    self._progress("her:record:skip", record_id=record.id, reason="llm_validation_failed")
                    continue
                self._progress("her:llm_rewrite_validate:done", record_id=record.id)
            relabelable += 1
            per_task_counts[record.task_id] += 1
            her_sft.append(relabeled)
            self._progress("her:record:done", record_id=record.id, output_count=len(her_sft))
        return her_sft, {
            "enabled": True,
            "input_failed_count": len(failed_records),
            "relabelable_count": relabelable,
            "output_her_count": len(her_sft),
            "failure_mode_counts": dict(failure_modes),
            "skipped_reasons": dict(skipped),
        }

    async def _llm_extract_outcomes(
        self,
        record: TrajectoryRecord,
        failure: dict[str, Any],
        heuristic_outcomes: list[PartialOutcome],
    ) -> list[PartialOutcome]:
        prompt = {
            "failure": failure,
            "heuristic_outcomes": [asdict(outcome) for outcome in heuristic_outcomes],
            "trajectory_prefix": record.raw.get("steps", [])[:8],
        }
        raw = await self._call_llm(
            "Extract factual partial achievements from a failed agent trajectory. Return JSON with key achievements, "
            "a list of objects with description, step_index, confidence.",
            prompt,
        )
        parsed = parse_judge_json(raw)
        outcomes = []
        for item in parsed.get("achievements", []):
            confidence = clamp(float(item.get("confidence", 0.0)))
            if confidence >= self.min_partial_score:
                outcomes.append(
                    PartialOutcome(
                        description=str(item.get("description", "")),
                        evidence={"llm_extracted": item},
                        step_index=int(item.get("step_index", 0)),
                        confidence=confidence,
                    )
                )
        return outcomes

    async def _llm_rewrite_and_validate(
        self,
        record: TrajectoryRecord,
        outcome: PartialOutcome,
        failure: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        rewrite_raw = await self._call_llm(
            "Rewrite this failed trajectory into a hindsight relabelled SFT example for the achieved subgoal. "
            "Return JSON with system_prompt, user_prompt, final_answer.",
            {"failure": failure, "partial_outcome": asdict(outcome), "fallback": fallback},
        )
        try:
            rewrite = parse_judge_json(rewrite_raw)
        except json.JSONDecodeError:
            return fallback
        candidate = dict(fallback)
        candidate["messages"] = [
            {"role": "system", "content": str(rewrite.get("system_prompt", fallback["messages"][0]["content"]))},
            {"role": "user", "content": str(rewrite.get("user_prompt", fallback["messages"][1]["content"]))},
            {"role": "assistant", "content": f"Final Answer: {rewrite.get('final_answer', outcome.description)}"},
        ]
        validate_raw = await self._call_llm(
            "Validate whether the hindsight prompt and final answer are fully supported by the trajectory evidence. "
            "Return JSON with valid (bool), score (0-1), reason.",
            {
                "candidate": candidate,
                "partial_outcome": asdict(outcome),
                "trajectory_prefix": record.raw.get("steps", [])[:8],
            },
        )
        try:
            validation = parse_judge_json(validate_raw)
        except json.JSONDecodeError:
            return None
        valid = bool(validation.get("valid", validation.get("passed", False)))
        score = clamp(float(validation.get("score", 0.0)))
        if not valid or score < 0.5:
            return None
        candidate["metadata"] = {
            **candidate.get("metadata", {}),
            "llm_rewrite": rewrite,
            "llm_validation": validation,
            "her_mode": self.mode,
        }
        return candidate

    async def _call_llm(self, system_prompt: str, payload: dict[str, Any]) -> str:
        if hasattr(self.llm_client, "achat"):
            return await self.llm_client.achat(
                model=getattr(self.llm_client, "model", "default"),
                messages=[
                    {"role": "system", "content": system_prompt + " Output ONLY valid JSON."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
        result = self.llm_client(system_prompt, payload)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)

    def _to_sft(self, record: TrajectoryRecord, outcome: PartialOutcome, failure: dict[str, Any]) -> dict[str, Any]:
        system = (
            "This is a hindsight-relabelled AgentHER trajectory. The original task failed, but the trajectory "
            "successfully achieved the subgoal below. Train only on the achieved subgoal."
        )
        user = (
            "Hindsight subgoal: reproduce the verified partial outcome achieved during the trajectory.\n\n"
            f"Partial outcome: {outcome.description}"
        )
        return {
            "task_id": f"her_{record.task_id}_{outcome.step_index}",
            "domain": record.domain,
            "difficulty": record.difficulty,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": f"Final Answer: {outcome.description}"},
            ],
            "metadata": {
                "her": True,
                "her_mode": self.mode,
                "source_record_id": record.id,
                "source_task_id": record.task_id,
                "source_raw_path": str(record.raw_path),
                "failure": failure,
                "partial_outcome": asdict(outcome),
            },
        }


class DeduplicatorMinhash:
    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        bands: int = 32,
        rows: int = 4,
        jaccard_threshold: float = 0.8,
        embedding_similarity_threshold: float = 0.9,
        enable_embedding_dedup: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_diagnostics_top_k: int = 50,
    ):
        if bands * rows != num_perm:
            raise ValueError("bands * rows must equal num_perm")
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.bands = bands
        self.rows = rows
        self.jaccard_threshold = jaccard_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.enable_embedding_dedup = enable_embedding_dedup
        self.embedding_model = embedding_model
        self.embedding_diagnostics_top_k = embedding_diagnostics_top_k
        self._seeds = [self._stable_hash64(f"seed:{i}") for i in range(num_perm)]

    def deduplicate(self, records: list[TrajectoryRecord]) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
        if len(records) <= 1:
            for record in records:
                record.metadata["level3"] = {"implemented": True, "kept": True, "duplicate_group_id": None}
            return records, {
                "duplicate_groups": 0,
                "filtered_duplicates": 0,
                "embedding_enabled": self.enable_embedding_dedup,
            }

        grams_by_id = {record.id: self._ngrams(record.dedup_text) for record in records}
        signatures = {record.id: self.signature_from_ngrams(grams_by_id[record.id]) for record in records}
        candidate_pairs = self._candidate_pairs(records, signatures)
        parent = {record.id: record.id for record in records}
        pair_metadata: dict[tuple[str, str], dict[str, float | str]] = {}

        for left, right in candidate_pairs:
            exact_jaccard = self._jaccard(grams_by_id[left], grams_by_id[right])
            if exact_jaccard >= self.jaccard_threshold:
                self._union(parent, left, right)
                pair_metadata[tuple(sorted((left, right)))] = {"jaccard": exact_jaccard, "reason": "minhash_lsh"}

        embedding_meta = self._apply_embedding_refinement(records, parent, pair_metadata)
        groups = self._groups(parent)
        record_by_id = {record.id: record for record in records}
        kept_ids: set[str] = set()
        duplicate_groups = 0

        for group_index, group_ids in enumerate(groups.values(), start=1):
            group_id = f"group_{group_index:04d}" if len(group_ids) > 1 else None
            best_id = self._best_record_id([record_by_id[record_id] for record_id in group_ids])
            if len(group_ids) > 1:
                duplicate_groups += 1
            kept_ids.add(best_id)
            for record_id in group_ids:
                record = record_by_id[record_id]
                kept = record_id == best_id
                pair_key = tuple(sorted((record_id, best_id)))
                duplicate_pair = pair_metadata.get(pair_key, {})
                dedup_reasons = []
                if not kept and len(group_ids) > 1:
                    dedup_reasons.append(str(duplicate_pair.get("reason") or "duplicate_lower_quality"))
                record.metadata["level3"] = {
                    "implemented": True,
                    "kept": kept,
                    "quality_score": record.quality_score,
                    "quality_score_source": record.quality_score_source,
                    "duplicate_group_id": group_id,
                    "duplicate_of": None if kept else best_id,
                    "dedup_reasons": dedup_reasons,
                    "similarity": duplicate_pair,
                }

        kept_records = [record for record in records if record.id in kept_ids]
        group_summaries = self._duplicate_group_summaries(groups, record_by_id)
        return kept_records, {
            "duplicate_groups": duplicate_groups,
            "filtered_duplicates": len(records) - len(kept_records),
            "candidate_pairs": len(candidate_pairs),
            "embedding_enabled": self.enable_embedding_dedup,
            "duplicate_group_summaries": group_summaries,
            **embedding_meta,
        }

    def signature(self, text: str) -> tuple[int, ...]:
        return self.signature_from_ngrams(self._ngrams(text))

    def signature_from_ngrams(self, grams: set[str]) -> tuple[int, ...]:
        if not grams:
            return tuple([2**64 - 1] * self.num_perm)
        signature = [2**64 - 1] * self.num_perm
        for gram in grams:
            base_hash = self._stable_hash64(gram)
            for index, seed in enumerate(self._seeds):
                signature[index] = min(signature[index], self._mix_hash(base_hash, seed))
        return tuple(signature)

    def _candidate_pairs(
        self,
        records: list[TrajectoryRecord],
        signatures: dict[str, tuple[int, ...]],
    ) -> set[tuple[str, str]]:
        buckets: dict[tuple[int, str], list[str]] = defaultdict(list)
        for record in records:
            signature = signatures[record.id]
            for band_index in range(self.bands):
                start = band_index * self.rows
                band = signature[start : start + self.rows]
                band_hash = hashlib.blake2b(repr(band).encode("utf-8"), digest_size=8).hexdigest()
                buckets[(band_index, band_hash)].append(record.id)

        pairs: set[tuple[str, str]] = set()
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            for index, left in enumerate(ids):
                for right in ids[index + 1 :]:
                    pairs.add(tuple(sorted((left, right))))
        return pairs

    def _apply_embedding_refinement(
        self,
        records: list[TrajectoryRecord],
        parent: dict[str, str],
        pair_metadata: dict[tuple[str, str], dict[str, float | str]],
    ) -> dict[str, Any]:
        if not self.enable_embedding_dedup or len(records) < 2:
            return {"embedding_pairs_above_threshold": 0, "embedding_error": None}
        pairs_above = 0
        all_similarities: list[float] = []
        top_pairs: list[dict[str, Any]] = []
        try:
            texts = [record.dedup_text for record in records]
            embeddings = encode_texts_with_transformers(texts, self.embedding_model)
            similarities = cosine_similarity_matrix(embeddings)
        except Exception as exc:
            return {"embedding_pairs_above_threshold": 0, "embedding_error": str(exc)}
        for i, left in enumerate(records):
            for j in range(i + 1, len(records)):
                right = records[j]
                similarity = float(similarities[i][j])
                all_similarities.append(similarity)
                if similarity > self.embedding_similarity_threshold:
                    self._union(parent, left.id, right.id)
                    pairs_above += 1
                    pair_metadata[tuple(sorted((left.id, right.id)))] = {
                        "embedding_similarity": similarity,
                        "reason": "embedding_similarity_above_threshold",
                    }
                top_pairs.append(
                    {
                        "left_id": left.id,
                        "left_task_id": left.task_id,
                        "right_id": right.id,
                        "right_task_id": right.task_id,
                        "same_task": left.task_id == right.task_id,
                        "similarity": round(similarity, 4),
                        "above_threshold": similarity > self.embedding_similarity_threshold,
                    }
                )
        top_pairs.sort(key=lambda pair: pair["similarity"], reverse=True)
        same_task_above = sum(1 for p in top_pairs if p["above_threshold"] and p["same_task"])
        cross_task_above = sum(1 for p in top_pairs if p["above_threshold"] and not p["same_task"])
        similarity_distribution = self._similarity_histogram(all_similarities)
        return {
            "embedding_pairs_above_threshold": pairs_above,
            "embedding_error": None,
            "embedding_same_task_pairs_above_threshold": same_task_above,
            "embedding_cross_task_pairs_above_threshold": cross_task_above,
            "embedding_similarity_distribution": similarity_distribution,
            "embedding_top_pairs": top_pairs[: self.embedding_diagnostics_top_k],
        }

    @staticmethod
    def _similarity_histogram(similarities: list[float], bucket_width: float = 0.05) -> dict[str, int]:
        if not similarities:
            return {}
        buckets: dict[str, int] = {}
        for sim in similarities:
            lower = int(sim / bucket_width) * bucket_width
            upper = lower + bucket_width
            label = f"{lower:.2f}-{upper:.2f}"
            buckets[label] = buckets.get(label, 0) + 1
        return dict(sorted(buckets.items()))

    def _best_record_id(self, records: list[TrajectoryRecord]) -> str:
        def key(record: TrajectoryRecord) -> tuple[float, float, int, str]:
            step_count = len(record.raw.get("steps") or [])
            success = 1.0 if record.raw.get("success") else 0.0
            return (record.quality_score, success, -step_count, str(record.raw_path))

        return max(records, key=key).id

    def _duplicate_group_summaries(
        self,
        groups: dict[str, list[str]],
        record_by_id: dict[str, TrajectoryRecord],
    ) -> list[dict[str, Any]]:
        summaries = []
        for group_ids in groups.values():
            if len(group_ids) <= 1:
                continue
            records = [record_by_id[record_id] for record_id in group_ids]
            task_ids = sorted({record.task_id for record in records})
            summaries.append(
                {
                    "size": len(records),
                    "unique_task_count": len(task_ids),
                    "same_task_group": len(task_ids) == 1,
                    "task_ids": task_ids[:10],
                    "kept_id": self._best_record_id(records),
                    "record_ids": sorted(group_ids)[:20],
                }
            )
        summaries.sort(key=lambda item: item["size"], reverse=True)
        return summaries[: self.embedding_diagnostics_top_k]

    def _groups(self, parent: dict[str, str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for record_id in parent:
            groups[self._find(parent, record_id)].append(record_id)
        return groups

    def _find(self, parent: dict[str, str], value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def _union(self, parent: dict[str, str], left: str, right: str) -> None:
        left_root = self._find(parent, left)
        right_root = self._find(parent, right)
        if left_root != right_root:
            parent[right_root] = left_root

    def _ngrams(self, text: str) -> set[str]:
        tokens = self._tokens(text)
        n = max(1, min(self.ngram_size, len(tokens)))
        return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
        return re.findall(r"[\w]+|[一-鿿]|[^\s\w]", normalized, flags=re.UNICODE)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _stable_hash64(text: str) -> int:
        return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")

    @staticmethod
    def _mix_hash(value: int, seed: int) -> int:
        mixed = (value ^ seed) & ((1 << 64) - 1)
        mixed ^= mixed >> 33
        mixed = (mixed * 0xff51afd7ed558ccd) & ((1 << 64) - 1)
        mixed ^= mixed >> 33
        mixed = (mixed * 0xc4ceb9fe1a85ec53) & ((1 << 64) - 1)
        mixed ^= mixed >> 33
        return mixed


class DiversityMonitor:
    def __init__(self, sample_size: int = 500):
        self.sample_size = sample_size
        self.metrics = DiversityMetrics()

    def calculate(self, texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {}
        try:
            metrics = self.metrics.get_all_metrics(texts)
        except Exception as exc:
            metrics = {"diversity_error": str(exc), "num_texts": len(texts)}
        for n in range(1, 5):
            entropy, normalized = self.ngram_entropy(texts, n)
            metrics[f"ngram_entropy_{n}"] = entropy
            metrics[f"normalized_ngram_entropy_{n}"] = normalized
        return metrics

    @staticmethod
    def ngram_entropy(texts: list[str], n: int) -> tuple[float, float]:
        counter: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            tokens = DeduplicatorMinhash._tokens(text)
            if len(tokens) < n:
                continue
            counter.update(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
        total = sum(counter.values())
        if total == 0:
            return 0.0, 0.0
        entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
        normalized = entropy / math.log2(len(counter)) if len(counter) > 1 else 0.0
        return entropy, normalized


class QualityFilter:
    def __init__(
        self,
        config: QualityFilterConfig | None = None,
        result_verifier: ResultVerifier | None = None,
        deduplicator: DeduplicatorMinhash | None = None,
        diversity_monitor: DiversityMonitor | None = None,
        process_reward_scorer: ProcessRewardScorer | None = None,
        difficulty_classifier: DifficultyClassifier | None = None,
        difficulty_sampler: DifficultyAwareSampler | None = None,
        her_relabeler: HERRelabeler | None = None,
        her_llm_client: Any | None = None,
        progress: Any | None = None,
    ):
        self.config = config or QualityFilterConfig()
        self.progress = progress
        self.result_verifier = result_verifier or ResultVerifier()
        self.deduplicator = deduplicator or DeduplicatorMinhash(
            num_perm=self.config.minhash_num_perm,
            ngram_size=self.config.minhash_ngram,
            bands=self.config.lsh_bands,
            rows=self.config.lsh_rows,
            jaccard_threshold=self.config.minhash_jaccard_threshold,
            embedding_similarity_threshold=self.config.embedding_similarity_threshold,
            enable_embedding_dedup=self.config.enable_embedding_dedup,
            embedding_model=self.config.embedding_model,
            embedding_diagnostics_top_k=self.config.embedding_diagnostics_top_k,
        )
        self.diversity_monitor = diversity_monitor or DiversityMonitor(self.config.diversity_sample_size)
        self.difficulty_classifier = difficulty_classifier or DifficultyClassifier(
            self.config.difficulty_pass32_buckets
        )
        self.process_reward_scorer = process_reward_scorer or ProcessRewardScorer(
            min_score=self.config.level2_min_score,
            mc_weight=self.config.level2_mc_weight,
            judge_weight=self.config.level2_judge_weight,
            judge_consensus=LLMJudgeConsensus(
                consensus_k=self.config.level2_judge_consensus_k,
                fail_closed=self.config.level2_fail_closed_on_judge_error,
            ),
        )
        self.difficulty_sampler = difficulty_sampler or DifficultyAwareSampler(
            ratio=self.config.difficulty_sampling_ratio,
            seed=self.config.level4_seed,
        )
        self.her_relabeler = her_relabeler or HERRelabeler(
            max_per_failed_task=self.config.her_max_per_failed_task,
            min_partial_score=self.config.her_min_partial_score,
            mode=self.config.her_mode,
            llm_client=her_llm_client,
            progress=self._progress,
        )

    async def run(self) -> dict[str, Any]:
        task_map = load_task_map(self.config.task_file) if self.config.task_file else {}
        self._progress("load_records:start")
        records, load_failures = self.load_records()
        self._progress("load_records:done", total=len(records), failures=len(load_failures))
        self._progress("difficulty:start", total=len(records))
        task_stats, difficulty_diagnostics = self.difficulty_classifier.classify(records)
        self._progress("difficulty:done", tasks=len(task_stats))
        self._progress("level1:start", total=len(records))
        level1_records, stage1 = await self._run_level1(records, task_map)
        self._progress("level1:done", passed=len(level1_records), total=len(records))
        level1_failed = [record for record in records if not record.metadata.get("level1", {}).get("passed")]
        self._progress("her:start", failed=len(level1_failed))
        her_sft, her_report = await self._run_her(level1_failed)
        self._progress("her:done", relabeled=len(her_sft))
        self._progress("level2:start", total=len(level1_records))
        level2_records, stage2 = await self._run_level2(level1_records, task_map, task_stats)
        self._progress("level2:done", passed=len(level2_records), total=len(level1_records))
        self._progress("level3:start", total=len(level2_records))
        level3_records, stage3 = self._run_level3(level2_records)
        self._progress("level3:done", passed=len(level3_records), total=len(level2_records))
        self._apply_composite_scores(level2_records)
        self._progress("level4:start", total=len(level3_records))
        final_records, stage4 = self._run_level4(level3_records)
        self._progress("level4:done", selected=len(final_records), total=len(level3_records))

        stage1.failures.extend(load_failures)
        stages = [stage1, stage2, stage3, stage4]
        base_sft_count = len([record for record in final_records if record.sft is not None])
        her_report["before_sft_count"] = base_sft_count
        her_report["after_sft_count"] = base_sft_count + len(her_sft)
        filtered_sft = [record.sft for record in final_records if record.sft is not None]
        if self.config.her_include_in_filtered_sft:
            filtered_sft.extend(her_sft)
        return {
            "created_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "summary": {
                "input_count": len(records),
                "final_count": len(final_records),
                "overall_pass_rate": len(final_records) / len(records) if records else 0.0,
                "her_count": len(her_sft),
                "filtered_sft_count": len([record for record in final_records if record.sft is not None]),
                "filtered_sft_with_her_count": len(filtered_sft),
            },
            "funnel": [stage.to_dict() for stage in stages],
            "difficulty_diagnostics": difficulty_diagnostics,
            "her_relabeling": her_report,
            "her_sft": her_sft,
            "kept": [record.summary() for record in final_records],
            "filtered": [record.summary() for record in records if record not in final_records],
            "failures": [failure for stage in stages for failure in stage.failures],
            "filtered_sft": filtered_sft,
        }

    def _progress(self, event: str, **payload: Any) -> None:
        if self.progress is not None:
            self.progress(event, **payload)

    def load_records(self) -> tuple[list[TrajectoryRecord], list[dict[str, Any]]]:
        failures: list[dict[str, Any]] = []
        records: list[TrajectoryRecord] = []
        raw_paths = sorted(self.config.input_dir.glob(self.config.raw_glob))
        if self.config.limit is not None:
            raw_paths = raw_paths[: self.config.limit]

        for raw_path in raw_paths:
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append({"path": str(raw_path), "reason": "raw_load_failed", "error": str(exc)})
                continue

            if self.config.domain and raw.get("domain") != self.config.domain:
                continue

            sft_path = Path(str(raw_path).replace("_raw.json", "_sft.json"))
            sft = None
            if sft_path.exists():
                try:
                    sft = json.loads(sft_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    failures.append({"path": str(sft_path), "reason": "sft_load_failed", "error": str(exc)})
            else:
                failures.append({"path": str(raw_path), "reason": "missing_sft_pair", "sft_path": str(sft_path)})
                sft_path = None

            final_answer = extract_final_answer(raw)
            record = TrajectoryRecord(
                id=raw_path.stem.replace("_raw", ""),
                task_id=str(raw.get("task_id", "")),
                domain=str(raw.get("domain", "")),
                difficulty=str(raw.get("difficulty", "")),
                raw_path=raw_path,
                sft_path=sft_path,
                raw=raw,
                sft=sft,
                final_answer=final_answer,
            )
            record.dedup_text = build_dedup_text(record, self.config.dedup_text_config)
            records.append(record)

        return records, failures

    async def _run_level1(
        self,
        records: list[TrajectoryRecord],
        task_map: dict[str, dict[str, Any]],
    ) -> tuple[list[TrajectoryRecord], StageResult]:
        semaphore = asyncio.Semaphore(self.config.level1_concurrency)

        async def verify(record: TrajectoryRecord) -> TrajectoryRecord:
            async with semaphore:
                task = task_map.get(record.task_id)
                return await self.result_verifier.verify_record(record, task, self.config.fail_open_missing_task)

        verified = await asyncio.gather(*(verify(record) for record in records))
        passed = [record for record in verified if record.metadata.get("level1", {}).get("passed")]
        failures = []
        for record in verified:
            level1 = record.metadata.get("level1", {})
            if level1.get("passed"):
                continue
            details = level1.get("details") or {}
            failures.append(
                {
                    "id": record.id,
                    "task_id": record.task_id,
                    "raw_path": str(record.raw_path),
                    "reason": level1.get("error") or details.get("reason") or "verification_failed",
                }
            )
        return passed, StageResult("level1_result_verifier", True, len(records), len(passed), failures=failures)

    async def _run_level2(
        self,
        records: list[TrajectoryRecord],
        task_map: dict[str, dict[str, Any]],
        task_stats: dict[str, TaskPassStats],
    ) -> tuple[list[TrajectoryRecord], StageResult]:
        if not self.config.enable_level2_prm:
            return self._skip_level2(records)
        scored = []
        failures = []
        scores = []
        step_score_count = 0
        for index, record in enumerate(records, start=1):
            self._progress("level2:record:start", index=index, total=len(records), record_id=record.id)
            result = await self.process_reward_scorer.score_record(
                record,
                task_map.get(record.task_id),
                task_stats.get(record.task_id),
            )
            record.quality_score = result.score
            record.quality_score_source = "level2_prm_soft_score"
            record.metadata["level2"] = {
                "implemented": True,
                "passed": result.passed,
                "score": result.score,
                "threshold": self.config.level2_min_score,
                "step_scores": [step.to_dict() for step in result.step_scores],
                "judge_consensus": result.judge_consensus,
                "mc_metadata": result.mc_metadata,
            }
            scores.append(result.score)
            step_score_count += len(result.step_scores)
            self._progress(
                "level2:record:done",
                index=index,
                total=len(records),
                record_id=record.id,
                score=round(result.score, 4),
                passed=result.passed,
            )
            if result.passed:
                scored.append(record)
            else:
                failures.append({"id": record.id, "task_id": record.task_id, "reason": "low_prm_score"})
        return scored, StageResult(
            "level2_prm_mc_llm_judge",
            True,
            len(records),
            len(scored),
            metadata={
                "min_score": self.config.level2_min_score,
                "mean_prm_score": sum(scores) / len(scores) if scores else 0.0,
                "min_prm_score": min(scores) if scores else 0.0,
                "max_prm_score": max(scores) if scores else 0.0,
                "filtered_below_threshold": len(records) - len(scored),
                "per_step_score_count": step_score_count,
                "judge_enabled": any(
                    not record.metadata.get("level2", {}).get("judge_consensus", {}).get("skipped", True)
                    for record in records
                ),
            },
            failures=failures,
        )

    @staticmethod
    def _skip_level2(records: list[TrajectoryRecord]) -> tuple[list[TrajectoryRecord], StageResult]:
        for record in records:
            record.metadata["level2"] = {
                "implemented": False,
                "skipped": True,
                "reason": "future_iteration",
            }
        return records, StageResult(
            "level2_prm_mc_llm_judge",
            False,
            len(records),
            len(records),
            skipped=True,
            metadata={
                "reason": "not_implemented_this_iteration",
                "planned_components": ["PRM Monte Carlo", "LLM-as-judge consensus"],
            },
        )

    def _run_level3(self, records: list[TrajectoryRecord]) -> tuple[list[TrajectoryRecord], StageResult]:
        diversity_before = self.diversity_monitor.calculate([record.dedup_text for record in records])
        kept, dedup_metadata = self.deduplicator.deduplicate(records)
        diversity_after = self.diversity_monitor.calculate([record.dedup_text for record in kept])
        return kept, StageResult(
            "level3_deduplication_diversity",
            True,
            len(records),
            len(kept),
            metadata={
                **dedup_metadata,
                "diversity_before": diversity_before,
                "diversity_after": diversity_after,
            },
        )

    async def _run_her(self, failed_records: list[TrajectoryRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.config.enable_her_relabeling:
            return [], {"enabled": False, "input_failed_count": len(failed_records), "output_her_count": 0}
        her_sft, report = await self.her_relabeler.relabel(failed_records)
        report["included_in_filtered_sft"] = self.config.her_include_in_filtered_sft
        return her_sft, report

    def _apply_composite_scores(self, records: list[TrajectoryRecord]) -> None:
        for record in records:
            correct = 1.0 if record.metadata.get("level1", {}).get("passed") else 0.0
            prm = float(record.metadata.get("level2", {}).get("score", record.quality_score) or 0.0)
            level3 = record.metadata.get("level3", {})
            diversity = 1.0 if level3.get("kept", True) else 0.0
            similarity = level3.get("similarity") or {}
            if similarity.get("embedding_similarity") is not None:
                diversity = max(0.0, 1.0 - float(similarity["embedding_similarity"]))
            composite = 0.4 * correct + 0.3 * prm + 0.2 * diversity
            record.metadata["composite_score"] = {
                "formula": "0.4*1[correct]+0.3*PRM+0.2*diversity",
                "correct_component": correct,
                "prm_component": prm,
                "diversity_component": diversity,
                "score": composite,
                "formula_weight_sum": 0.9,
            }

    def _run_level4(self, records: list[TrajectoryRecord]) -> tuple[list[TrajectoryRecord], StageResult]:
        if not self.config.enable_level4_sampling:
            return self._skip_level4(records)
        selected, metadata = self.difficulty_sampler.sample(records, self.config.level4_target_count)
        return selected, StageResult(
            "level4_difficulty_aware_sampling",
            True,
            len(records),
            len(selected),
            metadata=metadata,
        )

    @staticmethod
    def _skip_level4(records: list[TrajectoryRecord]) -> tuple[list[TrajectoryRecord], StageResult]:
        for record in records:
            record.metadata["level4"] = {
                "implemented": False,
                "skipped": True,
                "reason": "future_iteration",
            }
        return records, StageResult(
            "level4_difficulty_aware_sampling",
            False,
            len(records),
            len(records),
            skipped=True,
            metadata={
                "reason": "not_implemented_this_iteration",
                "planned_buckets": ["easy", "medium", "hard", "extreme"],
                "planned_sampling_ratio": "1:3:4:2",
            },
        )


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return json.loads(match.group(0) if match else text)


def record_composite_score(record: TrajectoryRecord) -> float:
    return float(record.metadata.get("composite_score", {}).get("score", record.quality_score) or 0.0)


def encode_texts_with_transformers(texts: list[str], model_name: str, batch_size: int = 32) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()

    embeddings: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.cpu().tolist())
    return embeddings


def cosine_similarity_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in embeddings:
        row = []
        for right in embeddings:
            row.append(sum(a * b for a, b in zip(left, right)))
        matrix.append(row)
    return matrix


def load_task_map(task_file: Path | str | None) -> dict[str, dict[str, Any]]:
    if task_file is None:
        return {}
    payload = json.loads(Path(task_file).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("prompts", "tasks", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Task file must contain a list or a dict with prompts/tasks/data")
    return {str(item.get("id")): item for item in payload if isinstance(item, dict) and item.get("id")}


def extract_final_answer(raw: dict[str, Any]) -> Any:
    for step in reversed(raw.get("steps") or []):
        action = step.get("action") or {}
        if action.get("action_type") == "final_answer":
            return action.get("answer")
    final_state = raw.get("final_state") or {}
    last_action = final_state.get("last_action") or {}
    if isinstance(last_action, dict) and last_action.get("action_type") == "final_answer":
        return last_action.get("answer")
    return None


def build_dedup_text(record: TrajectoryRecord, config: DedupTextConfig | None = None) -> str:
    config = config or DedupTextConfig()
    options = config.resolved()
    parts = []

    if options["include_prompt"]:
        prompt = extract_prompt_text(record)
        if prompt:
            parts.append(f"prompt: {prompt}")
    if options["include_thought"]:
        parts.extend(extract_thought_texts(record, include_final_answer=options["include_final_answer"]))
    if options["include_action"]:
        parts.extend(extract_action_texts(record))
    if options["include_observation"]:
        parts.extend(extract_observation_texts(record, include_errors=options["include_error"]))
    elif options["include_error"]:
        parts.extend(extract_error_texts(record))
    if options["include_final_answer"] and record.final_answer is not None:
        parts.append(f"final_answer: {record.final_answer}")

    return normalize_dedup_text("\n".join(parts))


def extract_thought_texts(record: TrajectoryRecord, include_final_answer: bool) -> list[str]:
    texts = []
    for step in record.raw.get("steps") or []:
        if step.get("thought"):
            texts.append(f"thought: {step['thought']}")
        action = step.get("action") or {}
        action_thought = action.get("thought")
        if action_thought and (include_final_answer or action.get("action_type") != "final_answer"):
            texts.append(f"thought: {action_thought}")
    if record.sft:
        for message in record.sft.get("messages") or []:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "assistant" and content:
                if not include_final_answer and "final answer" in content.lower():
                    continue
                texts.append(f"assistant: {content}")
    return texts


def extract_action_texts(record: TrajectoryRecord) -> list[str]:
    texts = []
    for step in record.raw.get("steps") or []:
        action = step.get("action") or {}
        if action:
            texts.append(f"action: {json.dumps(action, ensure_ascii=False, sort_keys=True)}")
    return texts


def extract_observation_texts(record: TrajectoryRecord, include_errors: bool) -> list[str]:
    texts = []
    for step in record.raw.get("steps") or []:
        observation = dict(step.get("observation") or {})
        if not include_errors:
            observation.pop("error", None)
        if observation:
            texts.append(f"observation: {json.dumps(observation, ensure_ascii=False, sort_keys=True)}")
    if record.sft:
        for message in record.sft.get("messages") or []:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role in {"tool", "observation"} and content:
                texts.append(f"{role}: {content}")
    return texts


def extract_error_texts(record: TrajectoryRecord) -> list[str]:
    texts = []
    for step in record.raw.get("steps") or []:
        error = (step.get("observation") or {}).get("error")
        if error:
            texts.append(f"error: {error}")
    return texts


def normalize_dedup_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_prompt_text(record: TrajectoryRecord) -> str:
    if record.sft:
        for message in record.sft.get("messages") or []:
            if message.get("role") == "user" and message.get("content"):
                return str(message.get("content"))
    for step in record.raw.get("steps") or []:
        snapshot = step.get("state_snapshot") or {}
        metadata = snapshot.get("metadata") or {}
        if metadata.get("task_prompt"):
            return str(metadata["task_prompt"])
    return str(record.raw.get("task_prompt") or "")


def write_quality_outputs(report: dict[str, Any], report_path: Path, filtered_path: Path | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if filtered_path is not None:
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_path.write_text(
            json.dumps(_json_safe(report.get("filtered_sft", [])), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
