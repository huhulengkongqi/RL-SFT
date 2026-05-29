"""Trajectory quality filtering funnel."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
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
            result = await self._verify_by_domain(domain, record.final_answer, test_cases)
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

    async def _verify_by_domain(self, domain: str, answer: Any, test_cases: list[dict[str, Any]]) -> VerificationResult:
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
            return result

        kwargs: dict[str, Any] = {}
        if test_cases and isinstance(test_cases[0].get("expected_output"), dict):
            kwargs["required_fields"] = list(test_cases[0]["expected_output"].keys())
        return await self.answer_verifier.verify(answer, mode=VerificationMode.FORMAT_VALIDATION, **kwargs)

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
    ):
        self.config = config or QualityFilterConfig()
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

    async def run(self) -> dict[str, Any]:
        task_map = load_task_map(self.config.task_file) if self.config.task_file else {}
        records, load_failures = self.load_records()
        level1_records, stage1 = await self._run_level1(records, task_map)
        level2_records, stage2 = self._skip_level2(level1_records)
        level3_records, stage3 = self._run_level3(level2_records)
        final_records, stage4 = self._skip_level4(level3_records)

        stage1.failures.extend(load_failures)
        stages = [stage1, stage2, stage3, stage4]
        return {
            "created_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "summary": {
                "input_count": len(records),
                "final_count": len(final_records),
                "overall_pass_rate": len(final_records) / len(records) if records else 0.0,
            },
            "funnel": [stage.to_dict() for stage in stages],
            "kept": [record.summary() for record in final_records],
            "filtered": [record.summary() for record in records if record not in final_records],
            "failures": [failure for stage in stages for failure in stage.failures],
            "filtered_sft": [record.sft for record in final_records if record.sft is not None],
        }

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
