"""Source-aware single-Gemma advisory grading for raw benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import load_records
from .grading import apply_single_model_advisory, grade_benchmark_artifact
from .mlx_judge_backend import (
    GEMMA_ADVISORY_RUBRIC_VERSION,
    GEMMA_ADVISORY_SCHEMA,
    GEMMA_GENERATION_CONFIG,
    MLXGemma4JudgeBackend,
    canonical_evidence_packet,
)
from .schemas import KnowledgeRecord
from .split_contract import EvalSuites, load_eval_suites
from .utils import atomic_write_text, sha256_json, sha256_text

VERIFICATION_STATUS = "single_local_judge_advisory"


def run_gemma_advisory(
    *,
    benchmark_path: Path,
    eval_dir: Path,
    knowledge_dir: Path,
    output_dir: Path,
    backend: MLXGemma4JudgeBackend | None = None,
) -> tuple[Path, Path]:
    """Judge every generated row blindly, then apply deterministic hard failures."""
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    if not isinstance(benchmark, dict) or benchmark.get("graded") is not False:
        raise ValueError("Expected an ungraded benchmark artifact")
    suites = load_eval_suites(eval_dir)
    records = load_records(knowledge_dir)
    benchmark_config = benchmark.get("config")
    if not isinstance(benchmark_config, dict):
        raise ValueError("Benchmark artifact is missing its bound configuration")
    allowed_parametric_records = tuple(
        str(value) for value in benchmark_config.get("parametric_source_record_ids", [])
    )
    deterministic = grade_benchmark_artifact(
        benchmark_path,
        eval_dir,
        suites,
        allowed_parametric_record_ids=allowed_parametric_records,
    )
    deterministic_by_key = {(row.question_id, row.arm): row for row in deterministic.rows}
    raw_rows = benchmark.get("results")
    if not isinstance(raw_rows, list):
        raise ValueError("Benchmark results must be a list")
    questions = {question.question_id: question for question in suites.all_questions()}
    evidence_by_question = {
        question_id: _canonical_evidence_for_question(
            records,
            suites,
            question=question,
        )
        for question_id, question in questions.items()
    }
    evidence_hashes = {
        question_id: sha256_text(packet) for question_id, packet in evidence_by_question.items()
    }
    shared_evidence_hashes: dict[str, set[str]] = defaultdict(set)
    attempts = []
    local_backend = backend or MLXGemma4JudgeBackend()
    acquired = False
    judge_snapshot_identity: Mapping[str, Any] | None = None
    try:
        local_backend.acquire()
        acquired = True
        judge_snapshot_identity = local_backend.snapshot_identity
        for raw in sorted(
            raw_rows,
            key=lambda row: hashlib.sha256(
                f"{row.get('question_id')}::{row.get('arm')}".encode()
            ).hexdigest(),
        ):
            question_id = str(raw.get("question_id", ""))
            arm = str(raw.get("arm", ""))
            key = (question_id, arm)
            if key not in deterministic_by_key or question_id not in questions:
                raise ValueError(f"Benchmark row is not bound to frozen grading: {key}")
            deterministic_row = deterministic_by_key[key]
            question = questions[question_id]
            evidence = evidence_by_question[question_id]
            evidence_hash = evidence_hashes[question_id]
            shared_evidence_hashes[question_id].add(evidence_hash)
            if deterministic_row.status == "unscored_generation_failure":
                attempts.append(
                    _failed_generation_attempt(
                        raw,
                        deterministic_row.to_dict(),
                        error=(
                            deterministic_row.reasons[0]
                            if deterministic_row.reasons
                            else "generation_failure"
                        ),
                    )
                )
                continue
            candidate = raw.get("output")
            if not isinstance(candidate, str) or not candidate.strip():
                attempts.append(
                    _failed_generation_attempt(
                        raw,
                        deterministic_row.to_dict(),
                        error="missing_final_answer",
                    )
                )
                continue
            question_text = str(raw.get("question", "")).strip()
            if not question_text:
                raise ValueError(f"Benchmark row has no rendered question text: {key}")
            logical_hash = sha256_json(
                {
                    "rubric_version": GEMMA_ADVISORY_RUBRIC_VERSION,
                    "question": question_text,
                    "reference_answer": question.expected,
                    "candidate_answer": candidate,
                    "canonical_evidence_hash": evidence_hash,
                    "as_of_date": raw.get("as_of_date"),
                }
            )
            result, parsed = local_backend.judge_with_evidence(
                question=question_text,
                reference_answer=question.expected,
                candidate_answer=candidate,
                supporting_evidence=evidence,
                max_output_tokens=1024,
                logical_prompt_hash=logical_hash,
            )
            advisory = apply_single_model_advisory(
                deterministic_row,
                parsed.to_dict(),
            )
            attempts.append(
                {
                    "question_id": question_id,
                    "arm": arm,
                    "suite": question.suite,
                    "record_id": question.record_id,
                    "question_family_id": question.question_family_id,
                    "generation_status": raw.get("generation_status"),
                    "generator_parse_status": raw.get("parse_status", "legacy_plain"),
                    "generator_finish_reason": raw.get("finish_reason"),
                    "generator_elapsed_seconds": raw.get("elapsed_seconds"),
                    "generator_prompt_tokens": raw.get("prompt_tokens"),
                    "generator_completion_tokens": raw.get("completion_tokens"),
                    "generator_peak_memory_gb": raw.get("peak_memory_gb"),
                    "deterministic_status": deterministic_row.status,
                    "deterministic_reasons": list(deterministic_row.reasons),
                    "canonical_evidence_hash": evidence_hash,
                    "logical_prompt_hash": logical_hash,
                    "judge_result": result.to_dict(),
                    "model_label": parsed.to_dict(),
                    "governed_final_score": advisory.final_score,
                    "score_source": advisory.score_source,
                    "all_attempted_score": advisory.all_attempted_score,
                }
            )
    finally:
        if acquired:
            local_backend.release()
    if any(len(values) != 1 for values in shared_evidence_hashes.values()):
        raise RuntimeError("Canonical judge evidence differs across arms")
    report = _build_report(
        benchmark_path=benchmark_path,
        benchmark_hash=hashlib.sha256(benchmark_bytes).hexdigest(),
        benchmark=benchmark,
        deterministic=deterministic.to_dict(),
        attempts=attempts,
        evidence_hashes=evidence_hashes,
        judge_snapshot_identity=judge_snapshot_identity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = benchmark_path.stem
    json_path = output_dir / f"{stem}--gemma4-advisory.json"
    markdown_path = output_dir / f"{stem}--gemma4-advisory.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Refusing to overwrite immutable Gemma advisory")
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    atomic_write_text(markdown_path, _render_markdown(report))
    return json_path, markdown_path


def run_gemma_preflight(*, output_path: Path) -> Path:
    """Exercise one developer-only structured judgment before measured judging."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Gemma preflight: {output_path}")
    backend = MLXGemma4JudgeBackend()
    acquired = False
    snapshot_identity: Mapping[str, Any] | None = None
    try:
        backend.acquire()
        acquired = True
        snapshot_identity = backend.snapshot_identity
        result, parsed = backend.judge_with_evidence(
            question="What is the supplied payment term?",
            reference_answer="Thirty calendar days after receipt.",
            candidate_answer="The payment term is thirty calendar days after receipt.",
            supporting_evidence=canonical_evidence_packet(
                {
                    "DEV-PREFLIGHT": {
                        "id": "DEV-PREFLIGHT",
                        "title": "Developer-only payment fixture",
                        "statement": "The payment term is thirty calendar days after receipt.",
                        "source_uri": "local://developer-preflight",
                        "status": "active",
                        "effective_from": "2026-01-01",
                        "effective_to": None,
                    }
                }
            ),
            max_output_tokens=1024,
        )
    finally:
        if acquired:
            backend.release()
    if not parsed.valid:
        raise RuntimeError(f"Gemma structured-output preflight failed: {parsed.error}")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "developer_fixture_only": True,
        "model_id": local_model_id(),
        "model_revision": local_model_revision(),
        "generation_config": GEMMA_GENERATION_CONFIG,
        "generation_config_hash": sha256_json(GEMMA_GENERATION_CONFIG),
        "rubric_version": GEMMA_ADVISORY_RUBRIC_VERSION,
        "snapshot_identity": snapshot_identity,
        "judge_result": result.to_dict(),
        "parsed_label": parsed.to_dict(),
        "human_approved": False,
        "promotion_eligible": False,
    }
    atomic_write_text(
        output_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return output_path


def _canonical_evidence_for_question(
    records: Sequence[KnowledgeRecord],
    suites: EvalSuites,
    *,
    question: Any,
) -> str:
    all_versions = (
        tuple(records) + tuple(suites.holdout_records) + tuple(suites.supersession_current_records)
    )
    selected = (
        tuple(record for record in all_versions if record.id == question.record_id)
        if question.record_id is not None
        else all_versions
    )
    if question.record_id is not None and not selected:
        raise ValueError(
            f"Frozen question refers to unknown canonical record: {question.record_id}"
        )
    authoritative = {
        f"{record.id}@{record.effective_from or 'undated'}@{record.effective_to or 'open'}": (
            _record_payload(record)
        )
        for record in selected
    }
    return canonical_evidence_packet(authoritative)


def _record_payload(record: KnowledgeRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "statement": record.statement,
        "source_uri": record.source_uri,
        "status": record.status,
        "effective_from": record.effective_from,
        "effective_to": record.effective_to,
    }


def _failed_generation_attempt(
    raw: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "question_id": str(raw.get("question_id", "")),
        "arm": str(raw.get("arm", "")),
        "suite": str(raw.get("suite", "")),
        "record_id": deterministic.get("record_id"),
        "question_family_id": deterministic.get("question_family_id"),
        "generation_status": "failed",
        "artifact_generation_status": raw.get("generation_status"),
        "generator_parse_status": raw.get("parse_status"),
        "generator_finish_reason": raw.get("finish_reason"),
        "generator_elapsed_seconds": raw.get("elapsed_seconds"),
        "generator_prompt_tokens": raw.get("prompt_tokens"),
        "generator_completion_tokens": raw.get("completion_tokens"),
        "generator_peak_memory_gb": raw.get("peak_memory_gb"),
        "deterministic_status": deterministic.get("status"),
        "deterministic_reasons": list(deterministic.get("reasons", [])),
        "canonical_evidence_hash": None,
        "logical_prompt_hash": None,
        "judge_result": None,
        "model_label": {
            "valid": False,
            "score": None,
            "reason": None,
            "unsupported_claims": [],
            "needs_human_attention": True,
            "confidence": None,
            "error": error or str(raw.get("generation_error") or "generation_failure"),
        },
        "governed_final_score": None,
        "score_source": "unscored_generation_failure",
        "all_attempted_score": 0.0,
    }


def _build_report(
    *,
    benchmark_path: Path,
    benchmark_hash: str,
    benchmark: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    evidence_hashes: Mapping[str, str],
    judge_snapshot_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    invalid = sum(
        row.get("judge_result") is not None and not bool(row["model_label"]["valid"])
        for row in attempts
    )
    generation_failures = sum(row["generation_status"] != "generated" for row in attempts)
    by_arm = _aggregate(attempts, key_fields=("arm", "suite"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "report_type": "source_aware_single_gemma_advisory",
        "status": ("complete_advisory" if invalid == 0 else "blocking_verifier_defect"),
        "reviewer_kind": "model",
        "reviewer_family": "gemma",
        "verification_status": VERIFICATION_STATUS,
        "human_approved": False,
        "promotion_eligible": False,
        "usable_for_judge_certification": False,
        "rubric_version": GEMMA_ADVISORY_RUBRIC_VERSION,
        "rubric_hash": sha256_json(
            {
                "version": GEMMA_ADVISORY_RUBRIC_VERSION,
                "schema": GEMMA_ADVISORY_SCHEMA,
            }
        ),
        "generation_config": GEMMA_GENERATION_CONFIG,
        "generation_config_hash": sha256_json(GEMMA_GENERATION_CONFIG),
        "canonical_evidence_index_hash": sha256_json(dict(sorted(evidence_hashes.items()))),
        "canonical_evidence_hashes_by_question": dict(sorted(evidence_hashes.items())),
        "evidence_identical_across_arms": True,
        "model_id": local_model_id(),
        "model_revision": local_model_revision(),
        "precision": "8-bit affine, group_size 64",
        "snapshot_identity": judge_snapshot_identity,
        "case_count": len(attempts),
        "invalid_verifier_output_count": invalid,
        "generation_failure_count": generation_failures,
        "semantic_coverage_complete": generation_failures == 0,
        "artifact_chain": {
            "benchmark_path": str(benchmark_path.resolve()),
            "benchmark_sha256": benchmark_hash,
            "deterministic_grading_sha256": sha256_json(deterministic),
            "fixture_hash": benchmark.get("fixture_hash"),
            "generator_config_hash": benchmark.get("config_hash"),
        },
        "by_arm_and_suite": by_arm,
        "attempts": sorted(
            attempts,
            key=lambda row: (str(row["question_id"]), str(row["arm"])),
        ),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Iterable[str],
) -> list[dict[str, Any]]:
    fields = tuple(key_fields)
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        all_scores = [float(row["all_attempted_score"]) for row in group]
        valid_scores = [
            float(row["governed_final_score"])
            for row in group
            if row["governed_final_score"] is not None
        ]
        counts = Counter(all_scores)
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                "n_attempted": len(group),
                "n_semantically_scored": len(valid_scores),
                "all_attempted_mean": sum(all_scores) / len(all_scores),
                "semantic_mean_when_scored": (
                    sum(valid_scores) / len(valid_scores) if valid_scores else None
                ),
                "fully_correct_count": counts[1.0],
                "partial_count": counts[0.5],
                "incorrect_or_runtime_failure_count": counts[0.0],
                "deterministic_hard_fail_count": sum(
                    row["deterministic_status"] == "deterministic_hard_fail" for row in group
                ),
                "runtime_or_parse_failure_count": sum(
                    row["governed_final_score"] is None for row in group
                ),
                "latency_seconds": sum(
                    float(row["judge_result"]["latency_seconds"])
                    for row in group
                    if row["judge_result"] is not None
                ),
                "judge_completion_tokens": sum(
                    int(row["judge_result"]["completion_tokens"])
                    for row in group
                    if row["judge_result"] is not None
                ),
                "generator_latency_seconds": sum(
                    float(row["generator_elapsed_seconds"] or 0.0) for row in group
                ),
                "generator_prompt_tokens": sum(
                    int(row["generator_prompt_tokens"] or 0) for row in group
                ),
                "generator_completion_tokens": sum(
                    int(row["generator_completion_tokens"] or 0) for row in group
                ),
                "generator_peak_memory_gb": max(
                    (
                        float(row["generator_peak_memory_gb"])
                        for row in group
                        if row["generator_peak_memory_gb"] is not None
                    ),
                    default=None,
                ),
            }
        )
    return output


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Local Gemma 4 advisory verification",
        "",
        f"- Status: **{report['status']}**",
        f"- Model: `{report['model_id']}@{report['model_revision']}`",
        f"- Verification: `{report['verification_status']}`",
        "- Human approved: **no**",
        "- Promotion eligible: **no**",
        "",
        "Deterministic hard failures remain authoritative. Invalid generation or "
        "verifier output stays in the all-attempted denominator.",
        "",
        "## Aggregate outcomes",
        "",
    ]
    for row in report["by_arm_and_suite"]:
        lines.extend(
            [
                f"### {row['arm']} / {row['suite']}",
                "",
                f"- Attempted: {row['n_attempted']}",
                f"- All-attempted mean: {row['all_attempted_mean']:.3f}",
                f"- Fully correct: {row['fully_correct_count']}",
                f"- Partial: {row['partial_count']}",
                f"- Runtime/parse failures: {row['runtime_or_parse_failure_count']}",
                "",
            ]
        )
    return "\n".join(lines)


def local_model_id() -> str:
    from .experiment_profiles import GEMMA_JUDGE_MODEL_ID

    return GEMMA_JUDGE_MODEL_ID


def local_model_revision() -> str:
    from .experiment_profiles import GEMMA_JUDGE_REVISION

    return GEMMA_JUDGE_REVISION
