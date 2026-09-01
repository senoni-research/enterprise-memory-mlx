"""Deterministic post-generation grading for answer-blind benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .gates import (
    DEFAULT_PROMOTION_GATES,
    GateObservation,
    GateResult,
    GateSpec,
    evaluate_gate,
)
from .provenance_grading import (
    ProvenanceGrade,
    ProvenanceGradeRequest,
    grade_provenance,
)
from .semantic_judging import (
    DualJudgeGrade,
    JudgeBackend,
    grade_with_dual_judges,
)
from .split_contract import EvalQuestion, EvalSuites, verify_frozen_assets
from .strict_grading import StrictGrade, grade_critical_slots
from .utils import atomic_write_text, sha256_json

GradingStatus = Literal[
    "deterministic_hard_fail",
    "semantic_review_required",
    "unscored_generation_failure",
]
SemanticScoreSource = Literal[
    "deterministic_hard_fail",
    "judge_agreement",
    "human_adjudication_required",
    "unscored_generation_failure",
]


@dataclass(frozen=True)
class DeterministicGradeRow:
    question_id: str
    arm: str
    suite: str
    record_id: str | None
    scenario_id: str | None
    question_family_id: str
    as_of_date: str | None
    generation_status: str
    retrieval_label: str
    status: GradingStatus
    deterministic_score: float | None
    strict: StrictGrade | None
    provenance: ProvenanceGrade | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "arm": self.arm,
            "suite": self.suite,
            "record_id": self.record_id,
            "scenario_id": self.scenario_id,
            "question_family_id": self.question_family_id,
            "as_of_date": self.as_of_date,
            "generation_status": self.generation_status,
            "retrieval_label": self.retrieval_label,
            "status": self.status,
            "deterministic_score": self.deterministic_score,
            "strict": _dataclass_payload(self.strict),
            "provenance": _dataclass_payload(self.provenance),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DeterministicGradingReport:
    raw_artifact_path: str
    raw_artifact_hash: str
    fixture_hash: str
    grader_config_hash: str
    promotion_eligible: bool
    semantic_grading: str
    rows: tuple[DeterministicGradeRow, ...]
    aggregates: tuple[dict[str, Any], ...]
    gates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "graded": True,
            "raw_artifact_path": self.raw_artifact_path,
            "raw_artifact_hash": self.raw_artifact_hash,
            "fixture_hash": self.fixture_hash,
            "grader_config_hash": self.grader_config_hash,
            "mode": "deterministic_only",
            "semantic_grading": self.semantic_grading,
            "promotion_eligible": self.promotion_eligible,
            "rows": [row.to_dict() for row in self.rows],
            "aggregates": list(self.aggregates),
            "gates": list(self.gates),
        }


def grade_benchmark_artifact(
    raw_artifact_path: Path,
    eval_dir: Path,
    suites: EvalSuites,
    *,
    allowed_parametric_record_ids: Iterable[str] = (),
) -> DeterministicGradingReport:
    """Join generated rows to frozen questions and apply hard checks only."""
    freeze_problems = verify_frozen_assets(eval_dir)
    if freeze_problems:
        raise ValueError(
            "Frozen evaluation assets failed verification:\n"
            + "\n".join(freeze_problems)
        )
    raw_bytes = raw_artifact_path.read_bytes()
    try:
        artifact = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Benchmark artifact is not valid JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("graded") is not False:
        raise ValueError("Expected an ungraded benchmark artifact")

    fixture_manifest = json.loads(
        (eval_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    fixture_hash = str(fixture_manifest.get("combined_hash", ""))
    if artifact.get("fixture_hash") != fixture_hash:
        raise ValueError("Benchmark fixture hash does not match frozen evaluation assets")

    questions = _questions_by_id(suites)
    artifact_rows = artifact.get("results")
    if not isinstance(artifact_rows, list):
        raise ValueError("Benchmark artifact results must be a list")
    _validate_result_matrix(artifact, artifact_rows, questions)

    configured_parametric = tuple(
        str(item)
        for item in artifact.get("config", {}).get(
            "parametric_source_record_ids", []
        )
    )
    allowed_parametric = tuple(allowed_parametric_record_ids) or configured_parametric
    rows = tuple(
        _grade_row(
            raw,
            questions[str(raw["question_id"])],
            allowed_parametric_record_ids=allowed_parametric,
        )
        for raw in artifact_rows
    )
    config = {
        "mode": "deterministic_only",
        "strict_grader": "strict_grading/v1",
        "provenance_grader": "provenance_grading/v1",
        "semantic_grading": "unavailable_pending_human_labels_and_certified_judges",
        "allowed_parametric_record_ids": list(allowed_parametric),
    }
    return DeterministicGradingReport(
        raw_artifact_path=str(raw_artifact_path.resolve()),
        raw_artifact_hash=hashlib.sha256(raw_bytes).hexdigest(),
        fixture_hash=fixture_hash,
        grader_config_hash=sha256_json(config),
        promotion_eligible=False,
        semantic_grading="unavailable_pending_human_labels_and_certified_judges",
        rows=rows,
        aggregates=_aggregate(rows),
        gates=_evaluate_report_gates(rows, questions, suites),
    )


@dataclass(frozen=True)
class SemanticJudgingOutcome:
    """One row's final score after deterministic checks and dual judging."""

    question_id: str
    arm: str
    deterministic_status: GradingStatus
    final_score: float | None
    score_source: SemanticScoreSource
    dual: DualJudgeGrade | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "arm": self.arm,
            "deterministic_status": self.deterministic_status,
            "final_score": self.final_score,
            "score_source": self.score_source,
            "dual": self.dual.to_dict() if self.dual is not None else None,
        }


def judge_benchmark_artifact(
    raw_artifact_path: Path,
    eval_dir: Path,
    suites: EvalSuites,
    *,
    judges: Sequence[JudgeBackend],
    evaluated_model_family: str,
    allowed_parametric_record_ids: Iterable[str] = (),
    max_output_tokens: int = 256,
    allow_uncertified_machinery: bool = False,
) -> tuple[DeterministicGradingReport, tuple[SemanticJudgingOutcome, ...]]:
    """Run dual judges on top of deterministic grading.

    The deterministic report is computed here and each row's hard-failure
    state is bound into the judge call, so a caller cannot skip the
    ``deterministic_hard_failure`` flag. Hard-failed rows never invoke a
    judge and are fixed at 0.0; unscored generation failures are skipped.
    """
    if not allow_uncertified_machinery:
        raise ValueError(
            "Semantic judging requires certified judges; pass "
            "allow_uncertified_machinery=True only for explicitly "
            "non-promotable machinery checks"
        )
    report = grade_benchmark_artifact(
        raw_artifact_path,
        eval_dir,
        suites,
        allowed_parametric_record_ids=allowed_parametric_record_ids,
    )
    artifact = json.loads(raw_artifact_path.read_text(encoding="utf-8"))
    outputs = {
        (str(row["question_id"]), str(row["arm"])): str(row.get("output") or "")
        for row in artifact["results"]
    }
    questions = _questions_by_id(suites)

    outcomes: list[SemanticJudgingOutcome] = []
    for row in report.rows:
        if row.generation_status != "generated":
            outcomes.append(
                SemanticJudgingOutcome(
                    question_id=row.question_id,
                    arm=row.arm,
                    deterministic_status=row.status,
                    final_score=None,
                    score_source="unscored_generation_failure",
                    dual=None,
                )
            )
            continue
        question = questions[row.question_id]
        hard_failure = row.status == "deterministic_hard_fail"
        dual = grade_with_dual_judges(
            question=question.question,
            reference_answer=question.expected,
            candidate_answer=outputs[(row.question_id, row.arm)],
            judges=judges,
            evaluated_model_family=evaluated_model_family,
            deterministic_hard_failure=hard_failure,
            max_output_tokens=max_output_tokens,
        )
        if hard_failure:
            final_score: float | None = 0.0
            source: SemanticScoreSource = "deterministic_hard_fail"
        elif dual.status == "agreed":
            final_score = dual.score
            source = "judge_agreement"
        else:
            final_score = None
            source = "human_adjudication_required"
        outcomes.append(
            SemanticJudgingOutcome(
                question_id=row.question_id,
                arm=row.arm,
                deterministic_status=row.status,
                final_score=final_score,
                score_source=source,
                dual=dual,
            )
        )
    return report, tuple(outcomes)


def write_grading_report(
    report: DeterministicGradingReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"deterministic-grading-{timestamp}.json"
    markdown_path = output_dir / f"deterministic-grading-{timestamp}.md"
    payload = report.to_dict()
    atomic_write_text(
        json_path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(markdown_path, _render_markdown(payload))
    return json_path, markdown_path


def _grade_row(
    raw: dict[str, Any],
    question: EvalQuestion,
    *,
    allowed_parametric_record_ids: tuple[str, ...],
) -> DeterministicGradeRow:
    generation_status = str(raw.get("generation_status", ""))
    retrieval_label = str(raw.get("retrieval_label", "not_applicable"))
    if generation_status != "generated":
        return DeterministicGradeRow(
            question_id=question.question_id,
            arm=str(raw.get("arm", "")),
            suite=question.suite,
            record_id=question.record_id,
            scenario_id=question.scenario_id,
            question_family_id=question.question_family_id,
            as_of_date=question.as_of_date,
            generation_status=generation_status,
            retrieval_label=retrieval_label,
            status="unscored_generation_failure",
            deterministic_score=None,
            strict=None,
            provenance=None,
            reasons=(str(raw.get("generation_error") or "answer was not generated"),),
        )

    output = str(raw.get("output", ""))
    strict = grade_critical_slots(output, question.critical_slots)
    arm = str(raw.get("arm", ""))
    provenance = grade_provenance(
        ProvenanceGradeRequest(
            generated_output=output,
            arm=arm,
            suite=question.suite,
            probe_kind=question.probe_kind,
            supplied_record_ids=tuple(
                str(item) for item in raw.get("selected_record_ids", [])
            ),
            supplied_source_uris=tuple(
                str(item) for item in raw.get("source_uris", [])
            ),
            gold_record_id=question.record_id,
            out_of_scope=question.suite == "unknown_oos",
            live_source=question.probe_kind == "live_source",
            citation_required=False,
            allowed_parametric_record_ids=(
                allowed_parametric_record_ids if arm == "parametric" else ()
            ),
        )
    )
    hard_fail = strict.status == "hard_fail" or provenance.status == "hard_fail"
    reasons = (
        tuple(f"strict:{reason}" for reason in strict.reasons)
        + tuple(f"provenance:{reason}" for reason in provenance.reasons)
    )
    return DeterministicGradeRow(
        question_id=question.question_id,
        arm=arm,
        suite=question.suite,
        record_id=question.record_id,
        scenario_id=question.scenario_id,
        question_family_id=question.question_family_id,
        as_of_date=question.as_of_date,
        generation_status=generation_status,
        retrieval_label=retrieval_label,
        status=(
            "deterministic_hard_fail" if hard_fail else "semantic_review_required"
        ),
        deterministic_score=0.0 if hard_fail else None,
        strict=strict,
        provenance=provenance,
        reasons=reasons,
    )


def _questions_by_id(suites: EvalSuites) -> dict[str, EvalQuestion]:
    questions = suites.all_questions()
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("Frozen evaluation contains duplicate question IDs")
    return by_id


def _validate_result_matrix(
    artifact: dict[str, Any],
    rows: list[Any],
    questions: dict[str, EvalQuestion],
) -> None:
    config = artifact.get("config")
    if not isinstance(config, dict):
        raise ValueError("Benchmark artifact has no configuration")
    suites = tuple(str(item) for item in config.get("suites", []))
    arms = tuple(str(item) for item in config.get("arms", []))
    selected_question_ids = {
        question_id
        for question_id, question in questions.items()
        if question.suite in suites
    }
    expected_pairs = {
        (question_id, arm)
        for question_id in selected_question_ids
        for arm in arms
    }
    actual_pairs: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Benchmark result row is not an object")
        question_id = str(row.get("question_id", ""))
        arm = str(row.get("arm", ""))
        if question_id not in selected_question_ids:
            raise ValueError(f"Unexpected or missing-frozen question ID: {question_id}")
        if row.get("suite") != questions[question_id].suite:
            raise ValueError(f"Suite mismatch for question: {question_id}")
        actual_pairs.append((question_id, arm))
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Benchmark artifact contains duplicate question/arm rows")
    if set(actual_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(actual_pairs))
        extra = sorted(set(actual_pairs) - expected_pairs)
        raise ValueError(
            f"Benchmark result matrix mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def _aggregate(
    rows: tuple[DeterministicGradeRow, ...],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[DeterministicGradeRow]] = {}
    for row in rows:
        grouped.setdefault((row.arm, row.suite), []).append(row)
    aggregates = []
    for (arm, suite), items in sorted(grouped.items()):
        statuses = Counter(item.status for item in items)
        strict_statuses = Counter(
            item.strict.status for item in items if item.strict is not None
        )
        provenance_statuses = Counter(
            item.provenance.status for item in items if item.provenance is not None
        )
        retrieval = Counter(item.retrieval_label for item in items)
        aggregates.append(
            {
                "arm": arm,
                "suite": suite,
                "total": len(items),
                "grading_status": dict(sorted(statuses.items())),
                "strict_status": dict(sorted(strict_statuses.items())),
                "provenance_status": dict(sorted(provenance_statuses.items())),
                "retrieval_labels": dict(sorted(retrieval.items())),
                "semantic_accuracy": None,
                "promotion_eligible": False,
            }
        )
    return tuple(aggregates)


def _evaluate_report_gates(
    rows: tuple[DeterministicGradeRow, ...],
    questions: dict[str, EvalQuestion],
    suites: EvalSuites,
) -> tuple[dict[str, Any], ...]:
    scenarios = {scenario.scenario_id: scenario for scenario in suites.scenarios}
    output = []
    for arm in sorted({row.arm for row in rows}):
        arm_rows = [row for row in rows if row.arm == arm]
        for spec in DEFAULT_PROMOTION_GATES:
            observations = _gate_observations(
                spec,
                arm_rows,
                questions,
                scenarios,
            )
            result: GateResult = evaluate_gate(spec, observations)
            output.append({"arm": arm, **result.to_dict()})
    return tuple(output)


def _gate_observations(
    spec: GateSpec,
    rows: list[DeterministicGradeRow],
    questions: dict[str, EvalQuestion],
    scenarios: dict[str, Any],
) -> list[GateObservation]:
    observations: list[GateObservation] = []
    for row in rows:
        question = questions[row.question_id]
        if row.generation_status != "generated":
            continue
        cluster_id = (
            row.scenario_id
            if spec.cluster_key == "scenario_id"
            else row.question_family_id
        )
        if not cluster_id:
            continue
        if spec.name == "hallucinated_citation" and row.provenance is not None:
            observations.append(
                GateObservation(
                    cluster_id=cluster_id,
                    passed=not row.provenance.hallucinated_citation,
                )
            )
            continue
        if question.suite != "supersession" or row.strict is None:
            continue
        scenario = scenarios.get(question.scenario_id)
        if scenario is None or question.probe_kind == "temporal":
            continue
        if spec.name == "new_current_value_strict":
            observations.append(
                GateObservation(
                    cluster_id=cluster_id,
                    passed=row.strict.status == "pass",
                )
            )
        elif spec.name in {"stale_current_answer", "old_new_conflation"}:
            relevant = [
                result
                for result in row.strict.slot_results
                if scenario.old_value in result.forbidden
            ]
            if relevant:
                observations.append(
                    GateObservation(
                        cluster_id=cluster_id,
                        passed=not any(
                            result.forbidden_present for result in relevant
                        ),
                    )
                )
    return observations


def _dataclass_payload(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return _convert(value.__dict__)
    return value


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_convert(item) for item in value]
    if hasattr(value, "__dict__"):
        return _convert(value.__dict__)
    return value


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deterministic benchmark grading",
        "",
        f"- Raw artifact: `{payload['raw_artifact_path']}`",
        f"- Raw SHA-256: `{payload['raw_artifact_hash']}`",
        f"- Fixture hash: `{payload['fixture_hash']}`",
        "- Semantic grading: unavailable",
        "- Promotion eligible: **no**",
        "",
        "## Aggregate outcomes",
        "",
    ]
    for item in payload["aggregates"]:
        lines.extend(
            [
                f"### {item['arm']} / {item['suite']}",
                "",
                f"- Total: {item['total']}",
                f"- Deterministic outcomes: `{json.dumps(item['grading_status'], sort_keys=True)}`",
                f"- Retrieval outcomes: `{json.dumps(item['retrieval_labels'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "A deterministic hard failure is a certain factual/provenance error.",
            "All other generated rows still require certified semantic review.",
            "No accuracy or superiority claim can be made from this report.",
            "",
        ]
    )
    return "\n".join(lines)
