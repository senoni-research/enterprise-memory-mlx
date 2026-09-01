"""Model-agnostic semantic-judge calibration and dual-judge grading.

Unit tests inject fake backends. This module does not load models, open
network sockets, or treat a large judge as reliable. Real classified
evaluation must remain local and sequential.

Deterministic critical-slot or provenance hard failures are outside judge
discretion and cannot be upgraded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .gates import clopper_pearson_upper
from .utils import atomic_write_text, sha256_json, sha256_text

LEGAL_SCORES: tuple[float, ...] = (0.0, 0.5, 1.0)
SCORE_INDEX: dict[float, int] = {0.0: 0, 0.5: 1, 1.0: 2}
ORDINAL_SCORES: tuple[float, ...] = LEGAL_SCORES
RUBRIC_PROMPT_VERSION = "v1"
HEADLINE_STRATUM = "judge_eligible"
AUDIT_STRATUM = "deterministic_audit"
SOURCE_CASES_FILE = "source_cases.jsonl"
HUMAN_LABEL_OVERLAY_FILE = "human_label_overlay.jsonl"
LABELED_DATASET_FILE = "labeled_dataset.jsonl"
_WILSON_Z_95 = 1.959963984540054
_DOMAIN_PREFIXES: dict[str, str] = {
    "FIN": "finance",
    "HR": "hr",
    "ENG": "engineering",
    "SUP": "support",
    "PROC": "procurement",
    "IT": "it",
    "LEG": "legal",
}

CalibrationStatus = Literal["certified_for_exploratory_grading", "not_certified"]
DualJudgeStatus = Literal["agreed", "human_adjudication_required", "deterministic_hard_fail"]


class CalibrationValidationError(ValueError):
    """Calibration inputs are incomplete, inconsistent, or otherwise unusable."""


class JudgeBackend(Protocol):
    """Injectable local judge. Implementations must not call external APIs."""

    model_id: str
    model_family: str
    revision: str
    local_only: bool

    def acquire(self) -> None: ...

    def release(self) -> None: ...

    def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        rubric_prompt_version: str,
        max_output_tokens: int,
    ) -> JudgeBackendResult: ...


@dataclass(frozen=True)
class JudgeBackendResult:
    raw_text: str
    model_id: str
    revision: str
    prompt_hash: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt_hash": self.prompt_hash,
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class JudgeIdentity:
    model_id: str
    model_family: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class ParsedJudgeOutput:
    valid: bool
    score: float | None
    reason: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    human_semantic_score: float | None
    human_reason: str | None
    human_reviewers: tuple[str, ...]
    human_approved: bool
    error_categories: tuple[str, ...] = ()
    domain: str = "none"
    certification_stratum: str = HEADLINE_STRATUM
    disputed: bool = False
    usable_for_training: bool = False
    deterministic_hard_failure: bool = False
    proposed_semantic_score: float | None = None
    proposed_reason: str | None = None
    source_record_ids: tuple[str, ...] = ()
    case_family_id: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationCase:
        case_id = str(value.get("case_id", "")).strip()
        if not case_id:
            raise CalibrationValidationError("Calibration case is missing case_id")
        reviewers = tuple(
            str(item).strip()
            for item in value.get("human_reviewers", ())
            if str(item).strip()
        )
        source_ids = tuple(
            str(item).strip()
            for item in value.get("source_record_ids", ())
            if str(item).strip()
        )
        domain = str(value.get("domain", "")).strip().lower()
        if not domain:
            domain = _domain_from_source_ids(source_ids)
        stratum = str(value.get("certification_stratum") or HEADLINE_STRATUM).strip()
        hard_fail = bool(value.get("deterministic_hard_failure")) or stratum == AUDIT_STRATUM
        family_id = str(value.get("case_family_id") or "").strip() or case_id
        return cls(
            case_id=case_id,
            question=str(value.get("question", "")),
            reference_answer=str(value.get("reference_answer", "")),
            candidate_answer=str(value.get("candidate_answer", "")),
            human_semantic_score=_optional_score(value.get("human_semantic_score")),
            human_reason=_optional_text(value.get("human_reason")),
            human_reviewers=reviewers,
            human_approved=bool(value.get("human_approved", False)),
            error_categories=tuple(
                str(item).strip()
                for item in value.get("error_categories", ())
                if str(item).strip()
            ),
            domain=domain,
            certification_stratum=stratum,
            disputed=bool(value.get("disputed", False)),
            usable_for_training=_training_flag(value),
            deterministic_hard_failure=hard_fail,
            proposed_semantic_score=_optional_score(value.get("proposed_semantic_score")),
            proposed_reason=_optional_text(value.get("proposed_reason")),
            source_record_ids=source_ids,
            case_family_id=family_id,
        )

    @property
    def requires_two_reviewers(self) -> bool:
        return self.disputed or self.deterministic_hard_failure or (
            self.certification_stratum == AUDIT_STRATUM
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "candidate_answer": self.candidate_answer,
            "case_family_id": self.case_family_id,
            "case_id": self.case_id,
            "certification_stratum": self.certification_stratum,
            "deterministic_hard_failure": self.deterministic_hard_failure,
            "domain": self.domain,
            "disputed": self.disputed,
            "error_categories": list(self.error_categories),
            "human_approved": self.human_approved,
            "human_reason": self.human_reason,
            "human_reviewers": list(self.human_reviewers),
            "human_semantic_score": self.human_semantic_score,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "source_record_ids": list(self.source_record_ids),
            "usable_for_training": self.usable_for_training,
        }

    def source_dict(self) -> dict[str, Any]:
        return {
            "candidate_answer": self.candidate_answer,
            "case_family_id": self.case_family_id,
            "case_id": self.case_id,
            "certification_stratum": self.certification_stratum,
            "deterministic_hard_failure": self.deterministic_hard_failure,
            "domain": self.domain,
            "disputed": self.disputed,
            "error_categories": list(self.error_categories),
            "proposed_reason": self.proposed_reason,
            "proposed_semantic_score": self.proposed_semantic_score,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "source_record_ids": list(self.source_record_ids),
            "usable_for_training": self.usable_for_training,
        }

    def overlay_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "human_approved": self.human_approved,
            "human_reason": self.human_reason,
            "human_reviewers": list(self.human_reviewers),
            "human_semantic_score": self.human_semantic_score,
        }


@dataclass(frozen=True)
class RateWithCI:
    numerator: int
    denominator: int
    rate: float | None
    ci_low: float | None
    ci_high: float | None
    interval: str = "clopper_pearson_two_sided"
    unit: str = "row"
    role: str = "descriptive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "interval": self.interval,
            "unit": self.unit,
            "role": self.role,
        }


@dataclass(frozen=True)
class ClassPrecisionRecall:
    score: float
    precision: RateWithCI
    recall: RateWithCI

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
        }


@dataclass(frozen=True)
class ConfusionMatrix:
    labels: tuple[float, ...]
    counts: tuple[tuple[int, int, int], ...]

    def cell(self, human: float, judge: float) -> int:
        return self.counts[SCORE_INDEX[human]][SCORE_INDEX[judge]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "counts": [list(row) for row in self.counts],
            "rows": "human",
            "columns": "judge",
        }


@dataclass(frozen=True)
class SliceMetrics:
    key: str
    n: int
    exact_agreement: RateWithCI
    fully_correct_false_pass_rate: RateWithCI
    incorrect_to_pass_rate: RateWithCI
    false_fail_rate: RateWithCI
    invalid_output_rate: RateWithCI
    family_clustered_false_pass_rate: RateWithCI

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n": self.n,
            "exact_agreement": self.exact_agreement.to_dict(),
            "fully_correct_false_pass_rate": self.fully_correct_false_pass_rate.to_dict(),
            "incorrect_to_pass_rate": self.incorrect_to_pass_rate.to_dict(),
            "false_fail_rate": self.false_fail_rate.to_dict(),
            "invalid_output_rate": self.invalid_output_rate.to_dict(),
            "family_clustered_false_pass_rate": (
                self.family_clustered_false_pass_rate.to_dict()
            ),
        }


@dataclass(frozen=True)
class FamilyClusteredMetrics:
    n_families: int
    exact_agreement: RateWithCI
    fully_correct_false_pass_rate: RateWithCI
    incorrect_to_pass_rate: RateWithCI
    false_fail_rate: RateWithCI
    invalid_output_rate: RateWithCI

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_families": self.n_families,
            "exact_agreement": self.exact_agreement.to_dict(),
            "fully_correct_false_pass_rate": self.fully_correct_false_pass_rate.to_dict(),
            "incorrect_to_pass_rate": self.incorrect_to_pass_rate.to_dict(),
            "false_fail_rate": self.false_fail_rate.to_dict(),
            "invalid_output_rate": self.invalid_output_rate.to_dict(),
        }


@dataclass(frozen=True)
class JudgeMetrics:
    n_cases: int
    n_valid: int
    n_invalid: int
    confusion_matrix: ConfusionMatrix
    exact_agreement: RateWithCI
    adjacent_disagreement: RateWithCI
    two_step_disagreement: RateWithCI
    class_precision_recall: tuple[ClassPrecisionRecall, ...]
    weighted_kappa: float | None
    weighted_kappa_ci_low: float | None
    weighted_kappa_ci_high: float | None
    weighted_kappa_interval: str
    mean_absolute_ordinal_error: float | None
    fully_correct_false_pass_rate: RateWithCI
    incorrect_to_pass_rate: RateWithCI
    false_fail_rate: RateWithCI
    invalid_output_rate: RateWithCI
    by_error_category: tuple[SliceMetrics, ...] = ()
    by_domain: tuple[SliceMetrics, ...] = ()
    family_clustered: FamilyClusteredMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "exact_agreement": self.exact_agreement.to_dict(),
            "adjacent_disagreement": self.adjacent_disagreement.to_dict(),
            "two_step_disagreement": self.two_step_disagreement.to_dict(),
            "class_precision_recall": [item.to_dict() for item in self.class_precision_recall],
            "weighted_kappa": self.weighted_kappa,
            "weighted_kappa_ci_low": self.weighted_kappa_ci_low,
            "weighted_kappa_ci_high": self.weighted_kappa_ci_high,
            "weighted_kappa_interval": self.weighted_kappa_interval,
            "mean_absolute_ordinal_error": self.mean_absolute_ordinal_error,
            "fully_correct_false_pass_rate": self.fully_correct_false_pass_rate.to_dict(),
            "incorrect_to_pass_rate": self.incorrect_to_pass_rate.to_dict(),
            "false_fail_rate": self.false_fail_rate.to_dict(),
            "invalid_output_rate": self.invalid_output_rate.to_dict(),
            "by_error_category": [item.to_dict() for item in self.by_error_category],
            "by_domain": [item.to_dict() for item in self.by_domain],
            "family_clustered": (
                None if self.family_clustered is None else self.family_clustered.to_dict()
            ),
        }


@dataclass(frozen=True)
class CaseDisagreement:
    case_id: str
    judge_a_score: float | None
    judge_b_score: float | None
    judge_a_valid: bool
    judge_b_valid: bool
    judge_a_reason: str | None
    judge_b_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "judge_a_score": self.judge_a_score,
            "judge_b_score": self.judge_b_score,
            "judge_a_valid": self.judge_a_valid,
            "judge_b_valid": self.judge_b_valid,
            "judge_a_reason": self.judge_a_reason,
            "judge_b_reason": self.judge_b_reason,
        }


@dataclass(frozen=True)
class PairMetrics:
    n_compared: int
    agreement_rate: RateWithCI
    disagreements: tuple[CaseDisagreement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_compared": self.n_compared,
            "agreement_rate": self.agreement_rate.to_dict(),
            "disagreements": [item.to_dict() for item in self.disagreements],
        }


@dataclass(frozen=True)
class ExploratoryCertificationThresholds:
    """Caller-supplied exploratory thresholds. Not production policy."""

    min_exact_agreement: float
    min_weighted_kappa: float
    max_incorrect_to_pass_rate: float
    max_invalid_output_rate: float
    max_critical_false_pass_rate: float


@dataclass(frozen=True)
class PerJudgeReport:
    identity: JudgeIdentity
    headline: JudgeMetrics
    audit: JudgeMetrics
    passed_thresholds: bool
    failed_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "headline": self.headline.to_dict(),
            "audit": self.audit.to_dict(),
            "passed_thresholds": self.passed_thresholds,
            "failed_requirements": list(self.failed_requirements),
        }


@dataclass(frozen=True)
class DatasetHashes:
    source_cases_hash: str
    human_label_overlay_hash: str
    labeled_dataset_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_cases_hash": self.source_cases_hash,
            "human_label_overlay_hash": self.human_label_overlay_hash,
            "labeled_dataset_hash": self.labeled_dataset_hash,
        }


@dataclass(frozen=True)
class CaseJudgeDecision:
    case_id: str
    case_family_id: str
    identity: JudgeIdentity
    raw_text: str
    parsed: ParsedJudgeOutput
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_family_id": self.case_family_id,
            "identity": self.identity.to_dict(),
            "raw_text": self.raw_text,
            "parsed": self.parsed.to_dict(),
            "prompt_hash": self.prompt_hash,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class JudgeCalibrationResult:
    status: CalibrationStatus
    dataset_hash: str
    source_cases_hash: str
    human_label_overlay_hash: str
    labeled_dataset_hash: str
    rubric_prompt_hash: str
    judge_identities: tuple[JudgeIdentity, ...]
    per_judge_metrics: tuple[PerJudgeReport, ...]
    pair_metrics: PairMetrics
    failed_requirements: tuple[str, ...]
    case_decisions: tuple[CaseJudgeDecision, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dataset_hash": self.dataset_hash,
            "source_cases_hash": self.source_cases_hash,
            "human_label_overlay_hash": self.human_label_overlay_hash,
            "labeled_dataset_hash": self.labeled_dataset_hash,
            "rubric_prompt_hash": self.rubric_prompt_hash,
            "judge_identities": [item.to_dict() for item in self.judge_identities],
            "per_judge_metrics": [item.to_dict() for item in self.per_judge_metrics],
            "pair_metrics": self.pair_metrics.to_dict(),
            "failed_requirements": list(self.failed_requirements),
            "case_decisions": [item.to_dict() for item in self.case_decisions],
        }


@dataclass(frozen=True)
class DualJudgeGrade:
    status: DualJudgeStatus
    score: float | None
    reasons: tuple[str, ...]
    parsed: tuple[ParsedJudgeOutput, ...]
    raw_results: tuple[JudgeBackendResult, ...]
    invocation_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "reasons": list(self.reasons),
            "parsed": [item.to_dict() for item in self.parsed],
            "raw_results": [item.to_dict() for item in self.raw_results],
            "invocation_order": list(self.invocation_order),
        }


@dataclass
class _InvocationTrace:
    order: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    loaded: set[str] = field(default_factory=set)
    in_flight: int = 0
    overlapping: bool = False


_INVOCATION = _InvocationTrace()


def parse_judge_output(raw_text: str) -> ParsedJudgeOutput:
    """Accept only a single legal JSON object. Do not repair invalid output."""
    if not isinstance(raw_text, str):
        return _invalid("judge output is not text")
    stripped = raw_text.strip()
    if not stripped:
        return _invalid("malformed output")
    decoder = json.JSONDecoder()
    try:
        payload, index = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return _invalid("malformed output")
    remainder = stripped[index:].strip()
    if remainder:
        return _invalid("prose outside JSON or multiple objects")
    if not isinstance(payload, dict):
        return _invalid("judge output is not a JSON object")
    if set(payload) != {"score", "reason"}:
        return _invalid("missing or extra schema fields")
    score = payload["score"]
    reason = payload["reason"]
    if isinstance(score, bool) or not isinstance(score, int | float):
        return _invalid("score is not a legal number")
    if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
        return _invalid("score is NaN or infinite")
    numeric = float(score)
    if numeric not in LEGAL_SCORES:
        return _invalid("score is not a legal value")
    if not isinstance(reason, str) or not reason.strip():
        return _invalid("reason is empty")
    return ParsedJudgeOutput(valid=True, score=numeric, reason=reason, error=None)


def build_rubric_prompt(version: str = RUBRIC_PROMPT_VERSION) -> str:
    if version != RUBRIC_PROMPT_VERSION:
        raise ValueError(f"Unsupported rubric prompt version: {version}")
    return (
        "You are a semantic judge. Score only the candidate answer against the "
        "reference. Use this rubric:\n"
        "- 1.0: core conclusion fully correct, no material omission, "
        "contradiction, unsupported exception, or false provenance.\n"
        "- 0.5: broadly correct conclusion but materially incomplete or "
        "containing a limited non-core flaw.\n"
        "- 0.0: incorrect, contradictory, unsupported, evasive, wrong refusal, "
        "or materially misleading.\n"
        "Deterministic critical-slot or provenance hard failures are outside "
        "judge discretion and cannot be upgraded to 0.5 or 1.0.\n"
        "Return exactly one JSON object and no other text:\n"
        '{"score": 1.0, "reason": "Concise evidence-based reason"}\n'
        "Legal scores are only 0.0, 0.5, and 1.0."
    )


def rubric_prompt_hash(version: str = RUBRIC_PROMPT_VERSION) -> str:
    return sha256_text(build_rubric_prompt(version))


def render_instance_prompt(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    rubric_prompt_version: str = RUBRIC_PROMPT_VERSION,
) -> str:
    return (
        f"{build_rubric_prompt(rubric_prompt_version)}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )


def instance_prompt_hash(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    rubric_prompt_version: str = RUBRIC_PROMPT_VERSION,
) -> str:
    return sha256_text(
        render_instance_prompt(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            rubric_prompt_version=rubric_prompt_version,
        )
    )


def hash_calibration_cases(cases: Sequence[CalibrationCase]) -> str:
    return sha256_json([case.canonical_dict() for case in cases])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def parse_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = data.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CalibrationValidationError(
                f"Invalid JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise CalibrationValidationError(f"Expected a JSON object at line {line_number}")
        rows.append(row)
    return rows


def split_case_bytes(cases: Sequence[CalibrationCase]) -> tuple[bytes, bytes]:
    source_rows = [case.source_dict() for case in cases]
    overlay_rows = [case.overlay_dict() for case in cases]
    return encode_jsonl_bytes(source_rows), encode_jsonl_bytes(overlay_rows)


@dataclass(frozen=True)
class ConvertedCalibrationDataset:
    """Hash-locked harness layout produced from one candidate cases file."""

    source_path: Path
    overlay_path: Path
    manifest_path: Path
    hashes: DatasetHashes
    case_count: int
    human_labels_present: bool


def convert_candidate_file(
    candidate_path: Path,
    output_dir: Path,
) -> ConvertedCalibrationDataset:
    """Split a single candidate ``cases.jsonl`` into the harness layout.

    Writes ``source_cases.jsonl`` (machine-proposed labels retained as
    proposals), ``human_label_overlay.jsonl`` (human fields exactly as found,
    which for a model-drafted candidate means no scores and
    ``human_approved: false``), and a deterministic ``manifest.json`` binding
    all three dataset hashes plus the candidate file's own hash. The converter
    never fabricates approval; ``calibrate_judges`` will still refuse the
    output until every overlay row carries a human-approved gold label.
    """
    candidate_bytes = candidate_path.read_bytes()
    rows = parse_jsonl_bytes(candidate_bytes)
    if not rows:
        raise CalibrationValidationError("candidate case file is empty")
    cases = tuple(CalibrationCase.from_dict(row) for row in rows)
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise CalibrationValidationError(
                f"duplicate candidate case_id: {case.case_id}"
            )
        seen.add(case.case_id)
    source_bytes, overlay_bytes = split_case_bytes(cases)
    # Round-trip through the same merge path the harness uses so the recorded
    # labeled-dataset hash is exactly what calibrate_judges will recompute.
    _, _, hashes = prepare_labeled_dataset(source_bytes, overlay_bytes)
    manifest = {
        "schema_version": 1,
        "candidate_file": candidate_path.name,
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "files": {
            SOURCE_CASES_FILE: hashes.source_cases_hash,
            HUMAN_LABEL_OVERLAY_FILE: hashes.human_label_overlay_hash,
            LABELED_DATASET_FILE: hashes.labeled_dataset_hash,
        },
        "case_count": len(cases),
        "human_labels_present": any(
            case.human_semantic_score is not None for case in cases
        ),
        "human_approved_count": sum(case.human_approved for case in cases),
    }
    source_path = output_dir / SOURCE_CASES_FILE
    overlay_path = output_dir / HUMAN_LABEL_OVERLAY_FILE
    manifest_path = output_dir / "manifest.json"
    atomic_write_text(source_path, source_bytes.decode("utf-8"))
    atomic_write_text(overlay_path, overlay_bytes.decode("utf-8"))
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return ConvertedCalibrationDataset(
        source_path=source_path,
        overlay_path=overlay_path,
        manifest_path=manifest_path,
        hashes=hashes,
        case_count=len(cases),
        human_labels_present=bool(manifest["human_labels_present"]),
    )


def prepare_labeled_dataset(
    source_cases_bytes: bytes,
    human_label_overlay_bytes: bytes,
) -> tuple[tuple[CalibrationCase, ...], bytes, DatasetHashes]:
    source_rows = parse_jsonl_bytes(source_cases_bytes)
    overlay_rows = parse_jsonl_bytes(human_label_overlay_bytes)
    overlay_by_id: dict[str, dict[str, Any]] = {}
    for row in overlay_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise CalibrationValidationError("human-label overlay row is missing case_id")
        if case_id in overlay_by_id:
            raise CalibrationValidationError(f"duplicate overlay case_id: {case_id}")
        overlay_by_id[case_id] = row
    source_ids = [str(row.get("case_id", "")).strip() for row in source_rows]
    if any(not item for item in source_ids):
        raise CalibrationValidationError("source case is missing case_id")
    if set(source_ids) != set(overlay_by_id):
        raise CalibrationValidationError(
            "source cases and human-label overlay case_id sets do not match"
        )
    cases: list[CalibrationCase] = []
    merged_rows: list[dict[str, Any]] = []
    for row in source_rows:
        case_id = str(row["case_id"]).strip()
        merged = {**row, **overlay_by_id[case_id]}
        case = CalibrationCase.from_dict(merged)
        cases.append(case)
        merged_rows.append(case.canonical_dict())
    labeled_bytes = encode_jsonl_bytes(merged_rows)
    hashes = DatasetHashes(
        source_cases_hash=sha256_bytes(source_cases_bytes),
        human_label_overlay_hash=sha256_bytes(human_label_overlay_bytes),
        labeled_dataset_hash=sha256_bytes(labeled_bytes),
    )
    return tuple(cases), labeled_bytes, hashes


def quadratic_weighted_kappa(human: Sequence[float], judge: Sequence[float]) -> float | None:
    matrix = _confusion_counts(human, judge)
    return _kappa_from_matrix(matrix)[0]


def compute_judge_metrics(
    cases: Sequence[CalibrationCase],
    parsed: Sequence[ParsedJudgeOutput],
    *,
    confidence: float = 0.95,
    include_slices: bool = True,
) -> JudgeMetrics:
    if len(cases) != len(parsed):
        raise ValueError("cases and parsed outputs must be aligned")
    valid_human: list[float] = []
    valid_judge: list[float] = []
    n_invalid = 0
    for case, output in zip(cases, parsed, strict=True):
        if not output.valid or output.score is None:
            n_invalid += 1
            continue
        if case.human_semantic_score not in LEGAL_SCORES:
            raise CalibrationValidationError(
                f"{case.case_id}: metrics require a legal human score"
            )
        valid_human.append(case.human_semantic_score)
        valid_judge.append(output.score)

    n_cases = len(cases)
    n_valid = len(valid_human)
    counts = _confusion_counts(valid_human, valid_judge)
    exact = sum(human == judge for human, judge in zip(valid_human, valid_judge, strict=True))
    adjacent = 0
    two_step = 0
    ordinal_error = 0
    for human, judge in zip(valid_human, valid_judge, strict=True):
        delta = abs(SCORE_INDEX[human] - SCORE_INDEX[judge])
        ordinal_error += delta
        if delta == 1:
            adjacent += 1
        elif delta == 2:
            two_step += 1

    false_pass = sum(
        judge == 1.0 and human != 1.0
        for human, judge in zip(valid_human, valid_judge, strict=True)
    )
    not_fully_correct = sum(human != 1.0 for human in valid_human)
    incorrect_to_pass = sum(
        judge == 1.0 and human == 0.0
        for human, judge in zip(valid_human, valid_judge, strict=True)
    )
    human_incorrect = sum(human == 0.0 for human in valid_human)
    false_fail = sum(
        judge == 0.0 and human == 1.0
        for human, judge in zip(valid_human, valid_judge, strict=True)
    )
    human_correct = sum(human == 1.0 for human in valid_human)

    class_metrics = []
    for score in LEGAL_SCORES:
        predicted = sum(judge == score for judge in valid_judge)
        actual = sum(human == score for human in valid_human)
        true_positive = sum(
            human == score and judge == score
            for human, judge in zip(valid_human, valid_judge, strict=True)
        )
        class_metrics.append(
            ClassPrecisionRecall(
                score=score,
                precision=_rate(true_positive, predicted, confidence),
                recall=_rate(true_positive, actual, confidence),
            )
        )

    kappa, kappa_low, kappa_high, kappa_interval = _kappa_from_matrix(counts, confidence)
    mae = (ordinal_error / n_valid) if n_valid else None
    family_clustered = _family_clustered_metrics(cases, parsed, confidence)
    return JudgeMetrics(
        n_cases=n_cases,
        n_valid=n_valid,
        n_invalid=n_invalid,
        confusion_matrix=ConfusionMatrix(labels=ORDINAL_SCORES, counts=counts),
        exact_agreement=_rate(exact, n_valid, confidence),
        adjacent_disagreement=_rate(adjacent, n_valid, confidence),
        two_step_disagreement=_rate(two_step, n_valid, confidence),
        class_precision_recall=tuple(class_metrics),
        weighted_kappa=kappa,
        weighted_kappa_ci_low=kappa_low,
        weighted_kappa_ci_high=kappa_high,
        weighted_kappa_interval=kappa_interval,
        mean_absolute_ordinal_error=mae,
        fully_correct_false_pass_rate=_rate(false_pass, not_fully_correct, confidence),
        incorrect_to_pass_rate=_rate(incorrect_to_pass, human_incorrect, confidence),
        false_fail_rate=_rate(false_fail, human_correct, confidence),
        invalid_output_rate=_rate(n_invalid, n_cases, confidence),
        by_error_category=(
            _slice_metrics(cases, parsed, "error_category", confidence)
            if include_slices
            else ()
        ),
        by_domain=(
            _slice_metrics(cases, parsed, "domain", confidence) if include_slices else ()
        ),
        family_clustered=family_clustered,
    )


def calibrate_judges(
    *,
    source_cases_bytes: bytes,
    human_label_overlay_bytes: bytes,
    manifest: Mapping[str, Any],
    expected_source_cases_hash: str,
    expected_human_label_overlay_hash: str,
    expected_labeled_dataset_hash: str,
    judges: Sequence[JudgeBackend],
    evaluated_model_family: str,
    thresholds: ExploratoryCertificationThresholds,
    min_class_count: int,
    expected_manifest_hash: str | None = None,
    manifest_bytes: bytes | None = None,
    rubric_prompt_version: str = RUBRIC_PROMPT_VERSION,
    max_output_tokens: int = 256,
    confidence: float = 0.95,
) -> JudgeCalibrationResult:
    parsed_cases, _labeled_bytes, hashes = prepare_labeled_dataset(
        source_cases_bytes,
        human_label_overlay_bytes,
    )
    _validate_calibration_inputs(
        parsed_cases,
        manifest=manifest,
        hashes=hashes,
        expected_source_cases_hash=expected_source_cases_hash,
        expected_human_label_overlay_hash=expected_human_label_overlay_hash,
        expected_labeled_dataset_hash=expected_labeled_dataset_hash,
        expected_manifest_hash=expected_manifest_hash,
        manifest_bytes=manifest_bytes,
        judges=judges,
        evaluated_model_family=evaluated_model_family,
        min_class_count=min_class_count,
    )

    headline_cases = tuple(
        case for case in parsed_cases if case.certification_stratum != AUDIT_STRATUM
    )
    audit_cases = tuple(
        case for case in parsed_cases if case.certification_stratum == AUDIT_STRATUM
    )
    outputs, decisions = _judge_cases_by_lifecycle(
        parsed_cases,
        judges,
        rubric_prompt_version=rubric_prompt_version,
        max_output_tokens=max_output_tokens,
    )

    reports: list[PerJudgeReport] = []
    failed: list[str] = []
    identities = tuple(_identity_of(judge) for judge in judges)
    for index in range(len(judges)):
        identity = identities[index]
        headline = compute_judge_metrics(
            headline_cases,
            _select_outputs(headline_cases, parsed_cases, outputs[index]),
            confidence=confidence,
        )
        audit = compute_judge_metrics(
            audit_cases,
            _select_outputs(audit_cases, parsed_cases, outputs[index]),
            confidence=confidence,
        )
        judge_failures = _threshold_failures(identity, headline, thresholds)
        reports.append(
            PerJudgeReport(
                identity=identity,
                headline=headline,
                audit=audit,
                passed_thresholds=not judge_failures,
                failed_requirements=judge_failures,
            )
        )
        failed.extend(judge_failures)

    if any(not report.passed_thresholds for report in reports):
        failed.append("no single-judge fallback: both judges must pass")

    pair = _pair_metrics(
        headline_cases,
        _select_outputs(headline_cases, parsed_cases, outputs[0]),
        _select_outputs(headline_cases, parsed_cases, outputs[1]),
        confidence=confidence,
    )
    status: CalibrationStatus = (
        "certified_for_exploratory_grading" if not failed else "not_certified"
    )
    return JudgeCalibrationResult(
        status=status,
        dataset_hash=hashes.labeled_dataset_hash,
        source_cases_hash=hashes.source_cases_hash,
        human_label_overlay_hash=hashes.human_label_overlay_hash,
        labeled_dataset_hash=hashes.labeled_dataset_hash,
        rubric_prompt_hash=rubric_prompt_hash(rubric_prompt_version),
        judge_identities=identities,
        per_judge_metrics=tuple(reports),
        pair_metrics=pair,
        failed_requirements=tuple(failed),
        case_decisions=tuple(decisions),
    )


def grade_with_dual_judges(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    judges: Sequence[JudgeBackend],
    evaluated_model_family: str,
    deterministic_hard_failure: bool = False,
    rubric_prompt_version: str = RUBRIC_PROMPT_VERSION,
    max_output_tokens: int = 256,
) -> DualJudgeGrade:
    if deterministic_hard_failure:
        return DualJudgeGrade(
            status="deterministic_hard_fail",
            score=0.0,
            reasons=(),
            parsed=(),
            raw_results=(),
            invocation_order=(),
        )
    _validate_judge_pair(judges, evaluated_model_family)
    raw_results: list[JudgeBackendResult] = []
    parsed: list[ParsedJudgeOutput] = []
    order: list[str] = []
    for judge in judges:
        _acquire_backend(judge)
        try:
            result = _invoke_judge(
                judge,
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                rubric_prompt_version=rubric_prompt_version,
                max_output_tokens=max_output_tokens,
            )
            expected_hash = instance_prompt_hash(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                rubric_prompt_version=rubric_prompt_version,
            )
            _verify_backend_result(judge, result, expected_hash)
            raw_results.append(result)
            parsed.append(parse_judge_output(result.raw_text))
            order.append(judge.model_id)
        finally:
            _release_backend(judge)

    first, second = parsed
    reasons = tuple(
        item.reason for item in parsed if item.valid and item.reason is not None
    )
    if (
        first.valid
        and second.valid
        and first.score is not None
        and second.score is not None
        and first.score == second.score
    ):
        return DualJudgeGrade(
            status="agreed",
            score=first.score,
            reasons=reasons,
            parsed=(first, second),
            raw_results=(raw_results[0], raw_results[1]),
            invocation_order=tuple(order),
        )
    return DualJudgeGrade(
        status="human_adjudication_required",
        score=None,
        reasons=reasons,
        parsed=(first, second),
        raw_results=(raw_results[0], raw_results[1]),
        invocation_order=tuple(order),
    )


def invocation_was_sequential() -> bool:
    return not _INVOCATION.overlapping


def recorded_invocation_order() -> tuple[str, ...]:
    return tuple(_INVOCATION.order)


def recorded_lifecycle_events() -> tuple[str, ...]:
    return tuple(_INVOCATION.events)


def reset_invocation_trace() -> None:
    _INVOCATION.order.clear()
    _INVOCATION.events.clear()
    _INVOCATION.loaded.clear()
    _INVOCATION.in_flight = 0
    _INVOCATION.overlapping = False


def _invalid(error: str) -> ParsedJudgeOutput:
    return ParsedJudgeOutput(valid=False, score=None, reason=None, error=error)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _optional_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if numeric in LEGAL_SCORES:
        return numeric
    return None


def _training_flag(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("usable_for_training")
        or value.get("use_for_training")
        or value.get("training_eligible")
    )


def _domain_from_source_ids(source_ids: Sequence[str]) -> str:
    if not source_ids:
        return "none"
    prefix = source_ids[0].split("-", maxsplit=1)[0]
    return _DOMAIN_PREFIXES.get(prefix, "none")


def _identity_of(judge: JudgeBackend) -> JudgeIdentity:
    return JudgeIdentity(
        model_id=judge.model_id,
        model_family=_normalize_family(judge.model_family),
        revision=judge.revision,
    )


def _normalize_family(value: str) -> str:
    return str(value).strip().lower()


def _validate_calibration_inputs(
    cases: Sequence[CalibrationCase],
    *,
    manifest: Mapping[str, Any],
    hashes: DatasetHashes,
    expected_source_cases_hash: str,
    expected_human_label_overlay_hash: str,
    expected_labeled_dataset_hash: str,
    expected_manifest_hash: str | None,
    manifest_bytes: bytes | None,
    judges: Sequence[JudgeBackend],
    evaluated_model_family: str,
    min_class_count: int,
) -> None:
    if not cases:
        raise CalibrationValidationError("Calibration dataset is empty")
    if hashes.source_cases_hash != expected_source_cases_hash:
        raise CalibrationValidationError("source-cases hash does not match the expected hash")
    if hashes.human_label_overlay_hash != expected_human_label_overlay_hash:
        raise CalibrationValidationError(
            "human-label-overlay hash does not match the expected hash"
        )
    if hashes.labeled_dataset_hash != expected_labeled_dataset_hash:
        raise CalibrationValidationError(
            "labeled-dataset hash does not match the expected hash"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CalibrationValidationError("manifest is missing files")
    recorded = {
        SOURCE_CASES_FILE: files.get(SOURCE_CASES_FILE),
        HUMAN_LABEL_OVERLAY_FILE: files.get(HUMAN_LABEL_OVERLAY_FILE),
        LABELED_DATASET_FILE: files.get(LABELED_DATASET_FILE),
    }
    expected_files = {
        SOURCE_CASES_FILE: hashes.source_cases_hash,
        HUMAN_LABEL_OVERLAY_FILE: hashes.human_label_overlay_hash,
        LABELED_DATASET_FILE: hashes.labeled_dataset_hash,
    }
    if recorded != expected_files:
        raise CalibrationValidationError(
            "manifest file hashes do not match the raw source, overlay, or labeled bytes"
        )
    if expected_manifest_hash is not None:
        actual_manifest_hash = (
            sha256_bytes(manifest_bytes)
            if manifest_bytes is not None
            else sha256_json(dict(manifest))
        )
        if actual_manifest_hash != expected_manifest_hash:
            raise CalibrationValidationError("manifest hash does not match the expected hash")

    _validate_judge_pair(judges, evaluated_model_family)

    if any(case.usable_for_training for case in cases):
        raise CalibrationValidationError("calibration rows must not be usable for model training")

    missing_human: list[str] = []
    for case in cases:
        if not case.human_approved:
            missing_human.append(f"{case.case_id}: human_approved is not true")
        if case.human_semantic_score not in LEGAL_SCORES:
            if case.proposed_semantic_score in LEGAL_SCORES:
                raise CalibrationValidationError(
                    f"{case.case_id}: model-proposed labels cannot replace human labels"
                )
            missing_human.append(f"{case.case_id}: missing legal human score")
        if not case.human_reason:
            if case.proposed_reason:
                raise CalibrationValidationError(
                    f"{case.case_id}: model-proposed labels cannot replace human labels"
                )
            missing_human.append(f"{case.case_id}: missing non-empty human reason")
        if case.requires_two_reviewers and len(set(case.human_reviewers)) < 2:
            raise CalibrationValidationError(
                f"{case.case_id}: disputed or critical cases require two named human reviewers"
            )
    if missing_human:
        raise CalibrationValidationError(
            "missing human labels block calibration: " + "; ".join(missing_human)
        )

    counts = {score: 0 for score in LEGAL_SCORES}
    eligible = [
        case for case in cases if case.certification_stratum == HEADLINE_STRATUM
    ]
    for case in eligible:
        if case.human_semantic_score in LEGAL_SCORES:
            counts[case.human_semantic_score] += 1
    short = [score for score, count in counts.items() if count < min_class_count]
    if short:
        raise CalibrationValidationError(
            "all three score classes must meet the approved minimum count on "
            f"judge_eligible rows; short classes: {short}"
        )


def _validate_judge_pair(judges: Sequence[JudgeBackend], evaluated_model_family: str) -> None:
    if len(judges) != 2:
        raise CalibrationValidationError("exactly two judges are required")
    families = [_normalize_family(judge.model_family) for judge in judges]
    evaluated = _normalize_family(evaluated_model_family)
    if not evaluated:
        raise CalibrationValidationError("evaluated model family is required")
    for judge, family in zip(judges, families, strict=True):
        if not getattr(judge, "local_only", False):
            raise CalibrationValidationError(
                f"{judge.model_id}: classified content cannot be sent to a non-local backend"
            )
        if family == evaluated:
            raise CalibrationValidationError(
                "judge model family must differ from the model being evaluated"
            )
    if families[0] == families[1]:
        raise CalibrationValidationError("the two judge families must differ from each other")


def _judge_cases_by_lifecycle(
    cases: Sequence[CalibrationCase],
    judges: Sequence[JudgeBackend],
    *,
    rubric_prompt_version: str,
    max_output_tokens: int,
) -> tuple[tuple[list[ParsedJudgeOutput], ...], list[CaseJudgeDecision]]:
    outputs: list[list[ParsedJudgeOutput]] = []
    decisions: list[CaseJudgeDecision] = []
    for judge in judges:
        parsed_for_judge: list[ParsedJudgeOutput] = []
        _acquire_backend(judge)
        try:
            for case in cases:
                result = _invoke_judge(
                    judge,
                    question=case.question,
                    reference_answer=case.reference_answer,
                    candidate_answer=case.candidate_answer,
                    rubric_prompt_version=rubric_prompt_version,
                    max_output_tokens=max_output_tokens,
                )
                expected_hash = instance_prompt_hash(
                    question=case.question,
                    reference_answer=case.reference_answer,
                    candidate_answer=case.candidate_answer,
                    rubric_prompt_version=rubric_prompt_version,
                )
                _verify_backend_result(judge, result, expected_hash)
                parsed = parse_judge_output(result.raw_text)
                parsed_for_judge.append(parsed)
                decisions.append(
                    CaseJudgeDecision(
                        case_id=case.case_id,
                        case_family_id=case.case_family_id,
                        identity=_identity_of(judge),
                        raw_text=result.raw_text,
                        parsed=parsed,
                        prompt_hash=result.prompt_hash,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        latency_seconds=result.latency_seconds,
                    )
                )
        finally:
            _release_backend(judge)
        outputs.append(parsed_for_judge)
    return tuple(outputs), decisions


def _acquire_backend(judge: JudgeBackend) -> None:
    if _INVOCATION.loaded:
        _INVOCATION.overlapping = True
    _INVOCATION.events.append(f"acquire:{judge.model_id}")
    _INVOCATION.loaded.add(judge.model_id)
    judge.acquire()


def _release_backend(judge: JudgeBackend) -> None:
    judge.release()
    _INVOCATION.events.append(f"release:{judge.model_id}")
    _INVOCATION.loaded.discard(judge.model_id)


def _verify_backend_result(
    judge: JudgeBackend,
    result: JudgeBackendResult,
    expected_prompt_hash: str,
) -> None:
    if result.model_id != judge.model_id:
        raise CalibrationValidationError(
            f"backend result model_id {result.model_id!r} does not match {judge.model_id!r}"
        )
    if result.revision != judge.revision:
        raise CalibrationValidationError(
            f"backend result revision {result.revision!r} does not match {judge.revision!r}"
        )
    if result.prompt_hash != expected_prompt_hash:
        raise CalibrationValidationError(
            "backend result prompt_hash does not match the exact instance-prompt hash"
        )


def _invoke_judge(
    judge: JudgeBackend,
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    rubric_prompt_version: str,
    max_output_tokens: int,
) -> JudgeBackendResult:
    _INVOCATION.in_flight += 1
    if _INVOCATION.in_flight > 1:
        _INVOCATION.overlapping = True
    try:
        _INVOCATION.order.append(judge.model_id)
        return judge.judge(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            rubric_prompt_version=rubric_prompt_version,
            max_output_tokens=max_output_tokens,
        )
    finally:
        _INVOCATION.in_flight -= 1


def _select_outputs(
    subset: Sequence[CalibrationCase],
    all_cases: Sequence[CalibrationCase],
    outputs: Sequence[ParsedJudgeOutput],
) -> list[ParsedJudgeOutput]:
    by_id = {case.case_id: output for case, output in zip(all_cases, outputs, strict=True)}
    return [by_id[case.case_id] for case in subset]


def _threshold_failures(
    identity: JudgeIdentity,
    metrics: JudgeMetrics,
    thresholds: ExploratoryCertificationThresholds,
) -> tuple[str, ...]:
    label = identity.model_id
    failed: list[str] = []
    clustered = metrics.family_clustered
    agreement = (
        clustered.exact_agreement.rate if clustered is not None else metrics.exact_agreement.rate
    )
    if agreement is None or agreement < thresholds.min_exact_agreement:
        failed.append(
            f"{label}: exact agreement "
            f"{_fmt_rate(agreement)} < {thresholds.min_exact_agreement}"
        )
    if (
        metrics.weighted_kappa is None
        or metrics.weighted_kappa < thresholds.min_weighted_kappa
    ):
        failed.append(
            f"{label}: weighted kappa "
            f"{_fmt_rate(metrics.weighted_kappa)} < {thresholds.min_weighted_kappa}"
        )
    incorrect = (
        clustered.incorrect_to_pass_rate.rate
        if clustered is not None
        else metrics.incorrect_to_pass_rate.rate
    )
    if incorrect is None or incorrect > thresholds.max_incorrect_to_pass_rate:
        failed.append(
            f"{label}: incorrect-to-pass rate "
            f"{_fmt_rate(incorrect)} > {thresholds.max_incorrect_to_pass_rate}"
        )
    invalid = (
        clustered.invalid_output_rate.rate
        if clustered is not None
        else metrics.invalid_output_rate.rate
    )
    if invalid is None or invalid > thresholds.max_invalid_output_rate:
        failed.append(
            f"{label}: invalid-output rate "
            f"{_fmt_rate(invalid)} > {thresholds.max_invalid_output_rate}"
        )
    for slice_metrics in metrics.by_error_category:
        clustered_rate = slice_metrics.family_clustered_false_pass_rate
        rate = clustered_rate.rate
        if (
            clustered_rate.denominator > 0
            and rate is not None
            and rate > thresholds.max_critical_false_pass_rate
        ):
            failed.append(
                f"{label}: critical-error category {slice_metrics.key} false-pass "
                f"{rate} > {thresholds.max_critical_false_pass_rate}"
            )
    return tuple(failed)


def _fmt_rate(value: float | None) -> str:
    return "undefined" if value is None else str(value)


def _pair_metrics(
    cases: Sequence[CalibrationCase],
    first: Sequence[ParsedJudgeOutput],
    second: Sequence[ParsedJudgeOutput],
    confidence: float,
) -> PairMetrics:
    disagreements: list[CaseDisagreement] = []
    agreed = 0
    for case, left, right in zip(cases, first, second, strict=True):
        same = left.valid and right.valid and left.score == right.score
        if same:
            agreed += 1
            continue
        disagreements.append(
            CaseDisagreement(
                case_id=case.case_id,
                judge_a_score=left.score,
                judge_b_score=right.score,
                judge_a_valid=left.valid,
                judge_b_valid=right.valid,
                judge_a_reason=left.reason,
                judge_b_reason=right.reason,
            )
        )
    return PairMetrics(
        n_compared=len(cases),
        agreement_rate=_rate(agreed, len(cases), confidence),
        disagreements=tuple(disagreements),
    )


def _slice_metrics(
    cases: Sequence[CalibrationCase],
    parsed: Sequence[ParsedJudgeOutput],
    kind: Literal["error_category", "domain"],
    confidence: float,
) -> tuple[SliceMetrics, ...]:
    buckets: dict[str, list[tuple[CalibrationCase, ParsedJudgeOutput]]] = {}
    for case, output in zip(cases, parsed, strict=True):
        keys = case.error_categories if kind == "error_category" else (case.domain,)
        if not keys:
            keys = ("none",)
        for key in keys:
            buckets.setdefault(key, []).append((case, output))
    slices: list[SliceMetrics] = []
    for key in sorted(buckets):
        paired = buckets[key]
        subset_cases = [item[0] for item in paired]
        subset_parsed = [item[1] for item in paired]
        metrics = compute_judge_metrics(
            subset_cases,
            subset_parsed,
            confidence=confidence,
            include_slices=False,
        )
        assert metrics.family_clustered is not None
        family_false_pass = metrics.family_clustered.fully_correct_false_pass_rate
        slices.append(
            SliceMetrics(
                key=key,
                n=len(paired),
                exact_agreement=metrics.exact_agreement,
                fully_correct_false_pass_rate=metrics.fully_correct_false_pass_rate,
                incorrect_to_pass_rate=metrics.incorrect_to_pass_rate,
                false_fail_rate=metrics.false_fail_rate,
                invalid_output_rate=metrics.invalid_output_rate,
                family_clustered_false_pass_rate=family_false_pass,
            )
        )
    return tuple(slices)


def _family_clustered_metrics(
    cases: Sequence[CalibrationCase],
    parsed: Sequence[ParsedJudgeOutput],
    confidence: float,
) -> FamilyClusteredMetrics:
    families: dict[str, list[tuple[CalibrationCase, ParsedJudgeOutput]]] = {}
    for case, output in zip(cases, parsed, strict=True):
        families.setdefault(case.case_family_id, []).append((case, output))
    n_families = len(families)
    exact_success = 0
    false_pass_num = 0
    false_pass_den = 0
    incorrect_num = 0
    incorrect_den = 0
    false_fail_num = 0
    false_fail_den = 0
    invalid_num = 0
    for members in families.values():
        if any(not output.valid or output.score is None for _case, output in members):
            invalid_num += 1
        valid = [
            (case, output)
            for case, output in members
            if (
                output.valid
                and output.score is not None
                and case.human_semantic_score in LEGAL_SCORES
            )
        ]
        if (
            valid
            and all(case.human_semantic_score == output.score for case, output in valid)
            and all(output.valid for _case, output in members)
        ):
            exact_success += 1
        if any(
            case.human_semantic_score != 1.0
            for case, output in valid
        ):
            false_pass_den += 1
            if any(
                output.score == 1.0 and case.human_semantic_score != 1.0
                for case, output in valid
            ):
                false_pass_num += 1
        if any(case.human_semantic_score == 0.0 for case, output in valid):
            incorrect_den += 1
            if any(
                output.score == 1.0 and case.human_semantic_score == 0.0
                for case, output in valid
            ):
                incorrect_num += 1
        if any(case.human_semantic_score == 1.0 for case, output in valid):
            false_fail_den += 1
            if any(
                output.score == 0.0 and case.human_semantic_score == 1.0
                for case, output in valid
            ):
                false_fail_num += 1
    return FamilyClusteredMetrics(
        n_families=n_families,
        exact_agreement=_clustered_rate(exact_success, n_families, confidence),
        fully_correct_false_pass_rate=_clustered_rate(
            false_pass_num, false_pass_den, confidence
        ),
        incorrect_to_pass_rate=_clustered_rate(incorrect_num, incorrect_den, confidence),
        false_fail_rate=_clustered_rate(false_fail_num, false_fail_den, confidence),
        invalid_output_rate=_clustered_rate(invalid_num, n_families, confidence),
    )


def _confusion_counts(
    human: Sequence[float],
    judge: Sequence[float],
) -> tuple[tuple[int, int, int], ...]:
    counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for left, right in zip(human, judge, strict=True):
        counts[SCORE_INDEX[left]][SCORE_INDEX[right]] += 1
    return (tuple(counts[0]), tuple(counts[1]), tuple(counts[2]))


def _quadratic_weights() -> tuple[tuple[float, float, float], ...]:
    weights = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(1.0 - ((i - j) / 2.0) ** 2)
        weights.append(tuple(row))
    return (weights[0], weights[1], weights[2])


def _kappa_from_matrix(
    counts: tuple[tuple[int, int, int], ...],
    confidence: float = 0.95,
) -> tuple[float | None, float | None, float | None, str]:
    total = sum(sum(row) for row in counts)
    interval = "fleiss_cohen_wald"
    if total == 0:
        return None, None, None, interval
    weights = _quadratic_weights()
    observed = 0.0
    for i in range(3):
        for j in range(3):
            observed += counts[i][j] * weights[i][j]
    po = observed / total
    row_n = [sum(row) for row in counts]
    col_n = [sum(counts[i][j] for i in range(3)) for j in range(3)]
    expected = 0.0
    for i in range(3):
        for j in range(3):
            expected += row_n[i] * col_n[j] * weights[i][j]
    pe = expected / (total * total)
    if math.isclose(pe, 1.0):
        return None, None, None, interval
    kappa = (po - pe) / (1.0 - pe)
    variance = _weighted_kappa_variance(counts, weights, po, pe, total)
    if variance is None or variance < 0.0:
        return kappa, None, None, interval
    z = _WILSON_Z_95 if math.isclose(confidence, 0.95) else _normal_z(confidence)
    se = math.sqrt(variance)
    return (
        kappa,
        max(-1.0, kappa - z * se),
        min(1.0, kappa + z * se),
        interval,
    )


def _weighted_kappa_variance(
    counts: tuple[tuple[int, int, int], ...],
    weights: tuple[tuple[float, float, float], ...],
    po: float,
    pe: float,
    total: int,
) -> float | None:
    if total <= 0 or math.isclose(pe, 1.0):
        return None
    row_p = [sum(row) / total for row in counts]
    col_p = [sum(counts[i][j] for i in range(3)) / total for j in range(3)]
    w_bar_row = [
        sum(weights[i][j] * col_p[j] for j in range(3))
        for i in range(3)
    ]
    w_bar_col = [
        sum(weights[i][j] * row_p[i] for i in range(3))
        for j in range(3)
    ]
    term = 0.0
    for i in range(3):
        for j in range(3):
            p_ij = counts[i][j] / total
            theta = weights[i][j] - (w_bar_row[i] + w_bar_col[j])
            term += p_ij * theta * theta
    return (term - (po - 2.0 * pe) ** 2) / (total * (1.0 - pe) ** 2)


def _normal_z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    # Inverse complementary error function via bisection for two-sided z.
    target = (1.0 + confidence) / 2.0
    low, high = 0.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _rate(numerator: int, denominator: int, confidence: float) -> RateWithCI:
    return _rate_with_role(
        numerator,
        denominator,
        confidence,
        unit="row",
        role="descriptive",
    )


def _clustered_rate(numerator: int, denominator: int, confidence: float) -> RateWithCI:
    return _rate_with_role(
        numerator,
        denominator,
        confidence,
        unit="case_family",
        role="clustered",
    )


def _rate_with_role(
    numerator: int,
    denominator: int,
    confidence: float,
    *,
    unit: str,
    role: str,
) -> RateWithCI:
    if denominator <= 0:
        return RateWithCI(
            numerator=numerator,
            denominator=0,
            rate=None,
            ci_low=None,
            ci_high=None,
            unit=unit,
            role=role,
        )
    rate = numerator / denominator
    low, high = clopper_pearson_two_sided(numerator, denominator, confidence)
    return RateWithCI(
        numerator=numerator,
        denominator=denominator,
        rate=rate,
        ci_low=low,
        ci_high=high,
        unit=unit,
        role=role,
    )


def clopper_pearson_two_sided(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    one_sided = 1.0 - (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else 1.0 - clopper_pearson_upper(
        trials - successes, trials, one_sided
    )
    high = 1.0 if successes == trials else clopper_pearson_upper(
        successes, trials, one_sided
    )
    return low, high
