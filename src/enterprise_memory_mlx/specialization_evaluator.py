"""Conservative evaluator contract for future specialization measurements.

Version 1 used required substrings in free-text fields as authoritative hard
failures. This version preserves machine authority only for syntax and
provenance checks. Natural-language obligation completeness is routed to a
calibrated semantic or human review.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .task_specialization import DECISIONS

EVALUATOR_V2_ID = "structured-policy-assessment/v2-obligation-review"
MULTIPART_SCHEMA_ID = "operational-assessment-multipart/v2"

MachineGradeStatus = Literal["hard_fail", "semantic_review_required"]
MULTIPART_FIELDS = frozenset(
    {"decisions", "required_actions", "missing_information", "exceptions", "evidence"}
)


@dataclass(frozen=True)
class MachineAssessmentGrade:
    status: MachineGradeStatus
    hard_failure_reasons: tuple[str, ...]
    semantic_review_reasons: tuple[str, ...]
    semantic_review_eligible: bool
    evaluator_id: str = EVALUATOR_V2_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "status": self.status,
            "hard_failure_reasons": list(self.hard_failure_reasons),
            "semantic_review_reasons": list(self.semantic_review_reasons),
            "semantic_review_eligible": self.semantic_review_eligible,
        }


def verify_evaluator_v2_contract(root: Path) -> dict[str, Any]:
    """Verify the draft evaluator and private-seed schema against their manifest."""
    asset_root = (
        root.resolve()
        / "knowledge"
        / "company_task_specialization"
        / "evaluator_v2"
    )
    manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((asset_root / "protocol.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or not isinstance(protocol, dict)
        or manifest.get("protocol_id") != "company-task-specialization/evaluator-v2"
        or protocol.get("protocol_id") != "company-task-specialization/evaluator-v2"
        or manifest.get("status") != "draft_awaiting_human_validation"
        or protocol.get("status") != "draft_awaiting_human_validation"
    ):
        raise ValueError("Evaluator-v2 identity or status is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Evaluator-v2 manifest has no file hashes")
    for name, expected_hash in files.items():
        path = asset_root / str(name)
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError(f"Evaluator-v2 asset hash mismatch: {name}")
    if (
        manifest.get("historical_pilot_scores_replaced") is not False
        or manifest.get("teacher_requalification_authorized") is not False
        or manifest.get("student_training_authorized") is not False
    ):
        raise ValueError("Evaluator-v2 must preserve the stop and block requalification")
    return manifest


def grade_legacy_assessment_v2(
    case: Mapping[str, Any],
    assessment: Mapping[str, Any] | None,
    *,
    parse_status: str,
) -> MachineAssessmentGrade:
    """Grade a v1-shaped answer without treating phrase overlap as semantics."""
    hard_failures: list[str] = []
    review_reasons: list[str] = []
    if assessment is None:
        hard_failures.append(f"parse:{parse_status}")
        return _grade(hard_failures, review_reasons)

    source_ids = _string_set(case.get("source_record_ids"))
    evidence = assessment.get("evidence")
    if not isinstance(evidence, list):
        hard_failures.append("schema:evidence must be an array")
        return _grade(hard_failures, review_reasons)
    cited_ids = [str(item.get("record_id", "")) for item in evidence if isinstance(item, Mapping)]
    if len(cited_ids) != len(evidence) or any(not record_id for record_id in cited_ids):
        hard_failures.append("schema:evidence entries require record_id")
    duplicate_ids = sorted(
        record_id for record_id in set(cited_ids) if cited_ids.count(record_id) > 1
    )
    if duplicate_ids:
        hard_failures.append(f"provenance:duplicate record IDs {duplicate_ids}")
    unauthorized = sorted(set(cited_ids) - source_ids)
    if unauthorized:
        hard_failures.append(f"provenance:unauthorized record IDs {unauthorized}")

    reference = case.get("reference")
    if not isinstance(reference, Mapping):
        hard_failures.append("contract:missing reference assessment")
        return _grade(hard_failures, review_reasons)
    if assessment.get("decision") != reference.get("decision"):
        review_reasons.append(
            "decision equivalence or correctness requires obligation-level review"
        )

    expected_ids = {
        str(item.get("record_id", ""))
        for item in reference.get("evidence", [])
        if isinstance(item, Mapping) and item.get("record_id")
    }
    if set(cited_ids) != expected_ids:
        review_reasons.append("evidence completeness requires semantic review")

    expected_terms = case.get("expected_field_terms")
    if isinstance(expected_terms, Mapping):
        for field in ("required_actions", "missing_information", "exceptions"):
            values = assessment.get(field)
            field_text = " ".join(values) if isinstance(values, list) else ""
            groups = expected_terms.get(field, [])
            if not isinstance(groups, list):
                hard_failures.append(f"contract:invalid expected terms for {field}")
                continue
            for group in groups:
                if isinstance(group, list) and not any(
                    _normalize(str(term)) in _normalize(field_text) for term in group
                ):
                    review_reasons.append(
                        f"{field}:possible missing obligation; lexical probe did not match {group}"
                    )

    review_reasons.append("free-text action completeness and polarity are not machine-certified")
    return _grade(hard_failures, review_reasons)


def parse_multipart_assessment_v2(
    text: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """Parse the draft multipart schema without scoring semantic correctness."""
    if not isinstance(text, str) or not text.strip():
        return None, "missing_output"
    normalized = text.strip()
    decoder = json.JSONDecoder()
    try:
        value, index = decoder.raw_decode(normalized)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if normalized[index:].strip():
        return None, "trailing_content"
    if not isinstance(value, dict) or set(value) != MULTIPART_FIELDS:
        return None, "schema_mismatch"
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return None, "invalid_decisions"
    item_ids: list[str] = []
    for item in decisions:
        if (
            not isinstance(item, dict)
            or set(item) != {"request_item_id", "decision"}
            or not isinstance(item.get("request_item_id"), str)
            or not item["request_item_id"].strip()
            or item.get("decision") not in DECISIONS
        ):
            return None, "invalid_decisions"
        item_ids.append(item["request_item_id"].strip())
    if len(item_ids) != len(set(item_ids)):
        return None, "duplicate_request_item"
    for field in ("required_actions", "missing_information", "exceptions"):
        items = value.get(field)
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            return None, f"invalid_{field}"
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return None, "invalid_evidence"
    for item in evidence:
        if (
            not isinstance(item, dict)
            or set(item) != {"record_id", "claim"}
            or not isinstance(item.get("record_id"), str)
            or not item["record_id"].strip()
            or not isinstance(item.get("claim"), str)
            or not item["claim"].strip()
        ):
            return None, "invalid_evidence"
    return value, "valid"


def _grade(
    hard_failures: list[str],
    review_reasons: list[str],
) -> MachineAssessmentGrade:
    return MachineAssessmentGrade(
        status="hard_fail" if hard_failures else "semantic_review_required",
        hard_failure_reasons=tuple(hard_failures),
        semantic_review_reasons=tuple(dict.fromkeys(review_reasons)),
        semantic_review_eligible=not hard_failures,
    )


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list | tuple):
        return set()
    return {str(item) for item in value if str(item)}


def _normalize(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .casefold()
        .replace("‑", " ")
        .replace("–", " ")
        .replace("-", " ")
    )
