"""Validation boundary for private, human-reviewed specialization seed cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .task_specialization import DECISIONS
from .utils import atomic_write_text, read_jsonl

REAL_WORK_SCHEMA_ID = "company-task-specialization-real-work-seed/v1"
TARGET_CASE_COUNT = 25
EVIDENCE_CLASSIFICATIONS = frozenset({"public", "internal_shared", "restricted_local_review"})


def validate_real_work_seed(
    *,
    root: Path,
    seed_path: Path,
    output_path: Path,
) -> Path:
    """Validate a private one-workflow seed and emit only a hash-bound manifest."""
    root = root.resolve()
    seed_path = seed_path.resolve()
    output_path = output_path.resolve()
    private_root = (root / "knowledge" / "private").resolve()
    try:
        seed_path.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("Real-work seed must remain under knowledge/private") from exc
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite real-work manifest: {output_path}")
    rows = read_jsonl(seed_path)
    if not rows:
        raise ValueError("Real-work seed is empty")
    seen: set[str] = set()
    workflows: set[str] = set()
    reviewer_sets: list[set[str]] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError("Real-work case IDs are missing or duplicated")
        seen.add(case_id)
        workflow = str(row.get("workflow", "")).strip()
        if not workflow:
            raise ValueError(f"{case_id} has no workflow")
        workflows.add(workflow)
        _validate_real_work_row(case_id, row)
        reviewer_sets.append({str(review["reviewer_id"]).strip() for review in row["reviews"]})
    if len(workflows) != 1:
        raise ValueError("The first real-work seed must focus on exactly one workflow")
    all_reviewers = sorted(set().union(*reviewer_sets))
    independently_reviewed = sum(len(reviewers) >= 2 for reviewers in reviewer_sets)
    manifest = {
        "schema_version": 1,
        "schema_id": REAL_WORK_SCHEMA_ID,
        "status": (
            "development_calibration_seed_ready"
            if len(rows) >= TARGET_CASE_COUNT
            else "insufficient_development_seed"
        ),
        "source_path": str(seed_path.relative_to(root)),
        "source_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "case_count": len(rows),
        "target_case_count": TARGET_CASE_COUNT,
        "workflow": next(iter(workflows)),
        "reviewers": all_reviewers,
        "independent_reviewer_count": len(all_reviewers),
        "cases_with_two_independent_reviewers": independently_reviewed,
        "independent_agreement_complete": independently_reviewed == len(rows),
        "contains_case_content": False,
        "source_is_private_and_gitignored": True,
        "authorizes_teacher_qualification": False,
        "authorizes_training": False,
        "next_action": (
            "calibrate evaluator and freeze fresh teacher-qualification cases"
            if len(rows) >= TARGET_CASE_COUNT
            else f"collect {TARGET_CASE_COUNT - len(rows)} additional approved cases"
        ),
    }
    atomic_write_text(
        output_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return output_path


def _validate_real_work_row(case_id: str, row: Mapping[str, Any]) -> None:
    required = {
        "case_id",
        "workflow",
        "request",
        "evidence",
        "accepted_resolution",
        "reviews",
        "redacted",
        "evidence_authorized_for_local_research",
    }
    if set(row) != required:
        raise ValueError(f"{case_id} has an invalid real-work schema")
    if (
        not isinstance(row["request"], str)
        or not row["request"].strip()
        or row["redacted"] is not True
        or row["evidence_authorized_for_local_research"] is not True
    ):
        raise ValueError(f"{case_id} must be redacted, authorized, and have a request")
    evidence = row["evidence"]
    if not isinstance(evidence, list):
        raise ValueError(f"{case_id} evidence must be an array")
    evidence_ids: set[str] = set()
    for item in evidence:
        if (
            not isinstance(item, dict)
            or set(item) != {"record_id", "title", "text", "source_uri", "classification"}
            or not all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("record_id", "title", "text", "source_uri")
            )
            or item.get("classification") not in EVIDENCE_CLASSIFICATIONS
        ):
            raise ValueError(f"{case_id} has invalid evidence")
        record_id = str(item["record_id"]).strip()
        if record_id in evidence_ids:
            raise ValueError(f"{case_id} has duplicate evidence IDs")
        evidence_ids.add(record_id)
    _validate_resolution(case_id, row["accepted_resolution"], evidence_ids)
    reviews = row["reviews"]
    if not isinstance(reviews, list) or not reviews:
        raise ValueError(f"{case_id} requires at least one human review")
    reviewer_ids: set[str] = set()
    for review in reviews:
        if (
            not isinstance(review, dict)
            or set(review)
            != {
                "reviewer_id",
                "domain_role",
                "reviewed_at",
                "human_attested",
                "approved",
            }
            or not all(
                isinstance(review.get(key), str) and review[key].strip()
                for key in ("reviewer_id", "domain_role", "reviewed_at")
            )
            or review.get("human_attested") is not True
            or review.get("approved") is not True
        ):
            raise ValueError(f"{case_id} has an invalid human review")
        try:
            datetime.fromisoformat(str(review["reviewed_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{case_id} has an invalid review timestamp") from exc
        reviewer_id = str(review["reviewer_id"]).strip()
        if reviewer_id in reviewer_ids:
            raise ValueError(f"{case_id} repeats a reviewer")
        reviewer_ids.add(reviewer_id)


def _validate_resolution(
    case_id: str,
    value: Any,
    evidence_ids: set[str],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "decision_items",
        "required_obligations",
        "missing_information",
        "applicable_exceptions",
        "supporting_record_ids",
    }:
        raise ValueError(f"{case_id} has an invalid accepted resolution")
    decisions = value["decision_items"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"{case_id} requires at least one decision item")
    item_ids: set[str] = set()
    for item in decisions:
        if (
            not isinstance(item, dict)
            or set(item) != {"request_item_id", "decision"}
            or not isinstance(item.get("request_item_id"), str)
            or not item["request_item_id"].strip()
            or item.get("decision") not in DECISIONS
        ):
            raise ValueError(f"{case_id} has an invalid decision item")
        item_id = str(item["request_item_id"]).strip()
        if item_id in item_ids:
            raise ValueError(f"{case_id} repeats a request item")
        item_ids.add(item_id)
    obligations = value["required_obligations"]
    if not isinstance(obligations, list):
        raise ValueError(f"{case_id} obligations must be an array")
    obligation_ids: set[str] = set()
    for obligation in obligations:
        if (
            not isinstance(obligation, dict)
            or set(obligation) != {"obligation_id", "description", "material"}
            or not isinstance(obligation.get("obligation_id"), str)
            or not obligation["obligation_id"].strip()
            or not isinstance(obligation.get("description"), str)
            or not obligation["description"].strip()
            or not isinstance(obligation.get("material"), bool)
        ):
            raise ValueError(f"{case_id} has an invalid obligation")
        obligation_id = str(obligation["obligation_id"]).strip()
        if obligation_id in obligation_ids:
            raise ValueError(f"{case_id} repeats an obligation")
        obligation_ids.add(obligation_id)
    for field in ("missing_information", "applicable_exceptions"):
        items = value[field]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"{case_id} has invalid {field}")
    supporting = value["supporting_record_ids"]
    if (
        not isinstance(supporting, list)
        or len(supporting) != len(set(supporting))
        or set(supporting) - evidence_ids
    ):
        raise ValueError(f"{case_id} resolution cites unauthorized evidence")
