"""Blinded human audit preparation for specialization pilot outputs."""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import load_records
from .utils import atomic_write_text, read_jsonl, sha256_json

AUDIT_SCHEMA_ID = "company-task-specialization-output-audit/v1"
OVERALL_OUTCOMES = frozenset({"acceptable", "minor_revision", "unacceptable", "cannot_assess"})
FAILURE_CLASSIFICATIONS = frozenset(
    {
        "no_failure",
        "genuine_task_error",
        "evaluator_false_failure",
        "reference_ambiguity",
        "output_contract_problem",
        "serialization_issue",
    }
)
SATISFACTION_VALUES = frozenset({"yes", "no", "unclear"})
INSTRUCTIONS = """\
# Blinded company-task specialization output audit

Review the candidate using only the operational request and supplied authorized
evidence. Model identity, prior deterministic results, Gemma labels, case IDs,
and reference answers are deliberately hidden.

For every answer:

1. list each required obligation implied by the request and evidence;
2. mark whether the candidate satisfies each obligation;
3. record every unsafe or unsupported claim;
4. decide whether the answer supports the next operational step;
5. classify any problem as a genuine task error, evaluator false failure,
   reference ambiguity, output-contract problem, or serialization issue.

Review acceptable answers as carefully as rejected answers. A single review is
development evidence, not independent agreement or production validation.
These labels must not replace the stopped pilot's scores.
"""


@dataclass(frozen=True)
class AuditPacketArtifacts:
    packet_path: Path
    mapping_path: Path
    template_path: Path


def prepare_specialization_audit(
    *,
    root: Path,
    pilot_path: Path,
    output_root: Path,
) -> AuditPacketArtifacts:
    """Blind all teacher and repair outputs from one immutable pilot."""
    root = root.resolve()
    pilot_path = pilot_path.resolve()
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    if not isinstance(pilot, dict) or pilot.get("protocol_id") != "company-task-specialization/v1":
        raise ValueError("Expected a company-task-specialization/v1 pilot")
    teacher = pilot.get("teacher_attempts")
    repairs = pilot.get("repair_attempts")
    if not isinstance(teacher, list) or not isinstance(repairs, list):
        raise ValueError("Pilot is missing teacher or repair attempts")
    if len(teacher) != 14 or len(repairs) != 8:
        raise ValueError("Audit requires the complete 14-teacher/8-repair corrected pilot")
    pilot_hash = hashlib.sha256(pilot_bytes).hexdigest()
    output_dir = output_root.resolve() / f"audit-{pilot_hash[:16]}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite specialization audit: {output_dir}")

    attempts = [
        (role, row) for role, rows in (("teacher", teacher), ("repair", repairs)) for row in rows
    ]
    salt = secrets.token_hex(32)
    blinded = []
    mapping_rows = []
    seen_ids: set[str] = set()
    for role, attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("Pilot attempt must be an object")
        review_id = _unique_review_id(seen_ids)
        seen_ids.add(review_id)
        candidate = attempt.get("assessment")
        if candidate is None:
            candidate = attempt.get("final_output") or attempt.get("raw_output")
        blinded.append(
            {
                "review_id": review_id,
                "operational_request": attempt.get("request"),
                "candidate_assessment": candidate,
                "source_record_ids": list(attempt.get("source_record_ids", [])),
            }
        )
        mapping_rows.append(
            {
                "review_id": review_id,
                "case_id": attempt.get("case_id"),
                "role": role,
                "attempt_sha256": sha256_json(attempt),
                "historical_deterministic": attempt.get("deterministic"),
                "historical_gemma_advisory": attempt.get("gemma_advisory"),
                "historical_governed_score": attempt.get("governed_score"),
            }
        )
    blinded.sort(key=lambda row: sha256_json({"salt": salt, "review_id": row["review_id"]}))
    source_ids = {str(record_id) for row in blinded for record_id in row["source_record_ids"]}
    records = {record.id: record for record in load_records(root / "knowledge")}
    sources = []
    for record_id in sorted(source_ids):
        record = records.get(record_id)
        if record is None:
            raise ValueError(f"Audit references an unknown source record: {record_id}")
        if record.sensitivity in {"restricted", "secret"}:
            raise ValueError(f"Audit would expose restricted evidence: {record_id}")
        sources.append(
            {
                "id": record.id,
                "domain": record.domain,
                "title": record.title,
                "statement": record.statement,
                "source_uri": record.source_uri,
                "status": record.status,
                "effective_from": record.effective_from,
                "effective_to": record.effective_to,
            }
        )
    template_rows = [_empty_review_row(str(row["review_id"])) for row in blinded]
    cases_bytes = _jsonl_bytes(blinded)
    sources_bytes = _jsonl_bytes(sources)
    template_bytes = _jsonl_bytes(template_rows)
    instructions_bytes = INSTRUCTIONS.encode("utf-8")
    mapping = {
        "schema_version": 1,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "source_pilot_sha256": pilot_hash,
        "blinding_salt": salt,
        "mapping": mapping_rows,
    }
    mapping_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    members = {
        "AUDIT_INSTRUCTIONS.md": instructions_bytes,
        "review_cases.jsonl": cases_bytes,
        "source_records.jsonl": sources_bytes,
        "review_template.jsonl": template_bytes,
    }
    manifest = {
        "schema_version": 1,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "status": "awaiting_human_review",
        "case_count": len(blinded),
        "teacher_answer_count": len(teacher),
        "repair_answer_count": len(repairs),
        "source_pilot_sha256": pilot_hash,
        "private_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in members.items()},
        "blinded_fields": [
            "model identity",
            "teacher versus repair role",
            "case ID",
            "reference answer",
            "deterministic result",
            "Gemma label",
            "governed score",
        ],
        "human_approved": False,
        "replaces_historical_scores": False,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True)
    packet_path = output_dir / "specialization-output-audit.zip"
    mapping_path = output_dir / "private-review-id-map.json"
    template_path = output_dir / "review-template.jsonl"
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(f"audit/{name}", content)
        archive.writestr("audit/packet_manifest.json", manifest_bytes)
    atomic_write_text(mapping_path, mapping_bytes.decode("utf-8"))
    atomic_write_text(template_path, template_bytes.decode("utf-8"))
    return AuditPacketArtifacts(
        packet_path=packet_path,
        mapping_path=mapping_path,
        template_path=template_path,
    )


def validate_specialization_audit_overlay(
    *,
    packet_path: Path,
    mapping_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> Path:
    """Validate one complete human overlay and write a separate audit report."""
    packet_path = packet_path.resolve()
    mapping_path = mapping_path.resolve()
    overlay_path = overlay_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")
    cases, manifest = _load_packet(packet_path)
    mapping_bytes = mapping_path.read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != manifest["private_mapping_sha256"]:
        raise ValueError("Private mapping hash does not match audit packet")
    mapping = json.loads(mapping_bytes)
    if (
        not isinstance(mapping, dict)
        or mapping.get("source_pilot_sha256") != manifest["source_pilot_sha256"]
    ):
        raise ValueError("Private mapping belongs to another pilot")
    mappings = mapping.get("mapping")
    if not isinstance(mappings, list):
        raise ValueError("Private mapping has no rows")
    mapping_by_review = {str(row.get("review_id")): row for row in mappings}
    expected_ids = {str(row["review_id"]) for row in cases}
    if set(mapping_by_review) != expected_ids:
        raise ValueError("Private mapping IDs do not match packet")

    overlay_rows = read_jsonl(overlay_path)
    overlay_by_review: dict[str, dict[str, Any]] = {}
    for row in overlay_rows:
        _validate_audit_row(row)
        review_id = str(row["review_id"])
        if review_id in overlay_by_review:
            raise ValueError(f"Duplicate audit review ID: {review_id}")
        overlay_by_review[review_id] = row
    if set(overlay_by_review) != expected_ids:
        raise ValueError("Audit overlay is incomplete or contains unknown review IDs")

    classifications = Counter(
        classification for row in overlay_rows for classification in row["failure_classifications"]
    )
    outcomes = Counter(str(row["overall_outcome"]) for row in overlay_rows)
    role_summary: dict[str, Counter[str]] = {
        "teacher": Counter(),
        "repair": Counter(),
    }
    for review_id, row in overlay_by_review.items():
        role = str(mapping_by_review[review_id]["role"])
        role_summary.setdefault(role, Counter())[str(row["overall_outcome"])] += 1
    reviewers = sorted({str(row["reviewer_id"]) for row in overlay_rows})
    report = {
        "schema_version": 1,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "status": "single_human_audit_complete_not_adjudicated",
        "created_at": datetime.now(UTC).isoformat(),
        "source_pilot_sha256": manifest["source_pilot_sha256"],
        "source_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "source_overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "case_count": len(overlay_rows),
        "reviewers": reviewers,
        "independent_reviewer_count": len(reviewers),
        "overall_outcomes": dict(sorted(outcomes.items())),
        "failure_classifications": dict(sorted(classifications.items())),
        "by_hidden_role": {
            role: dict(sorted(counts.items())) for role, counts in role_summary.items()
        },
        "supports_next_step": dict(
            sorted(Counter(str(row["supports_next_step"]) for row in overlay_rows).items())
        ),
        "unsafe_claim_case_count": sum(bool(row["unsafe_claims"]) for row in overlay_rows),
        "replaces_historical_scores": False,
        "human_approved": False,
        "requires_second_review_and_adjudication": True,
    }
    atomic_write_text(
        output_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return output_path


def _load_packet(packet_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(packet_path) as archive:
        manifest = json.loads(archive.read("audit/packet_manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("audit_schema_id") != AUDIT_SCHEMA_ID:
            raise ValueError("Invalid specialization audit packet")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Audit packet has no file hashes")
        for name, expected_hash in files.items():
            content = archive.read(f"audit/{name}")
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f"Audit packet member hash mismatch: {name}")
        cases = _read_jsonl_bytes(archive.read("audit/review_cases.jsonl"))
    if len(cases) != manifest.get("case_count"):
        raise ValueError("Audit packet case count mismatch")
    return cases, manifest


def _validate_audit_row(row: Mapping[str, Any]) -> None:
    required = {
        "review_id",
        "reviewer_id",
        "reviewed_at",
        "human_attested",
        "overall_outcome",
        "failure_classifications",
        "required_obligations",
        "unsafe_claims",
        "supports_next_step",
        "notes",
    }
    if set(row) != required:
        raise ValueError("Audit overlay row has an invalid schema")
    if (
        not str(row["review_id"]).strip()
        or not str(row["reviewer_id"]).strip()
        or not str(row["reviewed_at"]).strip()
        or row["human_attested"] is not True
    ):
        raise ValueError("Audit reviewer identity, timestamp, and attestation are required")
    if row["overall_outcome"] not in OVERALL_OUTCOMES:
        raise ValueError("Audit overall outcome is invalid")
    classifications = row["failure_classifications"]
    if (
        not isinstance(classifications, list)
        or not classifications
        or any(value not in FAILURE_CLASSIFICATIONS for value in classifications)
        or ("no_failure" in classifications and len(classifications) != 1)
    ):
        raise ValueError("Audit failure classifications are invalid")
    obligations = row["required_obligations"]
    if not isinstance(obligations, list):
        raise ValueError("Audit required obligations must be an array")
    for obligation in obligations:
        if (
            not isinstance(obligation, dict)
            or set(obligation) != {"description", "material", "satisfied"}
            or not isinstance(obligation.get("description"), str)
            or not obligation["description"].strip()
            or not isinstance(obligation.get("material"), bool)
            or obligation.get("satisfied") not in SATISFACTION_VALUES
        ):
            raise ValueError("Audit obligation entry is invalid")
    unsafe = row["unsafe_claims"]
    if not isinstance(unsafe, list) or any(
        not isinstance(value, str) or not value.strip() for value in unsafe
    ):
        raise ValueError("Audit unsafe claims are invalid")
    if row["supports_next_step"] not in SATISFACTION_VALUES:
        raise ValueError("Audit next-step assessment is invalid")
    if not isinstance(row["notes"], str):
        raise ValueError("Audit notes must be a string")


def _empty_review_row(review_id: str) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "reviewer_id": "",
        "reviewed_at": "",
        "human_attested": False,
        "overall_outcome": None,
        "failure_classifications": [],
        "required_obligations": [],
        "unsafe_claims": [],
        "supports_next_step": None,
        "notes": "",
    }


def _unique_review_id(seen: set[str]) -> str:
    while True:
        value = f"CTSA-{secrets.token_hex(6).upper()}"
        if value not in seen:
            return value


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _read_jsonl_bytes(content: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in content.decode("utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Audit packet JSONL row must be an object")
            rows.append(value)
    return rows
