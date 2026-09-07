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

AUDIT_SCHEMA_ID = "company-task-specialization-output-audit/v2-two-phase"
OVERALL_OUTCOMES = frozenset({"acceptable", "minor_revision", "unacceptable", "cannot_assess"})
SATISFACTION_VALUES = frozenset({"yes", "no", "unclear"})
INSTRUCTIONS = """\
# Blinded company-task specialization output audit — Phase A

Review the candidate using only the operational request and supplied authorized
evidence. Model identity, prior deterministic results, Gemma labels, case IDs,
and reference answers are deliberately hidden.

Judge the answer independently. Do not guess what a hidden reference said,
whether the old evaluator passed or failed the answer, or why its result may
have disagreed with yours. Accept semantically equivalent wording.

For every answer, list each material obligation, record whether it is satisfied,
identify unsafe or unsupported claims, decide whether the answer supports the
next operational step, and describe any ambiguity in the request, supplied
evidence, or candidate output.

Review acceptable answers as carefully as rejected answers. A single review is
development evidence, not independent agreement or production validation.
These labels must not replace the stopped pilot's scores.

After Phase A labels are complete and frozen, the maintainer—not the blinded
reviewer—uses the private mapping to compare human outcomes with historical
results. That Phase B diagnosis is written to a separate report.
"""
SCHEMA_GUIDE = """\
# Review template schema

Complete one JSON object per line in `review_template.jsonl`. Preserve each
`review_id` and fill every other field.

- `reviewer_id`: the actual human reviewer's identifier.
- `reviewed_at`: an ISO-8601 timestamp such as `2026-09-06T18:00:00+00:00`.
- `human_attested`: `true`, entered by the actual human reviewer.
- `overall_outcome`: exactly one of `acceptable`, `minor_revision`,
  `unacceptable`, or `cannot_assess`.
- `required_obligations`: a JSON list. Each entry has exactly:
  - `description`: a specific obligation stated as text;
  - `material`: Boolean `true` or `false`;
  - `satisfied`: exactly `yes`, `no`, or `unclear`.
- `unsafe_claims`: a JSON list of specific unsafe or unsupported claims, or
  `[]` when there are none.
- `supports_next_step`: exactly `yes`, `no`, or `unclear`.
- `ambiguities`: a JSON list describing ambiguity in the request, evidence, or
  candidate output, or `[]` when there is none.
- `notes`: optional free-text review notes; use an empty string when unneeded.

Do not add evaluator-failure or reference-error diagnoses. Those require hidden
historical information and belong to the maintainer's Phase B comparison.
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
    output_dir = output_root.resolve() / f"audit-v2-two-phase-{pilot_hash[:16]}"
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
    schema_guide_bytes = SCHEMA_GUIDE.encode("utf-8")
    mapping = {
        "schema_version": 2,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "source_pilot_sha256": pilot_hash,
        "blinding_salt": salt,
        "mapping": mapping_rows,
    }
    mapping_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    members = {
        "AUDIT_INSTRUCTIONS.md": instructions_bytes,
        "REVIEW_SCHEMA.md": schema_guide_bytes,
        "review_cases.jsonl": cases_bytes,
        "source_records.jsonl": sources_bytes,
        "review_template.jsonl": template_bytes,
    }
    manifest = {
        "schema_version": 2,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "supersedes_audit_schema_id": "company-task-specialization-output-audit/v1",
        "supersession_reason": (
            "Blinded reviewers cannot diagnose hidden evaluator or reference failures"
        ),
        "status": "awaiting_blinded_phase_a_human_review",
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
        "phase_a_excludes_causal_diagnosis": True,
        "phase_b_requires_frozen_labels_and_private_mapping": True,
        "human_approved": False,
        "replaces_historical_scores": False,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True)
    packet_path = output_dir / "specialization-output-audit-v2.zip"
    mapping_path = output_dir / "private-review-id-map-v2.json"
    template_path = output_dir / "review-template-v2.jsonl"
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
        or mapping.get("audit_schema_id") != manifest.get("audit_schema_id")
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

    outcomes = Counter(str(row["overall_outcome"]) for row in overlay_rows)
    role_summary: dict[str, Counter[str]] = {
        "teacher": Counter(),
        "repair": Counter(),
    }
    diagnosis_summary: Counter[str] = Counter()
    diagnosed_cases: list[dict[str, Any]] = []
    for review_id, row in overlay_by_review.items():
        mapping_row = mapping_by_review[review_id]
        role = str(mapping_row["role"])
        role_summary.setdefault(role, Counter())[str(row["overall_outcome"])] += 1
        historical = mapping_row.get("historical_deterministic")
        historical_status = (
            str(historical.get("status")) if isinstance(historical, Mapping) else "missing"
        )
        diagnosis = diagnose_phase_b_disagreement(
            human_outcome=str(row["overall_outcome"]),
            historical_status=historical_status,
        )
        diagnosis_summary[diagnosis] += 1
        diagnosed_cases.append(
            {
                "review_id": review_id,
                "case_id": mapping_row.get("case_id"),
                "hidden_role": role,
                "human_outcome": row["overall_outcome"],
                "historical_deterministic_status": historical_status,
                "phase_b_diagnosis": diagnosis,
            }
        )
    reviewers = sorted({str(row["reviewer_id"]) for row in overlay_rows})
    report = {
        "schema_version": 2,
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "status": "phase_b_comparison_complete_not_adjudicated",
        "created_at": datetime.now(UTC).isoformat(),
        "source_pilot_sha256": manifest["source_pilot_sha256"],
        "source_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "source_overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "case_count": len(overlay_rows),
        "reviewers": reviewers,
        "independent_reviewer_count": len(reviewers),
        "labels_per_case": 1,
        "independent_agreement_available": False,
        "overall_outcomes": dict(sorted(outcomes.items())),
        "by_hidden_role": {
            role: dict(sorted(counts.items())) for role, counts in role_summary.items()
        },
        "supports_next_step": dict(
            sorted(Counter(str(row["supports_next_step"]) for row in overlay_rows).items())
        ),
        "ambiguity_case_count": sum(bool(row["ambiguities"]) for row in overlay_rows),
        "unsafe_claim_case_count": sum(bool(row["unsafe_claims"]) for row in overlay_rows),
        "phase_b_diagnoses": dict(sorted(diagnosis_summary.items())),
        "diagnosed_cases": sorted(diagnosed_cases, key=lambda row: str(row["review_id"])),
        "diagnoses_are_investigation_categories_not_replacement_scores": True,
        "replaces_historical_scores": False,
        "human_approved": False,
        "requires_second_review_and_adjudication": True,
    }
    atomic_write_text(
        output_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return output_path


def diagnose_phase_b_disagreement(
    *,
    human_outcome: str,
    historical_status: str,
) -> str:
    """Map frozen human and historical outcomes to an investigation category."""
    historical_passed = historical_status == "pass"
    historical_failed = historical_status == "hard_fail"
    if not historical_passed and not historical_failed:
        return "historical_status_requires_investigation"
    if human_outcome == "acceptable":
        return (
            "human_machine_agreement_acceptable"
            if historical_passed
            else "possible_evaluator_false_failure"
        )
    if human_outcome == "unacceptable":
        return (
            "possible_evaluator_false_acceptance"
            if historical_passed
            else "supported_model_failure"
        )
    if human_outcome == "cannot_assess":
        return "request_evidence_reference_or_contract_ambiguity"
    if human_outcome == "minor_revision":
        return (
            "human_minor_issue_historical_pass"
            if historical_passed
            else "possible_materiality_or_representation_disagreement"
        )
    raise ValueError(f"Unknown human outcome: {human_outcome}")


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
        "required_obligations",
        "unsafe_claims",
        "supports_next_step",
        "ambiguities",
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
    ambiguities = row["ambiguities"]
    if not isinstance(ambiguities, list) or any(
        not isinstance(value, str) or not value.strip() for value in ambiguities
    ):
        raise ValueError("Audit ambiguities are invalid")
    if not isinstance(row["notes"], str):
        raise ValueError("Audit notes must be a string")


def _empty_review_row(review_id: str) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "reviewer_id": "",
        "reviewed_at": "",
        "human_attested": False,
        "overall_outcome": None,
        "required_obligations": [],
        "unsafe_claims": [],
        "supports_next_step": None,
        "ambiguities": [],
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
