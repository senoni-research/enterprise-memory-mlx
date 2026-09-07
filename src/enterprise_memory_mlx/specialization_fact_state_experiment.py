"""Bounded fact-state amendment comparison for the structured-remedy prompt."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import MLXBenchmarkBackend
from .experiment_profiles import QWEN_27B_MODEL_ID, QWEN_27B_REVISION
from .remedy_logic import parse_remedy_assessment, remedy_machine_grade, render_remedy
from .specialization_remedy_experiment import (
    CHALLENGER_PROMPT_VERSION,
    CHALLENGER_SYSTEM_PROMPT,
    load_remedy_experiment_assets,
)
from .task_specialization import GeneratorBackend
from .utils import atomic_write_text, read_jsonl, sha256_json, sha256_text

PROTOCOL_ID = "company-task-specialization/v4-fact-state"
CONTROL_ARM = "v3_structured_control"
CANDIDATE_ARM = "fact_state_amendment"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)
AMENDED_PROMPT_VERSION = "operational-assessment-remedy-logic/v1+fact-state-v1"
REVIEW_SCHEMA_ID = "fact-state-arm-label-blinded-review/v1"
OUTCOMES = frozenset({"acceptable", "minor_revision", "unacceptable", "cannot_assess"})
JUDGMENTS = frozenset({"yes", "no", "unclear"})
SCOPED_JUDGMENTS = frozenset({"yes", "no", "unclear", "not_applicable"})

STATE_AUTHORITY_GUIDE = {
    "authorized_workflow_record": (
        "Authenticated workflow state. Treat the recorded fact as established unless "
        "another unsuperseded authorized record conflicts with it."
    ),
    "workflow_status_observation": (
        "Authenticated observation of the workflow field. A stated absence of status "
        "means that condition is unresolved, not complete or incomplete."
    ),
    "requester_claim": (
        "Unverified requester statement. Do not treat it as authenticated workflow "
        "state unless the supplied policy expressly permits self-attestation."
    ),
    "document_observation": (
        "Establishes only whether the named document is available; it does not by "
        "itself establish completion unless the supplied policy says so."
    ),
}

FACT_STATE_AMENDMENT = """\

Fact-state amendment:
Apply the supplied policy to the supplied operational state. Preserve facts
established by an authorized workflow record. A workflow status observation
with no recorded status is genuinely unresolved. A requester claim is
unverified and must not be treated as authenticated workflow state unless the
policy expressly permits that. A document observation establishes only
document availability.

Missing documentation does not make an established condition unknown unless
the policy requires that documentation or its verification. A statement that
all other controls in the specified workflow or policy are complete covers
those controls unless contradicted; do not reopen them.

Do not invent additional conditions for proceeding. Every blocker must identify
an applicable policy requirement and the supplied state showing that it is
unmet. Every information request must concern something genuinely unresolved.
Surface unresolved conflicts between authorized records; honor explicit
supersession.

Keep decisions scoped to the supplied policy and evidence. "No blocker under
this policy" does not mean unconditional approval under every possible company
rule.
"""

AMENDED_SYSTEM_PROMPT = CHALLENGER_SYSTEM_PROMPT + FACT_STATE_AMENDMENT

REVIEW_INSTRUCTIONS = """\
# Arm-label-blinded fact-state review

Use only the supplied fictional policies, state-authority guide, operational
state, request item, and candidate assessment. Arm, case, contrast-pair,
reference, machine status, timing, and tokens are hidden. Both candidates use
the same structured-remedy schema.

Judge whether the candidate:

1. applies only requirements present in the supplied policy;
2. preserves facts established by authenticated workflow records;
3. treats genuinely absent workflow status as unknown;
4. does not infer completion merely from document presence;
5. does not infer non-completion merely from document absence;
6. enforces a document or verification requirement when policy explicitly
   requires it;
7. respects collective "all other controls complete" statements within their
   stated scope;
8. distinguishes unverified requester claims from authenticated workflow state;
9. surfaces unresolved authoritative conflicts and honors explicit supersession;
10. preserves valid remedy alternatives and cumulative requirements without an
    unsafe waiver.

Equivalent wording is acceptable. Correct structure does not prove correct
policy application. Labels are model-advisory and not human evidence.
"""

REVIEW_SCHEMA_GUIDE = """\
# Review output schema

Write one JSON object per line in template order with no extra fields.

- `reviewer_kind`: `model_advisory`
- `reviewer_id`: exact model name/version
- `reviewed_at`: ISO-8601 timestamp
- `human_attested`: `false`
- `overall_outcome`: `acceptable`, `minor_revision`, `unacceptable`, or
  `cannot_assess`
- `required_obligations`: objects with `description`, Boolean `material`, and
  `satisfied` (`yes`, `no`, or `unclear`)
- `unsafe_claims`: concrete unsupported requirements or unsafe waivers, or `[]`
- `supports_next_step`: `yes`, `no`, or `unclear`
- `fact_state_assessment.established_facts_preserved`: `yes`, `no`, or `unclear`
- `fact_state_assessment.unknowns_handled_correctly`: `yes`, `no`, `unclear`, or
  `not_applicable`
- `fact_state_assessment.documentation_scope_correct`: `yes`, `no`, `unclear`,
  or `not_applicable`
- `fact_state_assessment.authority_boundary_correct`: `yes`, `no`, `unclear`,
  or `not_applicable`
- `fact_state_assessment.no_invented_requirement`: `yes`, `no`, or `unclear`
- `fact_state_assessment.no_unsafe_waiver`: `yes`, `no`, or `unclear`
- `fact_state_assessment.remedy_logic_preserved`: `yes`, `no`, `unclear`, or
  `not_applicable`
- `ambiguities`: specific ambiguities, or `[]`
- `notes`: concise case-specific reasoning
"""


@dataclass(frozen=True)
class FactStateAssets:
    root: Path
    protocol: dict[str, Any]
    manifest: dict[str, Any]
    cases: tuple[dict[str, Any], ...]
    records: dict[str, dict[str, Any]]
    protocol_hash: str
    cases_hash: str
    records_hash: str
    manifest_hash: str


@dataclass(frozen=True)
class FactStateArtifacts:
    directory: Path
    report_path: Path


@dataclass(frozen=True)
class FactStateReviewArtifacts:
    directory: Path
    packet_path: Path
    mapping_path: Path
    template_path: Path


@dataclass(frozen=True)
class FactStateDecisionArtifacts:
    directory: Path
    report_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class FactStateRegressionArtifacts:
    directory: Path
    report_path: Path


@dataclass(frozen=True)
class FactStateCorrectionArtifacts:
    directory: Path
    report_path: Path


def load_fact_state_assets(root: Path) -> FactStateAssets:
    root = root.resolve()
    asset_root = root / "knowledge" / "company_task_specialization" / "v4_fact_state"
    protocol_path = asset_root / "protocol.json"
    cases_path = asset_root / "contrast_cases.jsonl"
    records_path = asset_root / "policy_records.jsonl"
    manifest_path = asset_root / "manifest.json"
    protocol = _read_object(protocol_path)
    manifest = _read_object(manifest_path)
    cases = tuple(read_jsonl(cases_path))
    record_rows = tuple(read_jsonl(records_path))
    records = {str(row.get("id")): row for row in record_rows}
    if len(records) != len(record_rows):
        raise ValueError("Fact-state policy record IDs are duplicated")
    protocol_hash = _file_hash(protocol_path)
    cases_hash = _file_hash(cases_path)
    records_hash = _file_hash(records_path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or manifest.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "frozen_before_model_generation"
        or manifest.get("status") != "frozen_before_model_generation"
    ):
        raise ValueError("Fact-state identity or freeze status is invalid")
    if manifest.get("files") != {
        "protocol.json": protocol_hash,
        "contrast_cases.jsonl": cases_hash,
        "policy_records.jsonl": records_hash,
    }:
        raise ValueError("Fact-state manifest hashes do not match")
    for binding_name in ("preserved_v3_report", "v3_regression_set"):
        _verify_binding(root, manifest.get(binding_name), binding_name)
    if (
        manifest.get("case_count") != len(cases)
        or manifest.get("pair_count") != len({case["pair_id"] for case in cases})
        or manifest.get("human_labels_present") is not False
        or manifest.get("training_eligible") is not False
    ):
        raise ValueError("Fact-state manifest scope is invalid")
    _validate_protocol(protocol)
    _validate_records(records)
    _validate_cases(cases, records)
    return FactStateAssets(
        root=root,
        protocol=protocol,
        manifest=manifest,
        cases=cases,
        records=records,
        protocol_hash=protocol_hash,
        cases_hash=cases_hash,
        records_hash=records_hash,
        manifest_hash=_file_hash(manifest_path),
    )


def run_fact_state_comparison(
    *,
    root: Path,
    output_root: Path,
    generator_factory: Callable[[str, str], GeneratorBackend] | None = None,
) -> FactStateArtifacts:
    assets = load_fact_state_assets(root)
    prompt_hashes = {
        CONTROL_ARM: sha256_text(CHALLENGER_SYSTEM_PROMPT),
        CANDIDATE_ARM: sha256_text(AMENDED_SYSTEM_PROMPT),
    }
    run_identity = sha256_json(
        {
            "protocol": assets.protocol_hash,
            "cases": assets.cases_hash,
            "records": assets.records_hash,
            "manifest": assets.manifest_hash,
            "prompts": prompt_hashes,
            "model": [QWEN_27B_MODEL_ID, QWEN_27B_REVISION],
        }
    )[:16]
    output_dir = output_root.resolve() / f"comparison-{run_identity}"
    report_path = output_dir / "fact-state-comparison.json"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fact-state run: {output_dir}")
    create_generator = generator_factory or (
        lambda model_id, revision: MLXBenchmarkBackend(model_id, revision=revision)
    )
    backend = create_generator(QWEN_27B_MODEL_ID, QWEN_27B_REVISION)
    try:
        attempts = {
            CONTROL_ARM: [
                _generate_attempt(
                    assets,
                    backend=backend,
                    case=case,
                    arm=CONTROL_ARM,
                    system_prompt=CHALLENGER_SYSTEM_PROMPT,
                )
                for case in assets.cases
            ],
            CANDIDATE_ARM: [
                _generate_attempt(
                    assets,
                    backend=backend,
                    case=case,
                    arm=CANDIDATE_ARM,
                    system_prompt=AMENDED_SYSTEM_PROMPT,
                )
                for case in assets.cases
            ],
        }
    finally:
        backend.close()
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "awaiting_arm_label_blinded_model_advisory",
        "created_at": datetime.now(UTC).isoformat(),
        "run_identity": run_identity,
        "artifact_chain": {
            "protocol_sha256": assets.protocol_hash,
            "contrast_cases_sha256": assets.cases_hash,
            "policy_records_sha256": assets.records_hash,
            "manifest_sha256": assets.manifest_hash,
            "preserved_v3_report": assets.manifest["preserved_v3_report"],
            "v3_regression_set": assets.manifest["v3_regression_set"],
        },
        "generator": {
            "model_id": QWEN_27B_MODEL_ID,
            "revision": QWEN_27B_REVISION,
            "thinking": False,
            "temperature": 0.0,
            "max_output_tokens": 1024,
        },
        "prompt_identity": {
            CONTROL_ARM: {
                "version": CHALLENGER_PROMPT_VERSION,
                "sha256": prompt_hashes[CONTROL_ARM],
            },
            CANDIDATE_ARM: {
                "version": AMENDED_PROMPT_VERSION,
                "sha256": prompt_hashes[CANDIDATE_ARM],
            },
        },
        "arm_summaries": {arm: _attempt_summary(rows) for arm, rows in attempts.items()},
        "attempts": attempts,
        "qualification": {
            "status": "not_scored_awaiting_model_advisory",
            "targeted_learning_rule": assets.protocol["targeted_learning_rule"],
            "substantive_gate": assets.protocol["substantive_gate"],
        },
        "v3_regression_authorized": False,
        "human_evidence": False,
        "training_eligible": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return FactStateArtifacts(output_dir, report_path)


def correct_fact_state_machine_grades(
    *,
    root: Path,
    comparison_path: Path,
    output_root: Path,
) -> FactStateCorrectionArtifacts:
    """Correct the supplied-state provenance set without changing generations."""
    assets = load_fact_state_assets(root)
    comparison_path = comparison_path.resolve()
    comparison_bytes = comparison_path.read_bytes()
    comparison_hash = hashlib.sha256(comparison_bytes).hexdigest()
    comparison = json.loads(comparison_bytes)
    _validate_comparison(comparison, assets)
    case_by_id = {str(case["case_id"]): case for case in assets.cases}
    corrected_attempts: dict[str, list[dict[str, Any]]] = {}
    changed = 0
    for arm in ARMS:
        rows = comparison["attempts"].get(arm)
        if not isinstance(rows, list) or len(rows) != len(assets.cases):
            raise ValueError(f"Fact-state arm is incomplete: {arm}")
        corrected_rows = []
        for source_attempt in rows:
            attempt = dict(source_attempt)
            case = case_by_id.get(str(attempt.get("case_id")))
            if case is None:
                raise ValueError("Fact-state comparison contains an unknown case")
            assessment, parse_status = parse_remedy_assessment(attempt.get("final_output"))
            authorized_ids = _authorized_record_ids(case)
            machine_grade = remedy_machine_grade(
                assessment=assessment,
                parse_status=parse_status,
                expected_request_item_ids=[
                    str(item["request_item_id"]) for item in case["request_items"]
                ],
                supplied_record_ids=authorized_ids,
            )
            if machine_grade != attempt.get("machine_grade"):
                changed += 1
            attempt["assessment"] = assessment
            attempt["structured_parse_status"] = parse_status
            attempt["machine_grade"] = machine_grade
            attempt["machine_grade_correction"] = {
                "reason": (
                    "Operational-state IDs are generator-visible supplied records and "
                    "belong in the provenance allowlist."
                ),
                "authorized_record_ids": authorized_ids,
                "generation_changed": False,
            }
            corrected_rows.append(attempt)
        corrected_attempts[arm] = corrected_rows
    corrected = {
        **comparison,
        "created_at": datetime.now(UTC).isoformat(),
        "run_identity": f"{comparison['run_identity']}-machine-grade-v2",
        "arm_summaries": {arm: _attempt_summary(rows) for arm, rows in corrected_attempts.items()},
        "attempts": corrected_attempts,
        "runtime_correction": {
            "kind": "model_free_provenance_allowlist_correction",
            "source_comparison_path": str(comparison_path),
            "source_comparison_sha256": comparison_hash,
            "attempt_count": sum(len(rows) for rows in corrected_attempts.values()),
            "changed_machine_grades": changed,
            "generation_calls_added": 0,
            "generation_outputs_changed": False,
            "prompt_or_case_changed": False,
        },
    }
    identity = sha256_json(
        {
            "source_comparison": comparison_hash,
            "correction": corrected["runtime_correction"]["kind"],
        }
    )[:16]
    output_dir = output_root.resolve() / f"comparison-corrected-{identity}"
    report_path = output_dir / "fact-state-comparison-corrected.json"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fact-state grade correction: {output_dir}")
    output_dir.mkdir(parents=True)
    atomic_write_text(
        report_path,
        json.dumps(corrected, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return FactStateCorrectionArtifacts(output_dir, report_path)


def prepare_fact_state_review(
    *,
    root: Path,
    comparison_path: Path,
    output_root: Path,
) -> FactStateReviewArtifacts:
    assets = load_fact_state_assets(root)
    comparison_bytes = comparison_path.resolve().read_bytes()
    comparison_hash = hashlib.sha256(comparison_bytes).hexdigest()
    comparison = json.loads(comparison_bytes)
    _validate_comparison(comparison, assets)
    case_by_id = {str(case["case_id"]): case for case in assets.cases}
    blinded: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for arm in ARMS:
        rows = comparison["attempts"].get(arm)
        if not isinstance(rows, list) or len(rows) != len(assets.cases):
            raise ValueError(f"Fact-state arm is incomplete: {arm}")
        for attempt in rows:
            case = case_by_id.get(str(attempt.get("case_id")))
            if case is None:
                raise ValueError("Fact-state comparison contains an unknown case")
            review_id = _unique_review_id(seen)
            blinded.append(
                {
                    "review_id": review_id,
                    "state_authority_guide": STATE_AUTHORITY_GUIDE,
                    "operational_state": case["operational_state"],
                    "request_items": case["request_items"],
                    "candidate_assessment": attempt.get("assessment")
                    or attempt.get("final_output")
                    or attempt.get("raw_output"),
                    "source_record_ids": case["source_record_ids"],
                    "authorized_record_ids": _authorized_record_ids(case),
                }
            )
            mapping_rows.append(
                {
                    "review_id": review_id,
                    "case_id": case["case_id"],
                    "pair_id": case["pair_id"],
                    "contrast_role": case["contrast_role"],
                    "arm": arm,
                    "reference": case["reference"],
                    "expected_fact_state": case["expected_fact_state"],
                    "evaluation_tags": case["evaluation_tags"],
                    "attempt_sha256": sha256_json(attempt),
                    "machine_grade": attempt["machine_grade"],
                    "total_elapsed_seconds": attempt["total_elapsed_seconds"],
                    "prompt_tokens": attempt["prompt_tokens"],
                    "completion_tokens": attempt["completion_tokens"],
                }
            )
    salt = secrets.token_hex(32)
    blinded.sort(key=lambda row: sha256_json({"salt": salt, "review_id": row["review_id"]}))
    source_ids = {str(record_id) for row in blinded for record_id in row["source_record_ids"]}
    source_rows = [assets.records[record_id] for record_id in sorted(source_ids)]
    template = [_empty_review_row(str(row["review_id"])) for row in blinded]
    members = {
        "REVIEW_INSTRUCTIONS.md": REVIEW_INSTRUCTIONS.encode(),
        "REVIEW_SCHEMA.md": REVIEW_SCHEMA_GUIDE.encode(),
        "review_cases.jsonl": _jsonl_bytes(blinded),
        "source_records.jsonl": _jsonl_bytes(source_rows),
        "review_template.jsonl": _jsonl_bytes(template),
    }
    mapping = {
        "schema_version": 1,
        "review_schema_id": REVIEW_SCHEMA_ID,
        "source_comparison_sha256": comparison_hash,
        "blinding_salt": salt,
        "mapping": mapping_rows,
    }
    mapping_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    manifest = {
        "schema_version": 1,
        "review_schema_id": REVIEW_SCHEMA_ID,
        "status": "awaiting_arm_label_blinded_model_advisory",
        "case_count": len(blinded),
        "distinct_scenario_count": len(assets.cases),
        "contrast_pair_count": 8,
        "source_comparison_sha256": comparison_hash,
        "private_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in members.items()},
        "arm_label_blinded": True,
        "same_output_schema": True,
        "hidden_fields": [
            "arm",
            "case and contrast-pair IDs",
            "reference and expected fact state",
            "machine status",
            "timing and tokens",
        ],
        "reviewer_kind": "model_advisory",
        "human_evidence": False,
        "training_eligible": False,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir = output_root.resolve() / f"review-{comparison_hash[:16]}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fact-state review: {output_dir}")
    output_dir.mkdir(parents=True)
    packet_path = output_dir / "fact-state-model-advisory.zip"
    mapping_path = output_dir / "private-review-map.json"
    template_path = output_dir / "model-advisory-template.jsonl"
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(f"review/{name}", content)
        archive.writestr("review/packet_manifest.json", manifest_bytes)
    atomic_write_text(mapping_path, mapping_bytes.decode())
    atomic_write_text(template_path, members["review_template.jsonl"].decode())
    return FactStateReviewArtifacts(output_dir, packet_path, mapping_path, template_path)


def score_fact_state_review(
    *,
    root: Path,
    comparison_path: Path,
    packet_path: Path,
    mapping_path: Path,
    advisory_path: Path,
    output_root: Path,
) -> FactStateDecisionArtifacts:
    assets = load_fact_state_assets(root)
    comparison_bytes = comparison_path.resolve().read_bytes()
    comparison_hash = hashlib.sha256(comparison_bytes).hexdigest()
    comparison = json.loads(comparison_bytes)
    _validate_comparison(comparison, assets)
    packet_rows, packet_manifest = _load_packet(packet_path.resolve())
    if packet_manifest["source_comparison_sha256"] != comparison_hash:
        raise ValueError("Fact-state packet belongs to another comparison")
    mapping_bytes = mapping_path.resolve().read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != packet_manifest["private_mapping_sha256"]:
        raise ValueError("Fact-state private mapping hash mismatch")
    mapping = json.loads(mapping_bytes)
    if not isinstance(mapping, dict):
        raise ValueError("Fact-state private mapping is invalid")
    mapping_rows = mapping.get("mapping")
    if (
        mapping.get("review_schema_id") != REVIEW_SCHEMA_ID
        or mapping.get("source_comparison_sha256") != comparison_hash
        or not isinstance(mapping_rows, list)
    ):
        raise ValueError("Fact-state private mapping is invalid")
    mapping_by_id = {str(row["review_id"]): row for row in mapping_rows}
    packet_ids = {str(row["review_id"]) for row in packet_rows}
    if set(mapping_by_id) != packet_ids:
        raise ValueError("Fact-state mapping does not match the packet")
    advisory_rows = read_jsonl(advisory_path.resolve())
    advisory_by_id: dict[str, dict[str, Any]] = {}
    for row in advisory_rows:
        _validate_review_row(row)
        review_id = str(row["review_id"])
        if review_id in advisory_by_id:
            raise ValueError(f"Duplicate fact-state review ID: {review_id}")
        advisory_by_id[review_id] = row
    if set(advisory_by_id) != packet_ids:
        raise ValueError("Fact-state advisory is incomplete or has unknown IDs")

    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for review_id, advisory in advisory_by_id.items():
        mapped = mapping_by_id[review_id]
        arm = str(mapped["arm"])
        case_id = str(mapped["case_id"])
        by_arm[arm][case_id] = {
            "review_id": review_id,
            "case_id": case_id,
            "pair_id": mapped["pair_id"],
            "overall_outcome": advisory["overall_outcome"],
            "unsafe_claim_count": len(advisory["unsafe_claims"]),
            "supports_next_step": advisory["supports_next_step"],
            "fact_state_assessment": advisory["fact_state_assessment"],
            "advisory_sha256": sha256_json(advisory),
        }
    expected_ids = {str(case["case_id"]) for case in assets.cases}
    if any(set(rows) != expected_ids for rows in by_arm.values()):
        raise ValueError("Fact-state advisory does not form a complete paired matrix")

    arm_results = {
        arm: _score_arm(
            rows=list(by_arm[arm].values()),
            attempts=comparison["attempts"][arm],
            gate=assets.protocol["substantive_gate"],
        )
        for arm in ARMS
    }
    paired = _paired_comparison(by_arm)
    learning = {
        "fact_state_wins": paired["fact_state_wins"],
        "fact_state_losses": paired["fact_state_losses"],
        "no_invented_requirement_wins": paired["no_invented_requirement_wins"],
        "no_invented_requirement_losses": paired["no_invented_requirement_losses"],
        "remedy_logic_losses": paired["remedy_logic_losses"],
        "unsafe_waiver_losses": paired["unsafe_waiver_losses"],
    }
    learning["improved"] = bool(
        learning["fact_state_wins"] > learning["fact_state_losses"]
        and learning["no_invented_requirement_wins"] >= learning["no_invented_requirement_losses"]
        and learning["remedy_logic_losses"] == 0
        and learning["unsafe_waiver_losses"] == 0
    )
    candidate_passes = bool(arm_results[CANDIDATE_ARM]["passes_substantive_gate"])
    fresh_success = bool(learning["improved"] and candidate_passes)
    if fresh_success:
        decision = "fact_state_amendment_passes_fresh_comparison"
    elif learning["improved"]:
        decision = "fact_state_improves_but_candidate_fails_gate"
    elif candidate_passes:
        decision = "candidate_passes_without_demonstrated_fact_state_improvement"
    else:
        decision = "no_success_stop_synthetic_prompt_iteration"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete_model_advisory_not_human_validated",
        "created_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "targeted_learning": learning,
        "qualification": {
            "candidate_passes_substantive_gate": candidate_passes,
            "fresh_comparison_success": fresh_success,
        },
        "arm_results": arm_results,
        "paired_comparison": paired,
        "v3_regression_authorized": fresh_success,
        "bounded_advisory_teacher_designation": None,
        "historical_v3_decision": "no_arm_passes_substantive_gate",
        "source_bindings": {
            "protocol_sha256": assets.protocol_hash,
            "cases_sha256": assets.cases_hash,
            "records_sha256": assets.records_hash,
            "comparison_sha256": comparison_hash,
            "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "advisory_sha256": hashlib.sha256(advisory_path.read_bytes()).hexdigest(),
        },
        "reviewer_kind": "model_advisory",
        "human_evidence": False,
        "training_authorized": False,
        "admission_protocol_approved": False,
        "stronger_model_authorized": False,
    }
    identity = sha256_json(report["source_bindings"])[:16]
    output_dir = output_root.resolve() / f"decision-{identity}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fact-state decision: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "fact-state-decision.json"
    markdown_path = output_dir / "fact-state-decision.md"
    atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    atomic_write_text(markdown_path, _render_decision(report))
    return FactStateDecisionArtifacts(output_dir, report_path, markdown_path)


def run_fact_state_regression(
    *,
    root: Path,
    fresh_decision_path: Path,
    output_root: Path,
    generator_factory: Callable[[str, str], GeneratorBackend] | None = None,
) -> FactStateRegressionArtifacts:
    """Run the sole amended-prompt v3 regression only after fresh success."""
    assets = load_fact_state_assets(root)
    fresh_decision_path = fresh_decision_path.resolve()
    fresh_decision = _read_object(fresh_decision_path)
    _validate_regression_authorization(fresh_decision, assets)
    v3_assets = load_remedy_experiment_assets(root)
    if (
        v3_assets.cases_hash != assets.manifest["v3_regression_set"]["sha256"]
        or len(v3_assets.cases) != 24
    ):
        raise ValueError("V3 regression set does not match the frozen binding")
    prompt_hash = sha256_text(AMENDED_SYSTEM_PROMPT)
    decision_hash = _file_hash(fresh_decision_path)
    run_identity = sha256_json(
        {
            "protocol": assets.protocol_hash,
            "fresh_decision": decision_hash,
            "v3_cases": v3_assets.cases_hash,
            "prompt": prompt_hash,
            "model": [QWEN_27B_MODEL_ID, QWEN_27B_REVISION],
        }
    )[:16]
    output_dir = output_root.resolve() / f"regression-{run_identity}"
    report_path = output_dir / "fact-state-v3-regression.json"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fact-state regression: {output_dir}")
    create_generator = generator_factory or (
        lambda model_id, revision: MLXBenchmarkBackend(model_id, revision=revision)
    )
    backend = create_generator(QWEN_27B_MODEL_ID, QWEN_27B_REVISION)
    try:
        attempts = [
            _generate_v3_regression_attempt(
                backend=backend,
                case=case,
                records=v3_assets.records,
            )
            for case in v3_assets.cases
        ]
    finally:
        backend.close()
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "regression_id": "company-task-specialization/v4-fact-state/v3-regression",
        "status": "awaiting_arm_label_blinded_model_advisory",
        "created_at": datetime.now(UTC).isoformat(),
        "run_identity": run_identity,
        "authorization": {
            "fresh_decision_path": str(fresh_decision_path),
            "fresh_decision_sha256": decision_hash,
            "fresh_comparison_success": True,
            "single_regression_pass": True,
        },
        "artifact_chain": {
            "v4_protocol_sha256": assets.protocol_hash,
            "v3_cases_sha256": v3_assets.cases_hash,
            "v3_manifest_sha256": v3_assets.manifest_hash,
        },
        "generator": {
            "model_id": QWEN_27B_MODEL_ID,
            "revision": QWEN_27B_REVISION,
            "thinking": False,
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "prompt_version": AMENDED_PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
        },
        "summary": _attempt_summary(attempts),
        "attempts": attempts,
        "historical_v3_scores_modified": False,
        "bounded_advisory_teacher_designation": None,
        "human_evidence": False,
        "training_eligible": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return FactStateRegressionArtifacts(output_dir, report_path)


def _generate_attempt(
    assets: FactStateAssets,
    *,
    backend: GeneratorBackend,
    case: Mapping[str, Any],
    arm: str,
    system_prompt: str,
) -> dict[str, Any]:
    payload = {
        "evidence": [assets.records[str(value)] for value in case["source_record_ids"]],
        "state_authority_guide": STATE_AUTHORITY_GUIDE,
        "operational_state": case["operational_state"],
        "request_items": case["request_items"],
    }
    question = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    started = time.perf_counter()
    try:
        answer = backend.generate(
            system_prompt=system_prompt,
            question=question,
            max_tokens=1024,
        )
        generation_error = None
    except Exception as exc:
        answer = None
        generation_error = f"{type(exc).__name__}: {exc}"
    final_output = answer.output if answer else None
    assessment, parse_status = parse_remedy_assessment(final_output)
    machine_grade = remedy_machine_grade(
        assessment=assessment,
        parse_status=parse_status,
        expected_request_item_ids=[str(item["request_item_id"]) for item in case["request_items"]],
        supplied_record_ids=_authorized_record_ids(case),
    )
    rendered = _render_assessment(assessment)
    total_elapsed = time.perf_counter() - started
    if answer is not None and answer.truncated:
        machine_grade = {
            **machine_grade,
            "status": "hard_fail",
            "hard_failure_reasons": [
                *machine_grade["hard_failure_reasons"],
                "generation:truncated_output",
            ],
            "semantic_review_eligible": False,
        }
        generation_error = generation_error or "truncated_output"
    generated = bool(answer and final_output is not None and not answer.truncated)
    return {
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "arm": arm,
        "model_id": backend.model_name,
        "model_revision": QWEN_27B_REVISION,
        "source_record_ids": case["source_record_ids"],
        "request_payload_hash": sha256_text(question),
        "generation_status": "generated" if generated else "failed",
        "generation_error": generation_error,
        "raw_output": answer.raw_output if answer else None,
        "final_output": final_output,
        "assessment": assessment,
        "structured_parse_status": parse_status,
        "generator_parse_status": answer.parse_status if answer else "exception",
        "finish_reason": answer.finish_reason if answer else None,
        "truncated": answer.truncated if answer else None,
        "prompt_tokens": answer.prompt_tokens if answer else None,
        "completion_tokens": answer.completion_tokens if answer else None,
        "generation_elapsed_seconds": answer.elapsed_seconds if answer else None,
        "total_elapsed_seconds": total_elapsed,
        "peak_memory_gb": answer.peak_memory_gb if answer else None,
        "rendered_prompt_hash": answer.prompt_hash if answer else None,
        "chat_template_hash": answer.rendered_template_hash if answer else None,
        "deterministic_render_sha256": sha256_text(rendered),
        "machine_grade": machine_grade,
    }


def _generate_v3_regression_attempt(
    *,
    backend: GeneratorBackend,
    case: Mapping[str, Any],
    records: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "evidence": [
            _knowledge_record_dict(records[str(value)]) for value in case["source_record_ids"]
        ],
        "operational_facts": case["operational_facts"],
        "request_items": case["request_items"],
    }
    question = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    started = time.perf_counter()
    try:
        answer = backend.generate(
            system_prompt=AMENDED_SYSTEM_PROMPT,
            question=question,
            max_tokens=1024,
        )
        generation_error = None
    except Exception as exc:
        answer = None
        generation_error = f"{type(exc).__name__}: {exc}"
    final_output = answer.output if answer else None
    assessment, parse_status = parse_remedy_assessment(final_output)
    machine_grade = remedy_machine_grade(
        assessment=assessment,
        parse_status=parse_status,
        expected_request_item_ids=[str(item["request_item_id"]) for item in case["request_items"]],
        supplied_record_ids=case["source_record_ids"],
    )
    rendered = _render_assessment(assessment)
    total_elapsed = time.perf_counter() - started
    if answer is not None and answer.truncated:
        machine_grade = {
            **machine_grade,
            "status": "hard_fail",
            "hard_failure_reasons": [
                *machine_grade["hard_failure_reasons"],
                "generation:truncated_output",
            ],
            "semantic_review_eligible": False,
        }
        generation_error = generation_error or "truncated_output"
    generated = bool(answer and final_output is not None and not answer.truncated)
    return {
        "case_id": case["case_id"],
        "arm": CANDIDATE_ARM,
        "model_id": backend.model_name,
        "model_revision": QWEN_27B_REVISION,
        "source_record_ids": case["source_record_ids"],
        "request_payload_hash": sha256_text(question),
        "generation_status": "generated" if generated else "failed",
        "generation_error": generation_error,
        "raw_output": answer.raw_output if answer else None,
        "final_output": final_output,
        "assessment": assessment,
        "structured_parse_status": parse_status,
        "generator_parse_status": answer.parse_status if answer else "exception",
        "finish_reason": answer.finish_reason if answer else None,
        "truncated": answer.truncated if answer else None,
        "prompt_tokens": answer.prompt_tokens if answer else None,
        "completion_tokens": answer.completion_tokens if answer else None,
        "generation_elapsed_seconds": answer.elapsed_seconds if answer else None,
        "total_elapsed_seconds": total_elapsed,
        "peak_memory_gb": answer.peak_memory_gb if answer else None,
        "rendered_prompt_hash": answer.prompt_hash if answer else None,
        "chat_template_hash": answer.rendered_template_hash if answer else None,
        "deterministic_render_sha256": sha256_text(rendered),
        "machine_grade": machine_grade,
    }


def _render_assessment(assessment: Mapping[str, Any] | None) -> str:
    if assessment is None:
        return ""
    rows = []
    for item in assessment["assessments"]:
        remedy = item["remedy"]
        if remedy is not None:
            rows.append(f"{item['request_item_id']}: {render_remedy(remedy)}")
    return "\n".join(rows)


def _attempt_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "generation_failures": sum(row["generation_status"] != "generated" for row in rows),
        "truncations": sum(row.get("truncated") is True for row in rows),
        "valid_structures": sum(row["structured_parse_status"] == "valid" for row in rows),
        "machine_hard_failures": sum(row["machine_grade"]["status"] == "hard_fail" for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "generation_elapsed_seconds": sum(
            float(row.get("generation_elapsed_seconds") or 0.0) for row in rows
        ),
        "total_elapsed_seconds": sum(
            float(row.get("total_elapsed_seconds") or 0.0) for row in rows
        ),
        "peak_memory_gb": max(
            (float(row.get("peak_memory_gb") or 0.0) for row in rows),
            default=0.0,
        ),
    }


def _score_arm(
    *,
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    outcomes = {
        outcome: sum(row["overall_outcome"] == outcome for row in rows)
        for outcome in sorted(OUTCOMES)
    }
    acceptable = outcomes["acceptable"]
    unsafe = sum(int(row["unsafe_claim_count"] > 0) for row in rows)
    failures = sum(row["generation_status"] != "generated" for row in attempts)
    truncations = sum(row.get("truncated") is True for row in attempts)
    invalid = sum(
        row["structured_parse_status"] != "valid" or row["machine_grade"]["status"] == "hard_fail"
        for row in attempts
    )
    checks = {
        "minimum_cases": len(rows) >= int(gate["minimum_cases_per_arm"]),
        "acceptable_rate": acceptable / len(rows) >= float(gate["minimum_acceptable_rate"]),
        "unacceptable_count": outcomes["unacceptable"] <= int(gate["maximum_unacceptable_count"]),
        "unsafe_claim_cases": unsafe <= int(gate["maximum_unsafe_or_unsupported_claim_cases"]),
        "all_contrast_cases_acceptable": acceptable == len(rows),
        "generation_failures": failures <= int(gate["maximum_generation_failures"]),
        "truncations": truncations <= int(gate["maximum_truncations"]),
        "invalid_structures": invalid <= int(gate["maximum_invalid_structures"]),
    }
    return {
        "case_count": len(rows),
        "outcomes": outcomes,
        "acceptable_rate": acceptable / len(rows),
        "unsafe_claim_case_count": unsafe,
        "generation_failures": failures,
        "truncations": truncations,
        "invalid_structures": invalid,
        "supports_next_step_yes": sum(row["supports_next_step"] == "yes" for row in rows),
        "fact_state_assessment_counts": {
            field: {
                value: sum(row["fact_state_assessment"][field] == value for row in rows)
                for value in sorted(SCOPED_JUDGMENTS)
            }
            for field in (
                "established_facts_preserved",
                "unknowns_handled_correctly",
                "documentation_scope_correct",
                "authority_boundary_correct",
                "no_invented_requirement",
                "no_unsafe_waiver",
                "remedy_logic_preserved",
            )
        },
        "total_elapsed_seconds": sum(
            float(row.get("total_elapsed_seconds") or 0.0) for row in attempts
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in attempts),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in attempts),
        "checks": checks,
        "passes_substantive_gate": all(checks.values()),
    }


def _paired_comparison(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    control = by_arm[CONTROL_ARM]
    candidate = by_arm[CANDIDATE_ARM]
    outcome_rank = {
        "cannot_assess": 0,
        "unacceptable": 1,
        "minor_revision": 2,
        "acceptable": 3,
    }
    judgment_rank = {"no": 0, "unclear": 1, "yes": 2}
    fields = (
        "established_facts_preserved",
        "unknowns_handled_correctly",
        "documentation_scope_correct",
        "authority_boundary_correct",
        "no_invented_requirement",
        "no_unsafe_waiver",
        "remedy_logic_preserved",
    )
    field_totals = {
        field: {"wins": 0, "losses": 0, "ties": 0, "not_comparable": 0} for field in fields
    }
    pairs = []
    outcome_wins = outcome_losses = 0
    fact_wins = fact_losses = 0
    for case_id in sorted(control):
        left = control[case_id]
        right = candidate[case_id]
        outcome_delta = (
            outcome_rank[str(right["overall_outcome"])] - outcome_rank[str(left["overall_outcome"])]
        )
        if outcome_delta > 0:
            outcome_wins += 1
        elif outcome_delta < 0:
            outcome_losses += 1
        deltas: dict[str, int | None] = {}
        for field in fields:
            before = str(left["fact_state_assessment"][field])
            after = str(right["fact_state_assessment"][field])
            if "not_applicable" in {before, after}:
                deltas[field] = None
                field_totals[field]["not_comparable"] += 1
                continue
            delta = judgment_rank[after] - judgment_rank[before]
            deltas[field] = delta
            if delta > 0:
                field_totals[field]["wins"] += 1
            elif delta < 0:
                field_totals[field]["losses"] += 1
            else:
                field_totals[field]["ties"] += 1
        fact_delta = sum(
            delta
            for field, delta in deltas.items()
            if field != "remedy_logic_preserved" and delta is not None
        )
        if fact_delta > 0:
            fact_wins += 1
        elif fact_delta < 0:
            fact_losses += 1
        pairs.append(
            {
                "case_id": case_id,
                "pair_id": left["pair_id"],
                "control_outcome": left["overall_outcome"],
                "candidate_outcome": right["overall_outcome"],
                "outcome_delta": outcome_delta,
                "fact_state_delta": fact_delta,
                "field_deltas": deltas,
            }
        )
    return {
        "scenario_count": len(pairs),
        "contrast_pair_count": len({row["pair_id"] for row in pairs}),
        "overall_outcome_wins": outcome_wins,
        "overall_outcome_losses": outcome_losses,
        "fact_state_wins": fact_wins,
        "fact_state_losses": fact_losses,
        "no_invented_requirement_wins": field_totals["no_invented_requirement"]["wins"],
        "no_invented_requirement_losses": field_totals["no_invented_requirement"]["losses"],
        "remedy_logic_losses": field_totals["remedy_logic_preserved"]["losses"],
        "unsafe_waiver_losses": field_totals["no_unsafe_waiver"]["losses"],
        "field_totals": field_totals,
        "pairs": pairs,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    generator = protocol.get("generator")
    arms = protocol.get("arms")
    fresh = protocol.get("fresh_comparison")
    if (
        not isinstance(generator, dict)
        or generator.get("model_id") != QWEN_27B_MODEL_ID
        or generator.get("revision") != QWEN_27B_REVISION
        or generator.get("thinking") is not False
        or generator.get("temperature") != 0.0
        or generator.get("max_output_tokens") != 1024
        or not isinstance(arms, dict)
        or arms.get(CONTROL_ARM, {}).get("prompt_version") != CHALLENGER_PROMPT_VERSION
        or arms.get(CANDIDATE_ARM, {}).get("prompt_version") != AMENDED_PROMPT_VERSION
        or any(arms.get(arm, {}).get("generation_calls_per_case") != 1 for arm in ARMS)
        or not isinstance(fresh, dict)
        or fresh.get("case_count") != 16
        or fresh.get("contrast_pair_count") != 8
        or fresh.get("output_count") != 32
        or fresh.get("frozen_before_generation") is not True
    ):
        raise ValueError("Fact-state model, arm, or budget contract drifted")
    if (
        protocol.get("substantive_gate", {}).get("frozen_before_generation") is not True
        or protocol.get("targeted_learning_rule", {}).get("frozen_before_generation") is not True
    ):
        raise ValueError("Fact-state scoring rules are not frozen")


def _validate_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    required = {
        "id",
        "title",
        "statement",
        "source_uri",
        "status",
        "effective_from",
        "effective_to",
    }
    if len(records) < 3:
        raise ValueError("Fact-state policy snapshot is incomplete")
    for record_id, record in records.items():
        if set(record) != required or record["id"] != record_id:
            raise ValueError("Fact-state policy record schema is invalid")


def _validate_cases(
    cases: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "case_id",
        "pair_id",
        "contrast_role",
        "split",
        "source_record_ids",
        "operational_state",
        "request_items",
        "reference",
        "expected_fact_state",
        "evaluation_tags",
        "authoring",
    }
    state_fields = {
        "state_id",
        "authority",
        "statement",
        "recorded_at",
        "supersedes_state_id",
    }
    seen_cases: set[str] = set()
    seen_states: set[str] = set()
    pairs: dict[str, int] = {}
    for case in cases:
        if set(case) != required:
            raise ValueError("Fact-state case schema is invalid")
        case_id = str(case["case_id"])
        if not case_id or case_id in seen_cases:
            raise ValueError("Fact-state case IDs are missing or duplicated")
        seen_cases.add(case_id)
        pair_id = str(case["pair_id"])
        pairs[pair_id] = pairs.get(pair_id, 0) + 1
        if case["split"] != "fresh_fact_state_contrast":
            raise ValueError(f"{case_id} has an invalid split")
        source_ids = case["source_record_ids"]
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or any(str(record_id) not in records for record_id in source_ids)
        ):
            raise ValueError(f"{case_id} has invalid policy evidence")
        states = case["operational_state"]
        if not isinstance(states, list) or not states:
            raise ValueError(f"{case_id} has no operational state")
        case_state_ids: set[str] = set()
        for state in states:
            if (
                not isinstance(state, dict)
                or set(state) != state_fields
                or state.get("authority") not in STATE_AUTHORITY_GUIDE
                or not str(state.get("state_id", "")).strip()
                or not str(state.get("statement", "")).strip()
                or not str(state.get("recorded_at", "")).strip()
            ):
                raise ValueError(f"{case_id} has invalid operational state")
            state_id = str(state["state_id"])
            if state_id in case_state_ids or state_id in seen_states:
                raise ValueError("Fact-state IDs must be globally unique")
            case_state_ids.add(state_id)
            seen_states.add(state_id)
        for state in states:
            supersedes = state["supersedes_state_id"]
            if supersedes is not None and supersedes not in case_state_ids:
                raise ValueError(f"{case_id} supersedes an unavailable state")
        request_items = case["request_items"]
        if (
            not isinstance(request_items, list)
            or not request_items
            or any(
                not isinstance(item, dict) or set(item) != {"request_item_id", "question"}
                for item in request_items
            )
        ):
            raise ValueError(f"{case_id} request items are invalid")
        request_ids = [str(item["request_item_id"]) for item in request_items]
        parsed, status = parse_remedy_assessment(json.dumps(case["reference"]))
        if parsed is None:
            raise ValueError(f"{case_id} reference is invalid: {status}")
        grade = remedy_machine_grade(
            assessment=parsed,
            parse_status=status,
            expected_request_item_ids=request_ids,
            supplied_record_ids=source_ids,
        )
        if grade["status"] == "hard_fail":
            raise ValueError(f"{case_id} reference fails structural checks")
        fact_state = case["expected_fact_state"]
        if (
            not isinstance(fact_state, dict)
            or set(fact_state)
            != {
                "established",
                "unknown",
                "unmet",
                "non_authoritative_or_scoped",
                "required_behavior",
            }
            or any(
                not isinstance(fact_state[field], list)
                for field in (
                    "established",
                    "unknown",
                    "unmet",
                    "non_authoritative_or_scoped",
                )
            )
            or not str(fact_state["required_behavior"]).strip()
        ):
            raise ValueError(f"{case_id} expected fact state is invalid")
        if (
            not isinstance(case["evaluation_tags"], list)
            or not case["evaluation_tags"]
            or case["authoring"].get("production_observation") is not False
        ):
            raise ValueError(f"{case_id} metadata is invalid")
    if len(cases) != 16 or len(pairs) != 8 or any(count != 2 for count in pairs.values()):
        raise ValueError("Fact-state suite must contain eight two-scenario pairs")


def _validate_comparison(comparison: Any, assets: FactStateAssets) -> None:
    if (
        not isinstance(comparison, dict)
        or comparison.get("protocol_id") != PROTOCOL_ID
        or comparison.get("status") != "awaiting_arm_label_blinded_model_advisory"
        or comparison.get("artifact_chain", {}).get("protocol_sha256") != assets.protocol_hash
        or comparison.get("artifact_chain", {}).get("contrast_cases_sha256") != assets.cases_hash
        or comparison.get("artifact_chain", {}).get("policy_records_sha256") != assets.records_hash
        or not isinstance(comparison.get("attempts"), dict)
    ):
        raise ValueError("Fact-state comparison does not match frozen assets")


def _validate_regression_authorization(
    decision: Mapping[str, Any],
    assets: FactStateAssets,
) -> None:
    bindings = decision.get("source_bindings")
    qualification = decision.get("qualification")
    if (
        decision.get("protocol_id") != PROTOCOL_ID
        or decision.get("status") != "complete_model_advisory_not_human_validated"
        or decision.get("v3_regression_authorized") is not True
        or not isinstance(qualification, dict)
        or qualification.get("fresh_comparison_success") is not True
        or not isinstance(bindings, dict)
        or bindings.get("protocol_sha256") != assets.protocol_hash
        or bindings.get("cases_sha256") != assets.cases_hash
        or bindings.get("records_sha256") != assets.records_hash
    ):
        raise ValueError("V3 regression is blocked until the frozen fresh comparison succeeds")


def _load_packet(packet_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(packet_path) as archive:
        manifest = json.loads(archive.read("review/packet_manifest.json"))
        if manifest.get("review_schema_id") != REVIEW_SCHEMA_ID:
            raise ValueError("Fact-state review packet is invalid")
        for name, expected in manifest["files"].items():
            content = archive.read(f"review/{name}")
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Fact-state packet member hash mismatch: {name}")
        rows = _read_jsonl_bytes(archive.read("review/review_cases.jsonl"))
    if len(rows) != manifest.get("case_count"):
        raise ValueError("Fact-state packet count mismatch")
    return rows, manifest


def _validate_review_row(row: Mapping[str, Any]) -> None:
    required = {
        "review_id",
        "reviewer_kind",
        "reviewer_id",
        "reviewed_at",
        "human_attested",
        "overall_outcome",
        "required_obligations",
        "unsafe_claims",
        "supports_next_step",
        "fact_state_assessment",
        "ambiguities",
        "notes",
    }
    if (
        set(row) != required
        or row["reviewer_kind"] != "model_advisory"
        or row["human_attested"] is not False
        or not str(row["reviewer_id"]).strip()
        or not str(row["reviewed_at"]).strip()
        or row["overall_outcome"] not in OUTCOMES
        or row["supports_next_step"] not in JUDGMENTS
    ):
        raise ValueError("Fact-state review row identity or outcome is invalid")
    obligations = row["required_obligations"]
    if not isinstance(obligations, list):
        raise ValueError("Fact-state review obligations must be an array")
    for obligation in obligations:
        if (
            not isinstance(obligation, dict)
            or set(obligation) != {"description", "material", "satisfied"}
            or not str(obligation.get("description", "")).strip()
            or not isinstance(obligation.get("material"), bool)
            or obligation.get("satisfied") not in JUDGMENTS
        ):
            raise ValueError("Fact-state review obligation is invalid")
    fact_state = row["fact_state_assessment"]
    expected_fields = {
        "established_facts_preserved",
        "unknowns_handled_correctly",
        "documentation_scope_correct",
        "authority_boundary_correct",
        "no_invented_requirement",
        "no_unsafe_waiver",
        "remedy_logic_preserved",
    }
    if not isinstance(fact_state, dict) or set(fact_state) != expected_fields:
        raise ValueError("Fact-state review assessment is invalid")
    if any(
        fact_state[field] not in SCOPED_JUDGMENTS
        for field in expected_fields
        - {"established_facts_preserved", "no_invented_requirement", "no_unsafe_waiver"}
    ) or any(
        fact_state[field] not in JUDGMENTS
        for field in (
            "established_facts_preserved",
            "no_invented_requirement",
            "no_unsafe_waiver",
        )
    ):
        raise ValueError("Fact-state review judgment is invalid")
    for field in ("unsafe_claims", "ambiguities"):
        if not isinstance(row[field], list) or any(
            not isinstance(value, str) or not value.strip() for value in row[field]
        ):
            raise ValueError(f"Fact-state review {field} is invalid")
    if not isinstance(row["notes"], str):
        raise ValueError("Fact-state review notes must be a string")


def _render_decision(report: Mapping[str, Any]) -> str:
    control = report["arm_results"][CONTROL_ARM]
    candidate = report["arm_results"][CANDIDATE_ARM]
    paired = report["paired_comparison"]
    return "\n".join(
        [
            "# Fact-state comparison — model-advisory decision",
            "",
            "**Synthetic model-advisory evidence only. Training remains blocked.**",
            "",
            f"Decision: **{report['decision']}**",
            "",
            f"Control acceptable: **{control['outcomes']['acceptable']}/16**",
            "",
            f"Candidate acceptable: **{candidate['outcomes']['acceptable']}/16**",
            "",
            f"Fact-state paired wins/losses: "
            f"**{paired['fact_state_wins']}/{paired['fact_state_losses']}**",
            "",
            f"Candidate gate passed: **{str(candidate['passes_substantive_gate']).lower()}**",
            "",
            f"V3 regression authorized: **{str(report['v3_regression_authorized']).lower()}**",
            "",
            "No teacher designation or training authorization is granted.",
            "",
        ]
    )


def _empty_review_row(review_id: str) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "reviewer_kind": "model_advisory",
        "reviewer_id": "",
        "reviewed_at": "",
        "human_attested": False,
        "overall_outcome": None,
        "required_obligations": [],
        "unsafe_claims": [],
        "supports_next_step": None,
        "fact_state_assessment": {
            "established_facts_preserved": None,
            "unknowns_handled_correctly": None,
            "documentation_scope_correct": None,
            "authority_boundary_correct": None,
            "no_invented_requirement": None,
            "no_unsafe_waiver": None,
            "remedy_logic_preserved": None,
        },
        "ambiguities": [],
        "notes": "",
    }


def _unique_review_id(seen: set[str]) -> str:
    while True:
        review_id = f"CTSF-{secrets.token_hex(6).upper()}"
        if review_id not in seen:
            seen.add(review_id)
            return review_id


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()


def _read_jsonl_bytes(content: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in content.decode().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Fact-state packet row must be an object")
            rows.append(value)
    return rows


def _verify_binding(root: Path, binding: Any, label: str) -> None:
    if not isinstance(binding, dict):
        raise ValueError(f"Fact-state manifest has no {label}")
    path = root / str(binding.get("path", ""))
    if not path.is_file() or _file_hash(path) != binding.get("sha256"):
        raise ValueError(f"Fact-state {label} hash binding is invalid")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _knowledge_record_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "statement": record.statement,
        "source_uri": record.source_uri,
        "status": record.status,
        "effective_from": record.effective_from,
        "effective_to": record.effective_to,
    }


def _authorized_record_ids(case: Mapping[str, Any]) -> list[str]:
    return [
        *[str(value) for value in case["source_record_ids"]],
        *[str(state["state_id"]) for state in case["operational_state"]],
    ]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
