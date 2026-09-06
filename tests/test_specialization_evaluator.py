from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from enterprise_memory_mlx.specialization_audit import (
    prepare_specialization_audit,
    validate_specialization_audit_overlay,
)
from enterprise_memory_mlx.specialization_evaluator import (
    grade_legacy_assessment_v2,
    parse_multipart_assessment_v2,
    verify_evaluator_v2_contract,
)
from enterprise_memory_mlx.specialization_real_work import validate_real_work_seed
from enterprise_memory_mlx.task_specialization import (
    grade_structured_assessment,
    load_specialization_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_v2_contract_is_hash_bound_and_keeps_training_blocked() -> None:
    manifest = verify_evaluator_v2_contract(ROOT)

    assert manifest["historical_pilot_scores_replaced"] is False
    assert manifest["teacher_requalification_authorized"] is False
    assert manifest["student_training_authorized"] is False


def test_paraphrased_obligation_requires_review_instead_of_hard_failure() -> None:
    case = next(
        case for case in load_specialization_assets(ROOT).cases if case["case_id"] == "CTS-DEV-006"
    )
    assessment = deepcopy(case["reference"])
    assessment["required_actions"] = [
        "Provide a human first response within the remaining five minutes and "
        "keep an assigned owner responsible continuously until stabilisation."
    ]

    assert (
        grade_structured_assessment(case, assessment, parse_status="valid")["status"] == "hard_fail"
    )
    grade = grade_legacy_assessment_v2(case, assessment, parse_status="valid")
    assert grade.status == "semantic_review_required"
    assert grade.hard_failure_reasons == ()


def test_reversed_action_is_not_machine_accepted_by_v2() -> None:
    case = next(
        case for case in load_specialization_assets(ROOT).cases if case["case_id"] == "CTS-DEV-004"
    )
    assessment = deepcopy(case["reference"])
    assessment["required_actions"] = [
        "Do not obtain written approval from the budget owner before booking."
    ]

    assert grade_structured_assessment(case, assessment, parse_status="valid")["status"] == "pass"
    grade = grade_legacy_assessment_v2(case, assessment, parse_status="valid")
    assert grade.status == "semantic_review_required"
    assert "polarity" in " ".join(grade.semantic_review_reasons)


def test_equivalent_relative_deadline_is_not_a_v2_hard_failure() -> None:
    case = next(
        case for case in load_specialization_assets(ROOT).cases if case["case_id"] == "CTS-DEV-010"
    )
    assessment = deepcopy(case["reference"])
    assessment["required_actions"][0] = (
        "Appoint the incident commander within the next three minutes."
    )

    assert (
        grade_structured_assessment(case, assessment, parse_status="valid")["status"] == "hard_fail"
    )
    assert (
        grade_legacy_assessment_v2(case, assessment, parse_status="valid").status
        == "semantic_review_required"
    )


def test_v2_keeps_unauthorized_evidence_as_a_hard_failure() -> None:
    case = load_specialization_assets(ROOT).cases[0]
    assessment = deepcopy(case["reference"])
    assessment["evidence"] = [{"record_id": "FAKE-001", "claim": "Invented."}]

    grade = grade_legacy_assessment_v2(case, assessment, parse_status="valid")

    assert grade.status == "hard_fail"
    assert any("unauthorized" in reason for reason in grade.hard_failure_reasons)
    assert grade.semantic_review_eligible is False


def test_multipart_schema_preserves_distinct_decisions() -> None:
    value = {
        "decisions": [
            {"request_item_id": "supplier_onboarding", "decision": "proceed"},
            {"request_item_id": "invoice_processing", "decision": "do_not_proceed"},
        ],
        "required_actions": ["Obtain an approved PO or Procurement exception."],
        "missing_information": [],
        "exceptions": [],
        "evidence": [
            {"record_id": "PROC-VEND-001", "claim": "Onboarding controls are complete."},
            {"record_id": "FIN-INV-002", "claim": "The invoice requires a PO or exception."},
        ],
    }

    parsed, status = parse_multipart_assessment_v2(json.dumps(value))

    assert status == "valid"
    assert parsed is not None
    assert [item["decision"] for item in parsed["decisions"]] == [
        "proceed",
        "do_not_proceed",
    ]


def _attempt(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "request": "May this action proceed?",
        "source_record_ids": ["PROC-VEND-001"],
        "assessment": {
            "decision": "proceed",
            "required_actions": [],
            "missing_information": [],
            "exceptions": [],
            "evidence": [{"record_id": "PROC-VEND-001", "claim": "The controls are complete."}],
        },
        "deterministic": {"status": "pass", "reasons": []},
        "gemma_advisory": {"valid": True, "score": 1.0},
        "governed_score": 1.0,
    }


def test_blinded_audit_packet_and_complete_overlay_validation(tmp_path: Path) -> None:
    pilot = {
        "protocol_id": "company-task-specialization/v1",
        "teacher_attempts": [_attempt(f"TEACH-{index:02d}") for index in range(14)],
        "repair_attempts": [_attempt(f"REPAIR-{index:02d}") for index in range(8)],
    }
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")

    artifacts = prepare_specialization_audit(
        root=ROOT,
        pilot_path=pilot_path,
        output_root=tmp_path / "audits",
    )

    with zipfile.ZipFile(artifacts.packet_path) as archive:
        cases_text = archive.read("audit/review_cases.jsonl").decode("utf-8")
    assert '"role"' not in cases_text
    assert "TEACH-00" not in cases_text
    assert "governed_score" not in cases_text
    assert "PROC-VEND-001" in cases_text

    rows = [
        json.loads(line)
        for line in artifacts.template_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row.update(
            {
                "reviewer_id": "Domain Reviewer",
                "reviewed_at": "2026-09-06T17:00:00+00:00",
                "human_attested": True,
                "overall_outcome": "acceptable",
                "failure_classifications": ["no_failure"],
                "required_obligations": [
                    {
                        "description": "Confirm all controls.",
                        "material": True,
                        "satisfied": "yes",
                    }
                ],
                "unsafe_claims": [],
                "supports_next_step": "yes",
                "notes": "",
            }
        )
    overlay = tmp_path / "overlay.jsonl"
    overlay.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit-report.json"

    validate_specialization_audit_overlay(
        packet_path=artifacts.packet_path,
        mapping_path=artifacts.mapping_path,
        overlay_path=overlay,
        output_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["case_count"] == 22
    assert report["overall_outcomes"] == {"acceptable": 22}
    assert report["replaces_historical_scores"] is False


def test_private_real_work_seed_is_hash_manifested_but_does_not_authorize_training(
    tmp_path: Path,
) -> None:
    private = tmp_path / "knowledge" / "private"
    private.mkdir(parents=True)
    seed_path = private / "real-work.jsonl"
    row = {
        "case_id": "REAL-001",
        "workflow": "supplier_onboarding",
        "request": "May this supplier proceed?",
        "evidence": [
            {
                "record_id": "POLICY-001",
                "title": "Supplier policy",
                "text": "A security review is required.",
                "source_uri": "internal://policy/1",
                "classification": "internal_shared",
            }
        ],
        "accepted_resolution": {
            "decision_items": [
                {"request_item_id": "supplier_onboarding", "decision": "do_not_proceed"}
            ],
            "required_obligations": [
                {
                    "obligation_id": "security-review",
                    "description": "Complete the security review.",
                    "material": True,
                }
            ],
            "missing_information": [],
            "applicable_exceptions": [],
            "supporting_record_ids": ["POLICY-001"],
        },
        "reviews": [
            {
                "reviewer_id": "Reviewer One",
                "domain_role": "Procurement",
                "reviewed_at": "2026-09-06T17:00:00+00:00",
                "human_attested": True,
                "approved": True,
            }
        ],
        "redacted": True,
        "evidence_authorized_for_local_research": True,
    }
    seed_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "manifest.json"

    validate_real_work_seed(root=tmp_path, seed_path=seed_path, output_path=output)

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_development_seed"
    assert manifest["contains_case_content"] is False
    assert manifest["authorizes_training"] is False
    with pytest.raises(ValueError, match="knowledge/private"):
        validate_real_work_seed(
            root=tmp_path,
            seed_path=tmp_path / "outside.jsonl",
            output_path=tmp_path / "other.json",
        )
