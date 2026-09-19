"""TRAIN-only typed-action annotation drafts. No training, no DEV/TEST, no inference."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import load_records
from .grpo_data import V3_CASES_RELATIVE, V4_CASES_RELATIVE, V4_POLICY_RELATIVE, file_sha256
from .grpo_landscape import EXPECTED_TRAIN, render_train_prompt, train_rows
from .specialization_fact_state_experiment import STATE_AUTHORITY_GUIDE
from .specialization_remedy_experiment import _source_dict
from .task_specialization import DECISIONS
from .utils import atomic_write_text, read_jsonl, sha256_json, sha256_text

ANNOTATION_ID = "company-task-specialization/qwen4b-typed-action-annotation-v1"
ANNOTATION_DIR = Path("artifacts/qwen4b-typed-action-annotation-v1")
SPLIT_RELATIVE = "knowledge/company_task_specialization/qwen4b_grpo_v1/split-manifest.json"
HISTORICAL_STOPS = {
    "grpo_v1": "stop_preflight",
    "landscape_v1": "abandon_standard_grpo_for_current_dataset",
}
DECISION_MEANINGS = {
    "proceed": (
        "the requested action is permitted now and no stated control remains unmet"
    ),
    "do_not_proceed": (
        "the requested action is prohibited or blocked by a known unmet control"
    ),
    "needs_action": (
        "an operational action or deadline must be executed, rather than approved"
    ),
    "needs_information": (
        "request facts are unknown and needed to decide under available policy"
    ),
    "refer_to_source": (
        "the answer requires an absent, live, or restricted authoritative source"
    ),
}
OPERATIONS = {
    "request_evidence": {
        "meaning": (
            "Establish an unknown condition status. Does not perform the control, "
            "does not record completion, and does not treat unknown as unmet."
        ),
        "arguments": ["condition_id", "source_record_id"],
    },
    "perform_unmet_control": {
        "meaning": (
            "Carry out a known unmet required control. Distinct from recording "
            "completion and from requesting evidence of unknown status."
        ),
        "arguments": ["condition_id", "source_record_id"],
    },
    "correct_nonconformance": {
        "meaning": (
            "Correct a document or other nonconformance so a required condition "
            "can be met. Does not grant an exception."
        ),
        "arguments": ["condition_id", "source_record_id", "artifact_kind"],
    },
    "record_completion": {
        "meaning": (
            "Record that a control is complete after appropriate evidence and "
            "permission already exist. Not a substitute for performing the control."
        ),
        "arguments": ["condition_id", "source_record_id"],
    },
    "request_alternative_authorization": {
        "meaning": (
            "Request that an authorized body record an alternative authorization "
            "or exception. Does not grant or waive the requirement."
        ),
        "arguments": ["condition_id", "source_record_id"],
    },
    "grant_or_waive": {
        "meaning": (
            "Grant an exception or waive a control in this assessment. No TRAIN "
            "source authorizes the assessor to do this; the operation remains in "
            "the grammar as a distinct, usually incorrect, choice."
        ),
        "arguments": ["condition_id", "source_record_id"],
        "used_in_train_targets": False,
    },
    "resolve_record_conflict": {
        "meaning": (
            "Resolve conflicting authorized records about one condition. Does not "
            "choose a status by fiat or treat conflict as known non-completion."
        ),
        "arguments": ["condition_id", "source_record_id"],
    },
    "refer_to_source": {
        "meaning": "Hand an unresolved matter to the authoritative source.",
        "arguments": ["source_record_id"],
    },
}
CONDITION_IDS = (
    "identity_check_complete",
    "bank_check_complete",
    "sanctions_check_complete",
    "conflict_check_complete",
    "bank_approval_not_by_requester",
    "security_questionnaire_complete",
    "data_processing_terms_signed",
    "baseline_checks_complete",
    "approved_purchase_order_quoted",
    "recorded_procurement_exception",
    "all_onboarding_controls_complete",
    "questionnaire_pdf_attached",
    "all_other_applicable_controls_complete",
)
CONDITION_STATUS = ("met", "unmet", "unknown", "conflicting", "not_applicable")
AUTHORITIES = (
    "authorized_workflow_record",
    "workflow_status_observation",
    "document_observation",
    "requester_claim",
    "request_fact",
    "none",
)
ARTIFACT_KINDS = ("invoice", "questionnaire_pdf", "not_applicable")


def _action(
    operation: str,
    condition_id: str,
    source_id: str,
    **arguments: str,
) -> dict[str, Any]:
    payload = {
        "node_type": "action",
        "operation": operation,
        "condition_id": condition_id,
        "source_record_id": source_id,
        "arguments": {"condition_id": condition_id, "source_record_id": source_id},
    }
    payload["arguments"].update(arguments)
    return payload


def _group(operator: str, *options: Mapping[str, Any]) -> dict[str, Any]:
    return {"node_type": "group", "operator": operator, "options": [dict(item) for item in options]}


def _state(
    condition_id: str,
    status: str,
    *,
    authority: str = "request_fact",
    document_availability: str = "not_applicable",
    supersession: str = "none",
    scope: str | None = None,
) -> dict[str, Any]:
    row = {
        "condition_id": condition_id,
        "status": status,
        "observation_authority": authority,
        "document_availability": document_availability,
        "supersession": supersession,
    }
    if scope:
        row["scope"] = scope
    return row


def load_train_split_ids(root: Path) -> tuple[str, ...]:
    split = json.loads((root / SPLIT_RELATIVE).read_text(encoding="utf-8"))
    ids = tuple(row["case_id"] for row in split["rows"] if row["split"] == "train")
    if ids != EXPECTED_TRAIN:
        raise ValueError(f"TRAIN allowlist drifted: {ids}")
    if any(row["split"] != "train" for row in split["rows"] if row["case_id"] in ids):
        raise ValueError("non-TRAIN row leaked into typed-action annotation")
    return ids


def load_train_cases(root: Path) -> list[dict[str, Any]]:
    allowed = set(load_train_split_ids(root))
    families = {row["case_id"]: row for row in train_rows(root)}
    records = {record.id: record for record in load_records(root / "knowledge")}
    policy_by_id = {
        str(row["id"]): row for row in read_jsonl(root / V4_POLICY_RELATIVE)
    }
    selected: list[dict[str, Any]] = []
    for path, protocol in ((V3_CASES_RELATIVE, "v3"), (V4_CASES_RELATIVE, "v4")):
        for case in read_jsonl(root / path):
            case_id = str(case["case_id"])
            if case_id not in allowed:
                continue
            if families[case_id]["split"] != "train":
                raise ValueError(f"refused non-TRAIN case {case_id}")
            sources = []
            if protocol == "v3":
                sources = [_source_dict(records[str(item)]) for item in case["source_record_ids"]]
            else:
                sources = [policy_by_id[str(item)] for item in case["source_record_ids"]]
            selected.append(
                {
                    **case,
                    "source_protocol": protocol,
                    "family_id": families[case_id]["family_id"],
                    "pair_id": families[case_id].get("pair_id"),
                    "sources": sources,
                    "case_record_sha256": sha256_json(case),
                    "prompt_sha256": sha256_text(render_train_prompt(root, case_id)["question"]),
                    "source_statement_sha256": {
                        str(row["id"]): sha256_text(str(row["statement"])) for row in sources
                    },
                }
            )
    selected.sort(key=lambda row: EXPECTED_TRAIN.index(row["case_id"]))
    if [row["case_id"] for row in selected] != list(EXPECTED_TRAIN):
        raise ValueError("TRAIN case inventory incomplete")
    return selected


def source_statements(cases: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    statements: dict[str, str] = {}
    for case in cases:
        for row in case["sources"]:
            statements[str(row["id"])] = str(row["statement"])
    return statements


def locate_excerpt(statement: str, fragment: str) -> dict[str, Any]:
    start = statement.find(fragment)
    if start < 0:
        raise ValueError(f"excerpt not found: {fragment!r}")
    return {
        "text": fragment,
        "start": start,
        "end": start + len(fragment),
        "statement_sha256": sha256_text(statement),
    }


def action_identity(node: Mapping[str, Any]) -> tuple[Any, ...]:
    if node.get("node_type") == "action":
        arguments = node.get("arguments") or {}
        return (
            "action",
            node.get("operation"),
            node.get("condition_id"),
            node.get("source_record_id"),
            arguments.get("artifact_kind"),
        )
    children = tuple(sorted(action_identity(child) for child in node.get("options") or []))
    return ("group", node.get("operator"), children)


def trees_equivalent(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    """Boolean equivalence over operation+condition identities, not mere tree shape."""
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return action_identity(left) == action_identity(right)


def _assessment(
    request_item_id: str,
    decision: str,
    *,
    blockers: Sequence[Mapping[str, str]] = (),
    missing: Sequence[str] = (),
    exceptions: Sequence[Mapping[str, str]] = (),
    remedy: Mapping[str, Any] | None = None,
    supporting_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "request_item_id": request_item_id,
        "decision": decision,
        "current_blockers": [dict(row) for row in blockers],
        "missing_information": [{"condition_id": item} for item in missing],
        "exceptions": [dict(row) for row in exceptions],
        "remedy": None if remedy is None else json.loads(json.dumps(remedy)),
        "supporting_source_ids": list(supporting_source_ids),
    }


def _draft(
    case: Mapping[str, Any],
    assessments: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    unresolved: Sequence[str] = (),
) -> dict[str, Any]:
    request_ids = [str(item["request_item_id"]) for item in case["request_items"]]
    actual_ids = [str(item["request_item_id"]) for item in assessments]
    if actual_ids != request_ids:
        raise ValueError(f"{case['case_id']} request-item coverage drifted")
    return {
        "annotation_id": ANNOTATION_ID,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "source_protocol": case["source_protocol"],
        "pair_id": case.get("pair_id"),
        "authorship": "model_authored",
        "human_attested": False,
        "independent_review": "pending",
        "training_authorized": False,
        "status": "draft_fully_mapped" if not unresolved else "draft_with_unresolved_notes",
        "mechanically_valid_is_not_semantically_validated": True,
        "hashes": {
            "case_record_sha256": case["case_record_sha256"],
            "prompt_sha256": case["prompt_sha256"],
            "source_statement_sha256": case["source_statement_sha256"],
        },
        "proposed_training_target": {"assessments": [dict(item) for item in assessments]},
        "annotation": {
            "condition_states": [dict(item) for item in states],
            "mappings": [dict(item) for item in mappings],
            "unresolved": list(unresolved),
            "prior_reference_role": "annotation_candidate_not_ground_truth",
            "landscape_reward_used": False,
        },
    }


def _cite(statements: Mapping[str, str], source_id: str, fragment: str) -> dict[str, Any]:
    return {"source_record_id": source_id, **locate_excerpt(statements[source_id], fragment)}


def build_drafts(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    statements = source_statements(cases)
    by_id = {row["case_id"]: row for row in cases}
    vend = "PROC-VEND-001"
    inv = "FIN-INV-002"
    att = "PROC-SEC-ATT-003"
    identity = _cite(statements, vend, "identity")
    bank = _cite(statements, vend, "bank-account")
    sanctions = _cite(statements, vend, "sanctions")
    conflict = _cite(statements, vend, "conflict-of-interest")
    questionnaire = _cite(statements, vend, "security questionnaire")
    terms = _cite(statements, vend, "approved data-processing terms")
    requester = _cite(
        statements,
        vend,
        "The requester may not approve the supplier's own bank details.",
    )
    po_quote = _cite(statements, inv, "quote an approved purchase order")
    exception = _cite(statements, inv, "Procurement has recorded an exception")
    pdf = _cite(statements, att, "PDF copy of the completed questionnaire")
    drafts = [
        _draft(
            by_id["CTS3-QUAL-001"],
            [
                _assessment(
                    "purchase_order_issue",
                    "proceed",
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state("identity_check_complete", "met"),
                _state("bank_check_complete", "met"),
                _state("sanctions_check_complete", "met"),
                _state("conflict_check_complete", "met"),
                _state("bank_approval_not_by_requester", "met"),
                _state("security_questionnaire_complete", "not_applicable"),
                _state("data_processing_terms_signed", "not_applicable"),
            ],
            [
                {
                    "explanation": (
                        "Non-data supplier; baseline checks and independent bank "
                        "approval are stated complete. No proposed action."
                    ),
                    "citations": [identity, bank, sanctions, conflict, requester],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-002"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {"condition_id": "identity_check_complete", "source_record_id": vend}
                    ],
                    remedy=_action("perform_unmet_control", "identity_check_complete", vend),
                    supporting_source_ids=[vend],
                )
            ],
            [_state("identity_check_complete", "unmet")],
            [
                {
                    "explanation": (
                        "Known unmet identity check. Mapped to perform_unmet_control, "
                        "not record_completion."
                    ),
                    "citations": [identity],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-003"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {"condition_id": "sanctions_check_complete", "source_record_id": vend},
                        {"condition_id": "conflict_check_complete", "source_record_id": vend},
                    ],
                    remedy=_group(
                        "all_of",
                        _action("perform_unmet_control", "sanctions_check_complete", vend),
                        _action("perform_unmet_control", "conflict_check_complete", vend),
                    ),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state("sanctions_check_complete", "unmet"),
                _state("conflict_check_complete", "unmet"),
            ],
            [
                {
                    "explanation": "Both remaining baseline checks are cumulative all_of.",
                    "citations": [sanctions, conflict],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-004"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "security_questionnaire_complete",
                            "source_record_id": vend,
                        },
                        {"condition_id": "data_processing_terms_signed", "source_record_id": vend},
                    ],
                    remedy=_group(
                        "all_of",
                        _action("perform_unmet_control", "security_questionnaire_complete", vend),
                        _action("perform_unmet_control", "data_processing_terms_signed", vend),
                    ),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state("security_questionnaire_complete", "unmet"),
                _state("data_processing_terms_signed", "unmet"),
            ],
            [
                {
                    "explanation": "Data supplier; both data-handling controls remain unmet.",
                    "citations": [questionnaire, terms],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-005"],
            [
                _assessment(
                    "purchase_order_issue",
                    "needs_information",
                    missing=["security_questionnaire_complete"],
                    remedy=_action("request_evidence", "security_questionnaire_complete", vend),
                    supporting_source_ids=[vend],
                )
            ],
            [_state("security_questionnaire_complete", "unknown")],
            [
                {
                    "explanation": (
                        "Unknown questionnaire status. request_evidence, not "
                        "perform_unmet_control and not record_completion."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-006"],
            [
                _assessment(
                    "purchase_order_issue",
                    "proceed",
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "met",
                    document_availability="absent",
                )
            ],
            [
                {
                    "explanation": (
                        "Workflow records the questionnaire complete. PROC-VEND-001 "
                        "does not require a PDF. Absent attachment is not non-completion."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-007"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "bank_approval_not_by_requester",
                            "source_record_id": vend,
                        }
                    ],
                    remedy=_action("perform_unmet_control", "bank_approval_not_by_requester", vend),
                    supporting_source_ids=[vend],
                )
            ],
            [_state("bank_approval_not_by_requester", "unmet")],
            [
                {
                    "explanation": (
                        "Requester approved bank details. Obtain independent approval; "
                        "do not waive the prohibition."
                    ),
                    "citations": [requester],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-008"],
            [
                _assessment(
                    "purchase_order_issue",
                    "proceed",
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state("baseline_checks_complete", "met"),
                _state("security_questionnaire_complete", "met"),
                _state("data_processing_terms_signed", "met"),
                _state("bank_approval_not_by_requester", "met"),
            ],
            [
                {
                    "explanation": "All stated applicable controls are complete. No action.",
                    "citations": [identity, questionnaire, terms, requester],
                }
            ],
        ),
        _any_of_invoice_blocked(by_id["CTS3-QUAL-009"], statements, verbal_claim=False),
        _draft(
            by_id["CTS3-QUAL-010"],
            [
                _assessment(
                    "invoice_processing",
                    "proceed",
                    supporting_source_ids=[inv],
                )
            ],
            [
                _state("approved_purchase_order_quoted", "met"),
                _state("recorded_procurement_exception", "unmet"),
            ],
            [
                {
                    "explanation": "Approved PO route satisfies any_of. No action.",
                    "citations": [po_quote],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-011"],
            [
                _assessment(
                    "invoice_processing",
                    "proceed",
                    exceptions=[
                        {
                            "condition_id": "recorded_procurement_exception",
                            "recorded_exception_id": "EX-31",
                        }
                    ],
                    supporting_source_ids=[inv],
                )
            ],
            [
                _state("approved_purchase_order_quoted", "unmet"),
                _state("recorded_procurement_exception", "met"),
            ],
            [
                {
                    "explanation": (
                        "Recorded exception satisfies any_of. Not a grant by the assessor."
                    ),
                    "citations": [exception],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-012"],
            [
                _assessment(
                    "invoice_processing",
                    "proceed",
                    exceptions=[
                        {
                            "condition_id": "recorded_procurement_exception",
                            "recorded_exception_id": "EX-44",
                        }
                    ],
                    supporting_source_ids=[inv],
                )
            ],
            [
                _state("approved_purchase_order_quoted", "met"),
                _state("recorded_procurement_exception", "met"),
            ],
            [
                {
                    "explanation": "Both any_of routes are already satisfied.",
                    "citations": [po_quote, exception],
                }
            ],
        ),
        _any_of_invoice_blocked(by_id["CTS3-QUAL-013"], statements, verbal_claim=True),
        _draft(
            by_id["CTS3-QUAL-016"],
            [
                _assessment(
                    "invoice_processing",
                    "needs_information",
                    missing=[
                        "approved_purchase_order_quoted",
                        "recorded_procurement_exception",
                    ],
                    remedy=_group(
                        "any_of",
                        _action("request_evidence", "approved_purchase_order_quoted", inv),
                        _action("request_evidence", "recorded_procurement_exception", inv),
                    ),
                    supporting_source_ids=[inv],
                )
            ],
            [
                _state("approved_purchase_order_quoted", "unknown"),
                _state("recorded_procurement_exception", "unknown"),
            ],
            [
                {
                    "explanation": (
                        "Both routes unknown. request_evidence on either route; "
                        "do not treat unknown as unmet and do not grant an exception."
                    ),
                    "citations": [po_quote, exception],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-017"],
            [
                _assessment(
                    "supplier_onboarding",
                    "proceed",
                    supporting_source_ids=[vend],
                ),
                _assessment(
                    "invoice_processing",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "approved_purchase_order_quoted",
                            "source_record_id": inv,
                        },
                        {
                            "condition_id": "recorded_procurement_exception",
                            "source_record_id": inv,
                        },
                    ],
                    remedy=_invoice_any_of(inv),
                    supporting_source_ids=[inv],
                ),
            ],
            [
                _state("all_onboarding_controls_complete", "met"),
                _state("approved_purchase_order_quoted", "unmet"),
                _state("recorded_procurement_exception", "unmet"),
            ],
            [
                {
                    "explanation": "Onboarding complete. Invoice still needs one any_of route.",
                    "citations": [po_quote, exception],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-018"],
            [
                _assessment(
                    "supplier_onboarding",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "security_questionnaire_complete",
                            "source_record_id": vend,
                        },
                        {"condition_id": "data_processing_terms_signed", "source_record_id": vend},
                    ],
                    remedy=_group(
                        "all_of",
                        _action("perform_unmet_control", "security_questionnaire_complete", vend),
                        _action("perform_unmet_control", "data_processing_terms_signed", vend),
                    ),
                    supporting_source_ids=[vend],
                ),
                _assessment(
                    "invoice_processing",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "approved_purchase_order_quoted",
                            "source_record_id": inv,
                        },
                        {
                            "condition_id": "recorded_procurement_exception",
                            "source_record_id": inv,
                        },
                    ],
                    remedy=_invoice_any_of(inv),
                    supporting_source_ids=[inv],
                ),
            ],
            [
                _state("security_questionnaire_complete", "unmet"),
                _state("data_processing_terms_signed", "unmet"),
                _state("approved_purchase_order_quoted", "unmet"),
                _state("recorded_procurement_exception", "unmet"),
            ],
            [
                {
                    "explanation": (
                        "Cumulative onboarding controls remain, and the invoice still "
                        "needs one alternative route."
                    ),
                    "citations": [questionnaire, terms, po_quote, exception],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-019"],
            [
                _assessment(
                    "supplier_onboarding",
                    "do_not_proceed",
                    blockers=[
                        {"condition_id": "sanctions_check_complete", "source_record_id": vend},
                        {"condition_id": "conflict_check_complete", "source_record_id": vend},
                    ],
                    remedy=_group(
                        "all_of",
                        _action("perform_unmet_control", "sanctions_check_complete", vend),
                        _action("perform_unmet_control", "conflict_check_complete", vend),
                    ),
                    supporting_source_ids=[vend],
                ),
                _assessment(
                    "invoice_processing",
                    "proceed",
                    exceptions=[
                        {
                            "condition_id": "recorded_procurement_exception",
                            "recorded_exception_id": "EX-52",
                        }
                    ],
                    supporting_source_ids=[inv],
                ),
            ],
            [
                _state("sanctions_check_complete", "unmet"),
                _state("conflict_check_complete", "unmet"),
                _state("approved_purchase_order_quoted", "unmet"),
                _state("recorded_procurement_exception", "met"),
            ],
            [
                {
                    "explanation": (
                        "Onboarding still blocked. Invoice proceeds on the already "
                        "recorded exception; the assessor does not grant it."
                    ),
                    "citations": [sanctions, conflict, exception],
                }
            ],
        ),
        _draft(
            by_id["CTS3-QUAL-024"],
            [
                _assessment(
                    "activate_supplier_and_process_invoice",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "security_questionnaire_complete",
                            "source_record_id": vend,
                        },
                        {"condition_id": "data_processing_terms_signed", "source_record_id": vend},
                    ],
                    remedy=_group(
                        "all_of",
                        _action("perform_unmet_control", "security_questionnaire_complete", vend),
                        _action("perform_unmet_control", "data_processing_terms_signed", vend),
                    ),
                    supporting_source_ids=[vend, inv],
                )
            ],
            [
                _state("security_questionnaire_complete", "unmet"),
                _state("data_processing_terms_signed", "unmet"),
                _state("approved_purchase_order_quoted", "met"),
                _state("recorded_procurement_exception", "unmet"),
            ],
            [
                {
                    "explanation": (
                        "Nested any_of invoice route is already satisfied by PO-551. "
                        "Only the unmet data-handling controls are proposed. The "
                        "satisfied alternative is not copied as a required action."
                    ),
                    "citations": [questionnaire, terms, po_quote],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-02A"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "security_questionnaire_complete",
                            "source_record_id": vend,
                        }
                    ],
                    remedy=_action(
                        "perform_unmet_control",
                        "security_questionnaire_complete",
                        vend,
                    ),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "unmet",
                    authority="authorized_workflow_record",
                    document_availability="present",
                ),
                _state(
                    "all_other_applicable_controls_complete",
                    "met",
                    authority="authorized_workflow_record",
                ),
            ],
            [
                {
                    "explanation": (
                        "Authorized record says incomplete. PDF presence does not "
                        "establish completion under PROC-VEND-001."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-02B"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {"condition_id": "questionnaire_pdf_attached", "source_record_id": att}
                    ],
                    remedy=_action(
                        "correct_nonconformance",
                        "questionnaire_pdf_attached",
                        att,
                        artifact_kind="questionnaire_pdf",
                    ),
                    supporting_source_ids=[att],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "met",
                    authority="authorized_workflow_record",
                    document_availability="absent",
                ),
                _state(
                    "questionnaire_pdf_attached",
                    "unmet",
                    authority="document_observation",
                    document_availability="absent",
                ),
            ],
            [
                {
                    "explanation": (
                        "Attachment-controlled policy requires the PDF. Correct the "
                        "missing attachment; do not redo the already complete questionnaire."
                    ),
                    "citations": [pdf],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-03A"],
            [
                _assessment(
                    "purchase_order_issue",
                    "do_not_proceed",
                    blockers=[
                        {
                            "condition_id": "security_questionnaire_complete",
                            "source_record_id": vend,
                        }
                    ],
                    remedy=_action(
                        "perform_unmet_control",
                        "security_questionnaire_complete",
                        vend,
                    ),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "unmet",
                    authority="authorized_workflow_record",
                ),
                _state(
                    "all_other_applicable_controls_complete",
                    "met",
                    authority="authorized_workflow_record",
                ),
            ],
            [
                {
                    "explanation": (
                        "Only the one incomplete control is proposed. Others stay closed."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-03B"],
            [
                _assessment(
                    "purchase_order_issue",
                    "needs_information",
                    missing=["security_questionnaire_complete"],
                    remedy=_action("request_evidence", "security_questionnaire_complete", vend),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "unknown",
                    authority="workflow_status_observation",
                ),
                _state(
                    "all_other_applicable_controls_complete",
                    "met",
                    authority="authorized_workflow_record",
                ),
            ],
            [
                {
                    "explanation": (
                        "No recorded questionnaire status. request_evidence only; "
                        "unknown is not unmet."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-04A"],
            [
                _assessment(
                    "purchase_order_issue",
                    "needs_information",
                    missing=["security_questionnaire_complete"],
                    remedy=_action(
                        "resolve_record_conflict",
                        "security_questionnaire_complete",
                        vend,
                    ),
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "conflicting",
                    authority="authorized_workflow_record",
                )
            ],
            [
                {
                    "explanation": (
                        "Two unsuperseded authorized records conflict. Surface the "
                        "conflict; do not pick a status or treat it as unmet."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
        _draft(
            by_id["CTS4-FRESH-04B"],
            [
                _assessment(
                    "purchase_order_issue",
                    "proceed",
                    supporting_source_ids=[vend],
                )
            ],
            [
                _state(
                    "security_questionnaire_complete",
                    "met",
                    authority="authorized_workflow_record",
                    supersession="explicitly_superseded",
                    scope="WF-04B-SEC-NEW supersedes WF-04B-SEC-OLD",
                )
            ],
            [
                {
                    "explanation": (
                        "Explicit supersession establishes completion. The older "
                        "incomplete record is not a live conflict."
                    ),
                    "citations": [questionnaire],
                }
            ],
        ),
    ]
    order = {case_id: index for index, case_id in enumerate(EXPECTED_TRAIN)}
    drafts.sort(key=lambda row: order[row["case_id"]])
    if [row["case_id"] for row in drafts] != list(EXPECTED_TRAIN):
        raise ValueError("draft inventory is not the 24 TRAIN cases")
    return drafts


def _invoice_any_of(source_id: str) -> dict[str, Any]:
    return _group(
        "any_of",
        _action(
            "correct_nonconformance",
            "approved_purchase_order_quoted",
            source_id,
            artifact_kind="invoice",
        ),
        _action(
            "request_alternative_authorization",
            "recorded_procurement_exception",
            source_id,
        ),
    )


def _any_of_invoice_blocked(
    case: Mapping[str, Any],
    statements: Mapping[str, str],
    *,
    verbal_claim: bool,
) -> dict[str, Any]:
    inv = "FIN-INV-002"
    states = [
        _state("approved_purchase_order_quoted", "unmet"),
        _state(
            "recorded_procurement_exception",
            "unmet",
            authority="requester_claim" if verbal_claim else "request_fact",
        ),
    ]
    explanation = (
        "Neither any_of route is recorded. Correct the invoice or request that "
        "Procurement record an exception. Requesting is not granting."
    )
    if verbal_claim:
        explanation = (
            "Verbal requester claim is not a recorded exception. Keep both "
            "alternatives; do not convert the claim into established authorization."
        )
    return _draft(
        case,
        [
            _assessment(
                "invoice_processing",
                "do_not_proceed",
                blockers=[
                    {"condition_id": "approved_purchase_order_quoted", "source_record_id": inv},
                    {"condition_id": "recorded_procurement_exception", "source_record_id": inv},
                ],
                remedy=_invoice_any_of(inv),
                supporting_source_ids=[inv],
            )
        ],
        states,
        [
            {
                "explanation": explanation,
                "citations": [
                    _cite(statements, inv, "quote an approved purchase order"),
                    _cite(statements, inv, "Procurement has recorded an exception"),
                ],
            }
        ],
    )


def collect_actions(node: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    if node.get("node_type") == "action":
        return [dict(node)]
    rows: list[dict[str, Any]] = []
    for option in node.get("options") or []:
        rows.extend(collect_actions(option))
    return rows


def validate_drafts(
    drafts: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Structural/logic validation. Does not certify source-to-action semantics."""
    cases_by_id = {row["case_id"]: row for row in cases}
    statements = source_statements(cases)
    failures: list[str] = []
    if [row["case_id"] for row in drafts] != list(EXPECTED_TRAIN):
        failures.append("drafts are not exactly the 24 TRAIN cases")
    for draft in drafts:
        case = cases_by_id[draft["case_id"]]
        expected = [str(item["request_item_id"]) for item in case["request_items"]]
        actual = [
            str(item["request_item_id"])
            for item in draft["proposed_training_target"]["assessments"]
        ]
        if actual != expected or len(actual) != len(set(actual)):
            failures.append(f"{draft['case_id']}: request-item coverage")
        authorized = set(case["source_record_ids"])
        if case.get("operational_state"):
            authorized.update(str(row["state_id"]) for row in case["operational_state"])
        seen_identities: list[tuple[Any, ...]] = []
        for assessment in draft["proposed_training_target"]["assessments"]:
            if assessment["decision"] not in DECISIONS:
                failures.append(f"{draft['case_id']}: unknown decision")
            for action in collect_actions(assessment.get("remedy")):
                identity = action_identity(action)
                if identity in seen_identities and action["operation"] not in {
                    "request_evidence",
                    "perform_unmet_control",
                    "correct_nonconformance",
                    "request_alternative_authorization",
                    "resolve_record_conflict",
                }:
                    failures.append(f"{draft['case_id']}: ambiguous unused identity")
                seen_identities.append(identity)
                if action["operation"] not in OPERATIONS:
                    failures.append(f"{draft['case_id']}: unsupported operation")
                if action["operation"] == "grant_or_waive":
                    failures.append(f"{draft['case_id']}: grant_or_waive invented in target")
                if action.get("condition_id") not in CONDITION_IDS:
                    failures.append(f"{draft['case_id']}: unknown condition")
                if action["source_record_id"] not in set(case["source_record_ids"]):
                    failures.append(f"{draft['case_id']}: source ID not in governing sources")
        for mapping in draft["annotation"]["mappings"]:
            for citation in mapping["citations"]:
                source_id = citation["source_record_id"]
                if source_id not in statements:
                    failures.append(f"{draft['case_id']}: citation source missing")
                    continue
                located = locate_excerpt(statements[source_id], citation["text"])
                if located["start"] != citation["start"] or located["end"] != citation["end"]:
                    failures.append(f"{draft['case_id']}: excerpt location mismatch")
        if draft["authorship"] != "model_authored" or draft["human_attested"] is not False:
            failures.append(f"{draft['case_id']}: authorship flags")
        if draft["annotation"]["landscape_reward_used"] is not False:
            failures.append(f"{draft['case_id']}: landscape reward used")
    substitution = _action_substitution_counterexample()
    if substitution["equivalent"]:
        failures.append("action substitution remained equivalent")
    return {
        "structural_logic_validation": "established" if not failures else "failed",
        "source_to_action_annotation": "requires_substantive_review",
        "circular_self_certification": False,
        "boolean_equivalence_scope": (
            "same operator and the same multiset of "
            "(operation, condition_id, source_record_id, artifact_kind); "
            "identical tree shape alone is not policy equivalence"
        ),
        "failures": failures,
        "action_substitution_counterexample": substitution,
        "request_evidence_versus_record_completion": _distinct_operations(
            "request_evidence", "record_completion"
        ),
        "request_exception_versus_grant": _distinct_operations(
            "request_alternative_authorization", "grant_or_waive"
        ),
        "equivalent_ordering_harmless": _order_invariance_holds(),
        "duplicates_do_not_complete": _duplicates_do_not_complete(),
    }


def _action_substitution_counterexample() -> dict[str, Any]:
    original = _action("perform_unmet_control", "identity_check_complete", "PROC-VEND-001")
    skipped = _action("record_completion", "identity_check_complete", "PROC-VEND-001")
    return {
        "same_source": True,
        "equivalent": trees_equivalent(original, skipped),
        "original_operation": original["operation"],
        "substituted_operation": skipped["operation"],
    }


def _distinct_operations(left: str, right: str) -> dict[str, Any]:
    first = _action(left, "security_questionnaire_complete", "PROC-VEND-001")
    second = _action(right, "security_questionnaire_complete", "PROC-VEND-001")
    return {
        "operations": [left, right],
        "same_source_and_condition": True,
        "equivalent": trees_equivalent(first, second),
    }


def _order_invariance_holds() -> bool:
    left = _group(
        "all_of",
        _action("perform_unmet_control", "sanctions_check_complete", "PROC-VEND-001"),
        _action("perform_unmet_control", "conflict_check_complete", "PROC-VEND-001"),
    )
    right = _group(
        "all_of",
        _action("perform_unmet_control", "conflict_check_complete", "PROC-VEND-001"),
        _action("perform_unmet_control", "sanctions_check_complete", "PROC-VEND-001"),
    )
    return trees_equivalent(left, right)


def _duplicates_do_not_complete() -> bool:
    required = _group(
        "all_of",
        _action("perform_unmet_control", "sanctions_check_complete", "PROC-VEND-001"),
        _action("perform_unmet_control", "conflict_check_complete", "PROC-VEND-001"),
    )
    duplicate = _group(
        "all_of",
        _action("perform_unmet_control", "sanctions_check_complete", "PROC-VEND-001"),
        _action("perform_unmet_control", "sanctions_check_complete", "PROC-VEND-001"),
    )
    return not trees_equivalent(required, duplicate)


def output_schema() -> dict[str, Any]:
    action_node = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "node_type",
            "operation",
            "condition_id",
            "source_record_id",
            "arguments",
        ],
        "properties": {
            "node_type": {"const": "action"},
            "operation": {"type": "string", "enum": sorted(OPERATIONS)},
            "condition_id": {"type": "string", "enum": list(CONDITION_IDS)},
            "source_record_id": {
                "type": "string",
                "description": "Must be an identifier visible in the future input.",
            },
            "arguments": {"type": "object"},
        },
    }
    group_node: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["node_type", "operator", "options"],
        "properties": {
            "node_type": {"const": "group"},
            "operator": {"type": "string", "enum": ["any_of", "all_of"]},
            "options": {
                "type": "array",
                "minItems": 1,
                "items": {"oneOf": [action_node, {"$ref": "#/$defs/group_node"}]},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{ANNOTATION_ID}/answer-blind-output",
        "title": "Typed-action operational assessment (answer-blind)",
        "description": (
            "Admits any global operation, condition, and decision. It must not "
            "encode the correct decision, condition status, or only the safe remedy."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": ["assessments"],
        "properties": {
            "assessments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "request_item_id",
                        "decision",
                        "current_blockers",
                        "missing_information",
                        "exceptions",
                        "remedy",
                        "supporting_source_ids",
                    ],
                    "properties": {
                        "request_item_id": {"type": "string"},
                        "decision": {"type": "string", "enum": sorted(DECISIONS)},
                        "current_blockers": {"type": "array", "items": {"type": "object"}},
                        "missing_information": {"type": "array", "items": {"type": "object"}},
                        "exceptions": {"type": "array", "items": {"type": "object"}},
                        "remedy": {"oneOf": [{"type": "null"}, action_node, group_node]},
                        "supporting_source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            }
        },
        "$defs": {"group_node": group_node},
        "must_not_encode": [
            "the correct current decision",
            "the correct condition status",
            "only the correct remedy",
            "only the safe action for a particular case",
        ],
    }


def wrong_but_admissible_examples() -> list[dict[str, Any]]:
    return [
        {
            "assessments": [
                _assessment(
                    "purchase_order_issue",
                    "proceed",
                    remedy=_action(
                        "record_completion",
                        "identity_check_complete",
                        "PROC-VEND-001",
                    ),
                    supporting_source_ids=["PROC-VEND-001"],
                )
            ]
        },
        {
            "assessments": [
                _assessment(
                    "invoice_processing",
                    "proceed",
                    remedy=_action(
                        "grant_or_waive",
                        "recorded_procurement_exception",
                        "FIN-INV-002",
                    ),
                    supporting_source_ids=["FIN-INV-002"],
                )
            ]
        },
    ]


def validate_against_schema(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(instance) != {"assessments"}:
        errors.append("top-level fields")
    assessments = instance.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        return ["assessments"]
    ops = set(OPERATIONS)
    conditions = set(CONDITION_IDS)
    for item in assessments:
        if item.get("decision") not in DECISIONS:
            errors.append("decision")
        for action in collect_actions(item.get("remedy")):
            if action.get("operation") not in ops:
                errors.append("operation")
            if action.get("condition_id") not in conditions:
                errors.append("condition_id")
    del schema
    return errors


REVIEW_INSTRUCTIONS = """\
# Typed-action TRAIN annotation review

Reviewer kind: model_advisory.

This packet is not human-attested and does not authorize training.
Do not treat the prior free-text reference as unquestionable ground truth.
The draft typed target is the candidate being assessed.

Cursor authored these drafts. Authorship is model_authored.

Check each case:
1. Does each typed operation express the source-supported action?
2. Are the condition and scope correct?
3. Are alternatives and mandatory steps preserved?
4. Are authority and uncertainty handled faithfully?
5. Is any required operation missing from the vocabulary?

Do not fabricate independent review, human approval, or completed labels.
"""


def write_annotation_bundle(root: Path) -> dict[str, Any]:
    destination = root / ANNOTATION_DIR
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "annotation_id": ANNOTATION_ID,
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "current working-tree bytes; not historical executed bytes",
        "historical_stops": dict(HISTORICAL_STOPS),
        "hashes": {
            SPLIT_RELATIVE: file_sha256(root / SPLIT_RELATIVE),
            V3_CASES_RELATIVE: file_sha256(root / V3_CASES_RELATIVE),
            V4_CASES_RELATIVE: file_sha256(root / V4_CASES_RELATIVE),
            V4_POLICY_RELATIVE: file_sha256(root / V4_POLICY_RELATIVE),
            "src/enterprise_memory_mlx/grpo_reward.py": file_sha256(
                root / "src/enterprise_memory_mlx/grpo_reward.py"
            ),
        },
        "grpo_reward_edited": False,
    }
    atomic_write_text(
        destination / "source-snapshot.json", json.dumps(snapshot, indent=2) + "\n"
    )
    cases = load_train_cases(root)
    drafts = build_drafts(cases)
    validation = validate_drafts(drafts, cases)
    if validation["failures"]:
        raise ValueError(f"draft validation failed: {validation['failures']}")
    schema = output_schema()
    for example in wrong_but_admissible_examples():
        if validate_against_schema(example, schema):
            raise ValueError("answer-blind schema rejected a substantively wrong example")
    catalogue = {
        "annotation_id": ANNOTATION_ID,
        "operations": OPERATIONS,
        "condition_ids": list(CONDITION_IDS),
        "decision_meanings": DECISION_MEANINGS,
        "condition_status": list(CONDITION_STATUS),
        "authorities": list(AUTHORITIES),
        "artifact_kinds": list(ARTIFACT_KINDS),
        "state_authority_guide": STATE_AUTHORITY_GUIDE,
        "outputs_are_proposed_actions_only": True,
    }
    atomic_write_text(
        destination / "action-catalogue.json", json.dumps(catalogue, indent=2) + "\n"
    )
    _write_jsonl(
        destination / "typed-targets.draft.jsonl",
        [
            {
                "case_id": row["case_id"],
                "source_protocol": row["source_protocol"],
                "family_id": row["family_id"],
                "authorship": row["authorship"],
                "human_attested": row["human_attested"],
                "independent_review": row["independent_review"],
                "training_authorized": row["training_authorized"],
                "status": row["status"],
                "hashes": row["hashes"],
                "proposed_training_target": row["proposed_training_target"],
            }
            for row in drafts
        ],
    )
    _write_jsonl(
        destination / "source-to-action-mappings.jsonl",
        [
            {
                "case_id": row["case_id"],
                "source_protocol": row["source_protocol"],
                "condition_states": row["annotation"]["condition_states"],
                "mappings": row["annotation"]["mappings"],
                "unresolved": row["annotation"]["unresolved"],
                "prior_reference_role": row["annotation"]["prior_reference_role"],
            }
            for row in drafts
        ],
    )
    unresolved = {
        "review_complete_cases": 0,
        "fully_mapped_draft_cases": sum(
            1 for row in drafts if row["status"] == "draft_fully_mapped"
        ),
        "cases_with_unresolved_notes": [
            row["case_id"] for row in drafts if row["annotation"]["unresolved"]
        ],
        "global_notes": [
            "grant_or_waive is catalogue-only; no TRAIN source authorizes the assessor to grant.",
            "record_completion is catalogue-only in these drafts; proceed cases emit no action.",
            "Source-to-action mappings require substantive review and are not self-certified.",
        ],
    }
    atomic_write_text(
        destination / "unresolved-issues.json", json.dumps(unresolved, indent=2) + "\n"
    )
    atomic_write_text(
        destination / "output-schema.json", json.dumps(schema, indent=2) + "\n"
    )
    counts = {
        "source_cases_inspected": len(cases),
        "fully_mapped_draft_cases": unresolved["fully_mapped_draft_cases"],
        "unresolved_cases": len(unresolved["cases_with_unresolved_notes"]),
        "review_complete_cases": 0,
    }
    validation_record = {
        **validation,
        "counts": counts,
        "schema_admits_wrong_examples": True,
        "historical_prompts_rendered_or_changed": False,
        "generations_run": False,
    }
    atomic_write_text(
        destination / "validation-results.json",
        json.dumps(validation_record, indent=2) + "\n",
    )
    packet_dir = destination / "review-packet"
    if packet_dir.exists():
        for path in packet_dir.rglob("*"):
            if path.is_file():
                path.unlink()
    packet_dir.mkdir(parents=True, exist_ok=True)
    _write_review_packet(packet_dir, cases, drafts, catalogue, schema, validation_record)
    zip_path = destination / "qwen4b-typed-action-annotation-v1-review.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(packet_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(packet_dir).as_posix())
    digest = file_sha256(zip_path)
    atomic_write_text(
        destination / "qwen4b-typed-action-annotation-v1-review.sha256",
        digest + "\n",
    )
    summary = {
        "annotation_id": ANNOTATION_ID,
        "historical_stops": dict(HISTORICAL_STOPS),
        "counts": counts,
        "review_zip": str(zip_path.relative_to(root)),
        "review_zip_sha256": digest,
        "training_authorized": False,
        "human_attested": False,
    }
    atomic_write_text(destination / "SUMMARY.json", json.dumps(summary, indent=2) + "\n")
    return summary


def _write_review_packet(
    packet_dir: Path,
    cases: Sequence[Mapping[str, Any]],
    drafts: Sequence[Mapping[str, Any]],
    catalogue: Mapping[str, Any],
    schema: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    atomic_write_text(packet_dir / "REVIEW_INSTRUCTIONS.md", REVIEW_INSTRUCTIONS)
    atomic_write_text(
        packet_dir / "action-catalogue.json", json.dumps(catalogue, indent=2) + "\n"
    )
    atomic_write_text(packet_dir / "output-schema.json", json.dumps(schema, indent=2) + "\n")
    requests = []
    sources: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        visible = {
            "case_id": case["case_id"],
            "source_protocol": case["source_protocol"],
            "request_items": case["request_items"],
            "source_record_ids": case["source_record_ids"],
        }
        if case["source_protocol"] == "v3":
            visible["operational_facts"] = case["operational_facts"]
        else:
            visible["operational_state"] = case["operational_state"]
            visible["state_authority_guide"] = STATE_AUTHORITY_GUIDE
        requests.append(visible)
        for row in case["sources"]:
            sources[str(row["id"])] = {
                "id": row["id"],
                "title": row.get("title"),
                "statement": row["statement"],
                "source_uri": row.get("source_uri"),
                "status": row.get("status"),
            }
    _write_jsonl(packet_dir / "train-requests.jsonl", requests)
    _write_jsonl(packet_dir / "governing-sources.jsonl", list(sources.values()))
    _write_jsonl(
        packet_dir / "typed-targets.draft.jsonl",
        [
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "authorship": row["authorship"],
                "human_attested": False,
                "proposed_training_target": row["proposed_training_target"],
                "prior_reference_role": "annotation_candidate_not_ground_truth",
            }
            for row in drafts
        ],
    )
    _write_jsonl(
        packet_dir / "source-to-action-mappings.jsonl",
        [
            {
                "case_id": row["case_id"],
                "condition_states": row["annotation"]["condition_states"],
                "mappings": row["annotation"]["mappings"],
                "unresolved": row["annotation"]["unresolved"],
            }
            for row in drafts
        ],
    )
    atomic_write_text(
        packet_dir / "unresolved-issues.json",
        json.dumps(
            {
                "review_complete_cases": 0,
                "cases_with_unresolved_notes": [
                    row["case_id"] for row in drafts if row["annotation"]["unresolved"]
                ],
                "global_notes": [
                    "grant_or_waive is unused in drafts.",
                    "Mappings require substantive review.",
                ],
            },
            indent=2,
        )
        + "\n",
    )
    template = [
        {
            "case_id": row["case_id"],
            "reviewer_kind": "model_advisory",
            "reviewer_id": "",
            "reviewed_at": "",
            "human_attested": False,
            "operation_expresses_source_action": None,
            "condition_and_scope_correct": None,
            "alternatives_and_mandatory_steps_preserved": None,
            "authority_and_uncertainty_faithful": None,
            "required_operation_missing_from_vocabulary": None,
            "overall_outcome": None,
            "notes": "",
        }
        for row in drafts
    ]
    _write_jsonl(packet_dir / "review-template.jsonl", template)
    members = {
        "REVIEW_INSTRUCTIONS.md": (packet_dir / "REVIEW_INSTRUCTIONS.md").read_bytes(),
        "action-catalogue.json": (packet_dir / "action-catalogue.json").read_bytes(),
        "output-schema.json": (packet_dir / "output-schema.json").read_bytes(),
        "train-requests.jsonl": (packet_dir / "train-requests.jsonl").read_bytes(),
        "governing-sources.jsonl": (packet_dir / "governing-sources.jsonl").read_bytes(),
        "typed-targets.draft.jsonl": (packet_dir / "typed-targets.draft.jsonl").read_bytes(),
        "source-to-action-mappings.jsonl": (
            packet_dir / "source-to-action-mappings.jsonl"
        ).read_bytes(),
        "unresolved-issues.json": (packet_dir / "unresolved-issues.json").read_bytes(),
        "review-template.jsonl": (packet_dir / "review-template.jsonl").read_bytes(),
        "validation-results.json": json.dumps(validation, indent=2).encode() + b"\n",
    }
    for name, payload in members.items():
        text = payload.decode("utf-8")
        if "CTS3-QUAL-014" in text or "CTS4-FRESH-01A" in text or "CTS4-FRESH-08A" in text:
            raise ValueError(f"{name} leaked a non-TRAIN case id")
        if any(token in name.lower() for token in ("score", "rollout", "adapter")):
            raise ValueError(f"forbidden packet member {name}")
        del text
    manifest = {
        "annotation_id": ANNOTATION_ID,
        "status": "awaiting_model_advisory_review",
        "authorship": "model_authored",
        "human_attested": False,
        "training_authorized": False,
        "review_complete_cases": 0,
        "case_count": len(drafts),
        "historical_stops": dict(HISTORICAL_STOPS),
        "file_sha256": {
            name: sha256_text(payload.decode("utf-8")) for name, payload in members.items()
        },
    }
    atomic_write_text(
        packet_dir / "packet_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    atomic_write_text(
        packet_dir / "validation-results.json", json.dumps(validation, indent=2) + "\n"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
