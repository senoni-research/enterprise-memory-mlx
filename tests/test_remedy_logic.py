from __future__ import annotations

import json
from copy import deepcopy

from enterprise_memory_mlx.remedy_logic import (
    compare_policy_logic,
    evaluate_policy_logic,
    parse_remedy_assessment,
    remedy_machine_grade,
    render_remedy,
    validate_remedy_node,
)


def _condition(condition_id: str) -> dict:
    return {"node_type": "condition", "condition_id": condition_id}


def _group(operator: str, *options: dict) -> dict:
    return {"node_type": "group", "operator": operator, "options": list(options)}


def _action(text: str, record_id: str = "FIN-INV-002") -> dict:
    return {
        "node_type": "action",
        "action": text,
        "source_record_id": record_id,
    }


def test_any_of_invoice_routes_apply_without_becoming_cumulative() -> None:
    invoice = _group(
        "any_of",
        _condition("approved_purchase_order"),
        _condition("recorded_procurement_exception"),
    )

    assert evaluate_policy_logic(invoice, set()) is False
    assert evaluate_policy_logic(invoice, {"approved_purchase_order"}) is True
    assert evaluate_policy_logic(invoice, {"recorded_procurement_exception"}) is True
    assert (
        evaluate_policy_logic(
            invoice,
            {"approved_purchase_order", "recorded_procurement_exception"},
        )
        is True
    )
    assert evaluate_policy_logic(invoice, {"verbal_procurement_agreement"}) is False


def test_nested_all_of_does_not_let_invoice_route_waive_security() -> None:
    nested = _group(
        "all_of",
        _condition("security_questionnaire_complete"),
        _group(
            "any_of",
            _condition("approved_purchase_order"),
            _condition("recorded_procurement_exception"),
        ),
    )

    assert evaluate_policy_logic(nested, {"recorded_procurement_exception"}) is False
    assert (
        evaluate_policy_logic(
            nested,
            {"approved_purchase_order", "recorded_procurement_exception"},
        )
        is False
    )
    assert (
        evaluate_policy_logic(
            nested,
            {"security_questionnaire_complete", "recorded_procurement_exception"},
        )
        is True
    )


def test_logic_comparison_detects_operator_route_and_group_mutations() -> None:
    reference = _group(
        "all_of",
        _condition("security_questionnaire_complete"),
        _group(
            "any_of",
            _condition("approved_purchase_order"),
            _condition("recorded_procurement_exception"),
        ),
    )

    changed_or_to_and = deepcopy(reference)
    changed_or_to_and["options"][1]["operator"] = "all_of"
    result = compare_policy_logic(reference, changed_or_to_and)
    assert result["status"] == "mismatch"
    assert result["operator_mismatches"][0]["expected"] == "any_of"
    assert result["operator_mismatches"][0]["actual"] == "all_of"

    dropped_route = deepcopy(reference)
    dropped_route["options"][1] = _condition("approved_purchase_order")
    result = compare_policy_logic(reference, dropped_route)
    assert result["missing_condition_ids"] == ["recorded_procurement_exception"]

    unsupported_route = deepcopy(reference)
    unsupported_route["options"][1]["options"].append(_condition("verbal_procurement_agreement"))
    result = compare_policy_logic(reference, unsupported_route)
    assert result["unsupported_condition_ids"] == ["verbal_procurement_agreement"]

    changed_all_to_or = deepcopy(reference)
    changed_all_to_or["operator"] = "any_of"
    result = compare_policy_logic(reference, changed_all_to_or)
    assert result["operator_mismatches"][0]["expected"] == "all_of"
    assert result["operator_mismatches"][0]["actual"] == "any_of"


def test_remedy_contract_checks_structure_and_source_membership_only() -> None:
    value = {
        "assessments": [
            {
                "request_item_id": "invoice_processing",
                "decision": "do_not_proceed",
                "current_blockers": [
                    {
                        "condition": "No approved PO or recorded exception exists.",
                        "source_record_id": "FIN-INV-002",
                    }
                ],
                "remedy": {
                    "node_type": "group",
                    "operator": "any_of",
                    "options": [
                        _action("Correct the invoice to quote an approved PO."),
                        _action("Have Procurement record an exception."),
                    ],
                },
                "missing_information": [],
                "exceptions": [],
                "evidence": [
                    {
                        "record_id": "FIN-INV-002",
                        "claim": "The invoice requires an approved PO or exception.",
                    }
                ],
            }
        ]
    }

    parsed, status = parse_remedy_assessment(json.dumps(value))

    assert status == "valid"
    assert parsed is not None
    grade = remedy_machine_grade(
        assessment=parsed,
        parse_status=status,
        expected_request_item_ids=["invoice_processing"],
        supplied_record_ids=["FIN-INV-002"],
    )
    assert grade["status"] == "semantic_review_required"
    assert "semantic review" in " ".join(grade["semantic_review_reasons"])

    parsed["assessments"][0]["remedy"]["options"][0]["source_record_id"] = "FAKE-001"
    grade = remedy_machine_grade(
        assessment=parsed,
        parse_status="valid",
        expected_request_item_ids=["invoice_processing"],
        supplied_record_ids=["FIN-INV-002"],
    )
    assert grade["status"] == "hard_fail"
    assert "unauthorized" in grade["hard_failure_reasons"][0]


def test_remedy_rendering_is_deterministic_and_groups_are_not_json_schema_keywords() -> None:
    remedy = {
        "node_type": "group",
        "operator": "any_of",
        "options": [
            _action("Correct the invoice to quote an approved purchase order."),
            _action("Have Procurement record an exception."),
        ],
    }

    assert validate_remedy_node(remedy) == ()
    assert render_remedy(remedy) == (
        "Either Correct the invoice to quote an approved purchase order.; "
        "or Have Procurement record an exception."
    )
    invalid = deepcopy(remedy)
    invalid["operator"] = "anyOf"
    assert validate_remedy_node(invalid)
