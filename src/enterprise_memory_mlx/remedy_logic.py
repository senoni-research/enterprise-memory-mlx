"""Typed remedy trees and conservative structural validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .task_specialization import DECISIONS

REMEDY_SCHEMA_ID = "operational-assessment-remedy-logic/v1"
OPERATORS = frozenset({"any_of", "all_of"})
ASSESSMENT_FIELDS = frozenset(
    {
        "request_item_id",
        "decision",
        "current_blockers",
        "remedy",
        "missing_information",
        "exceptions",
        "evidence",
    }
)


def parse_remedy_assessment(
    text: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """Parse the structured-remedy contract without judging policy semantics."""
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
    if not isinstance(value, dict) or set(value) != {"assessments"}:
        return None, "schema_mismatch"
    assessments = value["assessments"]
    if not isinstance(assessments, list) or not assessments:
        return None, "invalid_assessments"
    request_ids: list[str] = []
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_FIELDS:
            return None, "invalid_assessments"
        request_id = assessment.get("request_item_id")
        if not isinstance(request_id, str) or not request_id.strip():
            return None, "invalid_request_item"
        request_ids.append(request_id)
        if assessment.get("decision") not in DECISIONS:
            return None, "invalid_decision"
        blockers = assessment.get("current_blockers")
        if not isinstance(blockers, list):
            return None, "invalid_current_blockers"
        for blocker in blockers:
            if (
                not isinstance(blocker, dict)
                or set(blocker) != {"condition", "source_record_id"}
                or not _nonempty_string(blocker.get("condition"))
                or not _nonempty_string(blocker.get("source_record_id"))
            ):
                return None, "invalid_current_blockers"
        remedy = assessment.get("remedy")
        if remedy is not None and validate_remedy_node(remedy):
            return None, "invalid_remedy"
        for field in ("missing_information", "exceptions"):
            values = assessment.get(field)
            if not isinstance(values, list) or any(not _nonempty_string(item) for item in values):
                return None, f"invalid_{field}"
        evidence = assessment.get("evidence")
        if not isinstance(evidence, list):
            return None, "invalid_evidence"
        for item in evidence:
            if (
                not isinstance(item, dict)
                or set(item) != {"record_id", "claim"}
                or not _nonempty_string(item.get("record_id"))
                or not _nonempty_string(item.get("claim"))
            ):
                return None, "invalid_evidence"
    if len(request_ids) != len(set(request_ids)):
        return None, "duplicate_request_item"
    return value, "valid"


def validate_remedy_node(node: Any) -> tuple[str, ...]:
    """Return structural errors for a recursive remedy node."""
    errors: list[str] = []
    _validate_remedy_node(node, path="$", errors=errors)
    return tuple(errors)


def remedy_machine_grade(
    *,
    assessment: Mapping[str, Any] | None,
    parse_status: str,
    expected_request_item_ids: Sequence[str],
    supplied_record_ids: Sequence[str],
) -> dict[str, Any]:
    """Enforce structure and source membership, leaving meaning to review."""
    hard_failures: list[str] = []
    warnings: list[str] = []
    if assessment is None:
        hard_failures.append(f"parse:{parse_status}")
    else:
        expected = set(expected_request_item_ids)
        actual = {str(item["request_item_id"]) for item in assessment["assessments"]}
        if actual != expected:
            hard_failures.append(f"request_items:expected {sorted(expected)}; got {sorted(actual)}")
        supplied = set(supplied_record_ids)
        used_ids: list[str] = []
        for item in assessment["assessments"]:
            used_ids.extend(
                str(blocker["source_record_id"]) for blocker in item["current_blockers"]
            )
            remedy = item["remedy"]
            if remedy is not None:
                used_ids.extend(remedy_source_record_ids(remedy))
            used_ids.extend(str(evidence["record_id"]) for evidence in item["evidence"])
        unauthorized = sorted(set(used_ids) - supplied)
        if unauthorized:
            hard_failures.append(f"provenance:unauthorized record IDs {unauthorized}")
        warnings.extend(
            [
                "operator correctness requires semantic review",
                "free-text action-to-policy mapping requires semantic review",
                "route completeness and mandatory-condition preservation require semantic review",
            ]
        )
    return {
        "evaluator_id": "remedy-logic-structure/v1",
        "status": "hard_fail" if hard_failures else "semantic_review_required",
        "hard_failure_reasons": hard_failures,
        "semantic_review_reasons": warnings,
        "semantic_review_eligible": not hard_failures,
    }


def evaluate_policy_logic(
    node: Mapping[str, Any],
    satisfied_condition_ids: set[str],
) -> bool:
    """Evaluate an evaluator-side typed policy tree."""
    node_type = node.get("node_type")
    if node_type == "condition":
        condition_id = node.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError("Policy condition node is invalid")
        return condition_id in satisfied_condition_ids
    if node_type != "group" or node.get("operator") not in OPERATORS:
        raise ValueError("Policy group node is invalid")
    options = node.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("Policy group requires at least two options")
    outcomes = [evaluate_policy_logic(option, satisfied_condition_ids) for option in options]
    return any(outcomes) if node["operator"] == "any_of" else all(outcomes)


def compare_policy_logic(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare evaluator-side typed trees after semantic route mapping."""
    reference_conditions = _condition_ids(reference)
    candidate_conditions = _condition_ids(candidate)
    reference_groups = _group_operators(reference)
    candidate_groups = _group_operators(candidate)
    operator_mismatches = []
    for condition_set in sorted(
        set(reference_groups) & set(candidate_groups),
        key=lambda values: tuple(sorted(values)),
    ):
        if reference_groups[condition_set] != candidate_groups[condition_set]:
            operator_mismatches.append(
                {
                    "condition_ids": sorted(condition_set),
                    "expected": reference_groups[condition_set],
                    "actual": candidate_groups[condition_set],
                }
            )
    return {
        "status": (
            "match"
            if reference_conditions == candidate_conditions and reference_groups == candidate_groups
            else "mismatch"
        ),
        "missing_condition_ids": sorted(reference_conditions - candidate_conditions),
        "unsupported_condition_ids": sorted(candidate_conditions - reference_conditions),
        "operator_mismatches": operator_mismatches,
        "missing_groups": [
            {
                "condition_ids": sorted(condition_set),
                "operator": reference_groups[condition_set],
            }
            for condition_set in sorted(
                set(reference_groups) - set(candidate_groups),
                key=lambda values: tuple(sorted(values)),
            )
        ],
        "unsupported_groups": [
            {
                "condition_ids": sorted(condition_set),
                "operator": candidate_groups[condition_set],
            }
            for condition_set in sorted(
                set(candidate_groups) - set(reference_groups),
                key=lambda values: tuple(sorted(values)),
            )
        ],
    }


def render_remedy(node: Mapping[str, Any]) -> str:
    """Render a validated model remedy deterministically without another model."""
    errors = validate_remedy_node(node)
    if errors:
        raise ValueError(f"Cannot render invalid remedy: {errors}")
    if node["node_type"] == "action":
        return str(node["action"])
    rendered = [render_remedy(option) for option in node["options"]]
    if node["operator"] == "any_of":
        return "Either " + "; or ".join(rendered)
    return "Complete all of: " + "; and ".join(rendered)


def remedy_source_record_ids(node: Mapping[str, Any]) -> list[str]:
    if node.get("node_type") == "action":
        return [str(node["source_record_id"])]
    return [
        record_id
        for option in node.get("options", [])
        for record_id in remedy_source_record_ids(option)
    ]


def _validate_remedy_node(node: Any, *, path: str, errors: list[str]) -> None:
    if not isinstance(node, dict):
        errors.append(f"{path}:node must be an object")
        return
    node_type = node.get("node_type")
    if node_type == "action":
        if set(node) != {"node_type", "action", "source_record_id"}:
            errors.append(f"{path}:action fields are invalid")
            return
        if not _nonempty_string(node.get("action")):
            errors.append(f"{path}:action text is required")
        if not _nonempty_string(node.get("source_record_id")):
            errors.append(f"{path}:source_record_id is required")
        return
    if node_type == "group":
        if set(node) != {"node_type", "operator", "options"}:
            errors.append(f"{path}:group fields are invalid")
            return
        if node.get("operator") not in OPERATORS:
            errors.append(f"{path}:operator must be any_of or all_of")
        options = node.get("options")
        if not isinstance(options, list) or len(options) < 2:
            errors.append(f"{path}:group requires at least two options")
            return
        for index, option in enumerate(options):
            _validate_remedy_node(option, path=f"{path}.options[{index}]", errors=errors)
        return
    errors.append(f"{path}:unknown node_type")


def _condition_ids(node: Mapping[str, Any]) -> set[str]:
    if node.get("node_type") == "condition":
        return {str(node["condition_id"])}
    return {condition for option in node.get("options", []) for condition in _condition_ids(option)}


def _group_operators(
    node: Mapping[str, Any],
) -> dict[frozenset[str], str]:
    if node.get("node_type") == "condition":
        return {}
    groups = {
        frozenset(_condition_ids(node)): str(node["operator"]),
    }
    for option in node.get("options", []):
        groups.update(_group_operators(option))
    return groups


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
