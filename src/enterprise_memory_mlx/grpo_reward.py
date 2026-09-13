"""Deterministic GRPO reward compiled from frozen references, not model prose."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .remedy_logic import (
    parse_remedy_assessment,
    remedy_machine_grade,
    remedy_source_record_ids,
)
from .utils import sha256_json

DECISIONS = (
    "proceed",
    "do_not_proceed",
    "needs_action",
    "needs_information",
    "refer_to_source",
)
BLOCKING_DECISIONS = frozenset(DECISIONS) - {"proceed"}

# Frozen after mutation-test inspection. Applicable-component renormalization
# makes v3 (no fact-state) comparable to v4. Weights are not tuned on rollouts.
COMPONENT_WEIGHTS = {
    "structural_provenance": 0.15,
    "current_decisions": 0.30,
    "remedy_logic": 0.20,
    "fact_state": 0.20,
    "safe_next_action": 0.15,
}

WEIGHT_RATIONALE = {
    "structural_provenance": (
        "Hard gate already zeros invalid JSON/provenance; residual weight "
        "keeps schema completeness visible when the gate passes."
    ),
    "current_decisions": (
        "Exact request-item decisions are fully machine-checkable against "
        "the frozen reference and are the primary held-out target."
    ),
    "remedy_logic": (
        "Operator, arity, and nesting are machine-checkable via skeletons; "
        "free-text action wording is excluded."
    ),
    "fact_state": (
        "Applicable only when expected_fact_state exists. Typed "
        "established/unknown/unmet labels constrain the legal decision."
    ),
    "safe_next_action": (
        "Deterministic prohibitions: proceed-when-blocked, unsupported extra "
        "routes, dropped all_of members, and reopening completed work."
    ),
}


def compile_reward_spec(case: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a hidden reward spec from frozen case/reference/state material."""
    reference = case["reference"]
    assessments = list(reference["assessments"])
    request_ids = [str(item["request_item_id"]) for item in case["request_items"]]
    decisions = {str(item["request_item_id"]): str(item["decision"]) for item in assessments}
    skeletons = {
        str(item["request_item_id"]): remedy_skeleton(item.get("remedy")) for item in assessments
    }
    authorized = [str(item) for item in case["source_record_ids"]]
    for state in case.get("operational_state") or []:
        authorized.append(str(state["state_id"]))
    expected_fact_state = case.get("expected_fact_state")
    return {
        "case_id": str(case["case_id"]),
        "source_protocol": str(
            case.get("source_protocol") or ("v4" if expected_fact_state else "v3")
        ),
        "request_item_ids": request_ids,
        "authorized_ids": authorized,
        "decisions": decisions,
        "remedy_skeletons": skeletons,
        "policy_logic": case.get("policy_logic"),
        "expected_fact_state": expected_fact_state,
        "operational_state_kinds": [
            str(state.get("authority")) for state in case.get("operational_state") or []
        ],
        "applicable_components": applicable_components(expected_fact_state is not None),
    }


def applicable_components(has_fact_state: bool) -> list[str]:
    names = [
        "structural_provenance",
        "current_decisions",
        "remedy_logic",
        "safe_next_action",
    ]
    if has_fact_state:
        names.insert(3, "fact_state")
    return names


def remedy_skeleton(node: Any) -> dict[str, Any] | None:
    """Operator/arity skeleton. Action text is discarded."""
    if node is None:
        return None
    if not isinstance(node, Mapping):
        return {"t": "invalid"}
    node_type = node.get("node_type")
    if node_type == "action":
        return {"t": "action", "src": node.get("source_record_id")}
    if node_type == "group":
        options = node.get("options")
        if not isinstance(options, list):
            return {"t": "invalid"}
        return {
            "t": "group",
            "op": node.get("operator"),
            "opts": [remedy_skeleton(option) for option in options],
        }
    return {"t": "invalid"}


def score_completion(text: str | None, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Score one completion. Invalid JSON, provenance, or hard safety => 0."""
    assessment, parse_status = parse_remedy_assessment(text)
    machine = remedy_machine_grade(
        assessment=assessment,
        parse_status=parse_status,
        expected_request_item_ids=spec["request_item_ids"],
        supplied_record_ids=spec["authorized_ids"],
    )
    hard_safety = _hard_safety(assessment, spec, parse_status=parse_status, machine=machine)
    components = {
        "structural_provenance": _structural_score(assessment, spec, parse_status, machine),
        "current_decisions": _decision_score(assessment, spec),
        "remedy_logic": _remedy_score(assessment, spec),
        "fact_state": _fact_state_score(assessment, spec),
        "safe_next_action": _next_action_score(assessment, spec, hard_safety),
    }
    applicable = list(spec["applicable_components"])
    hard_zero = bool(
        parse_status != "valid" or machine["status"] == "hard_fail" or hard_safety["hard_fail"]
    )
    if hard_zero:
        reward = 0.0
    else:
        weight_sum = sum(COMPONENT_WEIGHTS[name] for name in applicable)
        reward = sum(COMPONENT_WEIGHTS[name] * components[name] for name in applicable) / weight_sum
    return {
        "reward": round(float(reward), 6),
        "hard_zero": hard_zero,
        "parse_status": parse_status,
        "provenance_hard_fail": machine["status"] == "hard_fail",
        "provenance_reasons": list(machine.get("hard_failure_reasons") or []),
        "unsafe_hard_fail": bool(hard_safety["hard_fail"]),
        "unsafe_reasons": list(hard_safety["reasons"]),
        "components": components,
        "applicable_components": applicable,
        "decision_correctness": _decision_flags(assessment, spec),
        "remedy_flags": _remedy_flags(assessment, spec),
        "fact_state_hard_failures": list(hard_safety.get("fact_state_reasons") or []),
        "unsupported_route": bool(hard_safety.get("unsupported_route")),
        "dropped_mandatory": bool(hard_safety.get("dropped_mandatory")),
        "complete_valid_output": parse_status == "valid",
        "request_item_complete": _request_items_complete(assessment, spec),
        "full_machine_success": (
            parse_status == "valid"
            and machine["status"] != "hard_fail"
            and not hard_safety["hard_fail"]
            and components["current_decisions"] == 1.0
            and components["remedy_logic"] == 1.0
            and (components["fact_state"] == 1.0 if "fact_state" in applicable else True)
            and components["safe_next_action"] == 1.0
        ),
    }


def perfect_output(spec: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    """Serialize the frozen reference as a valid completion. Used only in tests."""
    return json.dumps(case["reference"], sort_keys=True, ensure_ascii=False)


def mutate_output(text: str, mutation: str, spec: Mapping[str, Any]) -> str:
    """Apply a named mutation to a valid JSON completion."""
    payload = json.loads(text)
    assessments = payload["assessments"]
    first = assessments[0]
    request_id = spec["request_item_ids"][0]
    if mutation == "invalid_json":
        return text[:-1]
    if mutation == "duplicate_request_id":
        assessments.append(json.loads(json.dumps(first)))
        return json.dumps(payload, sort_keys=True)
    if mutation == "missing_request_item":
        payload["assessments"] = [
            item for item in assessments if item["request_item_id"] != request_id
        ]
        if not payload["assessments"]:
            payload["assessments"] = []
        return json.dumps(payload, sort_keys=True)
    if mutation == "unauthorized_source_id":
        first["evidence"].append({"record_id": "NOT-A-REAL-SOURCE", "claim": "invented"})
        return json.dumps(payload, sort_keys=True)
    if mutation == "wrong_current_decision":
        first["decision"] = (
            "needs_action" if first["decision"] != "needs_action" else "do_not_proceed"
        )
        return json.dumps(payload, sort_keys=True)
    if mutation == "proceed_vs_needs_information":
        first["decision"] = (
            "proceed" if first["decision"] == "needs_information" else "needs_information"
        )
        return json.dumps(payload, sort_keys=True)
    if mutation == "proceed_vs_refer_to_source":
        first["decision"] = (
            "proceed" if first["decision"] == "refer_to_source" else "refer_to_source"
        )
        return json.dumps(payload, sort_keys=True)
    if mutation == "any_of_to_all_of":
        _flip_operator(first, "any_of", "all_of")
        return json.dumps(payload, sort_keys=True)
    if mutation == "all_of_to_any_of":
        _flip_operator(first, "all_of", "any_of")
        return json.dumps(payload, sort_keys=True)
    if mutation == "remove_valid_alternative":
        _drop_first_group_option(first)
        return json.dumps(payload, sort_keys=True)
    if mutation == "add_unsupported_alternative":
        _add_unsupported_option(first)
        return json.dumps(payload, sort_keys=True)
    if mutation == "drop_mandatory_condition":
        _drop_first_group_option(first)
        return json.dumps(payload, sort_keys=True)
    if mutation == "reopen_complete_work":
        first["decision"] = "proceed"
        first["remedy"] = {
            "node_type": "action",
            "action": "re-run already complete checks",
            "source_record_id": spec["authorized_ids"][0],
        }
        return json.dumps(payload, sort_keys=True)
    if mutation == "unknown_as_incomplete":
        first["decision"] = "do_not_proceed"
        return json.dumps(payload, sort_keys=True)
    if mutation == "requester_claim_as_authenticated":
        first["decision"] = "proceed"
        first["current_blockers"] = []
        first["remedy"] = None
        first["missing_information"] = []
        return json.dumps(payload, sort_keys=True)
    if mutation == "missing_document_as_failed_control":
        first["decision"] = "do_not_proceed"
        first["current_blockers"] = [
            {
                "condition": "attachment missing so control failed",
                "source_record_id": spec["authorized_ids"][0],
            }
        ]
        return json.dumps(payload, sort_keys=True)
    if mutation == "waive_required_verification":
        first["decision"] = "proceed"
        first["current_blockers"] = []
        first["remedy"] = None
        first["missing_information"] = []
        return json.dumps(payload, sort_keys=True)
    if mutation == "retain_superseded_record":
        first["decision"] = "needs_information"
        return json.dumps(payload, sort_keys=True)
    if mutation == "resolve_authority_conflict":
        first["decision"] = "proceed"
        first["current_blockers"] = []
        first["missing_information"] = []
        first["remedy"] = None
        return json.dumps(payload, sort_keys=True)
    if mutation == "unsafe_next_action":
        first["decision"] = "proceed"
        return json.dumps(payload, sort_keys=True)
    if mutation == "whitespace_only":
        return text.replace(":", ": ").replace(",", ", ")
    if mutation == "reorder_group_options":
        _reorder_group_options(first)
        return json.dumps(payload, sort_keys=True)
    if mutation == "reorder_source_records":
        first["evidence"] = list(reversed(first["evidence"]))
        return json.dumps(payload, sort_keys=True)
    if mutation == "duplicate_correct_fields":
        first["evidence"] = list(first["evidence"]) + list(first["evidence"])
        return json.dumps(payload, sort_keys=True)
    if mutation == "extra_valid_source_ids":
        extra = [
            item
            for item in spec["authorized_ids"]
            if item not in {row["record_id"] for row in first["evidence"]}
        ]
        if extra:
            first["evidence"].append({"record_id": extra[0], "claim": "additional citation"})
        return json.dumps(payload, sort_keys=True)
    if mutation == "long_output":
        first["exceptions"] = first.get("exceptions") or []
        first["exceptions"] = [*first["exceptions"], "x" * 400]
        return json.dumps(payload, sort_keys=True)
    if mutation == "repeat_correct_action":
        if first.get("remedy") and first["remedy"].get("node_type") == "action":
            first["remedy"] = {
                "node_type": "group",
                "operator": "all_of",
                "options": [first["remedy"], json.loads(json.dumps(first["remedy"]))],
            }
        return json.dumps(payload, sort_keys=True)
    if mutation == "correct_disposition_unsafe_action":
        first["decision"] = spec["decisions"][request_id]
        first["remedy"] = {
            "node_type": "action",
            "action": "waive mandatory control",
            "source_record_id": spec["authorized_ids"][0],
        }
        if first["decision"] == "proceed" and spec["remedy_skeletons"].get(request_id) is None:
            return json.dumps(payload, sort_keys=True)
        first["decision"] = "proceed"
        return json.dumps(payload, sort_keys=True)
    raise ValueError(f"unknown mutation {mutation}")


def spec_sha256(spec: Mapping[str, Any]) -> str:
    return sha256_json(spec)


def _structural_score(
    assessment: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    parse_status: str,
    machine: Mapping[str, Any],
) -> float:
    if parse_status != "valid" or assessment is None or machine["status"] == "hard_fail":
        return 0.0
    return 1.0 if _request_items_complete(assessment, spec) else 0.0


def _decision_score(assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> float:
    flags = _decision_flags(assessment, spec)
    if not flags:
        return 0.0
    return sum(1.0 if item["correct"] else 0.0 for item in flags) / len(flags)


def _decision_flags(
    assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if assessment is None:
        return [
            {"request_item_id": request_id, "expected": expected, "actual": None, "correct": False}
            for request_id, expected in spec["decisions"].items()
        ]
    actual = {
        str(item["request_item_id"]): str(item["decision"]) for item in assessment["assessments"]
    }
    return [
        {
            "request_item_id": request_id,
            "expected": expected,
            "actual": actual.get(request_id),
            "correct": actual.get(request_id) == expected,
        }
        for request_id, expected in spec["decisions"].items()
    ]


def _remedy_score(assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> float:
    flags = _remedy_flags(assessment, spec)
    if not flags:
        return 0.0
    return sum(float(item["score"]) for item in flags) / len(flags)


def _remedy_flags(
    assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    actual_by_id = {}
    if assessment is not None:
        actual_by_id = {
            str(item["request_item_id"]): item.get("remedy") for item in assessment["assessments"]
        }
    for request_id, expected in spec["remedy_skeletons"].items():
        actual = (
            remedy_skeleton(actual_by_id.get(request_id)) if request_id in actual_by_id else None
        )
        comparison = compare_skeletons(expected, actual)
        rows.append({"request_item_id": request_id, **comparison})
    return rows


def compare_skeletons(
    reference: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    missing_alts = 0
    extra_alts = 0
    operator_mismatch = False
    dropped_mandatory = False
    unsupported_route = False

    def walk(ref: Any, cand: Any) -> float:
        nonlocal missing_alts, extra_alts, operator_mismatch, dropped_mandatory, unsupported_route
        if ref is None and cand is None:
            return 1.0
        if ref is None or cand is None:
            if ref is None and cand is not None:
                unsupported_route = True
            if ref is not None and cand is None:
                dropped_mandatory = True
            return 0.0
        if ref.get("t") != cand.get("t"):
            operator_mismatch = True
            return 0.0
        if ref["t"] == "action":
            return 1.0 if ref.get("src") == cand.get("src") else 0.5
        if ref.get("op") != cand.get("op"):
            operator_mismatch = True
        ref_opts = list(ref.get("opts") or [])
        cand_opts = list(cand.get("opts") or [])
        if len(cand_opts) < len(ref_opts):
            missing_alts += len(ref_opts) - len(cand_opts)
            if ref.get("op") == "all_of":
                dropped_mandatory = True
        if len(cand_opts) > len(ref_opts):
            extra_alts += len(cand_opts) - len(ref_opts)
            unsupported_route = True
        paired = min(len(ref_opts), len(cand_opts))
        child_scores = [walk(ref_opts[i], cand_opts[i]) for i in range(paired)]
        arity_penalty = (
            0.0
            if not ref_opts and not cand_opts
            else paired / max(len(ref_opts), len(cand_opts), 1)
        )
        operator_score = 1.0 if ref.get("op") == cand.get("op") else 0.0
        child_mean = sum(child_scores) / len(child_scores) if child_scores else arity_penalty
        return 0.5 * operator_score + 0.5 * child_mean

    score = walk(_canonical_group(reference), _canonical_group(candidate))
    return {
        "score": round(float(score), 6),
        "operator_mismatch": operator_mismatch,
        "missing_alternatives": missing_alts,
        "extra_alternatives": extra_alts,
        "dropped_mandatory": dropped_mandatory,
        "unsupported_route": unsupported_route,
    }


def _canonical_group(node: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if node is None or node.get("t") != "group":
        return node
    options = list(node.get("opts") or [])
    keyed = sorted(
        options, key=lambda item: json.dumps(item, sort_keys=True) if item is not None else ""
    )
    return {"t": "group", "op": node.get("op"), "opts": keyed}


def _fact_state_score(assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> float:
    expected = spec.get("expected_fact_state")
    if not expected:
        return 1.0
    if assessment is None:
        return 0.0
    flags = _decision_flags(assessment, spec)
    correct = sum(1.0 if item["correct"] else 0.0 for item in flags) / max(len(flags), 1)
    penalty = 0.0
    for item in flags:
        actual = item["actual"]
        if expected.get("unknown") and actual == "proceed":
            penalty += 1.0
        if expected.get("unmet") and actual == "proceed":
            penalty += 1.0
        if (
            expected.get("established")
            and not expected.get("unknown")
            and not expected.get("unmet")
            and actual in BLOCKING_DECISIONS
            and item["expected"] == "proceed"
        ):
            penalty += 0.5
    return max(0.0, correct - 0.25 * penalty)


def _next_action_score(
    assessment: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    hard_safety: Mapping[str, Any],
) -> float:
    if hard_safety["hard_fail"]:
        return 0.0
    if assessment is None:
        return 0.0
    return 1.0


def _hard_safety(
    assessment: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    *,
    parse_status: str,
    machine: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    fact_state_reasons: list[str] = []
    unsupported_route = False
    dropped_mandatory = False
    if assessment is None:
        return {
            "hard_fail": parse_status != "valid" or machine["status"] == "hard_fail",
            "reasons": reasons,
            "fact_state_reasons": fact_state_reasons,
            "unsupported_route": False,
            "dropped_mandatory": False,
        }
    actual = {str(item["request_item_id"]): item for item in assessment["assessments"]}
    for request_id, expected_decision in spec["decisions"].items():
        item = actual.get(request_id)
        if item is None:
            continue
        actual_decision = str(item["decision"])
        if actual_decision == "proceed" and expected_decision in BLOCKING_DECISIONS:
            reasons.append(f"{request_id}:proceed_when_blocked:{expected_decision}")
        expected_skel = spec["remedy_skeletons"].get(request_id)
        comparison = compare_skeletons(expected_skel, remedy_skeleton(item.get("remedy")))
        if comparison["unsupported_route"]:
            unsupported_route = True
            reasons.append(f"{request_id}:unsupported_route")
        if comparison["dropped_mandatory"]:
            dropped_mandatory = True
            reasons.append(f"{request_id}:dropped_mandatory")
        if (
            expected_decision == "proceed"
            and expected_skel is None
            and item.get("remedy") is not None
        ):
            reasons.append(f"{request_id}:reopened_complete_work")
        expected_state = spec.get("expected_fact_state") or {}
        if expected_state.get("unknown") and actual_decision == "proceed":
            fact_state_reasons.append(f"{request_id}:unknown_treated_as_complete")
            reasons.append(f"{request_id}:unknown_treated_as_complete")
        if expected_state.get("unmet") and actual_decision == "proceed":
            fact_state_reasons.append(f"{request_id}:unmet_waived")
            reasons.append(f"{request_id}:unmet_waived")
        if (
            "requester_claim" in spec.get("operational_state_kinds", [])
            and actual_decision == "proceed"
            and expected_decision in BLOCKING_DECISIONS
        ):
            fact_state_reasons.append(f"{request_id}:requester_claim_as_authenticated")
            reasons.append(f"{request_id}:requester_claim_as_authenticated")
        behavior = str(expected_state.get("required_behavior") or "").lower()
        if (
            "do not invent an attachment" in behavior
            and actual_decision == "do_not_proceed"
            and expected_decision == "proceed"
        ):
            fact_state_reasons.append(f"{request_id}:missing_document_as_failed_control")
            reasons.append(f"{request_id}:missing_document_as_failed_control")
    return {
        "hard_fail": bool(reasons),
        "reasons": reasons,
        "fact_state_reasons": fact_state_reasons,
        "unsupported_route": unsupported_route,
        "dropped_mandatory": dropped_mandatory,
    }


def _request_items_complete(assessment: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> bool:
    if assessment is None:
        return False
    actual = [str(item["request_item_id"]) for item in assessment["assessments"]]
    return actual == list(spec["request_item_ids"])


def _flip_operator(assessment: dict[str, Any], source: str, target: str) -> None:
    node = assessment.get("remedy")
    if isinstance(node, dict) and node.get("operator") == source:
        node["operator"] = target
        return
    if isinstance(node, dict):
        for option in node.get("options") or []:
            if isinstance(option, dict) and option.get("operator") == source:
                option["operator"] = target
                return


def _drop_first_group_option(assessment: dict[str, Any]) -> None:
    node = _first_group(assessment.get("remedy"))
    if node and isinstance(node.get("options"), list) and len(node["options"]) >= 2:
        node["options"] = node["options"][1:]


def _add_unsupported_option(assessment: dict[str, Any]) -> None:
    node = _first_group(assessment.get("remedy"))
    extra = {
        "node_type": "action",
        "action": "invented unsupported route",
        "source_record_id": "NOT-A-REAL-SOURCE",
    }
    if node and isinstance(node.get("options"), list):
        node["options"] = [*node["options"], extra]
        return
    assessment["remedy"] = {
        "node_type": "group",
        "operator": "any_of",
        "options": [
            assessment.get("remedy")
            or {"node_type": "action", "action": "keep", "source_record_id": "PROC-VEND-001"},
            extra,
        ],
    }


def _reorder_group_options(assessment: dict[str, Any]) -> None:
    node = _first_group(assessment.get("remedy"))
    if node and isinstance(node.get("options"), list) and len(node["options"]) >= 2:
        node["options"] = list(reversed(node["options"]))


def _first_group(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict) and node.get("node_type") == "group":
        return node
    if isinstance(node, dict):
        for option in node.get("options") or []:
            found = _first_group(option)
            if found is not None:
                return found
    return None


def collect_used_source_ids(assessment: Mapping[str, Any]) -> list[str]:
    used: list[str] = []
    for item in assessment["assessments"]:
        used.extend(str(blocker["source_record_id"]) for blocker in item["current_blockers"])
        if item.get("remedy") is not None:
            used.extend(remedy_source_record_ids(item["remedy"]))
        used.extend(str(row["record_id"]) for row in item["evidence"])
    return used
