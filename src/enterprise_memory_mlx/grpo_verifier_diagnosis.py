"""Separately versioned diagnostics for the frozen GRPO v1 reward.

This module does not change ``score_completion``, landscape thresholds, or
historical stop decisions. It records how the frozen checker measures, and
proposes a narrower typed-action contract for later authorization.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .grpo_data import file_sha256
from .grpo_landscape import EXPECTED_TRAIN, train_specs
from .grpo_reward import collect_used_source_ids, remedy_skeleton, score_completion
from .remedy_logic import parse_remedy_assessment, remedy_machine_grade
from .utils import atomic_write_text

DIAGNOSIS_ID = "company-task-specialization/qwen4b-grpo-verifier-diagnosis-v1"
LANDSCAPE_REWARD_SHA256 = (
    "e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b"
)
HISTORICAL_STOPS = {
    "grpo_v1": "stop_preflight",
    "landscape_v1": "abandon_standard_grpo_for_current_dataset",
}
JSON_SYNTAX_STATUSES = frozenset({"missing_output", "malformed_json", "trailing_content"})
SCHEMA_STATUSES = frozenset(
    {
        "schema_mismatch",
        "invalid_assessments",
        "invalid_request_item",
        "invalid_decision",
        "invalid_current_blockers",
        "invalid_remedy",
        "invalid_missing_information",
        "invalid_exceptions",
        "invalid_evidence",
        "duplicate_request_item",
    }
)
ACTION_TYPES = (
    "request_evidence",
    "record_completion",
    "waive_control",
    "obtain_alternative_authorization",
    "refer_to_source",
)
OBSERVED_STATE_CATEGORIES = (
    "established",
    "unknown",
    "unmet",
    "non_authoritative_or_scoped",
)
LANDSCAPE_RUN = Path("artifacts/qwen4b-grpo-landscape-v1/run-20260914T153149Z")
OVERLAY_DIR = Path("artifacts/qwen4b-grpo-landscape-v1/verifier-diagnosis-v1")


def current_reward_sha256() -> str:
    return file_sha256(Path(__file__).with_name("grpo_reward.py"))


def reward_matches_landscape_binding() -> bool:
    return current_reward_sha256() == LANDSCAPE_REWARD_SHA256


def invented_action_spec() -> dict[str, Any]:
    """Single-action TRAIN-shaped fixture. Not a DEV/TEST case."""
    return {
        "case_id": "DIAG-INV-ACTION-01",
        "source_protocol": "v3",
        "request_item_ids": ["purchase_order_issue"],
        "authorized_ids": ["SRC-POLICY-001"],
        "decisions": {"purchase_order_issue": "needs_action"},
        "remedy_skeletons": {
            "purchase_order_issue": {"t": "action", "src": "SRC-POLICY-001"},
        },
        "policy_logic": None,
        "expected_fact_state": None,
        "operational_state_kinds": [],
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
    }


def invented_any_of_three_spec() -> dict[str, Any]:
    """Three-alternative any_of fixture with distinct source IDs."""
    leaves = [
        {"t": "action", "src": "SRC-ALT-A"},
        {"t": "action", "src": "SRC-ALT-B"},
        {"t": "action", "src": "SRC-ALT-C"},
    ]
    return {
        "case_id": "DIAG-INV-ANYOF-03",
        "source_protocol": "v3",
        "request_item_ids": ["invoice_processing"],
        "authorized_ids": ["SRC-ALT-A", "SRC-ALT-B", "SRC-ALT-C"],
        "decisions": {"invoice_processing": "needs_action"},
        "remedy_skeletons": {
            "invoice_processing": {"t": "group", "op": "any_of", "opts": leaves},
        },
        "policy_logic": None,
        "expected_fact_state": None,
        "operational_state_kinds": [],
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
    }


def invented_all_of_two_spec() -> dict[str, Any]:
    leaves = [
        {"t": "action", "src": "SRC-MAND-A"},
        {"t": "action", "src": "SRC-MAND-B"},
    ]
    return {
        "case_id": "DIAG-INV-ALLOF-02",
        "source_protocol": "v3",
        "request_item_ids": ["purchase_order_issue"],
        "authorized_ids": ["SRC-MAND-A", "SRC-MAND-B"],
        "decisions": {"purchase_order_issue": "needs_action"},
        "remedy_skeletons": {
            "purchase_order_issue": {"t": "group", "op": "all_of", "opts": leaves},
        },
        "policy_logic": None,
        "expected_fact_state": None,
        "operational_state_kinds": [],
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
    }


def invented_two_item_spec() -> dict[str, Any]:
    return {
        "case_id": "DIAG-INV-MULTI-02",
        "source_protocol": "v3",
        "request_item_ids": ["supplier_onboarding", "invoice_processing"],
        "authorized_ids": ["SRC-ONB-001", "SRC-INV-001"],
        "decisions": {
            "supplier_onboarding": "proceed",
            "invoice_processing": "needs_action",
        },
        "remedy_skeletons": {
            "supplier_onboarding": None,
            "invoice_processing": {"t": "action", "src": "SRC-INV-001"},
        },
        "policy_logic": None,
        "expected_fact_state": None,
        "operational_state_kinds": [],
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
    }


def _item(
    request_item_id: str,
    decision: str,
    *,
    remedy: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    blockers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(evidence) if evidence is not None else []
    return {
        "request_item_id": request_item_id,
        "decision": decision,
        "current_blockers": list(blockers or []),
        "remedy": None if remedy is None else json.loads(json.dumps(remedy)),
        "missing_information": [],
        "exceptions": [],
        "evidence": [json.loads(json.dumps(row)) for row in rows],
    }


def encode_assessments(items: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps({"assessments": [dict(item) for item in items]}, sort_keys=True)


def valid_action_output(
    spec: Mapping[str, Any],
    *,
    action_text: str = "Complete mandatory verification V-1 and record the result",
) -> str:
    request_id = spec["request_item_ids"][0]
    source_id = spec["authorized_ids"][0]
    return encode_assessments(
        [
            _item(
                request_id,
                spec["decisions"][request_id],
                remedy={
                    "node_type": "action",
                    "action": action_text,
                    "source_record_id": source_id,
                },
                evidence=[{"record_id": source_id, "claim": "authorized policy citation"}],
            )
        ]
    )


def valid_any_of_three_output(spec: Mapping[str, Any]) -> str:
    request_id = spec["request_item_ids"][0]
    options = [
        {
            "node_type": "action",
            "action": f"Obtain alternative {source_id}",
            "source_record_id": source_id,
        }
        for source_id in spec["authorized_ids"]
    ]
    evidence = [
        {"record_id": source_id, "claim": f"supported alternative {source_id}"}
        for source_id in spec["authorized_ids"]
    ]
    return encode_assessments(
        [
            _item(
                request_id,
                spec["decisions"][request_id],
                remedy={"node_type": "group", "operator": "any_of", "options": options},
                evidence=evidence,
            )
        ]
    )


def valid_all_of_two_output(spec: Mapping[str, Any]) -> str:
    request_id = spec["request_item_ids"][0]
    options = [
        {
            "node_type": "action",
            "action": f"Complete mandatory control {source_id}",
            "source_record_id": source_id,
        }
        for source_id in spec["authorized_ids"]
    ]
    evidence = [
        {"record_id": source_id, "claim": f"mandatory control {source_id}"}
        for source_id in spec["authorized_ids"]
    ]
    return encode_assessments(
        [
            _item(
                request_id,
                spec["decisions"][request_id],
                remedy={"node_type": "group", "operator": "all_of", "options": options},
                evidence=evidence,
            )
        ]
    )


def substitute_action_text(text: str, action_text: str) -> str:
    payload = json.loads(text)
    payload["assessments"][0]["remedy"]["action"] = action_text
    return json.dumps(payload, sort_keys=True)


def drop_named_alternative(text: str, source_id: str) -> str:
    payload = json.loads(text)
    remedy = payload["assessments"][0]["remedy"]
    remedy["options"] = [
        option for option in remedy["options"] if option["source_record_id"] != source_id
    ]
    payload["assessments"][0]["evidence"] = [
        row
        for row in payload["assessments"][0]["evidence"]
        if row["record_id"] != source_id
    ]
    return json.dumps(payload, sort_keys=True)


def swap_group_operator(text: str, operator: str) -> str:
    payload = json.loads(text)
    payload["assessments"][0]["remedy"]["operator"] = operator
    return json.dumps(payload, sort_keys=True)


def valid_two_item_output(spec: Mapping[str, Any]) -> str:
    onb, inv = spec["request_item_ids"]
    return encode_assessments(
        [
            _item(
                onb,
                spec["decisions"][onb],
                evidence=[{"record_id": "SRC-ONB-001", "claim": "onboarding complete"}],
            ),
            _item(
                inv,
                spec["decisions"][inv],
                remedy={
                    "node_type": "action",
                    "action": "Quote an approved purchase order",
                    "source_record_id": "SRC-INV-001",
                },
                evidence=[{"record_id": "SRC-INV-001", "claim": "invoice exception path"}],
            ),
        ]
    )


def _canonical_group(node: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if node is None or node.get("t") != "group":
        return node
    options = list(node.get("opts") or [])
    keyed = sorted(
        options,
        key=lambda item: json.dumps(item, sort_keys=True) if item is not None else "",
    )
    return {"t": "group", "op": node.get("op"), "opts": keyed}


def corrected_structure_compare(
    reference: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Exact-arity structural compare. Prefix pairing does not hide omissions."""
    missing_alts = 0
    extra_alts = 0
    operator_mismatch = False
    dropped_mandatory = False
    unsupported_route = False

    def equivalent(ref: Any, cand: Any) -> bool:
        if ref is None and cand is None:
            return True
        if ref is None or cand is None:
            return False
        if ref.get("t") != cand.get("t"):
            return False
        if ref["t"] == "action":
            return ref.get("src") == cand.get("src")
        if ref.get("op") != cand.get("op"):
            return False
        unused = list(cand.get("opts") or [])
        for child in ref.get("opts") or []:
            match_at = next(
                (index for index, option in enumerate(unused) if equivalent(child, option)),
                None,
            )
            if match_at is None:
                return False
            unused.pop(match_at)
        return not unused

    def walk(ref: Any, cand: Any) -> None:
        nonlocal missing_alts, extra_alts, operator_mismatch
        nonlocal dropped_mandatory, unsupported_route
        if ref is None and cand is None:
            return
        if ref is None or cand is None:
            if ref is None and cand is not None:
                unsupported_route = True
                extra_alts += 1
            if ref is not None and cand is None:
                dropped_mandatory = True
                missing_alts += 1
            return
        if ref.get("t") != cand.get("t") or (
            ref.get("t") == "group" and ref.get("op") != cand.get("op")
        ):
            operator_mismatch = True
        if ref.get("t") == "action" or cand.get("t") == "action":
            if ref.get("src") != cand.get("src") or ref.get("t") != cand.get("t"):
                unsupported_route = True
            return
        unused = list(cand.get("opts") or [])
        for child in ref.get("opts") or []:
            match_at = next(
                (index for index, option in enumerate(unused) if equivalent(child, option)),
                None,
            )
            if match_at is None:
                missing_alts += 1
                if ref.get("op") == "all_of":
                    dropped_mandatory = True
            else:
                walk(child, unused.pop(match_at))
        if unused:
            extra_alts += len(unused)
            unsupported_route = True

    walk(_canonical_group(reference), _canonical_group(candidate))
    return {
        "operator_mismatch": operator_mismatch,
        "missing_alternatives": missing_alts,
        "extra_alternatives": extra_alts,
        "dropped_mandatory": dropped_mandatory,
        "unsupported_route": unsupported_route,
        "structure_complete": not (
            operator_mismatch
            or missing_alts
            or extra_alts
            or dropped_mandatory
            or unsupported_route
        ),
    }


def diagnose_completion(text: str | None, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Corrected taxonomy. Does not replace the frozen scalar reward."""
    assessment, parse_status = parse_remedy_assessment(text)
    machine = remedy_machine_grade(
        assessment=assessment,
        parse_status=parse_status,
        expected_request_item_ids=spec["request_item_ids"],
        supplied_record_ids=spec["authorized_ids"],
    )
    json_syntax_failure = parse_status in JSON_SYNTAX_STATUSES
    schema_failure = parse_status in SCHEMA_STATUSES
    if assessment is None:
        source_membership = "not_assessable"
        unauthorized = False
        coverage_failure = False
        coverage_status = "not_assessable"
        decision_mismatch = False
        structure = {
            "operator_mismatch": False,
            "missing_alternatives": 0,
            "extra_alternatives": 0,
            "dropped_mandatory": False,
            "unsupported_route": False,
            "structure_complete": False,
        }
        structure_status = "not_assessable"
        has_action = False
    else:
        used = set(collect_used_source_ids(assessment))
        authorized = set(spec["authorized_ids"])
        unauthorized = bool(used - authorized)
        source_membership = "unauthorized" if unauthorized else "authorized_only"
        expected_ids = list(spec["request_item_ids"])
        actual_ids = [str(item["request_item_id"]) for item in assessment["assessments"]]
        coverage_failure = actual_ids != expected_ids
        coverage_status = "fail" if coverage_failure else "pass"
        actual_decisions = {
            str(item["request_item_id"]): str(item["decision"])
            for item in assessment["assessments"]
        }
        decision_mismatch = any(
            actual_decisions.get(request_id) != expected
            for request_id, expected in spec["decisions"].items()
        )
        flags = []
        has_action = False
        for request_id, expected in spec["remedy_skeletons"].items():
            actual_node = next(
                (
                    item.get("remedy")
                    for item in assessment["assessments"]
                    if str(item["request_item_id"]) == request_id
                ),
                None,
            )
            if actual_node is not None:
                has_action = has_action or _contains_action(actual_node)
            flags.append(
                corrected_structure_compare(expected, remedy_skeleton(actual_node))
            )
        structure = {
            "operator_mismatch": any(row["operator_mismatch"] for row in flags),
            "missing_alternatives": sum(row["missing_alternatives"] for row in flags),
            "extra_alternatives": sum(row["extra_alternatives"] for row in flags),
            "dropped_mandatory": any(row["dropped_mandatory"] for row in flags),
            "unsupported_route": any(row["unsupported_route"] for row in flags),
            "structure_complete": all(row["structure_complete"] for row in flags),
        }
        structure_status = "pass" if structure["structure_complete"] else "fail"
    provenance_reasons = list(machine.get("hard_failure_reasons") or [])
    old_provenance_hard_fail = machine["status"] == "hard_fail"
    return {
        "parse_status": parse_status,
        "json_syntax_failure": json_syntax_failure,
        "schema_failure": schema_failure,
        "request_item_coverage_failure": coverage_failure,
        "request_item_coverage_status": coverage_status,
        "unauthorized_source_id_use": unauthorized,
        "source_membership_assessment": source_membership,
        "exact_decision_mismatch": decision_mismatch,
        "remedy_structure_mismatch": structure_status == "fail",
        "remedy_structure_status": structure_status,
        "omitted_alternatives": int(structure["missing_alternatives"]),
        "corrected_structure": structure,
        "action_meaning_not_assessed": True,
        "action_node_present": has_action if assessment is not None else False,
        "old_provenance_hard_fail": old_provenance_hard_fail,
        "old_provenance_reasons": provenance_reasons,
        "old_provenance_conflates": old_provenance_hard_fail
        and (
            json_syntax_failure
            or schema_failure
            or coverage_failure
            or unauthorized
            or parse_status != "valid"
        ),
        "positive_provenance_success": (
            parse_status == "valid"
            and source_membership == "authorized_only"
            and not coverage_failure
        ),
        "invented_citation": unauthorized,
    }


def _contains_action(node: Any) -> bool:
    if not isinstance(node, Mapping):
        return False
    if node.get("node_type") == "action":
        return True
    return any(_contains_action(option) for option in node.get("options") or [])


def overlay_row(
    score_row: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    diagnosis = diagnose_completion(score_row.get("raw_output"), spec)
    old = score_completion(score_row.get("raw_output"), spec)
    return {
        "diagnosis_id": DIAGNOSIS_ID,
        "case_id": score_row["case_id"],
        "regime_id": score_row.get("regime_id"),
        "replicate": score_row.get("replicate"),
        "raw_output_sha256": score_row.get("raw_output_sha256"),
        "original_reward": score_row["reward"],
        "original_full_machine_success": score_row.get("full_machine_success"),
        "original_parse_status": score_row.get("parse_status"),
        "original_provenance_hard_fail": score_row.get("provenance_hard_fail"),
        "original_unsafe_hard_fail": score_row.get("unsafe_hard_fail"),
        "original_hard_zero": score_row.get("hard_zero"),
        "original_complete_valid_output": score_row.get("complete_valid_output"),
        "original_truncated": score_row.get("truncated"),
        "original_remedy_flags": score_row.get("remedy_flags"),
        "original_components": score_row.get("components"),
        "rescored_reward_unchanged": old["reward"] == score_row["reward"],
        **diagnosis,
        "structural_contradiction_despite_reward_one": (
            float(score_row["reward"]) == 1.0
            and (
                diagnosis["remedy_structure_mismatch"]
                or diagnosis["omitted_alternatives"] > 0
            )
        ),
        "semantic_correctness_inferred": False,
        "admitted_to_training": False,
    }


def _overlap_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "json_syntax_failure",
        "schema_failure",
        "request_item_coverage_failure",
        "unauthorized_source_id_use",
        "exact_decision_mismatch",
        "remedy_structure_mismatch",
    )
    counts: dict[str, int] = {}
    for left in keys:
        for right in keys:
            if keys.index(left) >= keys.index(right):
                continue
            counts[f"{left}+{right}"] = sum(
                1 for row in rows if row[left] and row[right]
            )
    return counts


def summarize_overlay(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_classifications: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    n = len(rows)
    reward_one = [row for row in rows if float(row["original_reward"]) == 1.0]
    assessable_membership = [
        row for row in rows if row["source_membership_assessment"] != "not_assessable"
    ]
    assessable_structure = [
        row for row in rows if row["remedy_structure_status"] != "not_assessable"
    ]
    assessable_coverage = [
        row for row in rows if row["request_item_coverage_status"] != "not_assessable"
    ]
    truncated = sum(1 for row in rows if row.get("original_truncated"))
    if isinstance(case_classifications, Mapping) and "rows" in case_classifications:
        classification_rows = list(case_classifications["rows"])
    else:
        classification_rows = list(case_classifications or [])
    classifications = Counter(
        str(row.get("classification")) for row in classification_rows
    )
    return {
        "diagnosis_id": DIAGNOSIS_ID,
        "historical_stops": dict(HISTORICAL_STOPS),
        "n_outputs": n,
        "denominators": {
            "all_outputs": n,
            "assessable_source_membership": len(assessable_membership),
            "assessable_request_item_coverage": len(assessable_coverage),
            "assessable_remedy_structure": len(assessable_structure),
            "original_reward_one": len(reward_one),
        },
        "counts": {
            "json_syntax_failure": sum(1 for row in rows if row["json_syntax_failure"]),
            "schema_failure": sum(1 for row in rows if row["schema_failure"]),
            "request_item_coverage_failure": sum(
                1 for row in assessable_coverage if row["request_item_coverage_failure"]
            ),
            "unauthorized_source_id_use": sum(
                1 for row in assessable_membership if row["unauthorized_source_id_use"]
            ),
            "source_membership_not_assessable": sum(
                1
                for row in rows
                if row["source_membership_assessment"] == "not_assessable"
            ),
            "exact_decision_mismatch": sum(
                1 for row in rows if row["exact_decision_mismatch"]
            ),
            "remedy_structure_mismatch": sum(
                1 for row in assessable_structure if row["remedy_structure_mismatch"]
            ),
            "omitted_alternatives": sum(
                1 for row in assessable_structure if row["omitted_alternatives"] > 0
            ),
            "action_meaning_not_assessed": sum(
                1 for row in rows if row["action_meaning_not_assessed"]
            ),
            "old_provenance_hard_fail": sum(
                1 for row in rows if row["old_provenance_hard_fail"]
            ),
            "positive_provenance_success": sum(
                1 for row in rows if row["positive_provenance_success"]
            ),
            "invented_citation": sum(1 for row in rows if row["invented_citation"]),
            "original_reward_one": len(reward_one),
            "reward_one_with_structural_contradiction": sum(
                1 for row in reward_one if row["structural_contradiction_despite_reward_one"]
            ),
            "reward_one_with_omitted_alternatives": sum(
                1 for row in reward_one if row["omitted_alternatives"] > 0
            ),
            "original_truncated": truncated,
            "rescored_reward_unchanged": sum(
                1 for row in rows if row["rescored_reward_unchanged"]
            ),
        },
        "overlaps": _overlap_counts(rows),
        "observations": {
            "truncation": {
                "status": "observed_pass" if truncated == 0 and n else "observed_fail",
                "n_truncated": truncated,
                "n_outputs": n,
                "note": (
                    "Observed zero truncation is not a truncation failure merely "
                    "because the landscape aggregate decision failed."
                ),
            },
            "landscape_no_truncation_gate": {
                "status": "not_evaluated_due_to_ineligibility",
                "frozen_gate_value": False,
                "note": (
                    "The landscape fail-closed every viability gate except "
                    "reward_byte_identical after no regime was eligible. That "
                    "false gate is not an observed truncation failure."
                ),
            },
        },
        "original_case_classifications": dict(classifications),
        "semantic_correctness_inferred": False,
        "winner_selected": False,
        "regime_eligibility_reinterpreted": False,
    }


def write_landscape_overlay(root: Path, *, scores_path: Path | None = None) -> dict[str, Any]:
    if not reward_matches_landscape_binding():
        raise RuntimeError(
            "working-tree reward SHA differs from the landscape binding; "
            "refuse to attribute overlay findings to the landscape run"
        )
    run = root / LANDSCAPE_RUN
    scores_file = scores_path or (run / "scores.jsonl")
    specs = train_specs(root)
    if set(specs) != set(EXPECTED_TRAIN):
        raise ValueError("overlay refused a non-TRAIN reward spec")
    rows = []
    for line in scores_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        score_row = json.loads(line)
        case_id = str(score_row["case_id"])
        if case_id not in specs:
            raise ValueError(f"overlay refused non-TRAIN score row {case_id}")
        rows.append(overlay_row(score_row, specs[case_id]))
    case_stats: Mapping[str, Any] | list[Any] = []
    stats_path = run / "case-stats.json"
    if stats_path.is_file():
        case_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    summary = summarize_overlay(rows, case_classifications=case_stats)
    signal_path = run / "signal-classification.json"
    if signal_path.is_file():
        summary["original_classifications_by_regime"] = json.loads(
            signal_path.read_text(encoding="utf-8")
        )
    summary["source_scores"] = str(scores_file)
    summary["n_expected"] = 1152
    destination = root / OVERLAY_DIR
    destination.mkdir(parents=True, exist_ok=True)
    overlay_path = destination / "overlay.jsonl"
    if overlay_path.exists():
        raise FileExistsError(f"refusing to overwrite append-only overlay {overlay_path}")
    overlay_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    atomic_write_text(
        destination / "overlay-summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def proposed_typed_action_contract() -> dict[str, Any]:
    """Compact TRAIN-only proposal. Mapping existing prose remains unresolved."""
    catalogue = [
        {
            "action_type": "request_evidence",
            "arguments": ["condition_id", "source_record_id"],
            "precondition": "observed_state[condition_id] in {unknown, unmet}",
            "meaning": "Ask for evidence; do not record completion.",
        },
        {
            "action_type": "record_completion",
            "arguments": ["condition_id", "source_record_id"],
            "precondition": "observed_state[condition_id] == established",
            "meaning": "Record an already established control. Distinct from requesting it.",
        },
        {
            "action_type": "waive_control",
            "arguments": ["condition_id", "source_record_id"],
            "precondition": "never_safe_without_explicit_exception_source",
            "meaning": "Unsafe unless an authorized exception source is independently established.",
        },
        {
            "action_type": "obtain_alternative_authorization",
            "arguments": ["condition_id", "source_record_id"],
            "precondition": "condition_id is a member of an any_of group",
            "meaning": "One supported alternative. Source ID is not the action identity.",
        },
        {
            "action_type": "refer_to_source",
            "arguments": ["source_record_id"],
            "precondition": "disposition == refer_to_source",
            "meaning": "Hand off; not a completion record.",
        },
    ]
    return {
        "contract_id": f"{DIAGNOSIS_ID}/typed-action-proposal",
        "status": "proposal",
        "scope": "TRAIN material only; no DEV/TEST",
        "observed_state_categories": list(OBSERVED_STATE_CATEGORIES),
        "unknown_is_not_false": True,
        "action_catalogue": catalogue,
        "catalogue_independent_of_correct_answer": True,
        "per_case_grammar_must_admit_incorrect_actions": True,
        "source_id_is_not_action_identity": True,
        "matching_tree_shape_is_not_policy_equivalence": True,
        "free_text_entailment_is_not_machine_verified": True,
        "condition_ids_from_train_policy_logic": [
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
            "questionnaire_pdf_present",
            "data_supplier",
            "attachment_controlled",
            "all_other_applicable_controls_complete",
            "security_questionnaire_complete_due_to_conflict",
            "superseded_incomplete_record",
        ],
        "operators": ["any_of", "all_of"],
        "unresolved_judgments": [
            (
                "Mapping existing TRAIN free-text remedy actions onto typed "
                "action_type+condition_id pairs requires a substantive author "
                "judgment. That mapping is not invented here."
            ),
            (
                "Several TRAIN remedy skeletons cite the same source ID on "
                "multiple leaves. Leaf identity therefore cannot be recovered "
                "from source ID alone."
            ),
        ],
        "not_built": [
            "agent platform",
            "rules framework",
            "RL trainer",
        ],
    }


def typed_action_catalogue() -> list[dict[str, Any]]:
    return list(proposed_typed_action_contract()["action_catalogue"])


def score_typed_action(
    actions: Sequence[Mapping[str, Any]],
    *,
    expected_actions: Sequence[Mapping[str, Any]],
    expected_disposition: str,
    actual_disposition: str,
    observed_state: Mapping[str, str],
    catalogue: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Model-free typed-action checker for invented fixtures."""
    allowed_types = {row["action_type"] for row in (catalogue or typed_action_catalogue())}
    failures: list[str] = []
    if actual_disposition != expected_disposition:
        failures.append("disposition_mismatch")
    for action in actions:
        if action.get("action_type") not in allowed_types:
            failures.append("unknown_action_type")
        if action.get("action_type") == "waive_control":
            failures.append("unsafe_waiver")
        condition_id = action.get("condition_id")
        if (
            condition_id
            and observed_state.get(condition_id) == "unknown"
            and action.get("action_type") == "record_completion"
        ):
            failures.append("unknown_treated_as_false_or_complete")
        if action.get("action_type") == "record_completion" and any(
            row.get("action_type") == "request_evidence"
            and row.get("condition_id") == condition_id
            for row in expected_actions
        ):
            failures.append("request_evidence_recorded_as_completion")
    expected_keys = _action_multiset(expected_actions)
    actual_keys = _action_multiset(actions)
    if actual_keys - expected_keys:
        extra = actual_keys - expected_keys
        if any(key[0] == "waive_control" for key in extra):
            failures.append("unsafe_waiver")
    missing = expected_keys - actual_keys
    if missing:
        failures.append("missing_required_action")
    extras = actual_keys - expected_keys
    if extras and not missing and any(key in expected_keys for key in extras):
        failures.append("duplicate_does_not_complete")
    if missing:
        # Duplicates of a present action cannot satisfy a different missing action.
        present_types = {key[0:2] for key in actual_keys}
        if any(key[0:2] not in present_types for key in missing):
            failures.append("duplicate_does_not_complete")
    equivalent_order = _action_multiset(actions) == _action_multiset(
        list(actions)[::-1]
    ) or _action_multiset(actions) == expected_keys
    return {
        "pass": not failures,
        "failures": sorted(set(failures)),
        "equivalent_ordering_harmless": equivalent_order,
        "source_id_alone_used_as_identity": False,
        "free_text_entailment_claimed": False,
    }


def _action_multiset(
    actions: Sequence[Mapping[str, Any]],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            str(action.get("action_type")),
            str(action.get("condition_id") or ""),
            str(action.get("source_record_id") or ""),
        )
        for action in actions
    )


def qualify_observation(*, observed: bool, eligible: bool) -> str:
    if not eligible:
        return "not_evaluated_due_to_ineligibility"
    return "observed_pass" if observed else "observed_fail"


def training_decision_recommendation(
    overlay_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del overlay_summary
    return {
        "execute_now": False,
        "recommended_next": "bounded_supervised_baseline_on_typed_action_task",
        "grpo_status": "deferred",
        "dpo_status": "deferred",
        "do_not_construct_dpo_from_old_reward_differences": True,
        "do_not_treat_reward_one_as_approved": True,
        "usable_now": [
            "Frozen TRAIN reward specs: request-item IDs, exact decisions, authorized source IDs.",
            "Frozen v3 TRAIN policy_logic condition IDs and any_of/all_of trees.",
            "Frozen v4 TRAIN expected_fact_state established/unknown/unmet labels.",
        ],
        "candidates_for_review": [
            "Frozen TRAIN reference assessments, after an author maps free-text "
            "actions onto the typed catalogue. These are review candidates, "
            "not approved SFT rows.",
        ],
        "need_independently_established_targets": [
            "action_type for every TRAIN remedy leaf",
            "condition_id binding for leaves that currently share a source ID",
            "per-case grammars that also admit incorrect catalogue actions",
        ],
        "missing": [
            "A trustworthy typed-action verifier over author-established targets.",
            "A useful supervised baseline on that narrower task.",
            "A new dataset/budget/evaluation authorization for any later GRPO or DPO.",
        ],
        "historical_stops_unchanged": dict(HISTORICAL_STOPS),
        "later_training_requires_own_authorization": True,
    }


def diagnosis_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
