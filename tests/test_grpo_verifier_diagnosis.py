from __future__ import annotations

import json
from pathlib import Path

from enterprise_memory_mlx.grpo_reward import score_completion
from enterprise_memory_mlx.grpo_verifier_diagnosis import (
    DIAGNOSIS_ID,
    HISTORICAL_STOPS,
    LANDSCAPE_REWARD_SHA256,
    diagnose_completion,
    drop_named_alternative,
    invented_action_spec,
    invented_all_of_two_spec,
    invented_any_of_three_spec,
    invented_two_item_spec,
    overlay_row,
    proposed_typed_action_contract,
    qualify_observation,
    reward_matches_landscape_binding,
    score_typed_action,
    substitute_action_text,
    summarize_overlay,
    swap_group_operator,
    training_decision_recommendation,
    typed_action_catalogue,
    valid_action_output,
    valid_all_of_two_output,
    valid_any_of_three_output,
    valid_two_item_output,
)


def test_landscape_reward_binding_is_unchanged() -> None:
    assert reward_matches_landscape_binding() is True
    assert LANDSCAPE_REWARD_SHA256 == (
        "e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b"
    )
    assert HISTORICAL_STOPS == {
        "grpo_v1": "stop_preflight",
        "landscape_v1": "abandon_standard_grpo_for_current_dataset",
    }


def test_action_meaning_substitution_is_invisible_to_old_reward() -> None:
    spec = invented_action_spec()
    correct = valid_action_output(spec)
    baseline = score_completion(correct, spec)
    assert baseline["reward"] == 1.0
    assert baseline["full_machine_success"] is True
    mutated = substitute_action_text(
        correct,
        "Skip mandatory verification V-1 and record it complete",
    )
    payload = json.loads(mutated)
    original = json.loads(correct)
    assert payload["assessments"][0]["decision"] == original["assessments"][0]["decision"]
    assert payload["assessments"][0]["remedy"]["source_record_id"] == (
        original["assessments"][0]["remedy"]["source_record_id"]
    )
    assert payload["assessments"][0]["remedy"]["node_type"] == "action"
    scored = score_completion(mutated, spec)
    assert scored["reward"] == 1.0
    assert scored["full_machine_success"] is True
    assert scored["unsafe_hard_fail"] is False
    assert scored["provenance_hard_fail"] is False
    diagnosis = diagnose_completion(mutated, spec)
    assert diagnosis["action_meaning_not_assessed"] is True
    assert diagnosis["remedy_structure_mismatch"] is False
    assert diagnosis["exact_decision_mismatch"] is False


def test_missing_any_of_alternative_can_retain_unit_old_score() -> None:
    spec = invented_any_of_three_spec()
    complete = valid_any_of_three_output(spec)
    baseline = score_completion(complete, spec)
    assert baseline["reward"] == 1.0
    assert baseline["full_machine_success"] is True
    dropped = drop_named_alternative(complete, "SRC-ALT-C")
    payload = json.loads(dropped)
    assert payload["assessments"][0]["decision"] == spec["decisions"]["invoice_processing"]
    assert len(payload["assessments"][0]["remedy"]["options"]) == 2
    scored = score_completion(dropped, spec)
    flags = scored["remedy_flags"][0]
    assert flags["missing_alternatives"] == 1
    assert flags["score"] == 1.0
    assert scored["unsafe_hard_fail"] is False
    assert scored["full_machine_success"] is True
    assert scored["reward"] == 1.0
    diagnosis = diagnose_completion(dropped, spec)
    assert diagnosis["omitted_alternatives"] == 1
    assert diagnosis["remedy_structure_mismatch"] is True
    assert diagnosis["corrected_structure"]["dropped_mandatory"] is False


def test_all_of_to_any_of_swap_leaves_unchanged() -> None:
    spec = invented_all_of_two_spec()
    complete = valid_all_of_two_output(spec)
    swapped = swap_group_operator(complete, "any_of")
    payload = json.loads(swapped)
    assert [row["source_record_id"] for row in payload["assessments"][0]["remedy"]["options"]] == [
        "SRC-MAND-A",
        "SRC-MAND-B",
    ]
    scored = score_completion(swapped, spec)
    flags = scored["remedy_flags"][0]
    assert flags["operator_mismatch"] is True
    assert flags["missing_alternatives"] == 0
    assert scored["components"]["remedy_logic"] == 0.5
    assert scored["unsafe_hard_fail"] is False
    assert scored["full_machine_success"] is False
    diagnosis = diagnose_completion(swapped, spec)
    assert diagnosis["remedy_structure_mismatch"] is True
    assert diagnosis["corrected_structure"]["operator_mismatch"] is True


def test_old_provenance_flag_conflates_distinct_failures() -> None:
    spec = invented_two_item_spec()
    complete = valid_two_item_output(spec)
    malformed = score_completion("{", spec)
    missing = json.loads(complete)
    missing["assessments"] = missing["assessments"][:1]
    missing_text = json.dumps(missing, sort_keys=True)
    extra = json.loads(complete)
    extra["assessments"].append(
        {
            **extra["assessments"][0],
            "request_item_id": "invented_extra_item",
        }
    )
    extra_text = json.dumps(extra, sort_keys=True)
    unauthorized = json.loads(complete)
    unauthorized["assessments"][1]["evidence"].append(
        {"record_id": "NOT-A-REAL-SOURCE", "claim": "invented citation"}
    )
    unauthorized_text = json.dumps(unauthorized, sort_keys=True)
    malformed_scored = malformed
    missing_scored = score_completion(missing_text, spec)
    extra_scored = score_completion(extra_text, spec)
    unauthorized_scored = score_completion(unauthorized_text, spec)
    assert malformed_scored["provenance_hard_fail"] is True
    assert missing_scored["provenance_hard_fail"] is True
    assert extra_scored["provenance_hard_fail"] is True
    assert unauthorized_scored["provenance_hard_fail"] is True
    malformed_diag = diagnose_completion("{", spec)
    missing_diag = diagnose_completion(missing_text, spec)
    extra_diag = diagnose_completion(extra_text, spec)
    unauthorized_diag = diagnose_completion(unauthorized_text, spec)
    assert malformed_diag["json_syntax_failure"] is True
    assert malformed_diag["source_membership_assessment"] == "not_assessable"
    assert malformed_diag["positive_provenance_success"] is False
    assert malformed_diag["invented_citation"] is False
    assert missing_diag["request_item_coverage_failure"] is True
    assert extra_diag["request_item_coverage_failure"] is True
    assert unauthorized_diag["unauthorized_source_id_use"] is True
    assert unauthorized_diag["json_syntax_failure"] is False
    assert unauthorized_diag["schema_failure"] is False
    assert {
        malformed_scored["provenance_hard_fail"],
        missing_scored["provenance_hard_fail"],
        extra_scored["provenance_hard_fail"],
        unauthorized_scored["provenance_hard_fail"],
    } == {True}


def test_overlay_preserves_original_reward_and_reports_overlaps() -> None:
    spec = invented_any_of_three_spec()
    complete = valid_any_of_three_output(spec)
    dropped = drop_named_alternative(complete, "SRC-ALT-C")
    scored = score_completion(dropped, spec)
    row = overlay_row(
        {
            "case_id": spec["case_id"],
            "regime_id": "R0_CANONICAL",
            "replicate": 0,
            "raw_output": dropped,
            "raw_output_sha256": "fixture",
            "reward": scored["reward"],
            "full_machine_success": scored["full_machine_success"],
            "parse_status": scored["parse_status"],
            "provenance_hard_fail": scored["provenance_hard_fail"],
            "unsafe_hard_fail": scored["unsafe_hard_fail"],
            "hard_zero": scored["hard_zero"],
            "complete_valid_output": scored["complete_valid_output"],
            "truncated": False,
            "remedy_flags": scored["remedy_flags"],
            "components": scored["components"],
        },
        spec,
    )
    assert row["original_reward"] == 1.0
    assert row["original_full_machine_success"] is True
    assert row["structural_contradiction_despite_reward_one"] is True
    assert row["admitted_to_training"] is False
    assert row["semantic_correctness_inferred"] is False
    summary = summarize_overlay([row])
    assert summary["counts"]["reward_one_with_structural_contradiction"] == 1
    assert summary["counts"]["reward_one_with_omitted_alternatives"] == 1
    assert summary["winner_selected"] is False


def test_typed_action_contract_distinguishes_same_source_actions() -> None:
    contract = proposed_typed_action_contract()
    assert contract["source_id_is_not_action_identity"] is True
    assert contract["catalogue_independent_of_correct_answer"] is True
    catalogue = typed_action_catalogue()
    types = {row["action_type"] for row in catalogue}
    assert types == {
        "request_evidence",
        "record_completion",
        "waive_control",
        "obtain_alternative_authorization",
        "refer_to_source",
    }
    same_source_request = [
        {
            "action_type": "request_evidence",
            "condition_id": "security_questionnaire_complete",
            "source_record_id": "SRC-POLICY-001",
        }
    ]
    same_source_record = [
        {
            "action_type": "record_completion",
            "condition_id": "security_questionnaire_complete",
            "source_record_id": "SRC-POLICY-001",
        }
    ]
    expected = same_source_request
    request_score = score_typed_action(
        same_source_request,
        expected_actions=expected,
        expected_disposition="needs_information",
        actual_disposition="needs_information",
        observed_state={"security_questionnaire_complete": "unknown"},
    )
    record_score = score_typed_action(
        same_source_record,
        expected_actions=expected,
        expected_disposition="needs_information",
        actual_disposition="needs_information",
        observed_state={"security_questionnaire_complete": "unknown"},
    )
    assert request_score["pass"] is True
    assert record_score["pass"] is False
    assert "request_evidence_recorded_as_completion" in record_score["failures"]


def test_typed_action_unsafe_waiver_and_missing_controls() -> None:
    expected = [
        {
            "action_type": "obtain_alternative_authorization",
            "condition_id": "approved_purchase_order_quoted",
            "source_record_id": "SRC-ALT-A",
        },
        {
            "action_type": "obtain_alternative_authorization",
            "condition_id": "recorded_procurement_exception",
            "source_record_id": "SRC-ALT-B",
        },
    ]
    waived = score_typed_action(
        [
            {
                "action_type": "waive_control",
                "condition_id": "approved_purchase_order_quoted",
                "source_record_id": "SRC-ALT-A",
            }
        ],
        expected_actions=expected,
        expected_disposition="needs_action",
        actual_disposition="needs_action",
        observed_state={"approved_purchase_order_quoted": "unmet"},
    )
    assert waived["pass"] is False
    assert "unsafe_waiver" in waived["failures"]
    missing = score_typed_action(
        expected[:1],
        expected_actions=expected,
        expected_disposition="needs_action",
        actual_disposition="needs_action",
        observed_state={},
    )
    assert missing["pass"] is False
    assert "missing_required_action" in missing["failures"]
    reordered = score_typed_action(
        list(reversed(expected)),
        expected_actions=expected,
        expected_disposition="needs_action",
        actual_disposition="needs_action",
        observed_state={},
    )
    assert reordered["pass"] is True
    assert reordered["equivalent_ordering_harmless"] is True
    duplicate = score_typed_action(
        [expected[0], expected[0]],
        expected_actions=expected,
        expected_disposition="needs_action",
        actual_disposition="needs_action",
        observed_state={},
    )
    assert duplicate["pass"] is False
    assert "duplicate_does_not_complete" in duplicate["failures"]


def test_qualification_terms_and_deferred_training() -> None:
    assert qualify_observation(observed=True, eligible=True) == "observed_pass"
    assert qualify_observation(observed=False, eligible=True) == "observed_fail"
    assert (
        qualify_observation(observed=True, eligible=False)
        == "not_evaluated_due_to_ineligibility"
    )
    rec = training_decision_recommendation()
    assert rec["execute_now"] is False
    assert rec["grpo_status"] == "deferred"
    assert rec["dpo_status"] == "deferred"
    assert rec["do_not_treat_reward_one_as_approved"] is True
    assert rec["historical_stops_unchanged"] == HISTORICAL_STOPS
    assert DIAGNOSIS_ID.endswith("verifier-diagnosis-v1")


def test_diagnosis_module_does_not_import_dev_test_cases(project_root: Path) -> None:
    source = (
        project_root / "src/enterprise_memory_mlx/grpo_verifier_diagnosis.py"
    ).read_text(encoding="utf-8")
    assert "CTS3-QUAL-014" not in source
    assert "CTS3-QUAL-015" not in source
    assert "CTS3-QUAL-020" not in source
    assert "CTS4-FRESH-01A" not in source
    assert "CTS4-FRESH-08A" not in source
