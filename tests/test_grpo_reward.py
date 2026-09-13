from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.grpo_data import load_parent_assets
from enterprise_memory_mlx.grpo_reward import (
    compile_reward_spec,
    mutate_output,
    perfect_output,
    score_completion,
)


def _cases(project_root: Path) -> list[dict]:
    assets = load_parent_assets(project_root)
    rows = [{**case, "source_protocol": "v3"} for case in assets["v3_cases"]]
    rows.extend({**case, "source_protocol": "v4"} for case in assets["v4_cases"])
    return rows


def _by_id(project_root: Path, case_id: str) -> dict:
    return next(case for case in _cases(project_root) if case["case_id"] == case_id)


def test_perfect_reference_receives_unit_reward(project_root: Path) -> None:
    for case in _cases(project_root):
        spec = compile_reward_spec(case)
        scored = score_completion(perfect_output(spec, case), spec)
        assert scored["reward"] == 1.0
        assert scored["full_machine_success"] is True
        assert scored["hard_zero"] is False


def test_invalid_json_and_provenance_hard_zero(project_root: Path) -> None:
    case = _by_id(project_root, "CTS3-QUAL-009")
    spec = compile_reward_spec(case)
    perfect = perfect_output(spec, case)
    invalid = score_completion(mutate_output(perfect, "invalid_json", spec), spec)
    assert invalid["reward"] == 0.0
    assert invalid["complete_valid_output"] is False
    unauthorized = score_completion(mutate_output(perfect, "unauthorized_source_id", spec), spec)
    assert unauthorized["reward"] == 0.0
    assert unauthorized["provenance_hard_fail"] is True
    missing = score_completion(mutate_output(perfect, "missing_request_item", spec), spec)
    assert missing["reward"] == 0.0
    duplicate = score_completion(mutate_output(perfect, "duplicate_request_id", spec), spec)
    assert duplicate["reward"] == 0.0


def test_decision_and_operator_mutations_change_named_components(project_root: Path) -> None:
    case = _by_id(project_root, "CTS3-QUAL-009")
    spec = compile_reward_spec(case)
    perfect = perfect_output(spec, case)
    baseline = score_completion(perfect, spec)
    wrong = score_completion(mutate_output(perfect, "wrong_current_decision", spec), spec)
    assert wrong["components"]["current_decisions"] < baseline["components"]["current_decisions"]
    flipped = score_completion(mutate_output(perfect, "any_of_to_all_of", spec), spec)
    assert flipped["components"]["remedy_logic"] < baseline["components"]["remedy_logic"]
    dropped = score_completion(mutate_output(perfect, "remove_valid_alternative", spec), spec)
    assert dropped["components"]["remedy_logic"] < baseline["components"]["remedy_logic"]
    extra = score_completion(mutate_output(perfect, "add_unsupported_alternative", spec), spec)
    assert extra["reward"] == 0.0
    assert extra["unsupported_route"] or extra["provenance_hard_fail"]


def test_proceed_when_information_or_referral_required_is_hard_fail(project_root: Path) -> None:
    unknown = _by_id(project_root, "CTS4-FRESH-01B")
    spec = compile_reward_spec(unknown)
    perfect = perfect_output(spec, unknown)
    proceed = score_completion(mutate_output(perfect, "unsafe_next_action", spec), spec)
    assert proceed["reward"] == 0.0
    assert proceed["unsafe_hard_fail"] is True


def test_reopened_complete_work_is_hard_fail(project_root: Path) -> None:
    case = _by_id(project_root, "CTS3-QUAL-001")
    spec = compile_reward_spec(case)
    perfect = perfect_output(spec, case)
    reopened = score_completion(mutate_output(perfect, "reopen_complete_work", spec), spec)
    assert reopened["reward"] == 0.0
    assert reopened["unsafe_hard_fail"] is True


def test_fact_state_mutations(project_root: Path) -> None:
    unknown = _by_id(project_root, "CTS4-FRESH-01B")
    spec = compile_reward_spec(unknown)
    perfect = perfect_output(spec, unknown)
    as_incomplete = score_completion(mutate_output(perfect, "unknown_as_incomplete", spec), spec)
    assert as_incomplete["components"]["current_decisions"] < 1.0
    claim = _by_id(project_root, "CTS4-FRESH-05A")
    claim_spec = compile_reward_spec(claim)
    claim_perfect = perfect_output(claim_spec, claim)
    authenticated = score_completion(
        mutate_output(claim_perfect, "requester_claim_as_authenticated", claim_spec),
        claim_spec,
    )
    assert authenticated["reward"] == 0.0
    attachment = _by_id(project_root, "CTS4-FRESH-01A")
    attachment_spec = compile_reward_spec(attachment)
    attachment_perfect = perfect_output(attachment_spec, attachment)
    failed = score_completion(
        mutate_output(attachment_perfect, "missing_document_as_failed_control", attachment_spec),
        attachment_spec,
    )
    assert failed["reward"] == 0.0
    verification = _by_id(project_root, "CTS4-FRESH-08A")
    verification_spec = compile_reward_spec(verification)
    waived = score_completion(
        mutate_output(
            perfect_output(verification_spec, verification),
            "waive_required_verification",
            verification_spec,
        ),
        verification_spec,
    )
    assert waived["reward"] == 0.0
    conflict = _by_id(project_root, "CTS4-FRESH-04A")
    conflict_spec = compile_reward_spec(conflict)
    resolved = score_completion(
        mutate_output(
            perfect_output(conflict_spec, conflict), "resolve_authority_conflict", conflict_spec
        ),
        conflict_spec,
    )
    assert resolved["reward"] == 0.0
    superseded = _by_id(project_root, "CTS4-FRESH-04B")
    superseded_spec = compile_reward_spec(superseded)
    retained = score_completion(
        mutate_output(
            perfect_output(superseded_spec, superseded), "retain_superseded_record", superseded_spec
        ),
        superseded_spec,
    )
    assert retained["components"]["current_decisions"] < 1.0


def test_invariance_and_anti_hacking(project_root: Path) -> None:
    case = _by_id(project_root, "CTS3-QUAL-009")
    spec = compile_reward_spec(case)
    perfect = perfect_output(spec, case)
    baseline = score_completion(perfect, spec)["reward"]
    assert (
        score_completion(mutate_output(perfect, "whitespace_only", spec), spec)["reward"]
        == baseline
    )
    assert (
        score_completion(mutate_output(perfect, "reorder_group_options", spec), spec)["reward"]
        == baseline
    )
    assert (
        score_completion(mutate_output(perfect, "reorder_source_records", spec), spec)["reward"]
        == baseline
    )
    duplicate = score_completion(mutate_output(perfect, "duplicate_correct_fields", spec), spec)
    assert duplicate["reward"] <= baseline
    extra = score_completion(mutate_output(perfect, "extra_valid_source_ids", spec), spec)
    assert extra["reward"] <= baseline
    long_out = score_completion(mutate_output(perfect, "long_output", spec), spec)
    assert long_out["reward"] <= baseline
    repeated = score_completion(mutate_output(perfect, "repeat_correct_action", spec), spec)
    assert repeated["reward"] <= baseline
    unsafe = score_completion(
        mutate_output(perfect, "correct_disposition_unsafe_action", spec), spec
    )
    assert unsafe["reward"] == 0.0 or unsafe["components"]["remedy_logic"] < 1.0


def test_all_of_swap_on_nested_case(project_root: Path) -> None:
    case = _by_id(project_root, "CTS3-QUAL-021")
    spec = compile_reward_spec(case)
    perfect = perfect_output(spec, case)
    flipped = score_completion(mutate_output(perfect, "all_of_to_any_of", spec), spec)
    assert flipped["components"]["remedy_logic"] < 1.0
    dropped = score_completion(mutate_output(perfect, "drop_mandatory_condition", spec), spec)
    assert dropped["dropped_mandatory"] or dropped["components"]["remedy_logic"] < 1.0


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid_json",
        "duplicate_request_id",
        "missing_request_item",
        "unauthorized_source_id",
        "wrong_current_decision",
        "proceed_vs_needs_information",
        "proceed_vs_refer_to_source",
        "any_of_to_all_of",
        "all_of_to_any_of",
        "remove_valid_alternative",
        "add_unsupported_alternative",
        "drop_mandatory_condition",
        "reopen_complete_work",
        "unknown_as_incomplete",
        "requester_claim_as_authenticated",
        "missing_document_as_failed_control",
        "waive_required_verification",
        "retain_superseded_record",
        "resolve_authority_conflict",
        "unsafe_next_action",
    ],
)
def test_named_mutations_are_defined(project_root: Path, mutation: str) -> None:
    case = _by_id(project_root, "CTS3-QUAL-009")
    spec = compile_reward_spec(case)
    mutated = mutate_output(perfect_output(spec, case), mutation, spec)
    scored = score_completion(mutated, spec)
    assert "reward" in scored
    assert 0.0 <= scored["reward"] <= 1.0
