from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from enterprise_memory_mlx.grpo_data import file_sha256
from enterprise_memory_mlx.typed_action_annotation import (
    HISTORICAL_STOPS,
    _action,
    build_drafts,
    load_train_cases,
)
from enterprise_memory_mlx.typed_action_annotation_v2 import (
    ADVISORY_FILES,
    ADVISORY_ZIP_SHA256,
    ANNOTATION_V2_ID,
    MAPPING_CORRECTIONS,
    V1_REVIEW_ZIP_SHA256,
    apply_mapping_corrections,
    catalogue_v2,
    closed_operation_arguments,
    closed_output_schema,
    focused_diff_v1_to_v2,
    schema_counterexamples,
    structural_identity_check,
    validate_drafts_v2,
    write_annotation_bundle_v2,
)

V1_DIR = "artifacts/qwen4b-typed-action-annotation-v1"
V2_DIR = "artifacts/qwen4b-typed-action-annotation-v2"


def _require_v1_artifacts(project_root: Path) -> None:
    """Skip honestly when the local, gitignored annotation artifacts are absent.

    These are model-authored working outputs, not public fixtures. On public
    CI they are absent by design; the structural logic is covered by the
    reconstruction-based tests below.
    """
    if not (project_root / V1_DIR / "typed-targets.draft.jsonl").is_file():
        pytest.skip("local annotation-v1 artifacts are not present")


def test_v1_packet_and_advisory_digests_are_pinned(project_root: Path) -> None:
    _require_v1_artifacts(project_root)
    v1_zip = project_root / V1_DIR / "qwen4b-typed-action-annotation-v1-review.zip"
    assert file_sha256(v1_zip) == V1_REVIEW_ZIP_SHA256


def test_historical_stops_and_reward_unchanged(project_root: Path) -> None:
    assert (
        file_sha256(project_root / "src/enterprise_memory_mlx/grpo_reward.py")
        == "e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b"
    )
    assert HISTORICAL_STOPS == {
        "grpo_v1": "stop_preflight",
        "landscape_v1": "abandon_standard_grpo_for_current_dataset",
    }


def _reconstructed_drafts_and_mappings(project_root: Path) -> tuple[list, list]:
    """Rebuild v1 drafts/mappings from repository knowledge files only."""
    cases = load_train_cases(project_root)
    drafts = build_drafts(cases)
    mappings = [
        {
            "case_id": row["case_id"],
            "condition_states": row["annotation"]["condition_states"],
            "mappings": row["annotation"]["mappings"],
            "unresolved": row["annotation"]["unresolved"],
        }
        for row in drafts
    ]
    return drafts, mappings


def test_mapping_corrections_apply_exactly_the_two_requested_rows(
    project_root: Path,
) -> None:
    _, originals = _reconstructed_drafts_and_mappings(project_root)
    corrected, applied = apply_mapping_corrections(originals)
    assert applied == ["ROW-013", "ROW-024"]
    changed = [
        row["case_id"]
        for row in originals
        if row != next(item for item in corrected if item["case_id"] == row["case_id"])
    ]
    assert changed == ["CTS3-QUAL-013", "CTS3-QUAL-024"]
    qual013 = next(row for row in corrected if row["case_id"] == "CTS3-QUAL-013")
    exception_state = next(
        item
        for item in qual013["condition_states"]
        if item["condition_id"] == "recorded_procurement_exception"
    )
    assert exception_state["observation_authority"] == "request_fact"
    qual024 = next(row for row in corrected if row["case_id"] == "CTS3-QUAL-024")
    exception_state_24 = next(
        item
        for item in qual024["condition_states"]
        if item["condition_id"] == "recorded_procurement_exception"
    )
    assert exception_state_24["status"] == "unknown"
    for row in corrected:
        if row["case_id"] not in {"CTS3-QUAL-013", "CTS3-QUAL-024"}:
            original = next(
                item for item in originals if item["case_id"] == row["case_id"]
            )
            assert row == original


def test_drafts_v2_keep_decisions_and_actions_but_fail_on_identity_problems(
    project_root: Path,
) -> None:
    drafts, _ = _reconstructed_drafts_and_mappings(project_root)
    cases = load_train_cases(project_root)
    result = validate_drafts_v2(
        [{**row, "annotation_id": ANNOTATION_V2_ID} for row in drafts], cases
    )
    assert result["failures"] == []
    assert result["structural_logic_validation"] == "established"


def test_negative_schema_fixtures_are_now_rejected() -> None:
    probes = schema_counterexamples()
    rejected = {row["id"] for row in probes if row["expect_rejected"]}
    assert rejected == {
        "SCHEMA-01-UNTYPED-EXTRA-INSTRUCTION",
        "SCHEMA-02-CONFLICTING-DUPLICATED-CONDITION",
        "SCHEMA-03-EMPTY-BLOCKER-OBJECT",
        "SCHEMA-04-DUPLICATE-REQUEST",
        "SCHEMA-05-MISSING-OPERATION-ARGUMENTS",
    }
    assert all(row["rejected"] for row in probes if row["expect_rejected"])
    passing = {row["id"] for row in probes if not row["expect_rejected"]}
    assert passing == {
        "POSITIVE-BASE",
        "POSITIVE-ATTACHMENT",
        "ADMISSIBLE-WRONG-1",
        "ADMISSIBLE-WRONG-2",
    }
    assert all(not row["rejected"] for row in probes if not row["expect_rejected"])


def test_grammar_stays_answer_blind() -> None:
    schema = closed_output_schema()
    assert schema["must_not_encode"] == [
        "the correct current decision",
        "the correct condition status",
        "only the correct remedy",
        "only the safe action for a particular case",
    ]
    catalogue = catalogue_v2()
    assert catalogue["outputs_are_proposed_actions_only"] is True
    assert catalogue["assessor_record_change_authority"] is False
    assert catalogue["requested_actions_do_not_establish_outcomes"] is True


def test_action_result_boundary_is_explicit() -> None:
    from enterprise_memory_mlx.typed_action_annotation_v2 import ACTION_RESULT_STATEMENT

    assert "proposed next action only" in ACTION_RESULT_STATEMENT
    assert "does not authorize the assessor to change records" in ACTION_RESULT_STATEMENT


def test_duplicate_request_items_and_inconsistent_ids_fail() -> None:
    base = {
        "assessments": [
            {
                "request_item_id": "purchase_order_issue",
                "decision": "proceed",
                "current_blockers": [],
                "missing_information": [],
                "exceptions": [],
                "remedy": None,
                "supporting_source_ids": ["PROC-VEND-001"],
            }
        ]
    }

    duplicate = json.loads(json.dumps(base))
    duplicate["assessments"].append(json.loads(json.dumps(base["assessments"][0])))
    failures = structural_identity_check(duplicate, ["purchase_order_issue"], ["PROC-VEND-001"])
    assert any("duplicate_request_item" in item for item in failures)
    unsupplied = json.loads(json.dumps(base))
    unsupplied["assessments"][0]["supporting_source_ids"] = ["NOT-SUPPLIED-01"]
    failures = structural_identity_check(unsupplied, ["purchase_order_issue"], ["PROC-VEND-001"])
    assert any("supporting_source_id_not_supplied" in item for item in failures)
    mismatched = json.loads(json.dumps(base))
    mismatched["assessments"][0]["remedy"] = _action(
        "perform_unmet_control", "identity_check_complete", "PROC-VEND-001"
    )
    mismatched["assessments"][0]["remedy"]["arguments"]["condition_id"] = (
        "sanctions_check_complete"
    )
    failures = structural_identity_check(mismatched, ["purchase_order_issue"], ["PROC-VEND-001"])
    assert any("condition_id_mismatch_top_level_vs_arguments" in item for item in failures)


def test_conflicting_state_ids_are_required_for_conflict_resolution() -> None:
    action = _action(
        "resolve_record_conflict",
        "security_questionnaire_complete",
        "PROC-VEND-001",
        conflicting_state_ids=["WF-04A-SEC-1", "WF-04A-SEC-2"],
    )
    assert action["arguments"]["conflicting_state_ids"] == ["WF-04A-SEC-1", "WF-04A-SEC-2"]
    closed = closed_operation_arguments(action)
    assert closed["conflicting_state_ids"] == ["WF-04A-SEC-1", "WF-04A-SEC-2"]
    instance = {
        "assessments": [
            {
                "request_item_id": "purchase_order_issue",
                "decision": "needs_information",
                "current_blockers": [],
                "missing_information": [{"condition_id": "security_questionnaire_complete"}],
                "exceptions": [],
                "remedy": action,
                "supporting_source_ids": ["PROC-VEND-001"],
            }
        ]
    }
    failures = structural_identity_check(
        instance,
        ["purchase_order_issue"],
        ["PROC-VEND-001", "WF-04A-SEC-1", "WF-04A-SEC-2"],
    )
    assert failures == []
    too_few = _action(
        "resolve_record_conflict",
        "security_questionnaire_complete",
        "PROC-VEND-001",
        conflicting_state_ids=["WF-04A-SEC-1"],
    )
    instance["assessments"][0]["remedy"] = too_few
    failures = structural_identity_check(
        instance,
        ["purchase_order_issue"],
        ["PROC-VEND-001", "WF-04A-SEC-1", "WF-04A-SEC-2"],
    )
    assert any("conflicting_state_ids_incomplete_or_not_supplied" in item for item in failures)


def test_bundle_v2_writes_all_deliverables(
    tmp_path: Path, project_root: Path
) -> None:
    _require_v1_artifacts(project_root)
    shutil.copytree(project_root / "knowledge", tmp_path / "knowledge")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "src/enterprise_memory_mlx").mkdir(parents=True)
    for name in ("grpo_reward.py", "typed_action_annotation.py"):
        shutil.copy2(
            project_root / f"src/enterprise_memory_mlx/{name}",
            tmp_path / f"src/enterprise_memory_mlx/{name}",
        )
    # The writer validates the pinned v1 review packet and reads the v1
    # drafts/mappings; stage copies so nothing is written into the real tree.
    (tmp_path / V1_DIR).mkdir(parents=True)
    for name in (
        "qwen4b-typed-action-annotation-v1-review.zip",
        "typed-targets.draft.jsonl",
        "source-to-action-mappings.jsonl",
    ):
        shutil.copy2(
            project_root / V1_DIR / name,
            tmp_path / V1_DIR / name,
        )
    summary = write_annotation_bundle_v2(tmp_path)
    assert summary["mapping_corrections"] == ["ROW-013", "ROW-024"]
    assert summary["training_targets_identical_to_v1"] is True
    assert summary["counts"]["drafts_validated"] == 24
    assert summary["counts"]["negative_fixtures_rejected"] == 5
    assert summary["counts"]["admissible_wrong_examples_still_pass"] == 4
    assert summary["human_attested"] is False
    assert summary["training_authorized"] is False
    destination = tmp_path / V2_DIR
    for name in (
        "review-record.json",
        "action-catalogue.json",
        "output-schema.json",
        "typed-targets.draft.jsonl",
        "source-to-action-mappings.jsonl",
        "mapping-corrections.json",
        "focused-diff.json",
        "validation-results.json",
        "SUMMARY.json",
        "source-snapshot.json",
    ):
        assert (destination / name).is_file(), name


def test_focused_diff_reports_only_the_two_mapping_changes(project_root: Path) -> None:
    v1_drafts, originals = _reconstructed_drafts_and_mappings(project_root)
    corrected, _ = apply_mapping_corrections(originals)
    v2_drafts = [
        {**row, "annotation_id": ANNOTATION_V2_ID} for row in v1_drafts
    ]
    diff = focused_diff_v1_to_v2(v1_drafts, v2_drafts, originals, corrected)
    assert diff["training_targets_identical"] is True
    assert diff["grammar_encoding_correct_choices"] is False
    assert [row["case_id"] for row in diff["mapping_only_changes"]] == [
        "CTS3-QUAL-013",
        "CTS3-QUAL-024",
    ]


def test_advisory_files_are_recorded_not_trusted() -> None:
    assert len(ADVISORY_FILES) == 6
    assert ADVISORY_ZIP_SHA256 == (
        "a7a7d4facbc00595d8b13b2420b25ec2e416ae956a9601537c5360d09b5d725d"
    )
    assert len(MAPPING_CORRECTIONS) == 2
    assert all(not row["training_target_changed"] for row in MAPPING_CORRECTIONS)