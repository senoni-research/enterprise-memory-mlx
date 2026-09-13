from __future__ import annotations

import json

import pytest

from enterprise_memory_mlx.learning_mechanics import (
    APPROVAL_RECORD_SHA256,
    EXPECTED_COMPLETE_TOKENS,
    FROZEN_SETTINGS,
    MEMORY_OPTIMIZATION_AUTHORIZATION_RELATIVE,
    MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256,
    PROPOSED_MAX_SEQ_LENGTH,
    RETRY_AUTHORIZATION_RELATIVE,
    RETRY_AUTHORIZATION_SHA256,
    build_batch,
    enable_gradient_checkpointing,
    load_approval,
    score_review,
    serialize_training_rows,
    sha256_file,
    verify_bound_pairs,
)
from enterprise_memory_mlx.learning_mechanics_exact import (
    default_loss_padding_contract,
    supervised_hidden_span,
)


def test_frozen_settings_are_constant_lr_and_do_not_inherit_smoke_schedule() -> None:
    assert FROZEN_SETTINGS["learning_rate"] == 5e-5
    assert FROZEN_SETTINGS["learning_rate_schedule"] == "constant"
    assert FROZEN_SETTINGS["do_not_inherit_smoke_warmup_cosine"] is True
    assert FROZEN_SETTINGS["measured_optimizer_updates"] == 40
    assert FROZEN_SETTINGS["max_seq_length"] == 35406
    assert EXPECTED_COMPLETE_TOKENS["DEV-004"] == 35406
    assert FROZEN_SETTINGS["batch_size"] == 1
    assert FROZEN_SETTINGS["gradient_accumulation_steps"] == 1


@pytest.mark.private_data
def test_approval_and_bound_hashes(project_root, private_approval_path) -> None:
    del private_approval_path
    approval = load_approval(project_root)
    assert approval["record_id"].endswith("owner-approval-development-reference-v1")
    verified = verify_bound_pairs(project_root, approval)
    assert [row["case_id"] for row in verified] == [
        "DEV-001",
        "DEV-002",
        "DEV-003",
        "DEV-004",
        "DEV-005",
    ]
    assert verified[3]["target_sha256"] == (
        "680044659f4ad9832e8c828a842599e22da30e697ed829b4e7dc5ffa6b9ab966"
    )


def test_serialize_training_rows_omits_audit_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    target_path = tmp_path / "target.json"
    input_path.write_text(
        json.dumps(
            {
                "system_prompt": "Synthetic system prompt.",
                "user_prompt": "Synthetic user prompt.",
            }
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                "proposed_target_review": {
                    "case_id": "DEV-002",
                    "requirement_assessments": [{"requirement_id": "R1", "status": "met"}],
                }
            }
        ),
        encoding="utf-8",
    )
    rows = serialize_training_rows(
        tmp_path,
        [
            {
                "case_id": "DEV-002",
                "input_path": str(input_path),
                "target_path": str(target_path),
                "input_sha256": "a" * 64,
                "target_sha256": "b" * 64,
            }
        ],
    )
    assistant = rows[0]["messages"][2]["content"]
    payload = json.loads(assistant)
    assert payload["case_id"] == "DEV-002"
    assert "requirement_assessments" in payload
    assert "citation_appendix" not in payload
    assert "owner_approved" not in payload
    assert "training_eligible" not in payload
    assert "candidate_id" not in payload


def test_serialize_training_rows_rejects_leaked_audit_keys(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    target_path = tmp_path / "target.json"
    input_path.write_text(
        json.dumps({"system_prompt": "sys", "user_prompt": "user"}),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                "proposed_target_review": {
                    "case_id": "DEV-002",
                    "owner_approved": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="leaked audit metadata"):
        serialize_training_rows(
            tmp_path,
            [
                {
                    "case_id": "DEV-002",
                    "input_path": str(input_path),
                    "target_path": str(target_path),
                    "input_sha256": "a" * 64,
                    "target_sha256": "b" * 64,
                }
            ],
        )


@pytest.mark.private_data
def test_assistant_target_is_only_proposed_review(project_root, private_approval_path) -> None:
    del private_approval_path
    approval = load_approval(project_root)
    rows = serialize_training_rows(project_root, verify_bound_pairs(project_root, approval))
    for row in rows:
        assistant = row["messages"][2]["content"]
        payload = json.loads(assistant)
        assert payload["case_id"] == row["case_id"]
        assert "requirement_assessments" in payload
        assert "citation_appendix" not in payload
        assert "owner_approved" not in payload
        assert "training_eligible" not in payload
        assert "candidate_id" not in payload


def test_batch_uses_per_example_length_not_global_cap() -> None:
    tokens = list(range(100))
    built = build_batch(tokens, offset=40, max_seq_length=PROPOSED_MAX_SEQ_LENGTH)
    assert built["unpadded_length"] == 100
    assert built["padded_shape"][0] == 1
    assert built["padded_shape"][1] < 200
    assert built["padded_shape"][1] != PROPOSED_MAX_SEQ_LENGTH
    assert built["padding_positions"] == built["padded_shape"][1] - 100
    assert built["lengths"] == [[40, 99]]
    with pytest.raises(RuntimeError, match="truncate"):
        build_batch(
            list(range(PROPOSED_MAX_SEQ_LENGTH + 1)),
            offset=10,
            max_seq_length=PROPOSED_MAX_SEQ_LENGTH,
        )


def test_gradient_checkpoint_wrapper_is_applied_once_per_layer_type() -> None:
    class Layer:
        pass

    class Model:
        layers = [Layer(), Layer(), Layer()]

    calls = []
    assert enable_gradient_checkpointing(Model(), calls.append) == [
        f"{Layer.__module__}.{Layer.__qualname__}"
    ]
    assert enable_gradient_checkpointing(Model(), calls.append) == [
        f"{Layer.__module__}.{Layer.__qualname__}"
    ]
    assert len(calls) == 1


def test_score_review_does_not_flag_approved_negative_advice() -> None:
    review = {
        "requirement_assessments": [
            {"requirement_id": "R5", "status": "insufficient_evidence"},
            {"requirement_id": "R6", "status": "not_met"},
        ],
        "overall_disposition": "refer_to_source",
        "next_actions": [
            "Do not execute the experiment until that authority is resolved.",
            "A request for evidence is not permission to run the models.",
        ],
        "unsupported_claims_or_invented_requirements": [],
    }
    scored = score_review(
        case_id="DEV-004",
        approved_labels={"R5": "insufficient_evidence", "R6": "not_met"},
        approved_disposition="refer_to_source",
        review=review,
        schema_valid=True,
    )
    assert scored["invented_or_harmful_actions"] == []
    assert scored["full_task_success"] is True


def test_score_review_rejects_harmful_advice_as_full_success() -> None:
    approved = {"R1": "met", "R5": "insufficient_evidence"}
    review = {
        "requirement_assessments": [
            {"requirement_id": "R1", "status": "met"},
            {"requirement_id": "R5", "status": "insufficient_evidence"},
        ],
        "overall_disposition": "needs_information",
        "next_actions": ["Execute the experiment before authority is resolved."],
        "unsupported_claims_or_invented_requirements": [],
    }
    scored = score_review(
        case_id="DEV-001",
        approved_labels=approved,
        approved_disposition="needs_information",
        review=review,
        schema_valid=True,
    )
    assert scored["label_matches"] == 2
    assert scored["disposition_agrees"] is True
    assert scored["full_task_success"] is False


def test_approval_binding_constant() -> None:
    assert APPROVAL_RECORD_SHA256 == (
        "af6a26c0cf18f050751410cabd29d355722717ee4cd237f8c40ade9dbe610371"
    )


@pytest.mark.private_data
def test_retry_authorization_binding(project_root) -> None:
    path = project_root / RETRY_AUTHORIZATION_RELATIVE
    if not path.is_file():
        pytest.skip("private retry authorization is not present")
    assert sha256_file(path) == RETRY_AUTHORIZATION_SHA256


@pytest.mark.private_data
def test_memory_optimization_authorization_binding(project_root) -> None:
    path = project_root / MEMORY_OPTIMIZATION_AUTHORIZATION_RELATIVE
    if not path.is_file():
        pytest.skip("private memory-optimization authorization is not present")
    assert sha256_file(path) == MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256


def test_supervised_hidden_span_accounts_for_next_token_shift() -> None:
    span = supervised_hidden_span(33494, 35406)
    assert span["supervised_token_count"] == 1912
    assert span["first_supervised_hidden_index"] == 33493
    assert span["first_hidden_index_equals_prompt_length"] is False
    assert span["first_supervised_target_index"] == 33494
    assert span["last_supervised_target_index_inclusive"] == 35405


def test_old_default_loss_length_can_include_first_padding_token() -> None:
    contract = default_loss_padding_contract(
        offset=40, unpadded_length=80, padded_length=97
    )
    assert contract["includes_first_padding_token_when_padded"] is True
    assert contract["corrected_upper_bound"] == 79
    unpadded_cap = default_loss_padding_contract(
        offset=33494, unpadded_length=35406, padded_length=35406
    )
    assert unpadded_cap["includes_first_padding_token_when_padded"] is False


def test_duplicate_unmarked_checkpoint_wrap_is_rejected() -> None:
    class Layer:
        pass

    def checkpointed_fn(self, *args, **kwargs):
        return None

    Layer.__call__ = checkpointed_fn

    class Model:
        layers = [Layer()]

    with pytest.raises(RuntimeError, match="Duplicate checkpoint"):
        enable_gradient_checkpointing(Model(), lambda layer: None)
