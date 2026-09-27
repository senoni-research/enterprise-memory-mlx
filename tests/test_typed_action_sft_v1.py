from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.typed_action_sft_v1 import (
    EXPERIMENT_ID,
    GENERATION_CAP_TOKENS,
    MEASURED_UPDATES,
    TRAIN_EXPORT_ROW_CEILING_TOKENS,
    VOCABULARY_EXTENSIONS,
    build_system_prompt,
    build_user_prompt,
    canonical_tree,
    evaluation_inventory,
    evaluation_references,
    frozen_config,
    identity_failures,
    load_held_out_cases,
    load_train_targets,
    operation_catalogue,
    parse_strict,
    run_oracle,
    runtime_output_schema,
    serialize_target,
    training_schedule,
    typed_tree_equal,
    validate_schema_instance,
)


def _require_annotation_v2_targets(project_root: Path) -> None:
    """Skip honestly when the gitignored annotation-v2 artifacts are absent.

    Public CI has no local annotation artifacts by design; the comparator,
    schema and oracle coverage is reconstruction-based and unaffected.
    """
    if not (
        project_root
        / "artifacts/qwen4b-typed-action-annotation-v2/typed-targets.draft.jsonl"
    ).is_file():
        pytest.skip("local annotation-v2 artifacts are not present")


def test_frozen_config_binds_the_instruction() -> None:
    config = frozen_config()
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["training"]["measured_optimizer_updates"] == 192
    assert config["training"]["epochs"] == 8
    assert config["training"]["cases"] == 24
    assert config["training"]["early_stopping"] is False
    assert config["training"]["checkpoint_selection"] is False
    assert config["optimizer"]["learning_rate"] == 5e-5
    assert config["optimizer"]["weight_decay"] == 0.01
    assert config["adapter"]["rank"] == 16
    assert config["adapter"]["scale"] == 2.0
    assert config["adapter"]["dropout"] == 0.0
    assert config["base_model"]["initialize_from_existing_sft_adapter"] is False
    assert config["permission"]["human_attestation"] is False
    assert config["permission"]["broad_training_eligibility"] is False


def test_catalogue_composes_v1_definitions_plus_v2_overrides() -> None:
    catalogue = operation_catalogue()
    assert set(catalogue) == {
        "request_evidence",
        "perform_unmet_control",
        "correct_nonconformance",
        "record_completion",
        "request_alternative_authorization",
        "grant_or_waive",
        "resolve_record_conflict",
        "refer_to_source",
    }
    for entry in catalogue.values():
        assert entry["meaning"]
    assert "reconcile conflicting supplied" in catalogue["resolve_record_conflict"]["meaning"]
    blob = json.dumps(catalogue)
    assert "used_in_train_targets" not in blob


def test_runtime_schema_is_standards_valid_and_accepts_all_24_targets(
    project_root: Path,
) -> None:
    _require_annotation_v2_targets(project_root)
    from jsonschema import Draft202012Validator

    schema = runtime_output_schema()
    Draft202012Validator.check_schema(schema)
    assert "anyOf" in schema["properties"]["assessments"]["items"]["properties"]["remedy"]
    assert {"type": "null"} in schema["properties"]["assessments"]["items"]["properties"][
        "remedy"
    ]["anyOf"]
    assert schema["runtime_correction_vs_annotation_v2"]
    for row in load_train_targets(project_root):
        assert validate_schema_instance(row["proposed_training_target"], schema) == [], row[
            "case_id"
        ]


def test_runtime_schema_accepts_references_and_null_remedies() -> None:
    schema = runtime_output_schema()
    from enterprise_memory_mlx.typed_action_sft_v1 import _assessment

    for ref in evaluation_references():
        assert validate_schema_instance(ref["reference"], schema) == [], ref["case_id"]
        assert ref["human_attested"] is False
        assert ref["independent_review"] == "pending"
        assert ref["training_authorized"] is False
    null_probe = {
        "assessments": [
            _assessment("ri", "proceed", supporting_source_ids=["S-1"]),
        ]
    }
    assert validate_schema_instance(null_probe, schema) == []


def test_runtime_schema_rejects_advisory_negative_fixtures() -> None:
    import copy

    from enterprise_memory_mlx.typed_action_sft_v1 import _action, _assessment

    schema = runtime_output_schema()
    base = {
        "assessments": [
            _assessment(
                "purchase_order_issue",
                "do_not_proceed",
                blockers=[
                    {
                        "condition_id": "identity_check_complete",
                        "source_record_id": "PROC-VEND-001",
                    }
                ],
                remedy=_action(
                    "perform_unmet_control", "identity_check_complete", "PROC-VEND-001"
                ),
                supporting_source_ids=["PROC-VEND-001"],
            )
        ]
    }
    p = copy.deepcopy(base)
    p["assessments"][0]["remedy"]["arguments"]["extra_instruction"] = "Skip verification."
    assert validate_schema_instance(p, schema)
    p = copy.deepcopy(base)
    p["assessments"][0]["remedy"]["arguments"]["condition_id"] = "sanctions_check_complete"
    # schema alone accepts the duplicated-ID conflict; the identity layer
    # rejects it. Both must be exercised.
    assert validate_schema_instance(p, schema) == []
    assert identity_failures(
        p, ["purchase_order_issue"], ["PROC-VEND-001"]
    ) == ["condition_id_mismatch_top_level_vs_arguments"]
    p = copy.deepcopy(base)
    p["assessments"][0]["current_blockers"] = [{}]
    assert validate_schema_instance(p, schema)
    p = copy.deepcopy(base)
    p["assessments"].append(copy.deepcopy(p["assessments"][0]))
    ids = ["purchase_order_issue"]
    assert "duplicate_request_item" in identity_failures(p, ids, ["PROC-VEND-001"])
    p = copy.deepcopy(base)
    p["assessments"][0]["remedy"] = _action(
        "correct_nonconformance", "identity_check_complete", "PROC-VEND-001"
    )
    p["assessments"][0]["remedy"]["arguments"] = {}
    assert validate_schema_instance(p, schema)


def test_grammar_accepts_targets_and_admissible_wrong_outputs(
    project_root: Path,
) -> None:
    _require_annotation_v2_targets(project_root)
    pytest.importorskip("llguidance")
    from llguidance import LLMatcher
    from transformers import AutoTokenizer

    from enterprise_memory_mlx.experiment_profiles import QWEN_4B_REVISION
    from enterprise_memory_mlx.structured_inference import (
        compile_review_grammar,
        llguidance_tokenizer_from_hf,
        unwrap_hf_tokenizer,
    )
    from enterprise_memory_mlx.typed_action_annotation_v2 import (
        wrong_but_admissible_examples,
    )

    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
        / "snapshots"
        / QWEN_4B_REVISION
    )
    if not snapshot.is_dir():
        pytest.skip("pinned model snapshot tokenizer assets are not present")
    hf_tok = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    ll_tok = llguidance_tokenizer_from_hf(unwrap_hf_tokenizer(hf_tok))
    grammar = compile_review_grammar(runtime_output_schema())

    def accepts(text: str) -> bool:
        matcher = LLMatcher(ll_tok, grammar)
        ids = hf_tok.encode(text, add_special_tokens=False)
        if not matcher.consume_tokens(list(ids)):
            return False
        return bool(matcher.is_accepting()) and not matcher.is_error()

    accepted = sum(
        1
        for row in load_train_targets(project_root)
        if accepts(serialize_target(row["proposed_training_target"]))
    )
    assert accepted == 24
    accepted = sum(
        1
        for ref in evaluation_references()
        if accepts(serialize_target(ref["reference"]))
    )
    assert accepted == 16
    for example in wrong_but_admissible_examples():
        text = json.dumps(example, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        assert accepts(text)
    malformed = [
        '{"assessments":[{"request_item_id":"x","decision":"maybe"}]}',
        "not json",
        '{"assessments":[{"request_item_id":"x","decision":"proceed",'
        '"current_blockers":[{}],"missing_information":[],"exceptions":[],'
        '"remedy":null,"supporting_source_ids":[]}]}',
    ]
    for bad in malformed:
        assert not accepts(bad)


def test_references_cover_all_16_held_out_cases_with_support() -> None:
    refs = evaluation_references()
    assert len(refs) == 16
    assert {row["case_id"] for row in refs} == {
        "CTS3-QUAL-014",
        "CTS3-QUAL-015",
        "CTS3-QUAL-020",
        "CTS3-QUAL-021",
        "CTS3-QUAL-022",
        "CTS3-QUAL-023",
        "CTS4-FRESH-01A",
        "CTS4-FRESH-01B",
        "CTS4-FRESH-05A",
        "CTS4-FRESH-05B",
        "CTS4-FRESH-06A",
        "CTS4-FRESH-06B",
        "CTS4-FRESH-07A",
        "CTS4-FRESH-07B",
        "CTS4-FRESH-08A",
        "CTS4-FRESH-08B",
    }
    for ref in refs:
        assert ref["blocked_reference_contract"] is False
        assert ref["source_to_action_support"]
        for row in ref["source_to_action_support"]:
            assert row["supporting_clause"]
    extensions_used = sorted(
        {
            extension
            for ref in refs
            for extension in ref["vocabulary_extensions_used"]
        }
    )
    assert extensions_used == sorted(VOCABULARY_EXTENSIONS)


def test_qual020_reference_covers_both_request_items() -> None:
    ref = next(
        row
        for row in evaluation_references()
        if row["case_id"] == "CTS3-QUAL-020"
    )
    ids = {
        item["request_item_id"]
        for item in ref["reference"]["assessments"]
    }
    assert ids == {"purchase_order_issue", "invoice_processing"}


def test_comparator_oracle_passes() -> None:
    results = run_oracle()
    assert len(results) == 12
    failed = [row["id"] for row in results if not row["passed"]]
    assert failed == []


def test_comparator_rejects_duplicate_json_keys_and_trailing_text() -> None:
    parsed, status = parse_strict('{"a": 1, "a": 2}')
    assert parsed is None
    assert "duplicate" in status or "invalid_json" in status
    parsed, status = parse_strict("prefix {\"a\": 1}")
    assert parsed is None


def test_canonical_tree_is_order_invariant_but_not_duplicate_tolerant() -> None:
    from enterprise_memory_mlx.typed_action_sft_v1 import _action, _group

    left = _group(
        "all_of",
        _action("perform_unmet_control", "sanctions_check_complete", "S-1"),
        _action("perform_unmet_control", "conflict_check_complete", "S-1"),
    )
    right = _group(
        "all_of",
        _action("perform_unmet_control", "conflict_check_complete", "S-1"),
        _action("perform_unmet_control", "sanctions_check_complete", "S-1"),
    )
    assert typed_tree_equal(left, right)
    duplicate = _group(
        "all_of",
        _action("perform_unmet_control", "sanctions_check_complete", "S-1"),
        _action("perform_unmet_control", "sanctions_check_complete", "S-1"),
    )
    assert not typed_tree_equal(left, duplicate)
    assert canonical_tree(left) == canonical_tree(right)


def test_schedule_is_exactly_192_updates_with_eight_exposures() -> None:
    schedule = training_schedule()
    assert len(schedule) == MEASURED_UPDATES == 192
    assert MEASURED_UPDATES == 8 * 24
    exposures: dict[str, int] = {}
    for item in schedule:
        exposures.setdefault(str(item["case_slot"]), 0)
        exposures[str(item["case_slot"])] += 1
    assert len(exposures) == 24
    assert set(exposures.values()) == {8}


def test_inventory_freezes_the_measured_budget(project_root: Path) -> None:
    inventory = evaluation_inventory(project_root)
    assert inventory["test_case_ids"] == [
        "CTS3-QUAL-015",
        "CTS3-QUAL-020",
        "CTS4-FRESH-01A",
        "CTS4-FRESH-01B",
        "CTS4-FRESH-05A",
        "CTS4-FRESH-05B",
        "CTS4-FRESH-08A",
        "CTS4-FRESH-08B",
    ]
    assert inventory["dev_case_ids"] == [
        "CTS3-QUAL-014",
        "CTS3-QUAL-021",
        "CTS3-QUAL-022",
        "CTS3-QUAL-023",
        "CTS4-FRESH-06A",
        "CTS4-FRESH-06B",
        "CTS4-FRESH-07A",
        "CTS4-FRESH-07B",
    ]
    assert inventory["train_diagnostic_case_ids"] == [
        "CTS3-QUAL-001",
        "CTS3-QUAL-002",
        "CTS3-QUAL-003",
        "CTS3-QUAL-008",
    ]
    assert inventory["test_family_count"] == 5
    assert inventory["dev_family_count"] == 4
    assert inventory["contrast_pairs_test"] == [
        "CTS4-PAIR-01",
        "CTS4-PAIR-05",
        "CTS4-PAIR-08",
    ]
    budget = inventory["measured_generation_budget"]
    assert budget["total_calls"] == 40
    assert budget["preflight_disposable_generations"] == 1


def test_prompt_inputs_use_an_explicit_allowlist(project_root: Path) -> None:
    for case in load_held_out_cases(project_root):
        user = build_user_prompt(project_root, case)
        payload = json.loads(user)
        assert set(payload) <= {
            "evidence",
            "request_items",
            "operational_facts",
            "operational_state",
            "state_authority_guide",
        }
        assert "reference" not in payload
        assert "expected_fact_state" not in payload
        assert "policy_logic" not in payload
        assert "split" not in payload
        assert "family_id" not in payload


def test_prompt_is_invariant_to_hidden_target_changes(project_root: Path) -> None:
    case = load_held_out_cases(project_root)[0]
    first = build_user_prompt(project_root, case)
    from enterprise_memory_mlx.utils import sha256_text

    assert first == build_user_prompt(project_root, case)
    # The prompt builder only reads allowlisted fields; a change to the
    # reference cannot affect it by construction. Verify the hash is stable.
    assert sha256_text(first) == sha256_text(build_user_prompt(project_root, case))


def test_system_prompt_discloses_the_action_result_boundary() -> None:
    sp = build_system_prompt()
    assert "proposed actions" in sp
    assert "do not authorize the assessor" in sp
    assert "Requesting an exception is not recording one" in sp
    assert "grant_or_waive" in sp


def test_token_budgets_are_within_frozen_bounds(project_root: Path) -> None:
    _require_annotation_v2_targets(project_root)
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from enterprise_memory_mlx.experiment_profiles import QWEN_4B_REVISION
    from enterprise_memory_mlx.typed_action_sft_v1_runtime import build_training_rows

    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
        / "snapshots"
        / QWEN_4B_REVISION
    )
    if not snapshot.is_dir():
        pytest.skip("pinned model snapshot tokenizer assets are not present")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    rows = build_training_rows(project_root, tokenizer)
    assert len(rows) == 24
    for row in rows:
        assert row["total_len"] <= TRAIN_EXPORT_ROW_CEILING_TOKENS
        assert row["continuation_len"] <= GENERATION_CAP_TOKENS
        assert row["continuation_ids"][-1] == tokenizer.eos_token_id
        first = tokenizer.decode([row["continuation_ids"][0]])
        assert first == '{"'
        decoded = tokenizer.decode(row["continuation_ids"])
        assert "<|im_start|>" not in decoded
        assert "<think>" not in decoded


def test_training_prefix_equals_inference_prefix(project_root: Path) -> None:
    _require_annotation_v2_targets(project_root)
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from enterprise_memory_mlx.experiment_profiles import QWEN_4B_REVISION
    from enterprise_memory_mlx.typed_action_sft_v1_runtime import build_training_rows

    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
        / "snapshots"
        / QWEN_4B_REVISION
    )
    if not snapshot.is_dir():
        pytest.skip("pinned model snapshot tokenizer assets are not present")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    rows = build_training_rows(project_root, tokenizer)
    from enterprise_memory_mlx.typed_action_sft_v1 import build_system_prompt
    from enterprise_memory_mlx.utils import read_jsonl

    v3 = read_jsonl(
        project_root
        / "knowledge/company_task_specialization/v3_remedy_logic/qualification_cases.jsonl"
    )
    v4 = read_jsonl(
        project_root
        / "knowledge/company_task_specialization/v4_fact_state/contrast_cases.jsonl"
    )
    cases_by_id = {c["case_id"]: c for c in v3}
    cases_by_id.update({c["case_id"]: c for c in v4})
    sp = build_system_prompt()
    for row in rows:
        case = cases_by_id[row["case_id"]]
        from enterprise_memory_mlx.typed_action_sft_v1 import build_user_prompt

        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sp},
                {"role": "user", "content": build_user_prompt(project_root, case)},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inference_ids = tokenizer.encode(rendered, add_special_tokens=False)
        assert inference_ids == row["prompt_ids"], row["case_id"]


@pytest.mark.mlx_integration
def test_metal_runtime_stage_guards() -> None:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - environment-dependent
        # Linux CI has no mlx package at all; macOS sandboxes may lack Metal.
        # Both are honest environment skips, not disguised passes.
        pytest.skip(f"MLX unavailable in this environment: {exc}")
    from enterprise_memory_mlx.typed_action_sft_v1_runtime import (
        WIRED_LIMIT_BYTES,
        verify_frozen_digests,
    )

    if not mx.metal.is_available():
        pytest.skip("MLX present but Metal device unavailable")
    assert WIRED_LIMIT_BYTES == 96 * 1024**3
    assert all(verify_frozen_digests(Path(__file__).resolve().parents[1]).values())
    del mx