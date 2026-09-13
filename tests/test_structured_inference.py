from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.learning_mechanics import load_approval, verify_bound_pairs
from enterprise_memory_mlx.learning_mechanics_scoring import (
    PERMITTED_DISPOSITIONS,
    PERMITTED_STATUSES,
    parse_review_strict,
    score_declared_review,
)
from enterprise_memory_mlx.structured_inference import (
    FORBIDDEN_TRAINING_NAMES,
    IMPLEMENTATION_ID,
    LOCAL_SNAPSHOT,
    PINNED_PACKAGES,
    PROTOCOL_ID,
    assert_module_cannot_train,
    compile_review_grammar,
    declared_review_schema,
    dummy_valid_review,
    extract_first_object,
    first_allowed_token_ids,
    llguidance_tokenizer_from_hf,
    matcher_accepts_text,
    package_versions,
    schema_contains_case_specific_labels,
    unwrap_hf_tokenizer,
    visible_identifiers,
)

ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_SOURCE = {
    "case_id": "DEV-002",
    "requirements": [{"id": "R1"}, {"id": "R2"}, {"id": "R3"}, {"id": "R4"}],
    "source_index": [{"source_id": "S1"}],
}
SYNTHETIC_LABELS = {"R1": "met", "R2": "insufficient_evidence", "R3": "met", "R4": "not_met"}
SYNTHETIC_DISPOSITION = "needs_information"


def _synthetic_identifiers() -> dict[str, object]:
    return visible_identifiers(SYNTHETIC_SOURCE)


def _verified():
    return verify_bound_pairs(ROOT, load_approval(ROOT))


def test_module_cannot_reach_training_symbols() -> None:
    assert_module_cannot_train()
    assert "linear_to_lora_layers(" not in (
        ROOT / "src/enterprise_memory_mlx/structured_inference.py"
    ).read_text(encoding="utf-8")
    assert FORBIDDEN_TRAINING_NAMES


def test_protocol_and_environment_pins() -> None:
    assert PROTOCOL_ID == "senoni-deliverable-review/structured-inference-v1"
    assert IMPLEMENTATION_ID == "qwen4b-structured-inference/v1"
    assert PINNED_PACKAGES == {
        "mlx": "0.32.2",
        "mlx-lm": "0.31.3",
        "mlx-metal": "0.32.2",
        "llguidance": "1.8.0",
    }


@pytest.mark.mlx_integration
def test_installed_environment_matches_pins() -> None:
    versions = package_versions()
    if any(versions.get(name) is None for name in PINNED_PACKAGES):
        pytest.skip("pinned Apple/MLX or llguidance packages are not installed")
    for name, required in PINNED_PACKAGES.items():
        assert versions.get(name) == required


def test_visible_identifiers_come_from_frozen_inputs_only() -> None:
    identifiers = _synthetic_identifiers()
    assert identifiers["case_id"] == "DEV-002"
    assert identifiers["requirement_ids"] == ["R1", "R2", "R3", "R4"]
    assert identifiers["source_ids"] == ["S1"]
    assert "labels" not in identifiers
    assert "disposition" not in identifiers


def test_schema_keeps_all_statuses_and_omits_approved_bindings() -> None:
    identifiers = _synthetic_identifiers()
    schema = declared_review_schema(identifiers)
    status_enum = set(
        schema["properties"]["requirement_assessments"]["items"]["properties"]["status"]["enum"]
    )
    disposition_enum = set(schema["properties"]["overall_disposition"]["enum"])
    assert status_enum == set(PERMITTED_STATUSES)
    assert disposition_enum == set(PERMITTED_DISPOSITIONS)
    assert schema["properties"]["requirement_assessments"]["minItems"] == len(
        identifiers["requirement_ids"]
    )
    assert not schema_contains_case_specific_labels(
        schema, SYNTHETIC_LABELS, SYNTHETIC_DISPOSITION
    )
    blob = json.dumps(schema, ensure_ascii=False)
    assert SYNTHETIC_DISPOSITION in disposition_enum
    assert '"DEV-002"' in blob
    for requirement_id, status in SYNTHETIC_LABELS.items():
        assert f'"{requirement_id}": "{status}"' not in blob


@pytest.mark.private_data
def test_private_frozen_inputs_match_visible_identifier_contract(
    private_approval_path,
) -> None:
    del private_approval_path
    for item in _verified():
        source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
        identifiers = visible_identifiers(source)
        assert identifiers["case_id"] == item["case_id"]
        assert identifiers["requirement_ids"] == list(item["labels"])
        assert identifiers["source_ids"]
        schema = declared_review_schema(identifiers)
        assert not schema_contains_case_specific_labels(
            schema, item["labels"], item["disposition"]
        )


def test_strict_scorer_still_rejects_prefixed_json() -> None:
    body = dummy_valid_review(_synthetic_identifiers())
    text = "</tool_call>\n\n" + json.dumps(body, indent=2)
    parsed, errors = parse_review_strict(text)
    assert parsed is None
    assert errors[0].startswith("invalid_json:")
    extracted = extract_first_object(text)
    assert extracted is not None
    isolated, start, _end = extracted
    assert start == len("</tool_call>\n\n")
    scored = score_declared_review(
        case_id="DEV-002",
        approved_labels={"R1": "met", "R2": "met", "R3": "met", "R4": "met"},
        approved_disposition="proceed",
        raw_text=json.dumps(isolated, indent=2),
        truncated=False,
        parse_status="plain",
    )
    assert scored["complete_valid"] is True


def _local_hf_tokenizer():
    pytest.importorskip("llguidance")
    pytest.importorskip("transformers")
    if not LOCAL_SNAPSHOT.exists():
        pytest.skip("pinned Qwen4B tokenizer snapshot is not local")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(LOCAL_SNAPSHOT),
        local_files_only=True,
        trust_remote_code=False,
    )


@pytest.mark.mlx_integration
def test_unwrap_skips_rust_backend_and_keeps_fast_tokenizer() -> None:
    tokenizer = _local_hf_tokenizer()
    from transformers import PreTrainedTokenizerFast

    class Wrapper:
        def __init__(self, inner):
            self._tokenizer = inner

    rust_backend = tokenizer._tokenizer
    wrapped = Wrapper(Wrapper(rust_backend))
    resolved = unwrap_hf_tokenizer(wrapped)
    assert isinstance(resolved, PreTrainedTokenizerFast)
    llguidance_tokenizer_from_hf(wrapped)


@pytest.mark.mlx_integration
def test_llguidance_grammar_requires_json_and_rejects_tool_call_prefix() -> None:
    tokenizer = _local_hf_tokenizer()
    identifiers = _synthetic_identifiers()
    schema = declared_review_schema(identifiers)
    grammar = compile_review_grammar(schema)
    ll_tokenizer = llguidance_tokenizer_from_hf(tokenizer)
    from llguidance import LLMatcher

    matcher = LLMatcher(ll_tokenizer, grammar)
    brace_ids = tokenizer.encode("{", add_special_tokens=False)
    tool_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    allowed = first_allowed_token_ids(
        matcher,
        ll_tokenizer,
        {
            "brace": brace_ids[0],
            "tool_call": tool_ids[0],
            "think": think_ids[0],
        },
    )
    assert allowed["brace"] is True
    assert allowed["tool_call"] is False
    assert allowed["think"] is False
    valid = json.dumps(dummy_valid_review(identifiers), indent=2) + "\n"
    assert matcher_accepts_text(grammar, ll_tokenizer, tokenizer, valid) is True
    assert (
        matcher_accepts_text(
            grammar,
            ll_tokenizer,
            tokenizer,
            "</tool_call>\n\n" + valid,
        )
        is False
    )
