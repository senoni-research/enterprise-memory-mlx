from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.context_baselines import (
    CONTEXT_INSTRUCTION,
    DEFAULT_MAX_UTF8_BYTES,
    build_full_context,
    build_oracle_context,
    build_retrieved_context,
    render_authoritative_record,
)
from enterprise_memory_mlx.schemas import KnowledgeRecord
from enterprise_memory_mlx.utils import read_jsonl

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _record(
    record_id: str,
    *,
    title: str | None = None,
    statement: str | None = None,
    source_uri: str | None = None,
    sensitivity: str = "internal_shared",
    status: str = "active",
    effective_from: str | None = None,
    effective_to: str | None = None,
) -> KnowledgeRecord:
    return KnowledgeRecord.from_dict(
        {
            "id": record_id,
            "domain": "operations",
            "title": title or f"Policy {record_id}",
            "statement": statement or f"The authoritative statement for {record_id}.",
            "source_uri": source_uri or f"https://source.example/{record_id}",
            "sensitivity": sensitivity,
            "status": status,
            "effective_from": effective_from,
            "effective_to": effective_to,
        }
    )


def _fake_count_tokens(text: str) -> int:
    return len(text.split())


def test_renderer_includes_governed_fields_and_provenance() -> None:
    record = _record(
        "OPS-002",
        title="Change window",
        statement="Deployments require an approved change window.",
        source_uri="https://source.example/change-window",
        effective_from="2026-01-01",
        effective_to="2026-12-31",
    )

    rendered = render_authoritative_record(record)

    assert "Record ID: OPS-002" in rendered
    assert "Domain: operations" in rendered
    assert "Title: Change window" in rendered
    assert "Status: active" in rendered
    assert "Sensitivity: internal_shared" in rendered
    assert "Effective interval: 2026-01-01 to 2026-12-31" in rendered
    assert "Canonical statement: Deployments require an approved change window." in rendered
    assert "Source URI: https://source.example/change-window" in rendered


def test_renderer_excludes_all_evaluation_and_retrieval_metadata() -> None:
    record = KnowledgeRecord.from_dict(
        {
            "id": "OPS-NO-LEAK",
            "domain": "operations",
            "title": "Governed title",
            "statement": "Only this governed statement may be rendered.",
            "source_uri": "https://source.example/no-leak",
            "aliases": ["ALIAS_SENTINEL"],
            "questions": [
                {
                    "question": "QUESTION_SENTINEL",
                    "answer": "EXPECTED_ANSWER_SENTINEL",
                    "keywords": ["KEYWORD_SENTINEL"],
                }
            ],
            "metadata": {
                "critical_slots": ["CRITICAL_SLOT_SENTINEL"],
                "evaluation_note": "METADATA_SENTINEL",
            },
        }
    )

    result = build_full_context([record])

    assert result.action == "use_context"
    assert "Only this governed statement may be rendered." in result.context
    for evaluation_text in (
        "ALIAS_SENTINEL",
        "QUESTION_SENTINEL",
        "EXPECTED_ANSWER_SENTINEL",
        "KEYWORD_SENTINEL",
        "CRITICAL_SLOT_SENTINEL",
        "METADATA_SENTINEL",
    ):
        assert evaluation_text not in result.context


def test_full_context_uses_stable_record_id_order() -> None:
    records = [_record("OPS-020"), _record("OPS-003"), _record("OPS-011")]

    first = build_full_context(records, max_utf8_bytes=10_000)
    second = build_full_context(reversed(records), max_utf8_bytes=10_000)

    assert first == second
    assert first.action == "use_context"
    assert first.mode == "full_context"
    assert first.selected_record_ids == ("OPS-003", "OPS-011", "OPS-020")
    assert first.context.index("OPS-003") < first.context.index("OPS-011")
    assert first.context.index("OPS-011") < first.context.index("OPS-020")
    assert first.context.startswith(CONTEXT_INSTRUCTION)
    assert "only the supplied governed records" in first.context
    assert "current source system" in first.context
    assert first.max_utf8_bytes == 10_000
    assert first.max_tokens is None
    assert first.token_count is None
    assert first.record_count == 3
    assert first.budget_exhausted is None
    assert first.instruction_utf8_bytes + first.records_utf8_bytes == first.utf8_bytes


def test_public_only_context_has_public_classification() -> None:
    result = build_full_context(
        [_record("OPS-PUBLIC", sensitivity="public")],
        max_utf8_bytes=10_000,
    )

    assert result.highest_sensitivity == "public"
    assert "Sensitivity: public" in result.context


def test_mixed_context_has_internal_shared_classification() -> None:
    result = build_full_context(
        [
            _record("OPS-INTERNAL", sensitivity="internal_shared"),
            _record("OPS-PUBLIC", sensitivity="public"),
        ],
        max_utf8_bytes=10_000,
    )

    assert result.highest_sensitivity == "internal_shared"
    assert "Sensitivity: internal_shared" in result.context
    assert "Sensitivity: public" in result.context


def test_full_context_excludes_restricted_secret_draft_and_retired_records() -> None:
    records = [
        _record("OPS-ACTIVE"),
        _record("OPS-RESTRICTED", sensitivity="restricted"),
        _record("OPS-SECRET", sensitivity="secret"),
        _record("OPS-DRAFT", status="draft"),
        _record("OPS-RETIRED", status="retired"),
    ]

    result = build_full_context(records, max_utf8_bytes=10_000)

    assert result.selected_record_ids == ("OPS-ACTIVE",)
    assert result.source_uris == ("https://source.example/OPS-ACTIVE",)
    assert "OPS-RESTRICTED" not in result.context
    assert "OPS-SECRET" not in result.context
    assert "OPS-DRAFT" not in result.context
    assert "OPS-RETIRED" not in result.context


def test_cost_split_is_instruction_overhead_and_rendered_record_content() -> None:
    records = (_record("OPS-001"), _record("OPS-002"))
    result = build_full_context(records)
    expected_records = "\n\n".join(render_authoritative_record(record) for record in records)

    assert result.instruction_utf8_bytes == len(
        (CONTEXT_INSTRUCTION + "\n\n").encode("utf-8")
    )
    assert result.records_utf8_bytes == len(expected_records.encode("utf-8"))
    assert result.instruction_utf8_bytes + result.records_utf8_bytes == result.utf8_bytes
    assert result.record_count == 2


def test_byte_budget_never_returns_a_partial_record() -> None:
    records = [_record("OPS-001"), _record("OPS-002")]
    complete = build_full_context(records, max_utf8_bytes=10_000)

    result = build_full_context(records, max_utf8_bytes=complete.utf8_bytes - 1)

    assert result.action == "context_too_large"
    assert result.context == ""
    assert result.selected_record_ids == ("OPS-001", "OPS-002")
    assert result.source_uris == (
        "https://source.example/OPS-001",
        "https://source.example/OPS-002",
    )
    assert result.highest_sensitivity == "internal_shared"
    assert result.utf8_bytes == complete.utf8_bytes
    assert result.max_utf8_bytes == complete.utf8_bytes - 1
    assert result.budget_exhausted == "bytes"

    exact_fit = build_full_context(records, max_utf8_bytes=complete.utf8_bytes)
    assert exact_fit.action == "use_context"
    assert exact_fit.context == complete.context
    assert exact_fit.budget_exhausted is None


def test_byte_and_token_budgets_fail_closed_independently_and_together() -> None:
    records = [_record("OPS-001"), _record("OPS-002")]
    complete = build_full_context(
        records,
        max_tokens=10_000,
        count_tokens=_fake_count_tokens,
    )
    assert complete.token_count == _fake_count_tokens(complete.context)
    assert complete.max_utf8_bytes == DEFAULT_MAX_UTF8_BYTES
    assert complete.max_tokens == 10_000
    assert complete.budget_exhausted is None
    assert complete.token_count is not None

    token_overflow = build_full_context(
        records,
        max_tokens=complete.token_count - 1,
        count_tokens=_fake_count_tokens,
    )
    assert token_overflow.action == "context_too_large"
    assert token_overflow.context == ""
    assert token_overflow.budget_exhausted == "tokens"
    assert token_overflow.token_count == complete.token_count
    assert token_overflow.max_tokens == complete.token_count - 1

    byte_overflow = build_full_context(
        records,
        max_utf8_bytes=complete.utf8_bytes - 1,
        max_tokens=complete.token_count,
        count_tokens=_fake_count_tokens,
    )
    assert byte_overflow.action == "context_too_large"
    assert byte_overflow.context == ""
    assert byte_overflow.budget_exhausted == "bytes"

    both_overflow = build_full_context(
        records,
        max_utf8_bytes=complete.utf8_bytes - 1,
        max_tokens=complete.token_count - 1,
        count_tokens=_fake_count_tokens,
    )
    assert both_overflow.action == "context_too_large"
    assert both_overflow.context == ""
    assert both_overflow.budget_exhausted == "bytes_and_tokens"
    assert both_overflow.selected_record_ids == complete.selected_record_ids
    assert both_overflow.source_uris == complete.source_uris
    assert both_overflow.highest_sensitivity == complete.highest_sensitivity
    assert both_overflow.utf8_bytes == complete.utf8_bytes
    assert both_overflow.token_count == complete.token_count


def test_oracle_context_selects_exact_ids_in_stable_order() -> None:
    records = [_record("OPS-003"), _record("OPS-001"), _record("OPS-002")]

    result = build_oracle_context(
        records,
        ["OPS-003", "OPS-001"],
        max_utf8_bytes=10_000,
    )

    assert result.action == "use_context"
    assert result.mode == "oracle_context"
    assert result.selected_record_ids == ("OPS-001", "OPS-003")
    assert "OPS-001" in result.context
    assert "OPS-003" in result.context
    assert "OPS-002" not in result.context


@pytest.mark.parametrize("record_ids", [None, [], ()])
def test_unknown_or_oos_oracle_case_requires_source(
    record_ids: list[str] | tuple[()] | None,
) -> None:
    result = build_oracle_context(
        [_record("OPS-001")],
        record_ids,
        max_utf8_bytes=10_000,
    )

    assert result.action == "source_required"
    assert result.context == ""
    assert result.selected_record_ids == ()
    assert result.source_uris == ()
    assert result.highest_sensitivity is None
    assert result.utf8_bytes == 0
    assert result.max_utf8_bytes == 10_000
    assert result.max_tokens is None
    assert result.token_count is None
    assert result.instruction_utf8_bytes == 0
    assert result.records_utf8_bytes == 0
    assert result.record_count == 0
    assert result.budget_exhausted is None


def test_unknown_oracle_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown oracle record IDs: OPS-UNKNOWN"):
        build_oracle_context(
            [_record("OPS-001")],
            ["OPS-UNKNOWN"],
            max_utf8_bytes=10_000,
        )


def test_ineligible_oracle_id_fails_closed() -> None:
    records = [
        _record("OPS-ACTIVE"),
        _record("OPS-RESTRICTED", sensitivity="restricted"),
    ]

    with pytest.raises(ValueError, match="Ineligible oracle record IDs: OPS-RESTRICTED"):
        build_oracle_context(
            records,
            ["OPS-ACTIVE", "OPS-RESTRICTED"],
            max_utf8_bytes=10_000,
        )


def test_oracle_budget_is_also_fail_closed() -> None:
    record = _record("OPS-001")
    complete = build_oracle_context([record], [record.id], max_utf8_bytes=10_000)

    result = build_oracle_context(
        [record],
        [record.id],
        max_utf8_bytes=complete.utf8_bytes - 1,
    )

    assert result.action == "context_too_large"
    assert result.context == ""
    assert result.selected_record_ids == ("OPS-001",)
    assert result.highest_sensitivity == "internal_shared"
    assert result.utf8_bytes == complete.utf8_bytes


def test_retrieved_context_preserves_rank_order() -> None:
    records = [_record("OPS-003"), _record("OPS-001"), _record("OPS-002")]

    result = build_retrieved_context(records, max_utf8_bytes=10_000)

    assert result.mode == "bm25_context"
    assert result.selected_record_ids == ("OPS-003", "OPS-001", "OPS-002")
    assert result.context.index("OPS-003") < result.context.index("OPS-001")
    assert result.context.index("OPS-001") < result.context.index("OPS-002")


def test_retrieved_context_empty_and_ineligible_fail_closed() -> None:
    empty = build_retrieved_context([], max_utf8_bytes=10_000)
    assert empty.action == "source_required"
    assert empty.highest_sensitivity is None

    with pytest.raises(ValueError, match="Ineligible retrieved record IDs"):
        build_retrieved_context(
            [_record("OPS-RESTRICTED", sensitivity="restricted")],
            max_utf8_bytes=10_000,
        )


def test_duplicate_record_ids_fail_closed() -> None:
    record = _record("OPS-DUPLICATE")

    with pytest.raises(
        ValueError,
        match="Duplicate knowledge record IDs: OPS-DUPLICATE",
    ):
        build_full_context([record, record])


def test_empty_eligible_corpus_requires_source() -> None:
    result = build_full_context(
        [
            _record("OPS-RESTRICTED", sensitivity="restricted"),
            _record("OPS-DRAFT", status="draft"),
        ],
        count_tokens=_fake_count_tokens,
    )

    assert result.action == "source_required"
    assert result.context == ""
    assert result.selected_record_ids == ()
    assert result.token_count == 0
    assert result.record_count == 0
    assert result.budget_exhausted is None


def test_zero_byte_budget_fails_when_an_eligible_record_exists() -> None:
    result = build_full_context([_record("OPS-001")], max_utf8_bytes=0)

    assert result.action == "context_too_large"
    assert result.context == ""
    assert result.max_utf8_bytes == 0
    assert result.utf8_bytes > 0
    assert result.budget_exhausted == "bytes"
    assert result.selected_record_ids == ("OPS-001",)


@pytest.mark.parametrize("invalid_budget", [True, False])
def test_boolean_byte_budget_is_rejected(invalid_budget: bool) -> None:
    with pytest.raises(TypeError, match="max_utf8_bytes must be an integer"):
        build_full_context([_record("OPS-001")], max_utf8_bytes=invalid_budget)


@pytest.mark.parametrize("invalid_budget", [True, False])
def test_boolean_token_budget_is_rejected(invalid_budget: bool) -> None:
    with pytest.raises(TypeError, match="max_tokens must be an integer or None"):
        build_full_context(
            [_record("OPS-001")],
            max_tokens=invalid_budget,
            count_tokens=_fake_count_tokens,
        )


def test_token_budget_requires_an_injected_counter() -> None:
    with pytest.raises(
        ValueError,
        match="count_tokens is required when max_tokens is supplied",
    ):
        build_full_context([_record("OPS-001")], max_tokens=100)


def test_utf8_budget_counts_multibyte_content() -> None:
    record = _record(
        "FIN-CAFÉ",
        title="Café allowance",
        statement="The Montréal café allowance is £25.",
    )
    complete = build_full_context([record], max_utf8_bytes=10_000)

    assert complete.utf8_bytes == len(complete.context.encode("utf-8"))
    assert complete.utf8_bytes > len(complete.context)

    too_small = build_full_context([record], max_utf8_bytes=complete.utf8_bytes - 1)
    assert too_small.action == "context_too_large"
    assert too_small.context == ""


def test_complete_authoritative_benchmark_store_fits_default_byte_budget() -> None:
    corpus_paths = (
        _REPOSITORY_ROOT / "knowledge" / "records.jsonl",
        _REPOSITORY_ROOT / "knowledge" / "eval_frozen" / "holdout_records.jsonl",
    )
    records = [
        KnowledgeRecord.from_dict(row)
        for corpus_path in corpus_paths
        for row in read_jsonl(corpus_path)
    ]

    result = build_full_context(records)

    assert result.action == "use_context"
    assert result.max_utf8_bytes == DEFAULT_MAX_UTF8_BYTES
    assert result.utf8_bytes <= DEFAULT_MAX_UTF8_BYTES
    assert result.selected_record_ids == (
        "ENG-INC-002",
        "ENG-REL-001",
        "FIN-CARD-003",
        "FIN-EXP-001",
        "FIN-INV-002",
        "HR-LEAVE-001",
        "HR-REMOTE-002",
        "IT-ACCESS-001",
        "LEG-CONF-001",
        "PROC-VEND-001",
        "SUP-SLA-001",
    )
    assert result.record_count == 11
    assert "SEC-KEY-999" not in result.selected_record_ids
    assert "SEC-KEY-999" not in result.context
