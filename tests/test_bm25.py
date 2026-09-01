from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from enterprise_memory_mlx.bm25 import (
    BM25Config,
    BM25FeasibilityConstraints,
    BM25Index,
    BM25ValidationLabel,
    BM25ValidationQuery,
    select_retrieval_operating_point,
    tokenize,
)
from enterprise_memory_mlx.schemas import KnowledgeRecord, Sensitivity, Status

# Deliberate diagnostic threshold: 0.0 accepts every scored hit. Tests state
# it explicitly because BM25Config no longer carries a default threshold.
_DIAGNOSTIC_CONFIG = BM25Config(score_threshold=0.0)
_SYNTHETIC_VALIDATION_HASH = "a" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SHIPPED_RECORDS_PATH = _REPOSITORY_ROOT / "knowledge" / "records.jsonl"
_HOLDOUT_RECORDS_PATH = (
    _REPOSITORY_ROOT / "knowledge" / "eval_frozen" / "holdout_records.jsonl"
)
_ELIGIBLE_SHIPPED_RECORD_IDS = (
    "ENG-INC-002",
    "ENG-REL-001",
    "FIN-EXP-001",
    "FIN-INV-002",
    "HR-LEAVE-001",
    "HR-REMOTE-002",
    "PROC-VEND-001",
    "SUP-SLA-001",
)


def _record(
    record_id: str,
    *,
    title: str,
    statement: str,
    summary: str = "governed policy",
    domain: str = "operations",
    source_uri: str | None = None,
    aliases: tuple[str, ...] = (),
    sensitivity: Sensitivity = "internal_shared",
    status: Status = "active",
    metadata: dict[str, Any] | None = None,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=record_id,
        domain=domain,
        title=title,
        statement=statement,
        summary=summary,
        source_uri=source_uri or f"memory://{record_id}",
        aliases=aliases,
        sensitivity=sensitivity,
        status=status,
        metadata=metadata or {},
    )


def _answerable(
    query_id: str,
    query: str,
    relevant_record_id: str,
) -> BM25ValidationQuery:
    return BM25ValidationQuery(
        query_id=query_id,
        query=query,
        label=BM25ValidationLabel(
            kind="answerable",
            relevant_record_ids=(relevant_record_id,),
        ),
    )


def _oos(query_id: str, query: str, oos_type: str = "synthetic_collision") -> BM25ValidationQuery:
    return BM25ValidationQuery(
        query_id=query_id,
        query=query,
        label=BM25ValidationLabel(kind="oos", oos_type=oos_type),
    )


def _select(
    index: BM25Index,
    queries: tuple[BM25ValidationQuery, ...],
    *,
    constraints: BM25FeasibilityConstraints | None = None,
):
    return select_retrieval_operating_point(
        index,
        queries,
        validation_dataset_hash=_SYNTHETIC_VALIDATION_HASH,
        frozen_question_texts=(),
        constraints=constraints,
    )


def _load_knowledge_records(*paths: Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            raw["questions"] = []
            records.append(KnowledgeRecord.from_dict(raw))
    return records


def _shipped_index() -> BM25Index:
    return BM25Index(
        _load_knowledge_records(_SHIPPED_RECORDS_PATH),
        config=_DIAGNOSTIC_CONFIG,
    )


def _retrieval_outcome(
    result,
    *,
    relevant_record_id: str | None,
) -> str:
    if relevant_record_id is None:
        return "oos_false_load" if result.hits else "correct_oos_rejection"
    if not result.hits:
        return "empty_retrieval"
    if relevant_record_id in result.selected_record_ids:
        return "correct_record"
    return "wrong_record"


def test_finance_query_ranks_finance_record_first_and_returns_provenance() -> None:
    finance = _record(
        "FIN-EXP-001",
        domain="finance",
        title="Travel expense allowance",
        summary="The international meal allowance is €150 per day.",
        statement="Finance reimburses approved international meals up to €150 daily.",
        aliases=("international per diem",),
        source_uri="memory://finance/expense-policy",
    )
    engineering = _record(
        "ENG-INC-002",
        domain="engineering",
        title="Incident response",
        statement="Page the incident commander for severity one outages.",
    )

    result = BM25Index([engineering, finance], config=_DIAGNOSTIC_CONFIG).search(
        "What is the €150 international meal allowance?"
    )

    assert result.action == "use_context"
    assert result.reason is None
    assert result.hits[0].record_id == finance.id
    assert result.hits[0].source_uri == finance.source_uri
    assert result.hits[0].rank == 1
    assert result.selected_records[0] is finance
    assert result.selected_record_ids[0] == finance.id


def test_engineering_release_query_ranks_release_record_first() -> None:
    release = _record(
        "ENG-REL-001",
        domain="engineering",
        title="Production release approvals",
        summary="High-risk releases require a release manager approval.",
        statement="A production deployment needs its release gate approved before launch.",
        aliases=("deployment gate",),
    )
    support = _record(
        "SUP-SLA-001",
        domain="support",
        title="Customer support response",
        statement="Urgent customer tickets receive a response within one hour.",
    )

    result = BM25Index([support, release], config=_DIAGNOSTIC_CONFIG).search(
        "Who approves a high-risk engineering production release?"
    )

    assert result.action == "use_context"
    assert result.hits[0].record_id == release.id


def test_tokenization_normalizes_unicode_and_preserves_structured_values() -> None:
    tokens = tokenize(
        "Straße référence FIN-OPS_42: €1,250.50 due 2026-09-30; growth 12%."
    )

    assert tokens == (
        "strasse",
        "référence",
        "fin-ops_42",
        "€1,250.50",
        "due",
        "2026-09-30",
        "growth",
        "12%",
    )


def test_tokenization_pins_sample_structured_forms_and_currency_matching() -> None:
    assert tokenize(
        "£500 £750 £2,000 at 10:00 and 15:00 on 31 March or 2026-09-01 FIN-EXP-001"
    ) == (
        "£500",
        "£750",
        "£2,000",
        "at",
        "10:00",
        "and",
        "15:00",
        "on",
        "31",
        "march",
        "or",
        "2026-09-01",
        "fin-exp-001",
    )
    index = BM25Index(
        [
            _record(
                "FIN-EXP-001",
                title="Expense cap",
                statement="The approved expense cap is £500.",
            )
        ],
        config=_DIAGNOSTIC_CONFIG,
    )

    result = index.search("Is the cap £500?")

    assert result.action == "use_context"
    assert result.selected_record_ids == ("FIN-EXP-001",)


def test_ineligible_records_are_absent_from_index() -> None:
    records = [
        _record("ACTIVE-001", title="Visible policy", statement="visible fact"),
        _record(
            "RESTRICTED-001",
            title="Restricted policy",
            statement="restricted-only-token",
            sensitivity="restricted",
        ),
        _record(
            "SECRET-001",
            title="Secret policy",
            statement="secret-only-token",
            sensitivity="secret",
        ),
        _record(
            "DRAFT-001",
            title="Draft policy",
            statement="draft-only-token",
            status="draft",
        ),
        _record(
            "RETIRED-001",
            title="Retired policy",
            statement="retired-only-token",
            status="retired",
        ),
    ]

    index = BM25Index(records, config=_DIAGNOSTIC_CONFIG)

    assert [record.id for record in index.records] == ["ACTIVE-001"]
    assert index.search("restricted-only-token").action == "source_required"
    assert index.search("secret-only-token").action == "source_required"
    assert index.search("draft-only-token").action == "source_required"
    assert index.search("retired-only-token").action == "source_required"


def test_equal_scores_tie_break_by_record_id() -> None:
    later = _record(
        "OPS-002",
        title="Shared process",
        statement="shared governed procedure",
    )
    earlier = _record(
        "OPS-001",
        title="Shared process",
        statement="shared governed procedure",
    )

    result = BM25Index([later, earlier], config=_DIAGNOSTIC_CONFIG).search("shared")

    assert [hit.record_id for hit in result.hits] == ["OPS-001", "OPS-002"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.hits[0].score == result.hits[1].score


def test_empty_unrelated_and_empty_index_queries_fail_closed() -> None:
    index = BM25Index(
        [_record("OPS-001", title="Vendor onboarding", statement="Approve new suppliers.")],
        config=_DIAGNOSTIC_CONFIG,
    )

    empty = index.search(" \N{EM DASH} ")
    unrelated = index.search("xylophone nebula")
    no_records = BM25Index([], config=_DIAGNOSTIC_CONFIG).search("vendor")

    assert (empty.action, empty.reason, empty.hits) == (
        "source_required",
        "query_has_no_index_tokens",
        (),
    )
    assert unrelated.reason == "query_has_no_index_tokens"
    assert unrelated.selected_records == ()
    assert no_records.reason == "no_eligible_records"


def test_function_word_only_query_has_no_index_tokens() -> None:
    index = BM25Index(
        [
            _record(
                "OPS-001",
                title="What is this",
                summary="This is what it is.",
                statement="How is it that this is the one?",
            )
        ],
        config=_DIAGNOSTIC_CONFIG,
    )

    assert tokenize("What is this and how is it?") == (
        "what",
        "is",
        "this",
        "and",
        "how",
        "is",
        "it",
    )
    result = index.search("What is this and how is it?")

    assert result.action == "source_required"
    assert result.reason == "query_has_no_index_tokens"
    assert result.hits == ()


def test_meaningful_domain_term_outranks_shared_function_wording() -> None:
    meaningful = _record(
        "OPS-002",
        title="Zephyr",
        summary="This is it.",
        statement="It is this.",
    )
    generic = _record(
        "OPS-001",
        title="What is this",
        summary="This is what it is and how it is.",
        statement="What is it and how is it that this is the one?",
    )

    result = BM25Index([generic, meaningful], config=_DIAGNOSTIC_CONFIG).search(
        "What is this and how is it about the Zephyr?"
    )

    assert result.action == "use_context"
    assert result.hits[0].record_id == meaningful.id
    assert [hit.record_id for hit in result.hits] == [meaningful.id]


def test_top_k_and_score_threshold_are_respected() -> None:
    records = [
        _record(
            f"OPS-00{number}",
            title=f"Shared policy {number}",
            statement="shared procedure",
        )
        for number in range(1, 4)
    ]
    index = BM25Index(records, config=BM25Config(top_k=2, score_threshold=0.0))

    limited = index.search("shared")
    rejected = index.search(
        "shared",
        score_threshold=max(hit.score for hit in limited.hits) + 1.0,
    )

    assert len(limited.hits) == 2
    assert [hit.rank for hit in limited.hits] == [1, 2]
    assert rejected.action == "source_required"
    assert rejected.reason == "no_match_above_threshold"


def test_search_fails_closed_without_an_explicit_score_threshold() -> None:
    """The old 0.0 default is gone: direct library use must state a threshold."""
    index = BM25Index(
        [_record("OPS-001", title="Amber handbook", statement="Amber procedure.")]
    )

    with pytest.raises(ValueError, match="chosen explicitly"):
        index.search("amber")

    assert index.search("amber", score_threshold=0.0).action == "use_context"


def test_wrong_record_hit_remains_visible_as_use_context() -> None:
    intended = _record(
        "OPS-001",
        title="Cobalt handbook",
        statement="Cobalt procedure.",
    )
    wrong = _record(
        "OPS-002",
        title="Amber handbook",
        statement="Amber procedure.",
    )

    result = BM25Index([intended, wrong], config=_DIAGNOSTIC_CONFIG).search("amber")

    assert result.action == "use_context"
    assert result.reason is None
    assert result.selected_record_ids[0] == wrong.id
    assert result.selected_record_ids[0] != intended.id


def test_weighted_title_match_outranks_statement_match() -> None:
    title_match = _record(
        "OPS-001",
        title="quasar",
        statement="neutral",
        summary="neutral",
    )
    statement_match = _record(
        "OPS-002",
        title="neutral",
        statement="quasar",
        summary="neutral",
    )

    result = BM25Index(
        [statement_match, title_match],
        config=_DIAGNOSTIC_CONFIG,
    ).search("quasar")

    assert [hit.record_id for hit in result.hits] == ["OPS-001", "OPS-002"]
    assert result.hits[0].score > result.hits[1].score


def test_retrieval_does_not_index_record_metadata() -> None:
    index = BM25Index(
        [
            _record(
                "OPS-001",
                title="Vendor onboarding",
                statement="Approve new suppliers.",
                metadata={"retrieval_hint": "xylophone"},
            )
        ],
        config=_DIAGNOSTIC_CONFIG,
    )

    result = index.search("xylophone")

    assert result.action == "source_required"
    assert result.reason == "query_has_no_index_tokens"


def test_results_are_deterministic_across_repeated_builds_and_input_order() -> None:
    records = [
        _record(
            "FIN-002",
            domain="finance",
            title="Invoice schedule",
            statement="Invoices are paid on the monthly schedule.",
        ),
        _record(
            "FIN-001",
            domain="finance",
            title="Invoice approval",
            statement="Invoices require finance approval.",
        ),
    ]

    first = BM25Index(records, config=_DIAGNOSTIC_CONFIG).search(
        "finance invoice approval"
    )
    second = BM25Index(reversed(records), config=_DIAGNOSTIC_CONFIG).search(
        "finance invoice approval"
    )

    assert first == second


def test_index_statistics_cover_only_eligible_weighted_documents() -> None:
    active_records = [
        _record("OPS-001", title="Amber policy", statement="Approve amber requests."),
        _record("OPS-002", title="Cobalt policy", statement="Approve cobalt requests."),
    ]
    ineligible = _record(
        "OPS-999",
        title="Hidden policy",
        statement="Hidden procedure.",
        status="draft",
    )

    stats = BM25Index([ineligible, *active_records]).stats

    assert stats.eligible_document_count == 2
    assert stats.unique_term_count > 0
    assert stats.total_weighted_length > 0
    assert stats.average_document_length == stats.total_weighted_length / 2
    assert math.isfinite(stats.build_elapsed_seconds)
    assert stats.build_elapsed_seconds >= 0


@pytest.mark.parametrize(
    ("kind", "record_ids", "oos_type", "message"),
    [
        ("answerable", (), None, "exactly one"),
        ("answerable", ("OPS-001", "OPS-002"), None, "exactly one"),
        ("oos", ("OPS-001",), "collision", "cannot have"),
        ("oos", (), None, "require"),
    ],
)
def test_validation_labels_enforce_single_record_or_explicit_oos(
    kind: str,
    record_ids: tuple[str, ...],
    oos_type: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BM25ValidationLabel(  # type: ignore[arg-type]
            kind=kind,
            relevant_record_ids=record_ids,
            oos_type=oos_type,
        )


def test_selector_rejects_normalized_frozen_question_overlap() -> None:
    index = BM25Index(
        [_record("OPS-001", title="Amber handbook", statement="Amber procedure.")]
    )
    queries = (
        _answerable("VAL-001", "How does the amber handbook work?", "OPS-001"),
        _oos("VAL-002", "Synthetic orbital gardening request"),
    )

    with pytest.raises(ValueError, match="VAL-001"):
        select_retrieval_operating_point(
            index,
            queries,
            validation_dataset_hash=_SYNTHETIC_VALIDATION_HASH,
            frozen_question_texts=("  HOW does the amber handbook work?  ",),
        )


def test_selector_requires_approved_fixed_bm25_scoring_parameters() -> None:
    index = BM25Index(
        [_record("OPS-001", title="Amber handbook", statement="Amber procedure.")],
        config=BM25Config(k1=1.5),
    )
    queries = (
        _answerable("VAL-001", "amber handbook", "OPS-001"),
        _oos("VAL-002", "Synthetic orbital gardening request"),
    )

    with pytest.raises(ValueError, match="fixed k1=1.2 and b=0.75"):
        _select(index, queries)


def test_candidate_metrics_distinguish_correct_wrong_empty_and_oos_false_load() -> None:
    records = [
        _record("OPS-001", title="Amber", summary="neutral", statement="neutral"),
        _record("OPS-002", title="Neutral", summary="neutral", statement="amber"),
        _record("OPS-003", title="Cobalt", summary="neutral", statement="neutral"),
    ]
    queries = (
        _answerable("VAL-001", "amber details", "OPS-001"),
        _answerable("VAL-002", "what is amber", "OPS-002"),
        _answerable("VAL-003", "xylophone nebula", "OPS-003"),
        _oos("VAL-004", "amber outside request"),
    )

    report = _select(BM25Index(records), queries)
    top_one = next(
        candidate
        for candidate in report.candidates
        if candidate.config.top_k == 1 and candidate.config.score_threshold == 0.0
    )
    top_two = next(
        candidate
        for candidate in report.candidates
        if candidate.config.top_k == 2 and candidate.config.score_threshold == 0.0
    )

    assert top_one.metrics.correct_record.count == 1
    assert top_one.metrics.wrong_record.count == 1
    assert top_one.metrics.answerable_empty_retrieval.count == 1
    assert (
        top_one.metrics.correct_record.count
        + top_one.metrics.wrong_record.count
        + top_one.metrics.answerable_empty_retrieval.count
        == top_one.metrics.answerable_query_count
    )
    assert top_one.metrics.oos_false_load.count == 1
    assert top_one.metrics.correct_oos_rejection.count == 0
    assert top_two.metrics.correct_record.count == 2
    assert top_two.metrics.wrong_record.count == 0
    assert top_two.metrics.distractor_record_count == 2
    assert top_one.metrics.correct_record.confidence_low_95 >= 0
    assert top_one.metrics.correct_record.confidence_high_95 <= 1


def test_no_feasible_operating_point_has_no_fallback_and_reports_every_candidate() -> None:
    index = BM25Index(
        [
            _record("OPS-001", title="Amber", summary="neutral", statement="neutral"),
            _record("OPS-002", title="Cobalt", summary="neutral", statement="neutral"),
        ]
    )
    queries = (
        _answerable("VAL-001", "amber", "OPS-001"),
        _answerable("VAL-002", "amber handbook", "OPS-002"),
        _oos("VAL-003", "amber external"),
    )

    report = _select(index, queries)

    assert report.status == "no_feasible_operating_point"
    assert report.selected_config is None
    assert report.validation_size == 3
    assert report.top_k_candidates == (1, 2, 3, 5)
    assert report.score_threshold_candidates[0] == 0.0
    assert report.score_threshold_candidates == tuple(
        sorted(set(report.score_threshold_candidates))
    )
    assert len(report.candidates) == 4 * len(report.score_threshold_candidates)
    assert [candidate.ordinal for candidate in report.candidates] == list(
        range(len(report.candidates))
    )
    assert all(candidate.constraint_failures for candidate in report.candidates)
    assert report.pareto_frontier
    assert set(report.pareto_frontier).issubset(report.candidates)
    all_reject_threshold = report.score_threshold_candidates[-1]
    assert all(
        candidate.metrics.retrieved_record_count == 0
        for candidate in report.candidates
        if candidate.config.score_threshold == all_reject_threshold
    )


def test_feasible_selection_uses_smaller_top_k_then_highest_threshold() -> None:
    index = BM25Index(
        [
            _record(
                "OPS-001",
                title="Sapphire",
                summary="neutral",
                statement="shared policy",
            ),
            _record(
                "OPS-002",
                title="Cobalt",
                summary="neutral",
                statement="shared policy",
            ),
        ]
    )
    queries = (
        _answerable("VAL-001", "sapphire", "OPS-001"),
        _answerable("VAL-002", "cobalt", "OPS-002"),
        _oos("VAL-003", "shared"),
    )

    first = _select(index, queries)
    second = _select(index, queries)
    feasible_top_one = [
        candidate
        for candidate in first.candidates
        if candidate.feasible and candidate.config.top_k == 1
    ]

    assert first == second
    assert first.status == "selected"
    assert first.selected_config is not None
    assert first.selected_config.top_k == 1
    assert first.selected_config.score_threshold == max(
        candidate.config.score_threshold for candidate in feasible_top_one
    )
    assert first.exploratory is True
    assert first.validation_dataset_hash == _SYNTHETIC_VALIDATION_HASH
    selected_candidate = next(
        candidate for candidate in first.candidates if candidate.config == first.selected_config
    )
    dominated_larger_top_k = next(
        candidate
        for candidate in first.candidates
        if candidate.config.top_k == 2
        and candidate.config.score_threshold == first.selected_config.score_threshold
    )
    assert selected_candidate in first.pareto_frontier
    assert dominated_larger_top_k not in first.pareto_frontier


def test_tie_break_prioritizes_oos_false_load_before_correct_record_rate() -> None:
    index = BM25Index(
        [
            _record(
                "OPS-001",
                title="Rareterm",
                summary="rareterm",
                statement="rareterm",
            ),
            _record("OPS-002", title="Shared", summary="neutral", statement="neutral"),
        ]
    )
    queries = (
        _answerable("VAL-001", "rareterm", "OPS-001"),
        _answerable("VAL-002", "shared", "OPS-002"),
        _oos("VAL-003", "what is shared"),
    )
    relaxed = BM25FeasibilityConstraints(
        minimum_correct_record_rate=0.0,
        maximum_wrong_record_rate=1.0,
        maximum_answerable_empty_retrieval_rate=1.0,
        maximum_oos_false_load_rate=1.0,
    )

    report = _select(index, queries, constraints=relaxed)
    selected = next(
        candidate for candidate in report.candidates if candidate.config == report.selected_config
    )

    assert selected.metrics.oos_false_load.rate == 0.0
    assert selected.metrics.correct_record.rate == 0.5
    assert any(
        candidate.feasible
        and candidate.metrics.oos_false_load.rate == 1.0
        and candidate.metrics.correct_record.rate == 1.0
        for candidate in report.candidates
    )


def test_tie_break_prioritizes_wrong_record_rate_before_correct_record_rate() -> None:
    index = BM25Index(
        [
            _record("OPS-001", title="Shared", summary="neutral", statement="neutral"),
            _record("OPS-002", title="Unrelated", summary="neutral", statement="neutral"),
            _record(
                "OPS-003",
                title="Rareterm",
                summary="rareterm",
                statement="rareterm",
            ),
        ]
    )
    queries = (
        _answerable("VAL-001", "shared", "OPS-001"),
        _answerable("VAL-002", "what is shared", "OPS-002"),
        _answerable("VAL-003", "rareterm", "OPS-003"),
        _oos("VAL-004", "xylophone nebula"),
    )
    relaxed = BM25FeasibilityConstraints(
        minimum_correct_record_rate=0.0,
        maximum_wrong_record_rate=1.0,
        maximum_answerable_empty_retrieval_rate=1.0,
        maximum_oos_false_load_rate=1.0,
    )

    report = _select(index, queries, constraints=relaxed)
    selected = next(
        candidate for candidate in report.candidates if candidate.config == report.selected_config
    )

    assert selected.metrics.wrong_record.rate == 0.0
    assert selected.metrics.correct_record.rate == pytest.approx(1 / 3)
    assert any(
        candidate.feasible
        and candidate.metrics.wrong_record.rate == pytest.approx(1 / 3)
        and candidate.metrics.correct_record.rate == pytest.approx(2 / 3)
        for candidate in report.candidates
    )


def test_shipped_index_contains_exactly_eight_eligible_records() -> None:
    records = _load_knowledge_records(_SHIPPED_RECORDS_PATH)
    index = BM25Index(records, config=_DIAGNOSTIC_CONFIG)

    assert "SEC-KEY-999" in {record.id for record in records}
    assert [record.id for record in index.records] == list(_ELIGIBLE_SHIPPED_RECORD_IDS)
    assert index.stats.eligible_document_count == 8
    assert "SEC-KEY-999" not in {record.id for record in index.records}
    assert index.search("SEC-KEY-999").action == "source_required"
    assert index.search("restricted cryptographic recovery").action == "source_required"


def test_shipped_corpus_travel_threshold_currency_queries_rank_expense_record() -> None:
    index = _shipped_index()
    five_hundred = index.search(
        "Does travel or subsistence spend above £500 need written budget-owner "
        "approval before booking?"
    )
    seven_fifty = index.search(
        "Does a £750 travel-threshold booking still need written subsistence approval?"
    )

    assert five_hundred.action == "use_context"
    assert five_hundred.hits[0].record_id == "FIN-EXP-001"
    assert seven_fifty.action == "use_context"
    assert seven_fifty.hits[0].record_id == "FIN-EXP-001"
    assert "SEC-KEY-999" not in five_hundred.selected_record_ids
    assert "SEC-KEY-999" not in seven_fifty.selected_record_ids


def test_shipped_corpus_corporate_card_query_ranks_holdout_record() -> None:
    index = BM25Index(
        _load_knowledge_records(_SHIPPED_RECORDS_PATH, _HOLDOUT_RECORDS_PATH),
        config=_DIAGNOSTIC_CONFIG,
    )

    result = index.search(
        "What corporate-card purchase is capped at £2,000 in a single transaction?"
    )

    assert "FIN-CARD-003" in {record.id for record in index.records}
    assert result.action == "use_context"
    assert result.hits[0].record_id == "FIN-CARD-003"
    assert "SEC-KEY-999" not in {record.id for record in index.records}
    assert "SEC-KEY-999" not in result.selected_record_ids


def test_shipped_corpus_production_release_query_ranks_release_record() -> None:
    result = _shipped_index().search(
        "What production-release rollback plan and security scan must be linked "
        "before deployment?"
    )

    assert result.action == "use_context"
    assert result.hits[0].record_id == "ENG-REL-001"
    assert "SEC-KEY-999" not in result.selected_record_ids


def test_shipped_corpus_lexical_collision_ranking_is_deterministic() -> None:
    query = "What written approval is required before a change can proceed?"
    records = _load_knowledge_records(_SHIPPED_RECORDS_PATH)
    first = BM25Index(records, config=_DIAGNOSTIC_CONFIG).search(query)
    second = BM25Index(reversed(records), config=_DIAGNOSTIC_CONFIG).search(query)

    assert first.action == "use_context"
    assert first == second
    assert [hit.record_id for hit in first.hits] == [
        "ENG-REL-001",
        "FIN-EXP-001",
        "HR-LEAVE-001",
        "ENG-INC-002",
        "PROC-VEND-001",
    ]
    assert [hit.rank for hit in first.hits] == [1, 2, 3, 4, 5]
    assert "SEC-KEY-999" not in first.selected_record_ids


def test_shipped_corpus_selected_operating_point_rejects_held_out_oos_pair() -> None:
    index = _shipped_index()
    validation = (
        _answerable(
            "SHIP-VAL-001",
            "Does travel or subsistence spend above £500 need written "
            "budget-owner approval before booking?",
            "FIN-EXP-001",
        ),
        _answerable(
            "SHIP-VAL-002",
            "What production-release rollback plan and security scan must be "
            "linked before deployment?",
            "ENG-REL-001",
        ),
        _answerable(
            "SHIP-VAL-003",
            "How many unused annual leave days may be carried until 31 March?",
            "HR-LEAVE-001",
        ),
        _oos(
            "SHIP-VAL-004",
            "Which restaurant needs written approval before serving lunch?",
            "unsupported_policy",
        ),
        _oos(
            "SHIP-VAL-005",
            "What is the cafeteria lunch menu for Thursday in Leeds?",
            "live_operational_state",
        ),
    )
    held_out_oos = (
        "Who is the budget owner of the orbital greenhouse?",
        "How many days of cricket leave may a stadium carry?",
    )

    report = _select(index, validation)

    assert report.status == "selected"
    assert report.selected_config is not None
    assert report.validation_size == 5
    assert report.answerable_size == 3
    assert report.oos_size == 2
    for query in held_out_oos:
        result = index.search(
            query,
            top_k=report.selected_config.top_k,
            score_threshold=report.selected_config.score_threshold,
        )
        assert result.action == "source_required"
        assert result.hits == ()
        assert result.reason in {
            "query_has_no_index_tokens",
            "no_match_above_threshold",
        }


def test_shipped_corpus_retrieval_outcomes_remain_distinct() -> None:
    index = _shipped_index()
    correct = index.search(
        "Does travel or subsistence spend above £500 need written budget-owner "
        "approval before booking?",
        top_k=1,
    )
    wrong = index.search(
        "Does a £500 subsistence booking need travel approval?",
        top_k=1,
    )
    empty = index.search("xylophone nebula")
    oos_false_load = index.search(
        "Which restaurant needs written approval before serving lunch?"
    )
    report = _select(
        index,
        (
            _answerable(
                "SHIP-OUT-001",
                "Does travel spend above £500 need subsistence approval?",
                "FIN-EXP-001",
            ),
            _answerable(
                "SHIP-OUT-002",
                "Does a £500 subsistence booking need travel approval?",
                "ENG-REL-001",
            ),
            _answerable(
                "SHIP-OUT-003",
                "xylophone nebula quartet",
                "ENG-INC-002",
            ),
            _oos(
                "SHIP-OUT-004",
                "Which restaurant needs written approval before serving lunch?",
            ),
        ),
    )
    top_one = next(
        candidate
        for candidate in report.candidates
        if candidate.config.top_k == 1 and candidate.config.score_threshold == 0.0
    )

    assert _retrieval_outcome(correct, relevant_record_id="FIN-EXP-001") == "correct_record"
    assert _retrieval_outcome(wrong, relevant_record_id="ENG-REL-001") == "wrong_record"
    assert correct.action == "use_context"
    assert wrong.action == "use_context"
    assert wrong.hits[0].record_id == "FIN-EXP-001"
    assert wrong.hits[0].record_id != "ENG-REL-001"
    assert _retrieval_outcome(empty, relevant_record_id="ENG-INC-002") == "empty_retrieval"
    assert empty.action == "source_required"
    assert empty.hits == ()
    assert _retrieval_outcome(oos_false_load, relevant_record_id=None) == "oos_false_load"
    assert oos_false_load.action == "use_context"
    assert oos_false_load.hits
    assert {
        _retrieval_outcome(correct, relevant_record_id="FIN-EXP-001"),
        _retrieval_outcome(wrong, relevant_record_id="ENG-REL-001"),
        _retrieval_outcome(empty, relevant_record_id="ENG-INC-002"),
        _retrieval_outcome(oos_false_load, relevant_record_id=None),
    } == {
        "correct_record",
        "wrong_record",
        "empty_retrieval",
        "oos_false_load",
    }
    assert top_one.metrics.correct_record.count == 1
    assert top_one.metrics.wrong_record.count == 1
    assert top_one.metrics.answerable_empty_retrieval.count == 1
    assert top_one.metrics.oos_false_load.count == 1
    assert (
        top_one.metrics.correct_record.count
        + top_one.metrics.wrong_record.count
        + top_one.metrics.answerable_empty_retrieval.count
        == top_one.metrics.answerable_query_count
    )
