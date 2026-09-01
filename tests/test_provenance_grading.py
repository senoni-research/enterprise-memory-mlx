from __future__ import annotations

from enterprise_memory_mlx.provenance_grading import (
    ProvenanceGradeRequest,
    extract_citations,
    grade_provenance,
)


def _request(**overrides: object) -> ProvenanceGradeRequest:
    payload: dict[str, object] = {
        "generated_output": "Travel spend above £500 needs written approval.",
        "arm": "oracle",
        "suite": "acquisition",
        "probe_kind": "recall",
        "supplied_record_ids": ("FIN-EXP-001",),
        "supplied_source_uris": ("synthetic://northstar/finance/expense-policy/v3",),
        "gold_record_id": "FIN-EXP-001",
        "citation_required": True,
    }
    payload.update(overrides)
    return ProvenanceGradeRequest(**payload)  # type: ignore[arg-type]


def test_valid_supplied_citation_is_structurally_correct_but_not_semantic_support() -> None:
    grade = grade_provenance(
        _request(generated_output="Written approval is required [record: FIN-EXP-001].")
    )

    assert [citation.record_id for citation in grade.citations] == ["FIN-EXP-001"]
    assert grade.unique_cited_record_ids == ("FIN-EXP-001",)
    assert grade.malformed_citation_fragments == ()
    assert grade.citation_membership[0].supplied is True
    assert grade.citation_membership[0].gold is True
    assert grade.citation_correctness == "correct"
    assert grade.citation_completeness == "complete"
    assert grade.hallucinated_citation is False
    assert grade.status == "pass"
    assert grade.support_review_required is True
    assert "support_review_required" in grade.reasons


def test_invented_and_unsupplied_ids_are_hard_failures() -> None:
    invented = grade_provenance(
        _request(generated_output="The rule is stored in [record: INV-999].")
    )
    unsupplied = grade_provenance(
        _request(
            generated_output="See also [record: ENG-REL-001].",
            supplied_record_ids=("FIN-EXP-001",),
        )
    )
    prose_id = grade_provenance(
        _request(generated_output="Ordinary prose mentioning FIN-EXP-001 is not a citation.")
    )
    closed_book_citation = grade_provenance(
        _request(
            generated_output="Closed-book citation [record: FIN-EXP-001].",
            arm="base",
            supplied_record_ids=(),
            supplied_source_uris=(),
            citation_required=False,
            allowed_parametric_record_ids=(),
        )
    )

    assert invented.status == "hard_fail"
    assert invented.hallucinated_citation is True
    assert invented.citation_correctness == "incorrect"
    assert "hallucinated_citation:INV-999" in invented.reasons
    assert unsupplied.status == "hard_fail"
    assert unsupplied.hallucinated_citation is True
    assert unsupplied.unique_cited_record_ids == ("ENG-REL-001",)
    assert prose_id.citations == ()
    assert prose_id.hallucinated_citation is False
    assert prose_id.citation_correctness == "not_applicable"
    assert closed_book_citation.status == "hard_fail"
    assert closed_book_citation.hallucinated_citation is True


def test_malformed_and_duplicate_citations() -> None:
    grade = grade_provenance(
        _request(
            generated_output=(
                "First [record: FIN-EXP-001], again [record: FIN-EXP-001], "
                "then [record:] and [record: FIN EXP 001]."
            )
        )
    )
    assert [citation.record_id for citation in grade.citations] == [
        "FIN-EXP-001",
        "FIN-EXP-001",
    ]
    assert grade.unique_cited_record_ids == ("FIN-EXP-001",)
    assert grade.malformed_citation_fragments == ("[record:]", "[record: FIN EXP 001]")
    assert grade.status == "hard_fail"
    assert grade.citation_correctness == "incorrect"
    assert "malformed_citation" in grade.reasons


def test_citation_to_supplied_but_non_gold_record_is_incomplete() -> None:
    grade = grade_provenance(
        _request(
            arm="full_context",
            generated_output="A related rule appears in [record: ENG-REL-001].",
            supplied_record_ids=("FIN-EXP-001", "ENG-REL-001"),
            gold_record_id="FIN-EXP-001",
            citation_required=True,
        )
    )

    assert grade.hallucinated_citation is False
    assert grade.citation_correctness == "correct"
    assert grade.citation_completeness == "incomplete"
    assert grade.citation_membership[0].supplied is True
    assert grade.citation_membership[0].gold is False
    assert grade.status == "human_or_semantic_review"
    assert grade.support_review_required is True
    assert "citation_incomplete" in grade.reasons


def test_gold_citation_plus_invented_second_citation_hard_fails() -> None:
    grade = grade_provenance(
        _request(
            generated_output=(
                "Use the expense rule [record: FIN-EXP-001] and [record: INV-999]."
            )
        )
    )

    assert grade.unique_cited_record_ids == ("FIN-EXP-001", "INV-999")
    assert grade.citation_completeness == "complete"
    assert grade.citation_correctness == "incorrect"
    assert grade.hallucinated_citation is True
    assert grade.status == "hard_fail"
    assert grade.citation_membership[0].gold is True
    assert grade.citation_membership[1].supplied is False


def test_superseded_record_cited_as_current_is_hard_failure() -> None:
    grade = grade_provenance(
        _request(
            generated_output="The current threshold is in [record: FIN-EXP-001].",
            supplied_record_ids=("FIN-EXP-001", "FIN-EXP-001B"),
            gold_record_id="FIN-EXP-001B",
            active_record_ids=("FIN-EXP-001B",),
            superseded_record_ids=("FIN-EXP-001",),
            suite="supersession",
            probe_kind="temporal",
        )
    )

    assert grade.superseded_citation is True
    assert grade.citation_membership[0].superseded is True
    assert grade.status == "hard_fail"
    assert "superseded_citation:FIN-EXP-001" in grade.reasons


def test_oos_answer_inventing_a_record_id_hard_fails() -> None:
    grade = grade_provenance(
        _request(
            generated_output="The live headcount is 480 people [record: HR-LIVE-001].",
            arm="full_context",
            suite="unknown_oos",
            probe_kind="live_source",
            supplied_record_ids=("FIN-EXP-001",),
            gold_record_id=None,
            out_of_scope=True,
            live_source=True,
            citation_required=False,
        )
    )

    assert grade.hallucinated_citation is True
    assert grade.status == "hard_fail"
    assert grade.unsupported_answer_signal is True
    assert "hallucinated_citation:HR-LIVE-001" in grade.reasons


def test_oos_source_system_referral_is_only_a_signal() -> None:
    grade = grade_provenance(
        _request(
            generated_output=(
                "The supplied records do not support this live question. "
                "Refer the user to the current source system."
            ),
            arm="full_context",
            suite="unknown_oos",
            probe_kind="live_source",
            supplied_record_ids=("FIN-EXP-001",),
            gold_record_id=None,
            out_of_scope=True,
            live_source=True,
            citation_required=False,
        )
    )

    assert grade.citations == ()
    assert grade.hallucinated_citation is False
    assert grade.refusal_detected is True
    assert grade.unsupported_answer_signal is False
    assert grade.citation_completeness == "not_required"
    assert grade.status == "human_or_semantic_review"
    assert grade.support_review_required is True
    assert "refusal_detected" in grade.reasons


def test_base_answer_claiming_retrieval_is_a_hard_failure() -> None:
    grade = grade_provenance(
        _request(
            generated_output="I retrieved the supplied document and the threshold is £500.",
            arm="base",
            supplied_record_ids=(),
            supplied_source_uris=(),
            gold_record_id="FIN-EXP-001",
            citation_required=False,
        )
    )

    assert grade.false_retrieval_claim is True
    assert grade.status == "hard_fail"
    assert "false_retrieval_claim" in grade.reasons


def test_benign_checked_calculation_is_not_a_retrieval_claim() -> None:
    grade = grade_provenance(
        _request(
            generated_output="I checked the calculation: twenty-five plus five is thirty.",
            arm="base",
            supplied_record_ids=(),
            supplied_source_uris=(),
            gold_record_id="HR-LEAVE-001",
            citation_required=False,
        )
    )

    assert grade.false_retrieval_claim is False
    assert grade.hallucinated_citation is False
    assert grade.status == "pass"
    assert grade.support_review_required is True


def test_required_versus_optional_citation_completeness() -> None:
    cited = "The leave carry-over date is 31 March [record: HR-LEAVE-001]."
    uncited = "The leave carry-over date is 31 March."
    required = grade_provenance(
        _request(
            generated_output=cited,
            supplied_record_ids=("HR-LEAVE-001",),
            gold_record_id="HR-LEAVE-001",
            citation_required=True,
        )
    )
    required_missing = grade_provenance(
        _request(
            generated_output=uncited,
            supplied_record_ids=("HR-LEAVE-001",),
            gold_record_id="HR-LEAVE-001",
            citation_required=True,
        )
    )
    optional_missing = grade_provenance(
        _request(
            generated_output=uncited,
            supplied_record_ids=("HR-LEAVE-001",),
            gold_record_id="HR-LEAVE-001",
            citation_required=False,
        )
    )
    optional_complete = grade_provenance(
        _request(
            generated_output=cited,
            supplied_record_ids=("HR-LEAVE-001",),
            gold_record_id="HR-LEAVE-001",
            citation_required=False,
        )
    )

    assert required.citation_completeness == "complete"
    assert required_missing.citation_completeness == "incomplete"
    assert required_missing.status == "human_or_semantic_review"
    assert optional_missing.citation_completeness == "not_required"
    assert optional_complete.citation_completeness == "complete"
    assert optional_complete.citation_correctness == "correct"
    assert optional_complete.support_review_required is True


def test_empty_output_is_reviewable_and_has_no_citations() -> None:
    grade = grade_provenance(_request(generated_output="   "))

    assert grade.citations == ()
    assert grade.malformed_citation_fragments == ()
    assert grade.unique_cited_record_ids == ()
    assert grade.citation_correctness == "not_applicable"
    assert grade.citation_completeness == "incomplete"
    assert grade.refusal_detected is False
    assert grade.unsupported_answer_signal is False
    assert grade.status == "human_or_semantic_review"
    assert "empty_output" in grade.reasons
    assert grade.support_review_required is True


def test_record_id_case_and_whitespace_are_normalized() -> None:
    grade = grade_provenance(
        _request(generated_output="Cite the expense rule [RECORD:  fin-exp-001 ].")
    )

    assert grade.citations[0].record_id == "fin-exp-001"
    assert grade.unique_cited_record_ids == ("FIN-EXP-001",)
    assert grade.citation_membership[0].supplied is True
    assert grade.citation_membership[0].gold is True
    assert grade.citation_correctness == "correct"
    assert grade.hallucinated_citation is False


def test_repeated_calls_are_deterministic() -> None:
    request = _request(
        generated_output=(
            "Use [record: FIN-EXP-001] and ignore [See Policy] while "
            "repeating [record: FIN-EXP-001]."
        )
    )

    assert grade_provenance(request) == grade_provenance(request)
    assert extract_citations(request.generated_output) == extract_citations(
        request.generated_output
    )


def test_parametric_allowlist_is_id_specific() -> None:
    permitted = grade_provenance(
        _request(
            generated_output="Closed-book recall of the expense rule [record: FIN-EXP-001].",
            arm="base",
            supplied_record_ids=(),
            supplied_source_uris=(),
            citation_required=False,
            allowed_parametric_record_ids=("FIN-EXP-001",),
        )
    )
    other_id = grade_provenance(
        _request(
            generated_output="Closed-book recall of a release rule [record: ENG-REL-001].",
            arm="base",
            supplied_record_ids=(),
            supplied_source_uris=(),
            citation_required=False,
            allowed_parametric_record_ids=("FIN-EXP-001",),
        )
    )

    assert permitted.hallucinated_citation is False
    assert permitted.citation_correctness == "correct"
    assert permitted.citation_membership[0].supplied is False
    assert permitted.status == "pass"
    assert other_id.hallucinated_citation is True
    assert other_id.status == "hard_fail"
    assert "hallucinated_citation:ENG-REL-001" in other_id.reasons


def test_retrieval_claim_depends_on_supplied_context_not_arm() -> None:
    claim = "I retrieved the supplied document and the threshold is £500."
    oracle_without_context = grade_provenance(
        _request(
            generated_output=claim,
            arm="oracle",
            supplied_record_ids=(),
            supplied_source_uris=(),
            citation_required=False,
        )
    )
    base_with_context = grade_provenance(
        _request(
            generated_output=claim,
            arm="base",
            supplied_record_ids=("FIN-EXP-001",),
            citation_required=False,
        )
    )

    assert oracle_without_context.false_retrieval_claim is True
    assert oracle_without_context.status == "hard_fail"
    assert base_with_context.false_retrieval_claim is False
    assert base_with_context.status == "pass"


def test_historical_marker_must_share_sentence_with_superseded_citation() -> None:
    shared_kwargs = {
        "supplied_record_ids": ("FIN-EXP-001", "FIN-EXP-001B"),
        "gold_record_id": "FIN-EXP-001B",
        "active_record_ids": ("FIN-EXP-001B",),
        "superseded_record_ids": ("FIN-EXP-001",),
        "suite": "supersession",
        "probe_kind": "temporal",
        "citation_required": True,
    }
    adjacent_marker = grade_provenance(
        _request(
            generated_output=(
                "The previous rule was superseded. The current threshold is in "
                "[record: FIN-EXP-001]."
            ),
            **shared_kwargs,
        )
    )
    same_sentence = grade_provenance(
        _request(
            generated_output=(
                "The previous threshold in [record: FIN-EXP-001] was superseded."
            ),
            **shared_kwargs,
        )
    )
    later_current_repeat = grade_provenance(
        _request(
            generated_output=(
                "The previous threshold in [record: FIN-EXP-001] was superseded. "
                "The current threshold is still [record: FIN-EXP-001]."
            ),
            **shared_kwargs,
        )
    )

    assert adjacent_marker.superseded_citation is True
    assert adjacent_marker.status == "hard_fail"
    assert same_sentence.superseded_citation is False
    assert same_sentence.citation_membership[0].superseded is True
    assert same_sentence.status == "human_or_semantic_review"
    assert later_current_repeat.superseded_citation is True
    assert later_current_repeat.status == "hard_fail"
