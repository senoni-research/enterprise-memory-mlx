"""Historical-reference tests for the scientifically invalid lexical scorer."""

from __future__ import annotations

import pytest

from enterprise_memory_mlx.evaluation import score_answer
from enterprise_memory_mlx.legacy_guard import LegacyPipelineDisabledError


def test_score_answer_fails_closed_by_default() -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="scientifically invalid"):
        score_answer(output="anything", expected="anything", keywords=[])


def test_legacy_known_answer_passes_on_keywords() -> None:
    score = score_answer(
        output="Written approval from the budget owner is required before booking.",
        expected="The relevant budget owner must give written approval before the booking.",
        keywords=["budget owner", "written approval", "before"],
        allow_scientifically_invalid=True,
    )
    assert score.passed
    assert score.keyword_coverage == 1.0


def test_legacy_unknown_answer_requires_refusal() -> None:
    good = score_answer(
        output="I do not have an authoritative record. Check the current source system.",
        expected="Check the source.",
        keywords=[],
        kind="unknown",
        allow_scientifically_invalid=True,
    )
    bad = score_answer(
        output="The bank balance is probably £2 million.",
        expected="Check the source.",
        keywords=[],
        kind="unknown",
        allow_scientifically_invalid=True,
    )
    assert good.passed
    assert not bad.passed


def test_legacy_scorer_documented_defect_substring_pass() -> None:
    """Known invalidity: a bare substring of the expected answer passes."""
    score = score_answer(
        output="no",
        expected="No. A transaction must not be split to avoid the threshold.",
        keywords=[],
        allow_scientifically_invalid=True,
    )
    assert score.passed  # documents the defect that forced replacement
