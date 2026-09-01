from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.compiler import load_records
from enterprise_memory_mlx.leakage import (
    assert_no_leakage,
    collect_training_question_texts,
    scan_leakage,
)
from enterprise_memory_mlx.split_contract import EvalQuestion, load_eval_suites


def _eval_question(question: str, expected: str, question_id: str = "Q-1") -> EvalQuestion:
    return EvalQuestion.from_dict(
        {
            "question_id": question_id,
            "suite": "acquisition",
            "record_id": "FIN-EXP-001",
            "question_family_id": "FIN-EXP-001:afX",
            "probe_kind": "recall",
            "question": question,
            "expected": expected,
            "generator": {"kind": "human", "identity": "reviewer-a"},
        }
    )


def test_shipped_fixtures_have_no_leakage(project_root: Path) -> None:
    records = load_records(project_root / "knowledge")
    training_texts = collect_training_question_texts(records, project_root / "knowledge")
    suites = load_eval_suites(project_root / "knowledge" / "eval_frozen")
    report = scan_leakage(training_texts, suites.all_questions())
    assert report.passed, "\n".join(
        f"{finding.check}: {finding.eval_question_id} vs {finding.train_ref} - {finding.detail}"
        for finding in report.findings
    )
    # The audit list must exist even when every automated check passes.
    assert report.audit_pairs


def test_exact_duplicate_detected(project_root: Path) -> None:
    records = load_records(project_root / "knowledge")
    training_texts = collect_training_question_texts(records, project_root / "knowledge")
    stolen = records[0].questions[0].question
    report = scan_leakage(training_texts, [_eval_question(stolen, "Some answer text here.")])
    assert not report.passed
    assert any(finding.check == "exact_duplicate" for finding in report.findings)
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(report)


def test_near_duplicate_detected() -> None:
    training_texts = {"r:q0": "Who must approve travel costing more than £500?"}
    near = _eval_question(
        "Who must approve travel costing more than £600?",
        "The budget owner approves it in writing before booking.",
    )
    report = scan_leakage(training_texts, [near])
    assert any(finding.check.startswith("near_duplicate") for finding in report.findings)


def test_template_instantiations_are_covered(project_root: Path) -> None:
    records = load_records(project_root / "knowledge")
    training_texts = collect_training_question_texts(records, project_root / "knowledge")
    # The legacy compiler pads alignment data with template questions; an eval
    # question colliding with an instantiated template must be caught.
    template = "What is the approved company rule for Travel and subsistence approval?"
    report = scan_leakage(training_texts, [_eval_question(template, "Some answer text here.")])
    assert any(finding.check == "exact_duplicate" for finding in report.findings)


def test_answer_cue_detected() -> None:
    report = scan_leakage(
        {},
        [
            _eval_question(
                "Is it true that the relevant budget owner must give written approval "
                "before booking travel above the threshold?",
                "The relevant budget owner must give written approval before booking.",
            )
        ],
    )
    assert any(finding.check == "answer_cue" for finding in report.findings)


def test_unrelated_question_passes() -> None:
    training_texts = {"r:q0": "Who must approve travel costing more than £500?"}
    clean = _eval_question(
        "Which team owns the incident review calendar?",
        "The engineering operations team owns it.",
    )
    report = scan_leakage(training_texts, [clean])
    assert report.passed
