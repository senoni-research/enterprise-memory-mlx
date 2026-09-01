from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enterprise_memory_mlx import cli
from enterprise_memory_mlx.grading import (
    grade_benchmark_artifact,
    judge_benchmark_artifact,
    write_grading_report,
)
from enterprise_memory_mlx.semantic_judging import (
    JudgeBackendResult,
    instance_prompt_hash,
)
from enterprise_memory_mlx.split_contract import load_eval_suites


class AlwaysPassJudge:
    """Fake local judge that would upgrade anything to 1.0 if allowed."""

    def __init__(self, model_id: str, model_family: str) -> None:
        self.model_id = model_id
        self.model_family = model_family
        self.revision = "rev-fake"
        self.local_only = True
        self.calls: list[str] = []

    def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None

    def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        rubric_prompt_version: str,
        max_output_tokens: int,
    ) -> JudgeBackendResult:
        self.calls.append(question)
        return JudgeBackendResult(
            raw_text=json.dumps({"score": 1.0, "reason": "fake agreement"}),
            model_id=self.model_id,
            revision=self.revision,
            prompt_hash=instance_prompt_hash(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                rubric_prompt_version=rubric_prompt_version,
            ),
            latency_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )


def _artifact(
    tmp_path: Path,
    eval_dir: Path,
    *,
    suite_name: str = "acquisition",
    arm: str = "base",
) -> tuple[Path, object]:
    suites = load_eval_suites(eval_dir)
    questions = getattr(suites, suite_name)
    fixture = json.loads(
        (eval_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        {
            "question_id": question.question_id,
            "arm": arm,
            "suite": question.suite,
            "question": question.question,
            "as_of_date": question.as_of_date,
            "generation_status": "generated",
            "output": question.expected,
            "selected_record_ids": (
                [question.record_id]
                if arm in {"oracle", "full_context"} and question.record_id
                else []
            ),
            "source_uris": [],
            "retrieval_label": "not_applicable",
        }
        for question in questions
    ]
    payload = {
        "graded": False,
        "fixture_hash": fixture["combined_hash"],
        "config": {"suites": [suite_name], "arms": [arm]},
        "results": rows,
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, suites


def test_deterministic_grading_never_invents_semantic_accuracy(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)

    report = grade_benchmark_artifact(raw, eval_dir, suites)

    assert len(report.rows) == len(suites.acquisition)
    assert report.promotion_eligible is False
    assert all(row.deterministic_score in {None, 0.0} for row in report.rows)
    assert all(
        row.status in {"deterministic_hard_fail", "semantic_review_required"}
        for row in report.rows
    )
    assert all(item["semantic_accuracy"] is None for item in report.aggregates)
    assert all(item["status"] != "pass" for item in report.gates)


def test_wrong_critical_currency_is_a_certain_hard_failure(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    payload["results"][0]["output"] = (
        "Written approval is needed above £999 from the budget owner."
    )
    raw.write_text(json.dumps(payload), encoding="utf-8")

    report = grade_benchmark_artifact(raw, eval_dir, suites)

    first = report.rows[0]
    assert first.status == "deterministic_hard_fail"
    assert first.deterministic_score == 0.0
    assert first.strict is not None and first.strict.status == "hard_fail"


def test_invented_base_citation_is_a_provenance_hard_failure(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    payload["results"][0]["output"] += " [record: INVENTED-999]"
    raw.write_text(json.dumps(payload), encoding="utf-8")

    report = grade_benchmark_artifact(raw, eval_dir, suites)

    first = report.rows[0]
    assert first.status == "deterministic_hard_fail"
    assert first.provenance is not None
    assert first.provenance.hallucinated_citation is True


def test_generation_failure_remains_unscored(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    payload["results"][0].update(
        {
            "generation_status": "failed",
            "generation_error": "backend stopped",
            "output": None,
        }
    )
    raw.write_text(json.dumps(payload), encoding="utf-8")

    report = grade_benchmark_artifact(raw, eval_dir, suites)

    first = report.rows[0]
    assert first.status == "unscored_generation_failure"
    assert first.deterministic_score is None
    assert first.strict is None
    assert first.provenance is None
    assert first.reasons == ("backend stopped",)


def test_temporal_v2_preserves_as_of_date(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen" / "v2"
    raw, suites = _artifact(
        tmp_path,
        eval_dir,
        suite_name="supersession",
        arm="oracle",
    )

    report = grade_benchmark_artifact(raw, eval_dir, suites)

    assert {row.as_of_date for row in report.rows} == {"2026-10-15"}
    assert all(row.suite == "supersession" for row in report.rows)
    assert all(item["status"] != "pass" for item in report.gates)


def test_fixture_hash_and_result_matrix_fail_closed(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    payload["fixture_hash"] = "0" * 64
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixture hash"):
        grade_benchmark_artifact(raw, eval_dir, suites)

    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    payload["results"].pop()
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix mismatch"):
        grade_benchmark_artifact(raw, eval_dir, suites)


def test_raw_artifact_remains_immutable_and_report_is_separate(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    report = grade_benchmark_artifact(raw, eval_dir, suites)

    json_path, markdown_path = write_grading_report(report, tmp_path / "graded")

    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["graded"] is True
    assert payload["mode"] == "deterministic_only"
    assert payload["promotion_eligible"] is False
    assert "No accuracy or superiority claim" in markdown_path.read_text()


def test_judge_handoff_binds_hard_failures_and_never_upgrades(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    payload = json.loads(raw.read_text())
    # Row 0: certain typed hard failure; row 1: generation failure; rest clean.
    payload["results"][0]["output"] = "Approval is needed above £999 only."
    payload["results"][1].update(
        {"generation_status": "failed", "generation_error": "boom", "output": None}
    )
    raw.write_text(json.dumps(payload), encoding="utf-8")
    judges = (
        AlwaysPassJudge("gemma-fake", "gemma"),
        AlwaysPassJudge("mistral-fake", "mistral"),
    )

    report, outcomes = judge_benchmark_artifact(
        raw,
        eval_dir,
        suites,
        judges=judges,
        evaluated_model_family="qwen",
        allow_uncertified_machinery=True,
    )

    by_id = {outcome.question_id: outcome for outcome in outcomes}
    hard_id = report.rows[0].question_id
    failed_id = report.rows[1].question_id

    hard = by_id[hard_id]
    assert hard.deterministic_status == "deterministic_hard_fail"
    assert hard.final_score == 0.0
    assert hard.score_source == "deterministic_hard_fail"
    assert hard.dual is not None and hard.dual.status == "deterministic_hard_fail"
    assert hard.dual.parsed == ()  # judges were never invoked for this row

    unscored = by_id[failed_id]
    assert unscored.score_source == "unscored_generation_failure"
    assert unscored.final_score is None
    assert unscored.dual is None

    reviewable = [
        outcome
        for outcome in outcomes
        if outcome.deterministic_status == "semantic_review_required"
    ]
    assert reviewable
    assert all(outcome.final_score == 1.0 for outcome in reviewable)
    assert all(outcome.score_source == "judge_agreement" for outcome in reviewable)
    # Judges saw exactly the reviewable rows: no hard-fail or unscored rows.
    assert len(judges[0].calls) == len(reviewable)
    assert len(judges[1].calls) == len(reviewable)
    hard_question = next(
        item.question for item in suites.acquisition if item.question_id == hard_id
    )
    assert hard_question not in judges[0].calls


def test_judge_handoff_requires_explicit_machinery_acknowledgement(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, suites = _artifact(tmp_path, eval_dir)
    judges = (
        AlwaysPassJudge("gemma-fake", "gemma"),
        AlwaysPassJudge("mistral-fake", "mistral"),
    )

    with pytest.raises(ValueError, match="certified judges"):
        judge_benchmark_artifact(
            raw,
            eval_dir,
            suites,
            judges=judges,
            evaluated_model_family="qwen",
        )
    assert judges[0].calls == []


def test_cli_grade_writes_deterministic_report(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    raw, _suites = _artifact(tmp_path, eval_dir)
    output = tmp_path / "grading"

    exit_code = cli.main(
        [
            "--root",
            str(project_root),
            "grade",
            "--benchmark",
            str(raw),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    assert list(output.glob("deterministic-grading-*.json"))
    assert list(output.glob("deterministic-grading-*.md"))
