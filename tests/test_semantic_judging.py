from __future__ import annotations

import json
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from enterprise_memory_mlx.semantic_judging import (
    AUDIT_STRATUM,
    HEADLINE_STRATUM,
    HUMAN_LABEL_OVERLAY_FILE,
    LABELED_DATASET_FILE,
    RUBRIC_PROMPT_VERSION,
    SOURCE_CASES_FILE,
    CalibrationCase,
    CalibrationValidationError,
    ExploratoryCertificationThresholds,
    JudgeBackendResult,
    ParsedJudgeOutput,
    build_rubric_prompt,
    calibrate_judges,
    clopper_pearson_two_sided,
    compute_judge_metrics,
    convert_candidate_file,
    grade_with_dual_judges,
    instance_prompt_hash,
    invocation_was_sequential,
    parse_judge_output,
    prepare_labeled_dataset,
    quadratic_weighted_kappa,
    recorded_invocation_order,
    recorded_lifecycle_events,
    reset_invocation_trace,
    rubric_prompt_hash,
    split_case_bytes,
)
from enterprise_memory_mlx.utils import sha256_json


@dataclass
class FakeJudgeBackend:
    model_id: str
    model_family: str
    revision: str
    outputs: list[str] | dict[str, str]
    local_only: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    acquired: bool = False
    acquire_count: int = 0
    release_count: int = 0
    _index: int = 0

    def acquire(self) -> None:
        if self.acquired:
            raise AssertionError(f"{self.model_id} acquired while already loaded")
        self.acquired = True
        self.acquire_count += 1

    def release(self) -> None:
        if not self.acquired:
            raise AssertionError(f"{self.model_id} released while not loaded")
        self.acquired = False
        self.release_count += 1

    def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        rubric_prompt_version: str,
        max_output_tokens: int,
    ) -> JudgeBackendResult:
        if not self.acquired:
            raise AssertionError(f"{self.model_id} judged without acquire")
        self.calls.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "candidate_answer": candidate_answer,
                "rubric_prompt_version": rubric_prompt_version,
                "max_output_tokens": max_output_tokens,
            }
        )
        if isinstance(self.outputs, dict):
            raw = self.outputs[candidate_answer]
        else:
            raw = self.outputs[self._index]
            self._index += 1
        prompt_hash = instance_prompt_hash(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            rubric_prompt_version=rubric_prompt_version,
        )
        return JudgeBackendResult(
            raw_text=raw,
            model_id=self.model_id,
            revision=self.revision,
            prompt_hash=prompt_hash,
            latency_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )


def _legal(score: float, reason: str = "Evidence-based reason") -> str:
    return json.dumps({"score": score, "reason": reason})


def _thresholds(
    *,
    min_exact_agreement: float = 0.85,
    min_weighted_kappa: float = 0.80,
    max_incorrect_to_pass_rate: float = 0.02,
    max_invalid_output_rate: float = 0.0,
    max_critical_false_pass_rate: float = 0.05,
) -> ExploratoryCertificationThresholds:
    return ExploratoryCertificationThresholds(
        min_exact_agreement=min_exact_agreement,
        min_weighted_kappa=min_weighted_kappa,
        max_incorrect_to_pass_rate=max_incorrect_to_pass_rate,
        max_invalid_output_rate=max_invalid_output_rate,
        max_critical_false_pass_rate=max_critical_false_pass_rate,
    )


def _case(
    case_id: str,
    human: float,
    *,
    stratum: str = HEADLINE_STRATUM,
    domain: str = "finance",
    error_categories: tuple[str, ...] = (),
    disputed: bool = False,
    usable_for_training: bool = False,
    approved: bool = True,
    reason: str | None = "Human-approved reason",
    reviewers: tuple[str, ...] = ("alice", "bob"),
    proposed: float | None = None,
    proposed_reason: str | None = None,
    question: str | None = None,
    candidate: str | None = None,
    case_family_id: str | None = None,
) -> CalibrationCase:
    return CalibrationCase(
        case_id=case_id,
        question=question or f"Question {case_id}",
        reference_answer="Reference",
        candidate_answer=candidate or f"Candidate {case_id}",
        human_semantic_score=human,
        human_reason=reason,
        human_reviewers=reviewers,
        human_approved=approved,
        error_categories=error_categories,
        domain=domain,
        certification_stratum=stratum,
        disputed=disputed,
        usable_for_training=usable_for_training,
        deterministic_hard_failure=stratum == AUDIT_STRATUM,
        proposed_semantic_score=proposed,
        proposed_reason=proposed_reason,
        source_record_ids=("FIN-EXP-001",),
        case_family_id=case_family_id or case_id,
    )


def _balanced_cases() -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    scores = (1.0, 0.5, 0.0)
    for index, score in enumerate(scores):
        cases.append(
            _case(
                f"C-{index + 1}",
                score,
                error_categories=() if score == 1.0 else ("incomplete_multipart",),
            )
        )
    return cases


def _bundle(cases: list[CalibrationCase]) -> tuple[bytes, bytes, dict[str, Any], Any]:
    source_bytes, overlay_bytes = split_case_bytes(cases)
    _parsed, _labeled, hashes = prepare_labeled_dataset(source_bytes, overlay_bytes)
    manifest = {
        "files": {
            SOURCE_CASES_FILE: hashes.source_cases_hash,
            HUMAN_LABEL_OVERLAY_FILE: hashes.human_label_overlay_hash,
            LABELED_DATASET_FILE: hashes.labeled_dataset_hash,
        },
        "schema_version": 1,
    }
    return source_bytes, overlay_bytes, manifest, hashes


def _calibrate(
    cases: list[CalibrationCase],
    judge_a_scores: list[float] | list[str],
    judge_b_scores: list[float] | list[str],
    *,
    thresholds: ExploratoryCertificationThresholds | None = None,
    min_class_count: int = 1,
    evaluated_model_family: str = "qwen",
    families: tuple[str, str] = ("gemma", "mistral"),
) -> Any:
    reset_invocation_trace()
    source_bytes, overlay_bytes, manifest, hashes = _bundle(cases)
    outputs_a = [
        item if isinstance(item, str) else _legal(item) for item in judge_a_scores
    ]
    outputs_b = [
        item if isinstance(item, str) else _legal(item) for item in judge_b_scores
    ]
    judges = (
        FakeJudgeBackend("gemma-fake", families[0], "rev-a", outputs_a),
        FakeJudgeBackend("mistral-fake", families[1], "rev-b", outputs_b),
    )
    return calibrate_judges(
        source_cases_bytes=source_bytes,
        human_label_overlay_bytes=overlay_bytes,
        manifest=manifest,
        expected_source_cases_hash=hashes.source_cases_hash,
        expected_human_label_overlay_hash=hashes.human_label_overlay_hash,
        expected_labeled_dataset_hash=hashes.labeled_dataset_hash,
        expected_manifest_hash=sha256_json(manifest),
        judges=judges,
        evaluated_model_family=evaluated_model_family,
        thresholds=thresholds or _thresholds(),
        min_class_count=min_class_count,
    )


def test_parser_accepts_only_legal_schema_and_scores() -> None:
    for score in (0.0, 0.5, 1.0, 0, 1):
        parsed = parse_judge_output(_legal(float(score) if score in {0, 1} else score))
        if score in {0, 1}:
            parsed = parse_judge_output(json.dumps({"score": score, "reason": "ok"}))
        assert parsed.valid
        assert parsed.score in {0.0, 0.5, 1.0}
        assert parsed.reason == ("ok" if score in {0, 1} else "Evidence-based reason")

    reordered = parse_judge_output('{"reason": "Both fields present", "score": 0.5}')
    assert reordered.valid
    assert reordered.score == 0.5


def test_invalid_and_malformed_outputs_remain_invalid() -> None:
    invalid_payloads = (
        "The candidate is fully correct.",
        '{"score": 1.0, "reason": "ok"} trailing prose',
        '{"score": 1.0, "reason": "ok"}{"score": 0.0, "reason": "no"}',
        '[{"score": 1.0, "reason": "ok"}]',
        '{"score": 1.0}',
        '{"score": 1.0, "reason": "ok", "extra": true}',
        '{"score": "1.0", "reason": "ok"}',
        '{"score": true, "reason": "ok"}',
        '{"score": NaN, "reason": "ok"}',
        '{"score": 0.25, "reason": "ok"}',
        '{"score": 1.0, "reason": ""}',
        '{"score": 1.0, "reason": "   "}',
        "",
        "{",
        "null",
    )
    for raw in invalid_payloads:
        parsed = parse_judge_output(raw)
        assert parsed.valid is False
        assert parsed.score is None
        assert parsed.error


def test_missing_human_labels_block_calibration() -> None:
    cases = _balanced_cases()
    cases[0] = _case("C-1", 1.0, approved=False, reason=None)
    with pytest.raises(CalibrationValidationError, match="missing human labels"):
        _calibrate(cases, [1.0, 0.5, 0.0], [1.0, 0.5, 0.0])


def test_model_proposed_labels_cannot_replace_human_labels() -> None:
    cases = [
        CalibrationCase(
            case_id="C-1",
            question="Question C-1",
            reference_answer="Reference",
            candidate_answer="Candidate C-1",
            human_semantic_score=None,
            human_reason=None,
            human_reviewers=("alice", "bob"),
            human_approved=True,
            proposed_semantic_score=1.0,
            proposed_reason="Model drafted this label",
            source_record_ids=("FIN-EXP-001",),
        ),
        _case("C-2", 0.5),
        _case("C-3", 0.0),
    ]
    with pytest.raises(
        CalibrationValidationError,
        match="model-proposed labels cannot replace human labels",
    ):
        _calibrate(cases, [1.0, 0.5, 0.0], [1.0, 0.5, 0.0])


def test_same_family_judge_and_evaluated_model_is_rejected() -> None:
    cases = _balanced_cases()
    with pytest.raises(
        CalibrationValidationError,
        match="differ from the model being evaluated",
    ):
        _calibrate(
            cases,
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            evaluated_model_family="gemma",
        )


def test_same_family_dual_judges_are_rejected() -> None:
    cases = _balanced_cases()
    with pytest.raises(
        CalibrationValidationError,
        match="judge families must differ",
    ):
        _calibrate(
            cases,
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            families=("gemma", "GEMMA"),
        )


def test_confusion_matrix_and_named_metrics_match_hand_computation() -> None:
    # Human\Judge  0.0  0.5  1.0
    # 0.0           2    0    1
    # 0.5           0    2    0
    # 1.0           1    0    2
    humans = (0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0)
    judges = (0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0)
    cases = [
        _case(f"H-{index}", human, domain="finance" if human != 0.5 else "hr")
        for index, human in enumerate(humans)
    ]
    parsed = [
        ParsedJudgeOutput(valid=True, score=score, reason="ok", error=None)
        for score in judges
    ]
    metrics = compute_judge_metrics(cases, parsed)

    assert metrics.confusion_matrix.counts == (
        (2, 0, 1),
        (0, 2, 0),
        (1, 0, 2),
    )
    assert metrics.exact_agreement.numerator == 6
    assert metrics.exact_agreement.denominator == 8
    assert metrics.exact_agreement.rate == 0.75
    assert metrics.adjacent_disagreement.rate == 0.0
    assert metrics.two_step_disagreement.numerator == 2
    assert metrics.two_step_disagreement.rate == 0.25
    assert metrics.mean_absolute_ordinal_error == 0.5

    by_score = {item.score: item for item in metrics.class_precision_recall}
    assert by_score[0.0].precision.rate == pytest.approx(2 / 3)
    assert by_score[0.0].recall.rate == pytest.approx(2 / 3)
    assert by_score[0.5].precision.rate == 1.0
    assert by_score[0.5].recall.rate == 1.0
    assert by_score[1.0].precision.rate == pytest.approx(2 / 3)
    assert by_score[1.0].recall.rate == pytest.approx(2 / 3)

    assert metrics.fully_correct_false_pass_rate.numerator == 1
    assert metrics.fully_correct_false_pass_rate.denominator == 5
    assert metrics.incorrect_to_pass_rate.numerator == 1
    assert metrics.incorrect_to_pass_rate.denominator == 3
    assert metrics.false_fail_rate.numerator == 1
    assert metrics.false_fail_rate.denominator == 3
    assert metrics.invalid_output_rate.rate == 0.0
    assert metrics.weighted_kappa == pytest.approx(1 / 3)
    assert metrics.weighted_kappa_ci_low is not None
    assert metrics.weighted_kappa_ci_high is not None
    assert metrics.by_domain
    assert metrics.by_error_category


def test_weighted_kappa_edge_cases() -> None:
    perfect_human = (0.0, 0.5, 1.0, 0.0, 0.5, 1.0)
    assert quadratic_weighted_kappa(perfect_human, perfect_human) == pytest.approx(1.0)

    undefined = quadratic_weighted_kappa((0.0,) * 6, (0.0,) * 6)
    assert undefined is None

    chance = quadratic_weighted_kappa((0.0,) * 5, (1.0,) * 5)
    assert chance == pytest.approx(0.0)

    empty = compute_judge_metrics([], [])
    assert empty.weighted_kappa is None
    assert empty.mean_absolute_ordinal_error is None


def test_false_pass_and_false_fail_denominators() -> None:
    cases = [
        _case("D-1", 1.0),
        _case("D-2", 1.0),
        _case("D-3", 0.5),
        _case("D-4", 0.0),
        _case("D-5", 0.0),
    ]
    parsed = [
        ParsedJudgeOutput(valid=True, score=score, reason="ok", error=None)
        for score in (1.0, 0.0, 1.0, 1.0, 0.0)
    ]
    metrics = compute_judge_metrics(cases, parsed)

    # False-pass uses human != 1.0, not all rows.
    assert metrics.fully_correct_false_pass_rate.numerator == 2
    assert metrics.fully_correct_false_pass_rate.denominator == 3
    # Incorrect-to-pass uses human == 0.0.
    assert metrics.incorrect_to_pass_rate.numerator == 1
    assert metrics.incorrect_to_pass_rate.denominator == 2
    # False-fail uses human == 1.0.
    assert metrics.false_fail_rate.numerator == 1
    assert metrics.false_fail_rate.denominator == 2


def test_threshold_pass_fail_and_no_single_judge_fallback() -> None:
    cases = [
        _case("T-1", 1.0),
        _case("T-2", 0.5, error_categories=("incomplete_multipart",)),
        _case("T-3", 0.0, error_categories=("unsupported_exception",)),
        _case("T-4", 1.0),
        _case("T-5", 0.5, error_categories=("incomplete_multipart",)),
        _case("T-6", 0.0, error_categories=("unsupported_exception",)),
    ]
    perfect = [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
    leaking = [1.0, 0.5, 1.0, 1.0, 0.5, 0.0]
    result = _calibrate(
        cases,
        perfect,
        leaking,
        thresholds=_thresholds(
            min_exact_agreement=0.5,
            min_weighted_kappa=0.0,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    assert result.status == "not_certified"
    assert any("incorrect-to-pass" in item for item in result.failed_requirements)
    assert any("no single-judge fallback" in item for item in result.failed_requirements)
    assert result.per_judge_metrics[0].passed_thresholds is True
    assert result.per_judge_metrics[1].passed_thresholds is False

    certified = _calibrate(
        cases,
        perfect,
        perfect,
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    assert certified.status == "certified_for_exploratory_grading"
    assert certified.failed_requirements == ()


def test_judge_agreement_versus_human_adjudication() -> None:
    judges = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(1.0, "Gemma agrees")]),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(1.0, "Mistral agrees")]),
    )
    agreed = grade_with_dual_judges(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        judges=judges,
        evaluated_model_family="qwen",
    )
    assert agreed.status == "agreed"
    assert agreed.score == 1.0
    assert agreed.reasons == ("Gemma agrees", "Mistral agrees")
    payload = agreed.to_dict()
    assert "raw_results" in payload
    assert payload["raw_results"][0]["model_id"] == "gemma-fake"
    assert payload["raw_results"][1]["revision"] == "rev-b"
    assert payload["raw_results"][0]["prompt_hash"]
    assert payload["raw_results"][0]["prompt_tokens"] == 0

    split = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(0.0, "zero")]),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(1.0, "one")]),
    )
    disagreed = grade_with_dual_judges(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        judges=split,
        evaluated_model_family="qwen",
    )
    assert disagreed.status == "human_adjudication_required"
    assert disagreed.score is None
    assert disagreed.score != 0.5

    invalid = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(0.5)]),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", ["not json"]),
    )
    needs_human = grade_with_dual_judges(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        judges=invalid,
        evaluated_model_family="qwen",
    )
    assert needs_human.status == "human_adjudication_required"
    assert needs_human.score is None


def test_hard_failure_inputs_cannot_be_upgraded() -> None:
    judges = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(1.0, "would upgrade")]),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(1.0, "would upgrade")]),
    )
    grade = grade_with_dual_judges(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        judges=judges,
        evaluated_model_family="gemma",
        deterministic_hard_failure=True,
    )
    assert grade.status == "deterministic_hard_fail"
    assert grade.score == 0.0
    assert grade.parsed == ()
    assert grade.raw_results == ()
    assert judges[0].calls == []
    assert judges[1].calls == []
    assert judges[0].acquire_count == 0
    assert judges[1].acquire_count == 0
    assert grade.to_dict()["raw_results"] == []


def test_determinism_and_prompt_hashing() -> None:
    first = rubric_prompt_hash()
    second = rubric_prompt_hash(RUBRIC_PROMPT_VERSION)
    assert first == second
    assert len(first) == 64
    assert "cannot be upgraded" in build_rubric_prompt()

    prompt_a = instance_prompt_hash(
        question="Q",
        reference_answer="R",
        candidate_answer="A1",
    )
    prompt_b = instance_prompt_hash(
        question="Q",
        reference_answer="R",
        candidate_answer="A1",
    )
    prompt_c = instance_prompt_hash(
        question="Q",
        reference_answer="R",
        candidate_answer="A2",
    )
    assert prompt_a == prompt_b
    assert prompt_a != prompt_c

    parsed_once = parse_judge_output(_legal(0.5, "stable"))
    parsed_twice = parse_judge_output(_legal(0.5, "stable"))
    assert parsed_once == parsed_twice


def test_fake_backends_are_invoked_sequentially() -> None:
    reset_invocation_trace()
    cases = _balanced_cases()
    result = _calibrate(cases, [1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    assert result.status in {
        "certified_for_exploratory_grading",
        "not_certified",
    }
    assert invocation_was_sequential()
    assert recorded_invocation_order() == (
        "gemma-fake",
        "gemma-fake",
        "gemma-fake",
        "mistral-fake",
        "mistral-fake",
        "mistral-fake",
    )
    assert recorded_lifecycle_events() == (
        "acquire:gemma-fake",
        "release:gemma-fake",
        "acquire:mistral-fake",
        "release:mistral-fake",
    )

    reset_invocation_trace()
    grade_with_dual_judges(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        judges=(
            FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(0.5)]),
            FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(0.5)]),
        ),
        evaluated_model_family="qwen",
    )
    assert invocation_was_sequential()
    assert recorded_invocation_order() == ("gemma-fake", "mistral-fake")
    assert recorded_lifecycle_events() == (
        "acquire:gemma-fake",
        "release:gemma-fake",
        "acquire:mistral-fake",
        "release:mistral-fake",
    )


def test_no_network_or_mlx_import_during_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    banned = [
        name
        for name in sys.modules
        if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm")
    ]
    assert banned == []
    import enterprise_memory_mlx.semantic_judging as module

    assert "mlx" not in module.__dict__
    assert "mlx_lm" not in module.__dict__

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is not allowed during judge tests")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    cases = _balanced_cases()
    result = _calibrate(
        cases,
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    assert result.dataset_hash == result.labeled_dataset_hash
    assert result.source_cases_hash != result.human_label_overlay_hash
    assert result.labeled_dataset_hash != result.source_cases_hash
    parse_judge_output(_legal(1.0))


def test_dataset_manifest_hash_mismatch_is_rejected() -> None:
    cases = _balanced_cases()
    source_bytes, overlay_bytes, manifest, hashes = _bundle(cases)
    judges = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", [_legal(1.0)] * 3),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(1.0)] * 3),
    )
    with pytest.raises(CalibrationValidationError, match="source-cases hash"):
        calibrate_judges(
            source_cases_bytes=source_bytes,
            human_label_overlay_bytes=overlay_bytes,
            manifest=manifest,
            expected_source_cases_hash="0" * 64,
            expected_human_label_overlay_hash=hashes.human_label_overlay_hash,
            expected_labeled_dataset_hash=hashes.labeled_dataset_hash,
            judges=judges,
            evaluated_model_family="qwen",
            thresholds=_thresholds(),
            min_class_count=1,
        )
    manifest["files"][SOURCE_CASES_FILE] = "1" * 64
    with pytest.raises(CalibrationValidationError, match="manifest file hashes"):
        calibrate_judges(
            source_cases_bytes=source_bytes,
            human_label_overlay_bytes=overlay_bytes,
            manifest=manifest,
            expected_source_cases_hash=hashes.source_cases_hash,
            expected_human_label_overlay_hash=hashes.human_label_overlay_hash,
            expected_labeled_dataset_hash=hashes.labeled_dataset_hash,
            judges=judges,
            evaluated_model_family="qwen",
            thresholds=_thresholds(),
            min_class_count=1,
        )


def test_training_rows_and_missing_critical_reviewers_are_rejected() -> None:
    training = _balanced_cases()
    training[0] = _case("C-1", 1.0, usable_for_training=True)
    with pytest.raises(CalibrationValidationError, match="usable for model training"):
        _calibrate(training, [1.0, 0.5, 0.0], [1.0, 0.5, 0.0])

    critical = _balanced_cases()
    critical.append(
        _case(
            "C-audit",
            0.0,
            stratum=AUDIT_STRATUM,
            reviewers=("only-one",),
            error_categories=("wrong_number",),
        )
    )
    with pytest.raises(CalibrationValidationError, match="two named human reviewers"):
        _calibrate(
            critical,
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
        )


def test_audit_stratum_is_reported_separately() -> None:
    cases = [
        *_balanced_cases(),
        _case(
            "C-audit",
            0.0,
            stratum=AUDIT_STRATUM,
            error_categories=("wrong_number",),
            reviewers=("alice", "bob"),
        ),
    ]
    result = _calibrate(
        cases,
        [1.0, 0.5, 0.0, 1.0],
        [1.0, 0.5, 0.0, 0.0],
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    headline = result.per_judge_metrics[0].headline
    audit = result.per_judge_metrics[0].audit
    assert headline.n_cases == 3
    assert audit.n_cases == 1
    assert audit.incorrect_to_pass_rate.numerator == 1
    assert result.per_judge_metrics[1].audit.incorrect_to_pass_rate.numerator == 0


def test_clopper_pearson_interval_is_exact_and_reported() -> None:
    low, high = clopper_pearson_two_sided(0, 10)
    assert low == 0.0
    assert high == pytest.approx(1.0 - 0.025 ** (1 / 10))
    metrics = compute_judge_metrics(
        [_case("E-1", 1.0)],
        [ParsedJudgeOutput(valid=True, score=1.0, reason="ok", error=None)],
    )
    assert metrics.exact_agreement.interval == "clopper_pearson_two_sided"
    assert metrics.exact_agreement.role == "descriptive"
    assert metrics.exact_agreement.unit == "row"
    assert metrics.exact_agreement.ci_low == 0.0 or metrics.exact_agreement.ci_low is not None
    assert metrics.exact_agreement.ci_high == 1.0
    assert metrics.family_clustered is not None
    assert metrics.family_clustered.exact_agreement.role == "clustered"
    assert metrics.family_clustered.exact_agreement.unit == "case_family"


def test_row_metrics_are_descriptive_and_family_intervals_are_clustered() -> None:
    cases = [
        _case("F1-a", 1.0, case_family_id="family-1"),
        _case("F1-b", 0.0, case_family_id="family-1"),
        _case("F2-a", 1.0, case_family_id="family-2"),
        _case("F2-b", 1.0, case_family_id="family-2"),
    ]
    parsed = [
        ParsedJudgeOutput(valid=True, score=score, reason="ok", error=None)
        for score in (1.0, 0.0, 1.0, 0.0)
    ]
    metrics = compute_judge_metrics(cases, parsed)
    assert metrics.exact_agreement.role == "descriptive"
    assert metrics.exact_agreement.numerator == 3
    assert metrics.exact_agreement.denominator == 4
    assert metrics.family_clustered is not None
    assert metrics.family_clustered.n_families == 2
    assert metrics.family_clustered.exact_agreement.numerator == 1
    assert metrics.family_clustered.exact_agreement.denominator == 2
    assert metrics.family_clustered.exact_agreement.interval == "clopper_pearson_two_sided"
    assert metrics.family_clustered.false_fail_rate.numerator == 1
    assert metrics.family_clustered.false_fail_rate.denominator == 2


def test_minimum_score_counts_use_judge_eligible_only() -> None:
    cases = [
        _case("E-1", 1.0),
        _case("E-2", 0.5),
        _case("E-3", 0.0),
        _case("A-1", 0.0, stratum=AUDIT_STRATUM),
        _case("A-2", 0.0, stratum=AUDIT_STRATUM),
    ]
    with pytest.raises(CalibrationValidationError, match="judge_eligible"):
        _calibrate(
            cases,
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0, 0.0],
            min_class_count=2,
        )

    eligible = [
        _case("E-1", 1.0),
        _case("E-2", 1.0),
        _case("E-3", 0.5),
        _case("E-4", 0.5),
        _case("E-5", 0.0),
        _case("E-6", 0.0),
        _case("A-1", 1.0, stratum=AUDIT_STRATUM),
    ]
    result = _calibrate(
        eligible,
        [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        min_class_count=2,
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    assert result.per_judge_metrics[0].headline.n_cases == 6
    assert result.per_judge_metrics[0].audit.n_cases == 1


def test_backend_result_identity_and_prompt_hash_are_verified() -> None:
    cases = _balanced_cases()
    source_bytes, overlay_bytes, manifest, hashes = _bundle(cases)

    class WrongHashBackend(FakeJudgeBackend):
        def judge(self, **kwargs: Any) -> JudgeBackendResult:  # type: ignore[override]
            result = super().judge(**kwargs)
            return JudgeBackendResult(
                raw_text=result.raw_text,
                model_id=result.model_id,
                revision=result.revision,
                prompt_hash="0" * 64,
                latency_seconds=result.latency_seconds,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

    with pytest.raises(CalibrationValidationError, match="instance-prompt hash"):
        calibrate_judges(
            source_cases_bytes=source_bytes,
            human_label_overlay_bytes=overlay_bytes,
            manifest=manifest,
            expected_source_cases_hash=hashes.source_cases_hash,
            expected_human_label_overlay_hash=hashes.human_label_overlay_hash,
            expected_labeled_dataset_hash=hashes.labeled_dataset_hash,
            judges=(
                WrongHashBackend("gemma-fake", "gemma", "rev-a", [_legal(1.0)] * 3),
                FakeJudgeBackend("mistral-fake", "mistral", "rev-b", [_legal(1.0)] * 3),
            ),
            evaluated_model_family="qwen",
            thresholds=_thresholds(),
            min_class_count=1,
        )


def test_case_decisions_persist_raw_parsed_and_identity() -> None:
    cases = _balanced_cases()
    result = _calibrate(
        cases,
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
    )
    assert len(result.case_decisions) == 6
    first = result.case_decisions[0]
    assert first.case_id == "C-1"
    assert first.case_family_id == "C-1"
    assert first.identity.model_id == "gemma-fake"
    assert first.identity.revision == "rev-a"
    assert first.parsed.score == 1.0
    assert first.raw_text
    assert first.prompt_hash == instance_prompt_hash(
        question=cases[0].question,
        reference_answer=cases[0].reference_answer,
        candidate_answer=cases[0].candidate_answer,
    )
    assert first.prompt_tokens == 0
    assert first.completion_tokens == 0
    assert first.latency_seconds == 0.0
    assert {item.identity.model_id for item in result.case_decisions} == {
        "gemma-fake",
        "mistral-fake",
    }


def test_models_never_overlap_during_calibration() -> None:
    reset_invocation_trace()
    cases = _balanced_cases()
    source_bytes, overlay_bytes, manifest, hashes = _bundle(cases)
    loaded: list[str] = []

    class TrackingFake(FakeJudgeBackend):
        def acquire(self) -> None:
            if loaded:
                raise AssertionError(f"{self.model_id} acquired while {loaded} still loaded")
            loaded.append(self.model_id)
            super().acquire()

        def release(self) -> None:
            super().release()
            loaded.remove(self.model_id)

    judges = (
        TrackingFake("gemma-fake", "gemma", "rev-a", [_legal(1.0), _legal(0.5), _legal(0.0)]),
        TrackingFake("mistral-fake", "mistral", "rev-b", [_legal(1.0), _legal(0.5), _legal(0.0)]),
    )
    calibrate_judges(
        source_cases_bytes=source_bytes,
        human_label_overlay_bytes=overlay_bytes,
        manifest=manifest,
        expected_source_cases_hash=hashes.source_cases_hash,
        expected_human_label_overlay_hash=hashes.human_label_overlay_hash,
        expected_labeled_dataset_hash=hashes.labeled_dataset_hash,
        judges=judges,
        evaluated_model_family="qwen",
        thresholds=_thresholds(
            min_exact_agreement=0.99,
            min_weighted_kappa=0.99,
            max_incorrect_to_pass_rate=0.0,
            max_critical_false_pass_rate=1.0,
        ),
        min_class_count=1,
    )
    assert judges[0].release_count == 1
    assert judges[1].acquire_count == 1
    assert judges[0].acquired is False
    assert judges[1].acquired is False
    assert invocation_was_sequential()
    assert recorded_lifecycle_events().index("release:gemma-fake") < (
        recorded_lifecycle_events().index("acquire:mistral-fake")
    )


def test_three_dataset_hashes_are_distinct_raw_byte_hashes() -> None:
    cases = _balanced_cases()
    source_bytes, overlay_bytes, _manifest, hashes = _bundle(cases)
    assert hashes.source_cases_hash != hashes.human_label_overlay_hash
    assert hashes.source_cases_hash != hashes.labeled_dataset_hash
    assert hashes.human_label_overlay_hash != hashes.labeled_dataset_hash
    from enterprise_memory_mlx.semantic_judging import sha256_bytes

    assert hashes.source_cases_hash == sha256_bytes(source_bytes)
    assert hashes.human_label_overlay_hash == sha256_bytes(overlay_bytes)


def test_convert_candidate_v3_matches_harness_and_keeps_human_gate(
    tmp_path: Path,
    project_root: Path,
) -> None:
    """The shipped v3 candidate file converts deterministically and still
    cannot be calibrated until humans approve gold labels."""
    candidate = project_root / "knowledge" / "judge_calibration" / "v3" / "cases.jsonl"
    first = convert_candidate_file(candidate, tmp_path / "one")
    second = convert_candidate_file(candidate, tmp_path / "two")

    assert first.case_count == 303
    assert first.human_labels_present is False
    assert first.hashes == second.hashes
    assert first.source_path.read_bytes() == second.source_path.read_bytes()
    assert first.overlay_path.read_bytes() == second.overlay_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()

    overlay_rows = [
        json.loads(line)
        for line in first.overlay_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(overlay_rows) == 303
    assert all(row["human_approved"] is False for row in overlay_rows)
    assert all(row["human_semantic_score"] is None for row in overlay_rows)

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"][SOURCE_CASES_FILE] == first.hashes.source_cases_hash
    judges = (
        FakeJudgeBackend("gemma-fake", "gemma", "rev-a", []),
        FakeJudgeBackend("mistral-fake", "mistral", "rev-b", []),
    )
    with pytest.raises(CalibrationValidationError, match="human labels"):
        calibrate_judges(
            source_cases_bytes=first.source_path.read_bytes(),
            human_label_overlay_bytes=first.overlay_path.read_bytes(),
            manifest=manifest,
            expected_source_cases_hash=first.hashes.source_cases_hash,
            expected_human_label_overlay_hash=first.hashes.human_label_overlay_hash,
            expected_labeled_dataset_hash=first.hashes.labeled_dataset_hash,
            judges=judges,
            evaluated_model_family="qwen",
            thresholds=_thresholds(),
            min_class_count=1,
        )
