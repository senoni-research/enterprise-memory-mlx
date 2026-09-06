from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from enterprise_memory_mlx.benchmark import GeneratedAnswer
from enterprise_memory_mlx.experiment_profiles import (
    GEMMA_JUDGE_MODEL_ID,
    GEMMA_JUDGE_REVISION,
)
from enterprise_memory_mlx.mlx_judge_backend import ParsedGemmaAdvisory
from enterprise_memory_mlx.semantic_judging import JudgeBackendResult
from enterprise_memory_mlx.task_specialization import (
    grade_structured_assessment,
    load_specialization_assets,
    parse_structured_assessment,
    run_specialization_pilot,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_specialization_contract_and_cases_verify() -> None:
    assets = load_specialization_assets(ROOT)

    assert assets.protocol["protocol_id"] == "company-task-specialization/v1"
    assert len(assets.cases) == 14
    assert assets.manifest["training_eligible"] is False
    assert all(case["split"] == "development_repair_pool" for case in assets.cases)


def test_structured_assessment_parser_is_strict() -> None:
    valid = {
        "decision": "refer_to_source",
        "required_actions": ["Consult the source."],
        "missing_information": ["No policy supplied."],
        "exceptions": [],
        "evidence": [],
    }

    assert parse_structured_assessment(json.dumps(valid))[1] == "valid"
    assert parse_structured_assessment(f"```json\n{json.dumps(valid)}\n```")[1] == "malformed_json"
    assert parse_structured_assessment(json.dumps(valid) + "\nextra")[1] == "trailing_content"
    extra = {**valid, "reason": "not allowed"}
    assert parse_structured_assessment(json.dumps(extra))[1] == "schema_mismatch"


def test_deterministic_assessment_requires_decision_terms_and_authorized_evidence() -> None:
    case = load_specialization_assets(ROOT).cases[0]
    passing = deepcopy(case["reference"])

    assert grade_structured_assessment(case, passing, parse_status="valid")["status"] == "pass"

    failing = deepcopy(passing)
    failing["decision"] = "proceed"
    failing["evidence"] = [{"record_id": "FAKE-001", "claim": "Invented."}]
    grade = grade_structured_assessment(case, failing, parse_status="valid")
    assert grade["status"] == "hard_fail"
    assert any(reason.startswith("decision:") for reason in grade["reasons"])
    assert any("unauthorized" in reason for reason in grade["reasons"])


class _FakeGenerator:
    def __init__(self, model_id: str, references: dict[str, dict], *, fail_first: bool) -> None:
        self.model_name = model_id
        self._references = references
        self._fail_first = fail_first
        self.closed = False

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        assert max_tokens == 1024
        assert system_prompt
        payload = json.loads(question)
        assessment = deepcopy(self._references[payload["request"]])
        if self._fail_first:
            assessment["decision"] = "proceed"
            self._fail_first = False
        output = json.dumps(assessment)
        return GeneratedAnswer(
            output=output,
            raw_output=output,
            parse_status="plain",
            prompt_tokens=10,
            completion_tokens=20,
            elapsed_seconds=0.1,
            finish_reason="stop",
            truncated=False,
            peak_memory_gb=1.0,
            prompt_hash="prompt",
            rendered_template_hash="template",
        )

    def close(self) -> None:
        self.closed = True


class _FakeJudge:
    snapshot_identity = {"test": True}

    def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None

    def judge_with_evidence(self, **_kwargs):
        parsed = ParsedGemmaAdvisory(
            valid=True,
            score=1.0,
            reason="Fully correct and supported.",
            unsupported_claims=(),
            needs_human_attention=False,
            confidence="high",
            error=None,
        )
        result = JudgeBackendResult(
            raw_text=json.dumps(parsed.to_dict()),
            final_text=json.dumps(parsed.to_dict()),
            model_id=GEMMA_JUDGE_MODEL_ID,
            revision=GEMMA_JUDGE_REVISION,
            prompt_hash="judge-prompt",
            latency_seconds=0.1,
            prompt_tokens=10,
            completion_tokens=10,
            finish_reason="stop",
            parse_status="valid",
            rendered_prompt_hash="rendered",
            chat_template_hash="template",
            generation_config_hash="config",
            peak_memory_gb=1.0,
        )
        return result, parsed


def test_pilot_repairs_deterministic_student_failure_without_training_admission(
    tmp_path: Path,
) -> None:
    assets = load_specialization_assets(ROOT)
    references = {case["request"]: case["reference"] for case in assets.cases}
    created: list[_FakeGenerator] = []

    def factory(model_id: str, _revision: str) -> _FakeGenerator:
        backend = _FakeGenerator(
            model_id,
            references,
            fail_first="4B" in model_id,
        )
        created.append(backend)
        return backend

    artifacts = run_specialization_pilot(
        root=ROOT,
        output_root=tmp_path,
        generator_factory=factory,
        judge_backend=_FakeJudge(),
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    candidates = [
        json.loads(line)
        for line in artifacts.candidate_repairs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert report["teacher_qualification"]["qualified_for_bounded_advisory_repair"] is True
    assert report["student_summary"]["deterministic_passes"] == 13
    assert report["repair_summary"]["case_count"] == 1
    assert report["repair_summary"]["advisory_accepted_count"] == 1
    assert candidates[0]["advisory_accepted"] is True
    assert candidates[0]["training_eligible"] is False
    assert all(backend.closed for backend in created)
