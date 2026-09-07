from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path

from enterprise_memory_mlx.benchmark import GeneratedAnswer
from enterprise_memory_mlx.specialization_fact_state_experiment import (
    AMENDED_SYSTEM_PROMPT,
    CANDIDATE_ARM,
    CONTROL_ARM,
    FACT_STATE_AMENDMENT,
    correct_fact_state_machine_grades,
    load_fact_state_assets,
    prepare_fact_state_review,
    run_fact_state_comparison,
    run_fact_state_regression,
    score_fact_state_review,
)
from enterprise_memory_mlx.specialization_remedy_experiment import (
    CHALLENGER_SYSTEM_PROMPT,
    load_remedy_experiment_assets,
)

ROOT = Path(__file__).resolve().parents[1]


class _FakeGenerator:
    def __init__(self, references: dict[str, dict]) -> None:
        self.model_name = "mlx-community/Qwen3.8-27B-4bit"
        self.references = references
        self.calls = 0
        self.closed = False

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        assert max_tokens == 1024
        assert system_prompt in {CHALLENGER_SYSTEM_PROMPT, AMENDED_SYSTEM_PROMPT}
        payload = json.loads(question)
        first_state_id = payload["operational_state"][0]["state_id"]
        value = deepcopy(self.references[first_state_id])
        value["assessments"][0]["evidence"].append(
            {
                "record_id": first_state_id,
                "claim": "The cited operational state was supplied to the generator.",
            }
        )
        output = json.dumps(value)
        self.calls += 1
        return GeneratedAnswer(
            output=output,
            raw_output=output,
            parse_status="plain",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_seconds=0.1,
            finish_reason="stop",
            truncated=False,
            peak_memory_gb=1.0,
            prompt_hash="prompt",
            rendered_template_hash="template",
        )

    def close(self) -> None:
        self.closed = True


class _FakeRegressionGenerator:
    def __init__(self, references: dict[str, dict]) -> None:
        self.model_name = "mlx-community/Qwen3.8-27B-4bit"
        self.references = references
        self.calls = 0
        self.closed = False

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        assert system_prompt == AMENDED_SYSTEM_PROMPT
        assert max_tokens == 1024
        payload = json.loads(question)
        output = json.dumps(self.references[payload["operational_facts"]])
        self.calls += 1
        return GeneratedAnswer(
            output=output,
            raw_output=output,
            parse_status="plain",
            prompt_tokens=100,
            completion_tokens=50,
            elapsed_seconds=0.1,
            finish_reason="stop",
            truncated=False,
            peak_memory_gb=1.0,
            prompt_hash="prompt",
            rendered_template_hash="template",
        )

    def close(self) -> None:
        self.closed = True


def test_frozen_fact_state_contract_has_eight_contrast_pairs() -> None:
    assets = load_fact_state_assets(ROOT)

    assert len(assets.cases) == 16
    pair_counts: dict[str, int] = {}
    for case in assets.cases:
        pair_counts[case["pair_id"]] = pair_counts.get(case["pair_id"], 0) + 1
    assert len(pair_counts) == 8
    assert set(pair_counts.values()) == {2}
    assert assets.manifest["fresh_comparison_limit"] == 1
    assert assets.manifest["conditional_regression_limit"] == 1
    assert assets.manifest["training_eligible"] is False
    assert AMENDED_SYSTEM_PROMPT == CHALLENGER_SYSTEM_PROMPT + FACT_STATE_AMENDMENT


def test_fact_state_run_review_and_separate_learning_gate(tmp_path: Path) -> None:
    assets = load_fact_state_assets(ROOT)
    references = {
        str(case["operational_state"][0]["state_id"]): case["reference"] for case in assets.cases
    }
    backend = _FakeGenerator(references)

    run = run_fact_state_comparison(
        root=ROOT,
        output_root=tmp_path / "runs",
        generator_factory=lambda _model, _revision: backend,
    )

    comparison = json.loads(run.report_path.read_text())
    assert backend.calls == 32
    assert backend.closed is True
    assert set(comparison["attempts"]) == {CONTROL_ARM, CANDIDATE_ARM}
    assert all(
        summary["valid_structures"] == 16 for summary in comparison["arm_summaries"].values()
    )
    assert all(
        summary["machine_hard_failures"] == 0 for summary in comparison["arm_summaries"].values()
    )
    assert comparison["v3_regression_authorized"] is False

    comparison["attempts"][CONTROL_ARM][0]["machine_grade"] = {
        "evaluator_id": "remedy-logic-structure/v1",
        "status": "hard_fail",
        "hard_failure_reasons": ["provenance:incorrect test allowlist"],
        "semantic_review_reasons": [],
        "semantic_review_eligible": False,
    }
    broken_path = tmp_path / "broken-comparison.json"
    broken_path.write_text(json.dumps(comparison))
    correction = correct_fact_state_machine_grades(
        root=ROOT,
        comparison_path=broken_path,
        output_root=tmp_path / "corrections",
    )
    corrected = json.loads(correction.report_path.read_text())
    assert corrected["runtime_correction"]["generation_calls_added"] == 0
    assert corrected["runtime_correction"]["generation_outputs_changed"] is False
    assert corrected["arm_summaries"][CONTROL_ARM]["machine_hard_failures"] == 0

    review = prepare_fact_state_review(
        root=ROOT,
        comparison_path=correction.report_path,
        output_root=tmp_path / "reviews",
    )
    with zipfile.ZipFile(review.packet_path) as archive:
        cases_text = archive.read("review/review_cases.jsonl").decode()
        manifest = json.loads(archive.read("review/packet_manifest.json"))
    assert len(cases_text.splitlines()) == 32
    assert '"arm"' not in cases_text
    assert '"case_id"' not in cases_text
    assert '"pair_id"' not in cases_text
    assert manifest["same_output_schema"] is True

    mapping = json.loads(review.mapping_path.read_text())
    mapping_by_id = {row["review_id"]: row for row in mapping["mapping"]}
    advisory = [json.loads(line) for line in review.template_path.read_text().splitlines()]
    degraded_control = None
    for row in advisory:
        mapped = mapping_by_id[row["review_id"]]
        row.update(
            {
                "reviewer_id": "Test advisory model",
                "reviewed_at": "2026-09-06T21:00:00Z",
                "overall_outcome": "acceptable",
                "required_obligations": [
                    {
                        "description": "Apply policy to authenticated state.",
                        "material": True,
                        "satisfied": "yes",
                    }
                ],
                "unsafe_claims": [],
                "supports_next_step": "yes",
                "fact_state_assessment": {
                    "established_facts_preserved": "yes",
                    "unknowns_handled_correctly": "yes",
                    "documentation_scope_correct": "yes",
                    "authority_boundary_correct": "yes",
                    "no_invented_requirement": "yes",
                    "no_unsafe_waiver": "yes",
                    "remedy_logic_preserved": "yes",
                },
                "ambiguities": [],
                "notes": "Complete.",
            }
        )
        if (
            degraded_control is None
            and mapped["arm"] == CONTROL_ARM
            and "no_invented_requirement" in mapped["evaluation_tags"]
        ):
            row["overall_outcome"] = "unacceptable"
            row["unsafe_claims"] = ["Invented an attachment requirement."]
            row["fact_state_assessment"]["established_facts_preserved"] = "no"
            row["fact_state_assessment"]["documentation_scope_correct"] = "no"
            row["fact_state_assessment"]["no_invented_requirement"] = "no"
            degraded_control = row["review_id"]
    assert degraded_control is not None
    advisory_path = tmp_path / "advisory.jsonl"
    advisory_path.write_text("".join(json.dumps(row) + "\n" for row in advisory))

    decision = score_fact_state_review(
        root=ROOT,
        comparison_path=correction.report_path,
        packet_path=review.packet_path,
        mapping_path=review.mapping_path,
        advisory_path=advisory_path,
        output_root=tmp_path / "decisions",
    )

    report = json.loads(decision.report_path.read_text())
    assert report["targeted_learning"]["improved"] is True
    assert report["qualification"]["candidate_passes_substantive_gate"] is True
    assert report["qualification"]["fresh_comparison_success"] is True
    assert report["v3_regression_authorized"] is True
    assert report["bounded_advisory_teacher_designation"] is None
    assert report["training_authorized"] is False

    v3_assets = load_remedy_experiment_assets(ROOT)
    regression_backend = _FakeRegressionGenerator(
        {str(case["operational_facts"]): case["reference"] for case in v3_assets.cases}
    )
    regression = run_fact_state_regression(
        root=ROOT,
        fresh_decision_path=decision.report_path,
        output_root=tmp_path / "regression",
        generator_factory=lambda _model, _revision: regression_backend,
    )
    regression_report = json.loads(regression.report_path.read_text())
    assert regression_backend.calls == 24
    assert regression_backend.closed is True
    assert regression_report["summary"]["valid_structures"] == 24
    assert regression_report["historical_v3_scores_modified"] is False
    assert regression_report["training_eligible"] is False
