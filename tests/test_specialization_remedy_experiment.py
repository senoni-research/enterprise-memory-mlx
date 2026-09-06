from __future__ import annotations

import json
import zipfile
from pathlib import Path

from enterprise_memory_mlx.benchmark import GeneratedAnswer
from enterprise_memory_mlx.remedy_logic import render_remedy
from enterprise_memory_mlx.specialization_qualification import BASELINE_SYSTEM_PROMPT
from enterprise_memory_mlx.specialization_remedy_experiment import (
    CHALLENGER_SYSTEM_PROMPT,
    load_remedy_experiment_assets,
    prepare_remedy_review,
    run_remedy_experiment,
    score_remedy_review,
)

ROOT = Path(__file__).resolve().parents[1]


def _flat_reference(reference: dict) -> dict:
    actions = []
    missing = []
    exceptions = []
    evidence = []
    decisions = []
    for assessment in reference["assessments"]:
        decisions.append(
            {
                "request_item_id": assessment["request_item_id"],
                "decision": assessment["decision"],
            }
        )
        remedy = assessment["remedy"]
        if remedy is not None:
            actions.append(f"{assessment['request_item_id']}: {render_remedy(remedy)}")
        missing.extend(assessment["missing_information"])
        exceptions.extend(assessment["exceptions"])
        evidence.extend(assessment["evidence"])
    return {
        "decisions": decisions,
        "required_actions": actions,
        "missing_information": missing,
        "exceptions": exceptions,
        "evidence": evidence,
    }


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
        payload = json.loads(question)
        reference = self.references[payload["operational_facts"]]
        value = _flat_reference(reference) if system_prompt == BASELINE_SYSTEM_PROMPT else reference
        assert system_prompt in {BASELINE_SYSTEM_PROMPT, CHALLENGER_SYSTEM_PROMPT}
        self.calls += 1
        output = json.dumps(value)
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


def test_frozen_remedy_experiment_has_fresh_coverage() -> None:
    assets = load_remedy_experiment_assets(ROOT)

    assert assets.protocol["protocol_id"] == "company-task-specialization/v3-remedy-logic"
    assert len(assets.cases) == 24
    assert len({case["scenario_family"] for case in assets.cases}) >= 18
    assert any("nested_logic" in case["evaluation_tags"] for case in assets.cases)
    assert assets.manifest["training_eligible"] is False


def test_two_arm_run_review_and_paired_gate(tmp_path: Path) -> None:
    assets = load_remedy_experiment_assets(ROOT)
    references = {str(case["operational_facts"]): case["reference"] for case in assets.cases}
    backend = _FakeGenerator(references)

    run = run_remedy_experiment(
        root=ROOT,
        output_root=tmp_path / "runs",
        generator_factory=lambda _model, _revision: backend,
    )

    comparison = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert backend.calls == 48
    assert backend.closed is True
    assert set(comparison["attempts"]) == {
        "flat_action_control",
        "structured_remedy_challenger",
    }
    assert all(
        summary["valid_structures"] == 24 for summary in comparison["arm_summaries"].values()
    )

    review = prepare_remedy_review(
        root=ROOT,
        comparison_path=run.report_path,
        output_root=tmp_path / "reviews",
    )
    with zipfile.ZipFile(review.packet_path) as archive:
        cases_text = archive.read("review/review_cases.jsonl").decode("utf-8")
        instructions = archive.read("review/REVIEW_INSTRUCTIONS.md").decode("utf-8")
        manifest = json.loads(archive.read("review/packet_manifest.json"))
    assert len(cases_text.splitlines()) == 48
    assert '"arm"' not in cases_text
    assert '"case_id"' not in cases_text
    assert manifest["arm_label_blinded"] is True
    assert manifest["perfect_treatment_blinding_claimed"] is False
    assert "perfectly" in instructions

    mapping = json.loads(review.mapping_path.read_text(encoding="utf-8"))
    mapping_by_id = {row["review_id"]: row for row in mapping["mapping"]}
    advisory = [
        json.loads(line) for line in review.template_path.read_text(encoding="utf-8").splitlines()
    ]
    degraded_control = None
    for row in advisory:
        mapped = mapping_by_id[row["review_id"]]
        row.update(
            {
                "reviewer_id": "Test advisory model",
                "reviewed_at": "2026-09-06T20:00:00+00:00",
                "overall_outcome": "acceptable",
                "required_obligations": [
                    {
                        "description": "Preserve the applicable policy logic.",
                        "material": True,
                        "satisfied": "yes",
                    }
                ],
                "unsafe_claims": [],
                "supports_next_step": "yes",
                "logic_assessment": {
                    "current_decisions_correct": "yes",
                    "valid_alternatives_preserved": "yes",
                    "mandatory_conditions_preserved": "yes",
                    "unsupported_route_added": "no",
                },
                "ambiguities": [],
                "notes": "Complete.",
            }
        )
        if (
            degraded_control is None
            and mapped["arm"] == "flat_action_control"
            and "alternative_remedies" in mapped["evaluation_tags"]
        ):
            row["overall_outcome"] = "minor_revision"
            row["unsafe_claims"] = ["One valid alternative was erased."]
            row["logic_assessment"]["valid_alternatives_preserved"] = "no"
            degraded_control = row["review_id"]
    assert degraded_control is not None
    advisory_path = tmp_path / "advisory.jsonl"
    advisory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in advisory),
        encoding="utf-8",
    )

    decision = score_remedy_review(
        root=ROOT,
        comparison_path=run.report_path,
        packet_path=review.packet_path,
        mapping_path=review.mapping_path,
        advisory_path=advisory_path,
        output_root=tmp_path / "decisions",
    )

    report = json.loads(decision.report_path.read_text(encoding="utf-8"))
    assert report["selected_configuration"] == "structured_remedy_challenger"
    assert (
        report["decision"]
        == "structured_remedy_challenger_qualified_for_bounded_advisory_generation"
    )
    assert report["paired_comparison"]["alternative_preservation_wins"] == 1
    assert report["training_authorized"] is False
