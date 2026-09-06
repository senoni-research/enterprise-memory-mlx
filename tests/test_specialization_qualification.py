from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path

from enterprise_memory_mlx.benchmark import GeneratedAnswer
from enterprise_memory_mlx.specialization_qualification import (
    BASELINE_SYSTEM_PROMPT,
    OBLIGATION_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    load_qualification_assets,
    machine_grade_qualification,
    prepare_qualification_review,
    run_qualification_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_supplier_qualification_contract_verifies() -> None:
    assets = load_qualification_assets(ROOT)

    assert assets.protocol["protocol_id"] == "company-task-specialization/v2-model-advisory"
    assert len(assets.cases) == 14
    assert {case["workflow"] for case in assets.cases} == {"supplier_onboarding_and_first_invoice"}
    assert any(len(case["request_items"]) > 1 for case in assets.cases)
    assert assets.manifest["training_eligible"] is False
    assert assets.manifest["human_labels_present"] is False


def test_qualification_machine_grade_keeps_semantics_out_of_hard_failures() -> None:
    case = load_qualification_assets(ROOT).cases[0]
    reference = deepcopy(case["reference"])

    grade = machine_grade_qualification(case, reference, parse_status="valid")

    assert grade["status"] == "semantic_review_required"
    assert grade["hard_failure_reasons"] == []

    reference["evidence"] = [{"record_id": "FAKE-001", "claim": "Invented."}]
    grade = machine_grade_qualification(case, reference, parse_status="valid")
    assert grade["status"] == "hard_fail"
    assert "unauthorized" in grade["hard_failure_reasons"][0]


class _FakeGenerator:
    def __init__(self, references: dict[str, dict]) -> None:
        self.model_name = "mlx-community/Qwen3.8-27B-4bit"
        self.references = references
        self.calls: list[str] = []
        self.closed = False

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        assert max_tokens == 1024
        assert system_prompt in {
            BASELINE_SYSTEM_PROMPT,
            OBLIGATION_SYSTEM_PROMPT,
            REVISION_SYSTEM_PROMPT,
        }
        payload = json.loads(question)
        request = payload["operational_facts"]
        self.calls.append(system_prompt)
        output = json.dumps(self.references[request])
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


def test_qualification_runs_three_bounded_arms_and_blinds_review(
    tmp_path: Path,
) -> None:
    assets = load_qualification_assets(ROOT)
    references = {str(case["request"]): case["reference"] for case in assets.cases}
    backend = _FakeGenerator(references)

    artifacts = run_qualification_comparison(
        root=ROOT,
        output_root=tmp_path / "runs",
        generator_factory=lambda _model, _revision: backend,
    )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert backend.closed is True
    assert len(backend.calls) == 42
    assert report["status"] == "awaiting_blinded_model_advisory"
    assert report["training_eligible"] is False
    assert set(report["attempts"]) == {
        "baseline_single_pass",
        "obligation_single_pass",
        "obligation_revision_pass",
    }
    assert all(summary["structured_valid"] == 14 for summary in report["arm_summaries"].values())
    assert all(
        row["end_to_end_elapsed_seconds"] == 0.2
        for row in report["attempts"]["obligation_revision_pass"]
    )

    review = prepare_qualification_review(
        root=ROOT,
        comparison_path=artifacts.report_path,
        output_root=tmp_path / "reviews",
    )
    with zipfile.ZipFile(review.packet_path) as archive:
        cases = archive.read("review/review_cases.jsonl").decode("utf-8")
        guide = archive.read("review/REVIEW_SCHEMA.md").decode("utf-8")
        manifest = json.loads(archive.read("review/packet_manifest.json"))
    assert len(cases.splitlines()) == 42
    assert '"arm"' not in cases
    assert '"case_id"' not in cases
    assert '"reference"' not in cases
    assert "model_advisory" in guide
    assert manifest["distinct_request_count"] == 14
    assert manifest["arm_count"] == 3
    assert manifest["human_evidence"] is False

    mapping = json.loads(review.mapping_path.read_text(encoding="utf-8"))
    assert {row["arm"] for row in mapping["mapping"]} == {
        "baseline_single_pass",
        "obligation_single_pass",
        "obligation_revision_pass",
    }
