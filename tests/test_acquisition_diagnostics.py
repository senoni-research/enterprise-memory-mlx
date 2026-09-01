from __future__ import annotations

import json
from pathlib import Path

from enterprise_memory_mlx.acquisition_diagnostics import (
    normalized_exact,
    run_general_diagnostic,
)
from enterprise_memory_mlx.acquisition_training import VerifiedAcquisitionAdapter
from enterprise_memory_mlx.benchmark import GeneratedAnswer


class FakeBackend:
    outputs = {
        "Base answer": "Wrong",
        "Adapted answer": "Expected.",
    }

    def __init__(self, _model, *, revision, adapter_path=None) -> None:
        self.key = "Adapted answer" if adapter_path else "Base answer"
        self.revision = revision

    def generate(self, *, system_prompt, question, max_tokens):
        assert system_prompt
        assert question
        assert max_tokens == 80
        return GeneratedAnswer(
            output=self.outputs[self.key],
            prompt_tokens=10,
            completion_tokens=1,
            elapsed_seconds=0.1,
        )

    def close(self) -> None:
        return


def test_normalized_exact_is_strict_but_format_tolerant() -> None:
    assert normalized_exact("Buenos días!", "Buenos dias.")
    assert not normalized_exact("The answer is 43.", "43.")


def test_general_diagnostic_is_explicitly_non_promotable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "enterprise_memory_mlx.acquisition_diagnostics.MLXBenchmarkBackend",
        FakeBackend,
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter = VerifiedAcquisitionAdapter(
        run_manifest_path=tmp_path / "run.json",
        model_id="fake/model",
        model_revision="revision",
        adapter_path=adapter_dir,
        adapter_hash="a" * 64,
        source_record_ids=("REC-1",),
        inherited_classification="internal_shared",
        promotion_eligible=False,
    )

    path = run_general_diagnostic(
        rows=[{"instruction": "Question", "response": "Expected."}],
        adapter=adapter,
        output_dir=tmp_path / "diagnostics",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "diagnostic_non_promotable"
    assert payload["promotion_eligible"] is False
    assert payload["base"]["normalized_exact"] == 0
    assert payload["parametric"]["normalized_exact"] == 1
