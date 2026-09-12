from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.review_ui import _load_and_verify_mapping, _load_and_verify_packet
from enterprise_memory_mlx.structured_inference import historical_eval_dir
from enterprise_memory_mlx.structured_inference_review import (
    BLINDED_FORBIDDEN,
    MEASURED_RUN_ID,
    collect_measured_outputs,
    measured_run_dir,
    output_path,
    prepare_measured_constrained_review,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for arm_id in ("arm-a-unadapted", "arm-b-adapter"):
        for case_id in ("DEV-001", "DEV-002", "DEV-003", "DEV-004", "DEV-005"):
            rows.append(
                {
                    "arm_id": arm_id,
                    "case_id": case_id,
                    "mapping_id": f"{arm_id}/{case_id}",
                    "source_output_path": (
                        f"artifacts/qwen4b-structured-inference-v1/{MEASURED_RUN_ID}/"
                        f"{arm_id}/outputs/{case_id}.json"
                    ),
                    "source_output_file_sha256": "a" * 64,
                    "raw_output_sha256": "b" * 64,
                    "raw_output": f'{{"case_id": "{case_id}", "overall_disposition": "proceed"}}',
                    "input_sha256": "c" * 64,
                    "target_sha256": "d" * 64,
                    "user_prompt": f"Synthetic frozen question for {case_id}.",
                    "source_index": [{"source_id": "S1", "path": "synthetic.txt"}],
                }
            )
    return rows


def test_measured_run_constants_do_not_point_at_historical_eval() -> None:
    historical = historical_eval_dir(ROOT)
    assert MEASURED_RUN_ID == "run-20260912T112818Z"
    assert MEASURED_RUN_ID not in str(historical)
    assert "eval-20260911T212334Z" not in MEASURED_RUN_ID
    sample = output_path(ROOT, "arm-b-adapter", "DEV-002")
    assert MEASURED_RUN_ID in str(sample)
    assert str(historical) not in str(sample)


def test_prepare_measured_review_is_ui_compatible_and_blinded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "enterprise_memory_mlx.structured_inference_review.collect_measured_outputs",
        lambda _root: _synthetic_rows(),
    )
    status = prepare_measured_constrained_review(
        ROOT,
        salt=b"structured-review-test",
        output_root=tmp_path / "packet",
    )
    assert status["semantic_assessment"] == "pending"
    packet_dir = tmp_path / "packet"
    packet = next(packet_dir.glob("*.zip"))
    mapping = packet_dir / "private" / "review_id_map.json"
    cases, sources, manifest = _load_and_verify_packet(packet)
    _load_and_verify_mapping(
        mapping, {str(row["review_id"]) for row in cases}, manifest
    )
    assert len(cases) == 10
    assert manifest["semantic_assessment"] == "pending"
    assert manifest["blinding"]["scores_excluded"] is True
    for row in cases:
        assert not (BLINDED_FORBIDDEN & row.keys())
        assert row["candidate_answer"].lstrip().startswith("{")
        assert "No reference review" in row["reference_answer"]
        assert "arm-a-unadapted" not in row["question"]
        assert "arm-b-adapter" not in row["question"]
    assert sources
    assert {item["id"] for item in sources}


@pytest.mark.private_data
def test_collect_measured_outputs_are_the_ten_constrained_answers(
    private_approval_path,
) -> None:
    del private_approval_path
    if not measured_run_dir(ROOT).is_dir():
        pytest.skip("measured structured-inference run is not present")
    rows = collect_measured_outputs(ROOT)
    assert len(rows) == 10
    assert {row["arm_id"] for row in rows} == {"arm-a-unadapted", "arm-b-adapter"}
    historical = historical_eval_dir(ROOT)
    for row in rows:
        assert MEASURED_RUN_ID in row["source_output_path"]
        assert "eval-20260911T212334Z" not in row["source_output_path"]
        assert str(historical) not in row["source_output_path"]
        assert row["raw_output"].lstrip().startswith("{")
