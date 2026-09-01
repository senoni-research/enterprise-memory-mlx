"""Historical-reference tests for the scientifically invalid legacy trainer.

These tests document the legacy configuration (including its rejected q/v-only
substrate and mis-calibrated scale) and prove the entry points fail closed
without an explicit acknowledgement. They do not endorse the legacy pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.compiler import compile_knowledge
from enterprise_memory_mlx.hardware import PRESETS
from enterprise_memory_mlx.legacy_guard import LegacyPipelineDisabledError
from enterprise_memory_mlx.training import build_stage_config, train_pipeline


def _compile_legacy(isolated_project: Path) -> None:
    compile_knowledge(
        isolated_project / "knowledge",
        isolated_project / "artifacts" / "datasets",
        allow_scientifically_invalid=True,
    )


def test_train_pipeline_fails_closed_by_default(isolated_project: Path) -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="scientifically invalid"):
        train_pipeline(isolated_project, stage="all", preset_name="smoke", dry_run=True)


def test_build_stage_config_fails_closed_by_default(isolated_project: Path) -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="scientifically invalid"):
        build_stage_config(isolated_project, "inject", preset=PRESETS["smoke"])


def test_legacy_config_documents_rejected_qv_only_substrate(
    isolated_project: Path,
) -> None:
    """Known invalidity: q/v-only keys and a mis-calibrated MLX scale.

    The assertion documents why the legacy trainer is disabled; the revised
    path uses all seven projections on all layers with scale 2.0.
    """
    _compile_legacy(isolated_project)
    config, config_path = build_stage_config(
        isolated_project,
        "inject",
        preset=PRESETS["smoke"],
        allow_scientifically_invalid=True,
    )
    assert config_path.name == "inject.yaml"
    assert config["train"] is True
    assert config["fine_tune_type"] == "lora"
    assert config["mask_prompt"] is True
    assert config["data"].endswith("artifacts/datasets/inject")
    # Documented defects, not desired behaviour:
    assert config["lora_parameters"]["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]
    assert config["lora_parameters"]["scale"] >= 20.0


def test_legacy_all_stage_dry_run_does_not_require_existing_adapter_files(
    isolated_project: Path,
) -> None:
    _compile_legacy(isolated_project)
    commands = train_pipeline(
        isolated_project,
        stage="all",
        preset_name="smoke",
        dry_run=True,
        allow_scientifically_invalid=True,
    )
    assert len(commands) == 3
    assert all(command[2:4] == ["mlx_lm", "lora"] for command in commands)


def test_legacy_vanilla_stage_starts_from_base_model(isolated_project: Path) -> None:
    _compile_legacy(isolated_project)
    config, _ = build_stage_config(
        isolated_project,
        "vanilla",
        preset=PRESETS["smoke"],
        allow_scientifically_invalid=True,
    )
    assert config["resume_adapter_file"] is None
    assert config["data"].endswith("artifacts/datasets/vanilla")
