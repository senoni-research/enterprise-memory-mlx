from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from enterprise_memory_mlx.acquisition_compiler import compile_acquisition_dataset
from enterprise_memory_mlx.acquisition_training import (
    ALL_QWEN_PROJECTIONS,
    AcquisitionConfig,
    build_mlx_acquisition_config,
    load_verified_acquisition_adapter,
    run_acquisition,
)
from enterprise_memory_mlx.cli import build_parser


class HashEmbeddingBackend:
    model_id = "fake/hash"
    revision = "fake-revision"

    def encode(self, texts):
        return [
            [1.0 if value & 1 else -1.0 for value in hashlib.sha256(text.encode()).digest()]
            for text in texts
        ]


def _compilation(tmp_path: Path, project_root: Path):
    return compile_acquisition_dataset(
        knowledge_dir=project_root / "knowledge",
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "acquisition",
        semantic_backend=HashEmbeddingBackend(),
        profile="smoke_non_promotable",
        target_exposures_per_fact=24,
        effective_batch_size=8,
    )


def _config() -> AcquisitionConfig:
    return AcquisitionConfig(
        model_id="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        model_revision="a" * 40,
        rank=16,
        scale=2.0,
        learning_rate=2e-4,
        dropout=0.05,
        num_layers=36,
        batch_size=1,
        grad_accumulation_steps=8,
        max_seq_length=2048,
        seed=42,
        profile="smoke_non_promotable",
    )


def test_mlx_config_uses_all_layers_and_all_projections(
    tmp_path: Path,
    project_root: Path,
) -> None:
    compilation = _compilation(tmp_path, project_root)

    config = build_mlx_acquisition_config(
        compilation=compilation,
        config=_config(),
        model_path="/models/qwen-pinned",
        adapter_path=tmp_path / "adapter",
    )

    assert config["model"] == "/models/qwen-pinned"
    assert config["fine_tune_type"] == "lora"
    assert config["num_layers"] == 36
    assert config["iters"] == 240
    assert config["batch_size"] == 1
    assert config["grad_accumulation_steps"] == 8
    assert config["mask_prompt"] is True
    assert config["grad_checkpoint"] is True
    assert config["learning_rate"] == 2e-4
    assert config["lora_parameters"] == {
        "keys": list(ALL_QWEN_PROJECTIONS),
        "rank": 16,
        "scale": 2.0,
        "dropout": 0.05,
    }
    assert config["lr_schedule"]["warmup"] == 3
    assert config["save_every"] == 80


def test_dry_run_writes_non_promotable_config_and_manifest(
    tmp_path: Path,
    project_root: Path,
) -> None:
    compilation = _compilation(tmp_path, project_root)

    run = run_acquisition(
        root=tmp_path,
        compilation=compilation,
        config=_config(),
        execute=False,
    )

    assert run.executed is False
    assert run.adapter_hash is None
    assert run.config_path.is_file()
    parsed_config = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
    assert parsed_config["lora_parameters"]["keys"] == list(ALL_QWEN_PROJECTIONS)
    manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["promotion_eligible"] is False
    assert manifest["model_revision"] == "a" * 40
    assert manifest["source_record_ids"] == [
        "ENG-INC-002",
        "ENG-REL-001",
        "FIN-EXP-001",
        "FIN-INV-002",
        "HR-LEAVE-001",
        "HR-REMOTE-002",
        "PROC-VEND-001",
        "SUP-SLA-001",
    ]
    assert manifest["adapter_hash"] is None


def test_execute_hashes_adapter_without_legacy_registry(
    tmp_path: Path,
    project_root: Path,
    monkeypatch,
) -> None:
    compilation = _compilation(tmp_path, project_root)
    adapter_bytes = b"synthetic adapter"

    monkeypatch.setattr(
        "enterprise_memory_mlx.acquisition_training.materialize_model_snapshot",
        lambda _model, _revision: "/models/pinned",
    )

    def fake_run(_command, *, cwd, log_path):
        assert cwd == tmp_path
        log_path.parent.mkdir(parents=True)
        log_path.write_text("synthetic training log\n", encoding="utf-8")
        adapter = (
            tmp_path
            / "artifacts"
            / "acquisition"
            / "adapters"
            / "smoke_non_promotable-v2-micro-iterations-r16-e24-s42"
        )
        adapter.mkdir(parents=True)
        (adapter / "adapters.safetensors").write_bytes(adapter_bytes)

    monkeypatch.setattr(
        "enterprise_memory_mlx.acquisition_training.execute_training_command",
        fake_run,
    )
    monkeypatch.setattr(
        "enterprise_memory_mlx.acquisition_training.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "enterprise_memory_mlx.acquisition_training.importlib.util.find_spec",
        lambda name: object() if name == "mlx_lm" else None,
    )

    run = run_acquisition(
        root=tmp_path,
        compilation=compilation,
        config=_config(),
        execute=True,
    )

    assert run.executed is True
    assert run.adapter_hash == hashlib.sha256(adapter_bytes).hexdigest()
    manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter_hash"] == run.adapter_hash
    assert manifest["training_log_hash"] == hashlib.sha256(
        b"synthetic training log\n"
    ).hexdigest()
    assert not (tmp_path / "artifacts" / "registry" / "adapters.json").exists()

    verified = load_verified_acquisition_adapter(run.run_manifest_path)
    assert verified.adapter_hash == run.adapter_hash
    assert verified.source_record_ids == tuple(manifest["source_record_ids"])
    assert verified.inherited_classification == "internal_shared"
    assert verified.promotion_eligible is False


def test_revised_trainer_does_not_import_legacy_training(project_root: Path) -> None:
    source = (
        project_root
        / "src"
        / "enterprise_memory_mlx"
        / "acquisition_training.py"
    ).read_text(encoding="utf-8")
    assert "from .training" not in source
    assert "from .hardware" not in source
    assert "register_adapter" not in source


def test_acquire_command_is_dry_run_unless_execute_is_explicit() -> None:
    parser = build_parser()
    safe = parser.parse_args(["acquire"])
    executing = parser.parse_args(["acquire", "--execute"])

    assert safe.command == "acquire"
    assert safe.execute is False
    assert safe.rank == 16
    assert safe.target_exposures == 24
    assert executing.execute is True
