from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import yaml

from .compiler import manifest_snapshot_hash
from .hardware import TrainingPreset, detect_hardware, resolve_preset
from .registry import register_adapter
from .utils import atomic_write_text, slugify

Stage = Literal["vanilla", "inject", "align", "recover"]
PIPELINE_STAGES: tuple[Stage, ...] = ("inject", "align", "recover")
VALID_STAGES: tuple[Stage, ...] = ("vanilla", *PIPELINE_STAGES)


def train_pipeline(
    root: Path,
    *,
    stage: str = "all",
    preset_name: str = "auto",
    model_override: str | None = None,
    domain: str = "global",
    dry_run: bool = False,
    allow_non_mac: bool = False,
    allow_scientifically_invalid: bool = False,
) -> list[list[str]]:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "training.train_pipeline",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    hardware = detect_hardware()
    if not allow_non_mac and not hardware.is_apple_silicon and not dry_run:
        raise RuntimeError("MLX training requires an Apple Silicon Mac")

    preset = resolve_preset(preset_name, hardware)
    if model_override:
        preset = TrainingPreset(**{**asdict(preset), "model": model_override})

    requested: tuple[Stage, ...]
    if stage == "all":
        requested = PIPELINE_STAGES
    elif stage in VALID_STAGES:
        requested = (stage,)  # type: ignore[assignment]
    else:
        raise ValueError(
            f"Unknown stage {stage!r}; choose vanilla, inject, align, recover, or all"
        )

    if not dry_run and importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError('mlx-lm is not installed. Run: pip install -e ".[mac]"')

    manifest_path = root / "artifacts" / "manifests" / "knowledge_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Compile the knowledge first with: emmlx compile")
    snapshot_hash = manifest_snapshot_hash(manifest_path)

    commands: list[list[str]] = []
    for current_stage in requested:
        config, config_path = build_stage_config(
            root,
            current_stage,
            preset=preset,
            domain=domain,
            require_resume_exists=not dry_run,
            allow_scientifically_invalid=allow_scientifically_invalid,
        )
        atomic_write_text(config_path, yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
        commands.append(command)
        if dry_run:
            continue

        subprocess.run(command, cwd=root, check=True)
        adapter_path = Path(config["adapter_path"])
        adapter_file = adapter_path / "adapters.safetensors"
        if not adapter_file.exists():
            raise RuntimeError(f"Training finished without producing {adapter_file}")
        register_adapter(
            root / "artifacts" / "registry" / "adapters.json",
            name=f"{slugify(domain)}-{current_stage}",
            stage=current_stage,
            domain=domain,
            base_model=preset.model,
            adapter_path=adapter_path,
            config=config,
            knowledge_snapshot_hash=snapshot_hash,
        )
    return commands


def build_stage_config(
    root: Path,
    stage: Stage,
    *,
    preset: TrainingPreset,
    domain: str = "global",
    require_resume_exists: bool = True,
    allow_scientifically_invalid: bool = False,
) -> tuple[dict[str, Any], Path]:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "training.build_stage_config",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    dataset_root = root / "artifacts" / "datasets"
    adapter_root = root / "artifacts" / "adapters"
    config_root = root / "artifacts" / "configs"
    domain_slug = slugify(domain)

    if domain != "global":
        dataset_root = dataset_root / "domains" / domain_slug
        adapter_root = adapter_root / "domains" / domain_slug
        config_root = config_root / "domains" / domain_slug

    data_path = dataset_root / stage
    if not (data_path / "train.jsonl").exists():
        raise FileNotFoundError(f"Missing compiled dataset: {data_path}")

    adapter_path = adapter_root / stage
    previous_stage = {"align": "inject", "recover": "align"}.get(stage)
    resume_file = adapter_root / previous_stage / "adapters.safetensors" if previous_stage else None
    if resume_file and require_resume_exists and not resume_file.exists():
        raise FileNotFoundError(
            f"{stage} requires {resume_file}. Train the preceding stage first or use --stage all."
        )

    iterations = {
        "vanilla": preset.align_iters,
        "inject": preset.inject_iters,
        "align": preset.align_iters,
        "recover": preset.recover_iters,
    }[stage]
    learning_rate = {
        "vanilla": preset.align_lr,
        "inject": preset.inject_lr,
        "align": preset.align_lr,
        "recover": preset.recover_lr,
    }[stage]

    config: dict[str, Any] = {
        "model": preset.model,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {
            "adamw": {
                "betas": [0.9, 0.98],
                "eps": 1e-6,
                "weight_decay": 0.01,
            }
        },
        "data": str(data_path),
        "seed": 42,
        "num_layers": preset.num_layers,
        "batch_size": preset.batch_size,
        "iters": iterations,
        "val_batches": -1,
        "learning_rate": learning_rate,
        "steps_per_report": 10,
        "steps_per_eval": max(1, min(iterations, max(20, iterations // 4))),
        "grad_accumulation_steps": preset.grad_accumulation_steps,
        "resume_adapter_file": str(resume_file) if resume_file else None,
        "adapter_path": str(adapter_path),
        "save_every": max(1, min(iterations, max(20, iterations // 2))),
        "test": False,
        "test_batches": -1,
        "max_seq_length": preset.max_seq_length,
        "grad_checkpoint": preset.grad_checkpoint,
        "mask_prompt": True,
        "lora_parameters": {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": preset.lora_rank,
            "scale": preset.lora_scale,
            "dropout": 0.0,
        },
    }
    config_path = config_root / f"{stage}.yaml"
    return config, config_path


def doctor_report() -> dict[str, Any]:
    hardware = detect_hardware()
    preset = resolve_preset("auto", hardware)
    return {
        "hardware": hardware.to_dict(),
        "recommended_preset": asdict(preset),
        "mlx_lm_installed": importlib.util.find_spec("mlx_lm") is not None,
        "mlx_lm_version": _package_version("mlx-lm"),
        "python": sys.version.split()[0],
    }


def doctor_json() -> str:
    return json.dumps(doctor_report(), indent=2)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
