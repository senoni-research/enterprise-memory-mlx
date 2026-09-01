"""Revised all-linear MLX acquisition path; never imports legacy training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .acquisition_compiler import AcquisitionCompilation
from .utils import atomic_write_text, sha256_json

DEFAULT_ACQUISITION_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
ACQUISITION_CONTRACT_VERSION = "v2-micro-iterations"
ALL_QWEN_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class AcquisitionConfig:
    model_id: str
    model_revision: str
    rank: int
    scale: float
    learning_rate: float
    dropout: float
    num_layers: int
    batch_size: int
    grad_accumulation_steps: int
    max_seq_length: int
    seed: int
    profile: str


@dataclass(frozen=True)
class AcquisitionRun:
    config_path: Path
    adapter_path: Path
    run_manifest_path: Path
    training_log_path: Path
    command: tuple[str, ...]
    executed: bool
    adapter_hash: str | None


@dataclass(frozen=True)
class VerifiedAcquisitionAdapter:
    run_manifest_path: Path
    model_id: str
    model_revision: str
    adapter_path: Path
    adapter_hash: str
    source_record_ids: tuple[str, ...]
    inherited_classification: str
    promotion_eligible: bool


def build_mlx_acquisition_config(
    *,
    compilation: AcquisitionCompilation,
    config: AcquisitionConfig,
    model_path: str,
    adapter_path: Path,
) -> dict[str, Any]:
    if config.rank not in {8, 16, 32}:
        raise ValueError("Acquisition rank must be one of 8, 16, or 32")
    if config.num_layers != 36:
        raise ValueError("Qwen3-4B acquisition must target all 36 layers")
    if config.scale != 2.0:
        raise ValueError("MLX LoRA scale must be 2.0 for alpha=32, rank=16 smoke")
    if config.batch_size <= 0 or config.grad_accumulation_steps <= 0:
        raise ValueError("Batch and accumulation must be positive")
    expected_batch = config.batch_size * config.grad_accumulation_steps
    if expected_batch != compilation.schedule.effective_batch_size:
        raise ValueError("Training batch does not match compiled exposure schedule")
    if config.batch_size != compilation.schedule.micro_batch_size:
        raise ValueError("Micro batch size does not match compiled exposure schedule")
    if (
        config.grad_accumulation_steps
        != compilation.schedule.gradient_accumulation_steps
    ):
        raise ValueError("Gradient accumulation does not match exposure schedule")
    optimizer_steps = compilation.schedule.optimizer_steps
    iterations = compilation.schedule.micro_iterations
    warmup = max(1, round(optimizer_steps * 0.1))
    decay_steps = max(1, optimizer_steps - warmup)
    return {
        "model": model_path,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {
            "adamw": {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "bias_correction": True,
            }
        },
        "data": str(compilation.dataset_dir),
        "seed": config.seed,
        "num_layers": config.num_layers,
        "batch_size": config.batch_size,
        "iters": iterations,
        "val_batches": 0,
        "learning_rate": config.learning_rate,
        "steps_per_report": max(
            1, compilation.schedule.micro_iterations_per_epoch // 2
        ),
        "steps_per_eval": iterations + 1,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "resume_adapter_file": None,
        "adapter_path": str(adapter_path),
        "save_every": compilation.schedule.micro_iterations_per_epoch,
        "test": False,
        "test_batches": 0,
        "max_seq_length": config.max_seq_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "keys": list(ALL_QWEN_PROJECTIONS),
            "rank": config.rank,
            "scale": config.scale,
            "dropout": config.dropout,
        },
        "lr_schedule": {
            "name": "cosine_decay",
            "warmup": warmup,
            "warmup_init": 0.0,
            "arguments": [
                config.learning_rate,
                decay_steps,
                config.learning_rate * 0.1,
            ],
        },
    }


def run_acquisition(
    *,
    root: Path,
    compilation: AcquisitionCompilation,
    config: AcquisitionConfig,
    execute: bool,
) -> AcquisitionRun:
    if config.profile != "smoke_non_promotable":
        raise ValueError("Only the non-promotable smoke profile is enabled")
    if compilation.promotion_eligible:
        raise ValueError("Smoke runner expected a non-promotable compilation")
    manifest = json.loads(compilation.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile") != config.profile:
        raise ValueError("Training profile does not match compiled manifest")
    if execute and platform.system() != "Darwin":
        raise RuntimeError("MLX acquisition execution requires macOS")
    if execute and importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError('MLX-LM is required: pip install -e ".[mac]"')

    run_name = (
        f"{config.profile}-{ACQUISITION_CONTRACT_VERSION}-r{config.rank}-"
        f"e{compilation.schedule.target_exposures_per_fact}-s{config.seed}"
    )
    adapter_path = root / "artifacts" / "acquisition" / "adapters" / run_name
    config_path = root / "artifacts" / "acquisition" / "configs" / f"{run_name}.yaml"
    run_manifest_path = (
        root / "artifacts" / "acquisition" / "runs" / f"{run_name}.json"
    )
    training_log_path = (
        root / "artifacts" / "acquisition" / "logs" / f"{run_name}.log"
    )
    model_path = config.model_id
    if execute:
        model_path = materialize_model_snapshot(
            config.model_id,
            config.model_revision,
        )
    mlx_config = build_mlx_acquisition_config(
        compilation=compilation,
        config=config,
        model_path=model_path,
        adapter_path=adapter_path,
    )
    atomic_write_text(config_path, yaml.safe_dump(mlx_config, sort_keys=False))
    command = (
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
    )
    adapter_hash = None
    if execute:
        execute_training_command(command, cwd=root, log_path=training_log_path)
        adapter_file = adapter_path / "adapters.safetensors"
        if not adapter_file.is_file():
            raise RuntimeError(f"Training produced no adapter: {adapter_file}")
        adapter_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()

    config_payload = {
        **config.__dict__,
        "target_modules": list(ALL_QWEN_PROJECTIONS),
        "mlx_config": mlx_config,
    }
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "trained" if execute else "dry_run",
        "profile": config.profile,
        "acquisition_contract_version": ACQUISITION_CONTRACT_VERSION,
        "promotion_eligible": False,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_path": model_path,
        "source_record_ids": manifest["record_ids"],
        "source_snapshot_hash": manifest["source_snapshot_hash"],
        "dataset_manifest_path": str(compilation.manifest_path),
        "dataset_manifest_hash": _file_hash(compilation.manifest_path),
        "dataset_hash": manifest["dataset_sha256"],
        "freeze_hash": manifest["fixture_hash"],
        "semantic_audit_path": str(compilation.audit_path),
        "semantic_audit_hash": _file_hash(compilation.audit_path),
        "inherited_classification": manifest["highest_sensitivity"],
        "training_config": config_payload,
        "training_config_hash": sha256_json(config_payload),
        "adapter_path": str(adapter_path),
        "adapter_hash": adapter_hash,
        "training_log_path": str(training_log_path) if execute else None,
        "training_log_hash": (
            _file_hash(training_log_path) if execute else None
        ),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "mlx_lm_version": _package_version("mlx-lm"),
        },
        "warning": (
            "Synthetic rank-16 smoke only. Fewer than 24 independent study "
            "views per fact and no certified semantic grading; cannot promote."
        ),
    }
    atomic_write_text(
        run_manifest_path,
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return AcquisitionRun(
        config_path=config_path,
        adapter_path=adapter_path,
        run_manifest_path=run_manifest_path,
        training_log_path=training_log_path,
        command=command,
        executed=execute,
        adapter_hash=adapter_hash,
    )


def materialize_model_snapshot(model_id: str, revision: str) -> str:
    if not revision:
        raise ValueError("Pinned model revision is required")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to materialize the model") from exc
    return snapshot_download(model_id, revision=revision)


def execute_training_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    log_path: Path,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout or ""
    print(output, end="")
    atomic_write_text(log_path, output)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=output,
        )


def load_verified_acquisition_adapter(
    run_manifest_path: Path,
) -> VerifiedAcquisitionAdapter:
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "trained":
        raise ValueError("Acquisition run manifest is not a trained run")
    if payload.get("promotion_eligible") is not False:
        raise ValueError("Smoke adapter must remain explicitly non-promotable")
    adapter_path = Path(str(payload.get("adapter_path", ""))).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = run_manifest_path.resolve().parents[3] / adapter_path
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Acquisition adapter missing: {adapter_file}")
    expected_hash = str(payload.get("adapter_hash", ""))
    actual_hash = _file_hash(adapter_file)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Acquisition adapter hash does not match run manifest")
    record_ids = tuple(str(item) for item in payload.get("source_record_ids", []))
    classification = str(payload.get("inherited_classification", ""))
    model_id = str(payload.get("model_id", ""))
    revision = str(payload.get("model_revision", ""))
    if not record_ids or not classification or not model_id or not revision:
        raise ValueError("Acquisition run manifest is missing governed provenance")
    return VerifiedAcquisitionAdapter(
        run_manifest_path=run_manifest_path.resolve(),
        model_id=model_id,
        model_revision=revision,
        adapter_path=adapter_path.resolve(),
        adapter_hash=actual_hash,
        source_record_ids=record_ids,
        inherited_classification=classification,
        promotion_eligible=False,
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
