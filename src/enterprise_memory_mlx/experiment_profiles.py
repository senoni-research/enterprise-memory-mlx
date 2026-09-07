"""Immutable profiles for the bounded model-upgrade exploratory experiment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .utils import sha256_json

MODEL_UPGRADE_PROTOCOL_ID = "model-upgrade-exploratory/v1"
MODEL_UPGRADE_PROFILE = "model_upgrade_exploratory_v1"
MODEL_UPGRADE_EXECUTION_REVISION = "v2-explicit-seeding-and-output-integrity"

QWEN_27B_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
QWEN_27B_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
GEMMA_JUDGE_MODEL_ID = "mlx-community/gemma-4-31b-it-8bit"
GEMMA_JUDGE_REVISION = "f5f3dc92ab4af76724c36c21eb6bedadb3a851be"
QWEN_4B_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
QWEN_4B_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"

FULL_ATTENTION_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
LINEAR_ATTENTION_TARGETS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
)
MLP_TARGETS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

LayerKind = Literal["full_attention", "linear_attention"]


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    revision: str
    upstream_model_id: str
    model_type: str
    text_model_type: str
    num_layers: int
    layer_pattern: tuple[LayerKind, ...]
    target_modules: tuple[str, ...]
    expected_target_count: int
    quantization: dict[str, Any]
    license: str

    def expected_targets_for_layer(self, layer_index: int) -> tuple[str, ...]:
        kind = self.layer_pattern[layer_index]
        attention = (
            LINEAR_ATTENTION_TARGETS if kind == "linear_attention" else FULL_ATTENTION_TARGETS
        )
        return attention + MLP_TARGETS


QWEN_4B_PROFILE = ModelProfile(
    profile_id="qwen3-4b-instruct-2507-4bit/v1",
    model_id=QWEN_4B_MODEL_ID,
    revision=QWEN_4B_REVISION,
    upstream_model_id="Qwen/Qwen3-4B-Instruct-2507",
    model_type="qwen3",
    text_model_type="qwen3",
    num_layers=36,
    layer_pattern=("full_attention",) * 36,
    target_modules=FULL_ATTENTION_TARGETS + MLP_TARGETS,
    expected_target_count=36 * 7,
    quantization={"bits": 4, "group_size": 64, "mode": "affine"},
    license="apache-2.0",
)

QWEN_27B_PROFILE = ModelProfile(
    profile_id="qwen3.8-27b-4bit/v1",
    model_id=QWEN_27B_MODEL_ID,
    revision=QWEN_27B_REVISION,
    upstream_model_id="Qwen/Qwen3.8-27B",
    model_type="qwen3_5",
    text_model_type="qwen3_5_text",
    num_layers=64,
    layer_pattern=tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(64)
    ),
    target_modules=(FULL_ATTENTION_TARGETS + LINEAR_ATTENTION_TARGETS + MLP_TARGETS),
    expected_target_count=(16 * 4) + (48 * 5) + (64 * 3),
    quantization={"bits": 4, "group_size": 64, "mode": "affine"},
    license="apache-2.0",
)

MODEL_PROFILES = {profile.model_id: profile for profile in (QWEN_4B_PROFILE, QWEN_27B_PROFILE)}


@dataclass(frozen=True)
class ModelUpgradeExperimentConfig:
    protocol_id: str = MODEL_UPGRADE_PROTOCOL_ID
    profile: str = MODEL_UPGRADE_PROFILE
    execution_revision: str = MODEL_UPGRADE_EXECUTION_REVISION
    rank: int = 16
    scale: float = 2.0
    dropout: float = 0.05
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_sequence_length: int = 2048
    seed: int = 42
    target_exposures_per_record: int = 96
    diagnostic_exposures: tuple[int, ...] = (24, 48, 96)
    generator_thinking: bool = False
    generator_temperature: float = 0.0
    generator_max_output_tokens: int = 1024
    judge_thinking: bool = False
    judge_temperature: float = 0.0
    judge_max_output_tokens: int = 1024
    application_memory_limit_gib: int = 96

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "profile": self.profile,
            "execution_revision": self.execution_revision,
            "rank": self.rank,
            "scale": self.scale,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_sequence_length": self.max_sequence_length,
            "seed": self.seed,
            "target_exposures_per_record": self.target_exposures_per_record,
            "diagnostic_exposures": list(self.diagnostic_exposures),
            "generator": {
                "thinking": self.generator_thinking,
                "temperature": self.generator_temperature,
                "max_output_tokens": self.generator_max_output_tokens,
                "fresh_state_per_question": True,
            },
            "judge": {
                "model_id": GEMMA_JUDGE_MODEL_ID,
                "revision": GEMMA_JUDGE_REVISION,
                "thinking": self.judge_thinking,
                "temperature": self.judge_temperature,
                "max_output_tokens": self.judge_max_output_tokens,
                "structured_json": True,
            },
            "application_memory_limit_gib": self.application_memory_limit_gib,
        }


MODEL_UPGRADE_CONFIG = ModelUpgradeExperimentConfig()


def model_profile(model_id: str, revision: str) -> ModelProfile:
    try:
        profile = MODEL_PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(f"No governed acquisition model profile for {model_id}") from exc
    if revision != profile.revision:
        raise ValueError(
            f"{model_id} must use pinned revision {profile.revision}; received {revision}"
        )
    return profile


def validate_snapshot_config(
    profile: ModelProfile,
    snapshot_config: dict[str, Any],
) -> None:
    if snapshot_config.get("model_type") != profile.model_type:
        raise ValueError("Pinned snapshot model_type does not match model profile")
    text = (
        snapshot_config.get("text_config")
        if isinstance(snapshot_config.get("text_config"), dict)
        else snapshot_config
    )
    if text.get("model_type", profile.model_type) != profile.text_model_type:
        raise ValueError("Pinned snapshot text model_type does not match model profile")
    if int(text.get("num_hidden_layers", -1)) != profile.num_layers:
        raise ValueError("Pinned snapshot layer count does not match model profile")
    layer_types = text.get("layer_types")
    if layer_types is not None and tuple(layer_types) != profile.layer_pattern:
        raise ValueError("Pinned snapshot layer pattern does not match model profile")
    quantization = snapshot_config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Pinned snapshot is missing quantization configuration")
    normalized = {
        "bits": int(quantization.get("bits", -1)),
        "group_size": int(quantization.get("group_size", -1)),
        "mode": str(quantization.get("mode", "affine")),
    }
    if normalized != profile.quantization:
        raise ValueError("Pinned snapshot quantization does not match model profile")


def immutable_run_name(
    *,
    experiment_config: ModelUpgradeExperimentConfig,
    profile: ModelProfile,
    dataset_hash: str,
    dataset_manifest_hash: str,
    seed: int,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_hash):
        raise ValueError("dataset_hash must be a SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_manifest_hash):
        raise ValueError("dataset_manifest_hash must be a SHA-256 digest")
    identity = {
        "experiment": experiment_config.to_dict(),
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "model_profile": profile.profile_id,
        "dataset_hash": dataset_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "seed": seed,
    }
    digest = sha256_json(identity)[:16]
    model_slug = profile.model_id.rsplit("/", maxsplit=1)[-1].lower()
    return (
        f"model-upgrade-v1--{model_slug}--{profile.revision[:10]}--"
        f"data-{dataset_hash[:12]}--r{experiment_config.rank}-"
        f"e{experiment_config.target_exposures_per_record}-s{seed}--{digest}"
    )
