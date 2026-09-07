"""On-device MLX acquisition runtime with fail-closed architecture coverage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import types
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .experiment_profiles import (
    MODEL_UPGRADE_CONFIG,
    QWEN_27B_MODEL_ID,
    ModelProfile,
    model_profile,
    validate_snapshot_config,
)
from .utils import atomic_write_text, sha256_json, sha256_text


class NonThinkingTokenizer:
    """Force the pinned template policy for both full rows and prompt masking."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        requested = kwargs.pop("enable_thinking", False)
        if requested is not False:
            raise ValueError("Acquisition training requires enable_thinking=False")
        return self._tokenizer.apply_chat_template(
            *args,
            enable_thinking=False,
            **kwargs,
        )


def build_target_coverage(
    model: Any,
    profile: ModelProfile,
    snapshot_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect the actual adapted graph and reject every silent omission."""
    try:
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError as exc:
        raise RuntimeError("MLX-LM is required for target coverage") from exc

    modules = []
    global_target_matches = {target: 0 for target in profile.target_modules}
    layer_matches: dict[int, set[str]] = {index: set() for index in range(profile.num_layers)}
    trainable_count = 0
    for path, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        layer_index, relative = _layer_and_relative_path(path)
        if layer_index is None or relative is None:
            raise RuntimeError(f"Adapted module is outside language layers: {path}")
        if not 0 <= layer_index < profile.num_layers:
            raise RuntimeError(f"Adapted module has invalid layer index: {path}")
        expected = set(profile.expected_targets_for_layer(layer_index))
        if relative not in expected:
            raise RuntimeError(f"Unexpected adapted module for layer type: {path}")
        layer_matches[layer_index].add(relative)
        global_target_matches[relative] += 1
        input_dims, rank = module.lora_a.shape
        rank_b, output_dims = module.lora_b.shape
        if rank != rank_b:
            raise RuntimeError(f"Incompatible LoRA shapes at {path}")
        trainable = int(module.lora_a.size + module.lora_b.size)
        trainable_count += trainable
        base = module.linear
        quantized = hasattr(base, "bits")
        modules.append(
            {
                "path": path,
                "layer_index": layer_index,
                "layer_type": profile.layer_pattern[layer_index],
                "relative_target": relative,
                "base_weight_shape": list(base.weight.shape),
                "logical_input_dims": int(input_dims),
                "logical_output_dims": int(output_dims),
                "rank": int(rank),
                "quantization": (
                    {
                        "bits": int(base.bits),
                        "group_size": int(base.group_size),
                        "mode": str(getattr(base, "mode", "affine")),
                    }
                    if quantized
                    else None
                ),
                "trainable_parameter_count": trainable,
            }
        )

    omissions = []
    for layer_index, matches in layer_matches.items():
        expected = set(profile.expected_targets_for_layer(layer_index))
        for target in sorted(expected - matches):
            omissions.append(
                {
                    "layer_index": layer_index,
                    "layer_type": profile.layer_pattern[layer_index],
                    "target": target,
                }
            )
    zero_global = [target for target, count in global_target_matches.items() if count == 0]
    actual_trainable = sum(
        int(value.size) for _name, value in tree_flatten(model.trainable_parameters())
    )
    full_layers = sum(kind == "full_attention" for kind in profile.layer_pattern)
    linear_layers = sum(kind == "linear_attention" for kind in profile.layer_pattern)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_profile": profile.profile_id,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "model_type": snapshot_config.get("model_type"),
        "layer_count": len(getattr(model, "layers", [])),
        "full_attention_layer_count": full_layers,
        "linear_attention_layer_count": linear_layers,
        "mlp_layer_count": profile.num_layers,
        "configured_targets": list(profile.target_modules),
        "global_target_matches": global_target_matches,
        "expected_adapted_module_count": profile.expected_target_count,
        "adapted_module_count": len(modules),
        "reported_trainable_parameter_count": trainable_count,
        "actual_trainable_parameter_count": actual_trainable,
        "unmatched_configured_targets": zero_global,
        "applicable_target_omissions": omissions,
        "modules": sorted(modules, key=lambda item: str(item["path"])),
    }
    if len(getattr(model, "layers", [])) != profile.num_layers:
        raise RuntimeError("Runtime model layer count does not match profile")
    if len(modules) != profile.expected_target_count:
        raise RuntimeError(
            "Adapter target count mismatch: "
            f"expected {profile.expected_target_count}, got {len(modules)}"
        )
    if omissions or zero_global:
        raise RuntimeError(
            f"Adapter target coverage has omissions={omissions[:5]} unmatched={zero_global}"
        )
    if actual_trainable != trainable_count or trainable_count <= 0:
        raise RuntimeError("Trainable parameter inventory does not match LoRA modules")
    return report


def run_training(config_path: Path, coverage_path: Path) -> None:
    config = _load_runtime_config(config_path)
    profile, snapshot_path, snapshot_config = _validated_profile(config)
    if Path(config["adapter_path"]).exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory: {config['adapter_path']}")
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np
        from mlx_lm import load
        from mlx_lm.tuner.datasets import CacheDataset, load_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers
        from mlx_lm.utils import save_config
    except ImportError as exc:
        raise RuntimeError('Training requires pip install -e ".[mac]"') from exc

    _set_memory_limit(mx)
    model = tokenizer = None
    try:
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        print("Loading untouched pinned base for measured training", flush=True)
        model, tokenizer = load(
            str(snapshot_path),
            tokenizer_config={"trust_remote_code": False},
        )
        wrapped = NonThinkingTokenizer(tokenizer)
        template = _verify_template_policy(wrapped, profile)
        args = types.SimpleNamespace(**config)
        train_set, valid_set, _test_set = load_dataset(args, wrapped)
        token_report = _validate_training_rows(train_set, config["max_seq_length"])

        model.freeze()
        linear_to_lora_layers(
            model,
            profile.num_layers,
            config["lora_parameters"],
        )
        coverage = build_target_coverage(model, profile, snapshot_config)
        coverage.update(
            {
                "status": "coverage_verified_before_measured_training",
                "training_data": token_report,
                "chat_template_hash": sha256_text(str(template)),
                "snapshot_identity": _snapshot_identity(snapshot_path, snapshot_config),
                "enable_thinking": False,
                "seeding": {
                    "seed": seed,
                    "python_random": True,
                    "numpy": True,
                    "mlx": True,
                    "applied_before_lora_initialization_and_dataset_iteration": True,
                },
                "training_config_hash": sha256_json(config),
            }
        )
        atomic_write_text(
            coverage_path,
            json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

        adapter_path = Path(config["adapter_path"])
        adapter_path.mkdir(parents=True, exist_ok=False)
        save_config(config, adapter_path / "adapter_config.json")
        learning_rate = build_schedule(config["lr_schedule"])
        optimizer = optim.AdamW(
            learning_rate=learning_rate,
            **config["optimizer_config"]["adamw"],
        )
        training_args = TrainingArgs(
            batch_size=int(config["batch_size"]),
            iters=int(config["iters"]),
            val_batches=int(config["val_batches"]),
            steps_per_report=int(config["steps_per_report"]),
            steps_per_eval=int(config["steps_per_eval"]),
            # The pinned trainer supports a fixed interval. Saving every epoch
            # includes required 24/48/96 checkpoints plus a disclosed 72 diagnostic.
            steps_per_save=int(config["diagnostic_save_iterations"][0]),
            adapter_file=str(adapter_path / "adapters.safetensors"),
            max_seq_length=int(config["max_seq_length"]),
            grad_checkpoint=bool(config["grad_checkpoint"]),
            grad_accumulation_steps=int(config["grad_accumulation_steps"]),
        )
        governed_limit = MODEL_UPGRADE_CONFIG.application_memory_limit_gib * 2**30
        original_set_wired_limit = mx.set_wired_limit

        def capped_set_wired_limit(requested: int) -> None:
            original_set_wired_limit(min(int(requested), governed_limit))

        mx.set_wired_limit = capped_set_wired_limit
        try:
            train(
                model=model,
                optimizer=optimizer,
                train_dataset=CacheDataset(train_set),
                val_dataset=CacheDataset(valid_set) if valid_set else None,
                args=training_args,
            )
        finally:
            mx.set_wired_limit = original_set_wired_limit
        actual_updates = int(config["iters"]) // int(config["grad_accumulation_steps"])
        expected_updates = int(config["expected_optimizer_updates"])
        if actual_updates != expected_updates:
            raise RuntimeError(
                f"Actual optimizer update count {actual_updates} != {expected_updates}"
            )
        print(
            "MEASURED_TRAINING_COMPLETE "
            f"micro_iterations={config['iters']} optimizer_updates={actual_updates}",
            flush=True,
        )
    finally:
        model = None
        tokenizer = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            pass


def run_preflight(
    config_path: Path,
    coverage_path: Path,
    report_path: Path,
) -> None:
    """Exercise the real backward path once; all outputs are disposable."""
    config = _load_runtime_config(config_path)
    profile, snapshot_path, snapshot_config = _validated_profile(config)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.datasets import load_dataset
        from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise RuntimeError('Preflight requires pip install -e ".[mac]"') from exc

    _set_memory_limit(mx)
    model = tokenizer = None
    started_memory = _memory_snapshot(mx)
    started_swap = _swap_snapshot()
    try:
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        model, tokenizer = load(
            str(snapshot_path),
            tokenizer_config={"trust_remote_code": False},
        )
        wrapped = NonThinkingTokenizer(tokenizer)
        template = _verify_template_policy(wrapped, profile)
        args = types.SimpleNamespace(**config)
        train_set, _valid, _test = load_dataset(args, wrapped)
        token_report = _validate_training_rows(train_set, config["max_seq_length"])
        model.freeze()
        linear_to_lora_layers(
            model,
            profile.num_layers,
            config["lora_parameters"],
        )
        coverage = build_target_coverage(model, profile, snapshot_config)
        coverage.update(
            {
                "status": "coverage_verified_in_disposable_preflight",
                "snapshot_identity": _snapshot_identity(snapshot_path, snapshot_config),
                "chat_template_hash": sha256_text(str(template)),
                "enable_thinking": False,
                "seeding": {
                    "seed": seed,
                    "python_random": True,
                    "numpy": True,
                    "mlx": True,
                    "applied_before_lora_initialization_and_dataset_iteration": True,
                },
                "training_config_hash": sha256_json(config),
            }
        )
        atomic_write_text(
            coverage_path,
            json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

        longest_index = max(
            range(len(train_set)),
            key=lambda index: len(train_set.process(train_set[index])[0]),
        )
        tokens, offset = train_set.process(train_set[longest_index])
        if len(tokens) > int(config["max_seq_length"]):
            raise RuntimeError("Longest supervised row would be truncated")
        configured_max_sequence_length = int(config["max_seq_length"])
        exercised_sequence_length = len(tokens)
        batch = mx.array([tokens])
        lengths = mx.array([[offset, exercised_sequence_length]])
        grad_checkpoint(model.layers[0])
        model.train()
        loss_and_grad = nn.value_and_grad(model, default_loss)
        trainable_parameters = list(tree_flatten(model.trainable_parameters()))
        before = {
            name: hashlib.sha256(bytes(value)).hexdigest() for name, value in trainable_parameters
        }
        optimizer = optim.AdamW(
            learning_rate=float(config["learning_rate"]),
            **config["optimizer_config"]["adamw"],
        )
        state = [model.state, optimizer.state, mx.random.state]

        def preflight_step(batch: Any, lengths: Any) -> tuple[Any, Any, Any, Any]:
            (step_loss, step_tokens), gradients = loss_and_grad(model, batch, lengths)
            gradient_values = [value for _name, value in tree_flatten(gradients)]
            finite_count = sum(
                mx.all(mx.isfinite(value)).astype(mx.int32) for value in gradient_values
            )
            nonzero_count = sum(mx.any(value != 0).astype(mx.int32) for value in gradient_values)
            optimizer.update(model, gradients)
            return step_loss, step_tokens, finite_count, nonzero_count

        compiled_step = mx.compile(preflight_step, inputs=state, outputs=state)
        loss, supervised_tokens, finite_count, nonzero_count = compiled_step(
            batch,
            lengths,
        )
        mx.eval(
            state,
            loss,
            supervised_tokens,
            finite_count,
            nonzero_count,
        )
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError("Preflight loss is non-finite")
        gradient_tensor_count = len(trainable_parameters)
        finite_gradients = int(finite_count.item()) == gradient_tensor_count
        nonzero_gradient_tensors = int(nonzero_count.item())
        if not finite_gradients or nonzero_gradient_tensors == 0:
            raise RuntimeError("Preflight gradients are non-finite or all zero")
        after = {
            name: hashlib.sha256(bytes(value)).hexdigest()
            for name, value in tree_flatten(model.trainable_parameters())
        }
        changed = sorted(name for name in before if before[name] != after[name])
        if not changed:
            raise RuntimeError("Preflight optimizer update changed no adapter tensors")

        disposable = report_path.parent / "preflight-adapters.safetensors"
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(disposable), weights)
        saved_hashes = {
            name: hashlib.sha256(bytes(value)).hexdigest() for name, value in weights.items()
        }
        for _name, value in tree_flatten(model.trainable_parameters()):
            value[:] = 0
        model.load_weights(str(disposable), strict=False)
        mx.eval(model.trainable_parameters())
        reloaded_hashes = {
            name: hashlib.sha256(bytes(value)).hexdigest()
            for name, value in tree_flatten(model.trainable_parameters())
        }
        if reloaded_hashes != saved_hashes:
            raise RuntimeError("Adapter save/reload parity check failed")

        model.eval()
        source_question = "State the exact normal payment term from the supplied company source."
        source_context = (
            "A valid supplier invoice is payable thirty calendar days after Finance receives it."
        )
        rendered = wrapped.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Answer only from this approved source: " + source_context,
                },
                {"role": "user", "content": source_question},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        fragments = []
        final_response = None
        for response in stream_generate(
            model,
            tokenizer,
            rendered,
            max_tokens=96,
            sampler=make_sampler(temp=0.0),
        ):
            fragments.append(response.text)
            final_response = response
        generated = "".join(fragments)
        if final_response is None or not generated.strip():
            raise RuntimeError("Preflight coherent-generation check returned no text")
        if final_response.finish_reason == "length":
            raise RuntimeError("Preflight source-grounded generation was truncated")
        if "thirty" not in generated.casefold() and "30" not in generated:
            raise RuntimeError("Preflight generation did not recover the supplied payment term")
        report = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "passed",
            "disposable": True,
            "model_id": profile.model_id,
            "model_revision": profile.revision,
            "model_profile": profile.profile_id,
            "execution_revision": MODEL_UPGRADE_CONFIG.execution_revision,
            "seeding": coverage["seeding"],
            "training_config_hash": sha256_json(config),
            "dataset": {
                "path": str(Path(config["data"]).resolve()),
                "train_sha256": hashlib.sha256(
                    (Path(config["data"]) / "train.jsonl").read_bytes()
                ).hexdigest(),
            },
            "coverage_sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
            "training_data": token_report,
            "template": {
                "enable_thinking": False,
                "template_hash": sha256_text(str(template)),
            },
            "backward": {
                "loss": loss_value,
                "supervised_tokens": int(supervised_tokens.item()),
                "exercised_sequence_length": exercised_sequence_length,
                "configured_max_sequence_length": configured_max_sequence_length,
                "configured_cap_fully_exercised": (
                    exercised_sequence_length == configured_max_sequence_length
                ),
                "configured_cap_note": (
                    "All fixed curriculum rows were checked for truncation. "
                    "The backward pass used the actual longest rendered row, "
                    "not artificial padding to the configured cap."
                ),
                "gradient_tensor_count": gradient_tensor_count,
                "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
                "finite_gradients": finite_gradients,
                "changed_tensor_count": len(changed),
                "zero_initial_factor_gradients_allowed": True,
            },
            "save_reload_parity": True,
            "generation": {
                "raw_output": generated,
                "finish_reason": final_response.finish_reason,
                "generated_tokens": final_response.generation_tokens,
                "peak_memory_gb": final_response.peak_memory,
            },
            "memory": {
                "application_limit_gib": MODEL_UPGRADE_CONFIG.application_memory_limit_gib,
                "before": started_memory,
                "after": _memory_snapshot(mx),
                "swap_before": started_swap,
                "swap_after": _swap_snapshot(),
            },
        }
        atomic_write_text(
            report_path,
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
    finally:
        model = None
        tokenizer = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            pass


def _load_runtime_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training config must be a mapping")
    required = {
        "model",
        "governed_model_id",
        "governed_model_revision",
        "experiment_profile",
        "execution_revision",
        "data",
        "adapter_path",
        "lora_parameters",
        "diagnostic_save_iterations",
        "expected_optimizer_updates",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Training config is missing governed fields: {missing}")
    if value.get("experiment_profile") != MODEL_UPGRADE_CONFIG.profile:
        raise ValueError("Runtime only accepts model-upgrade-exploratory/v1")
    if value.get("execution_revision") != MODEL_UPGRADE_CONFIG.execution_revision:
        raise ValueError("Runtime execution revision does not match the frozen experiment")
    if value.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("Training config must explicitly disable thinking")
    expected_scalars = {
        "batch_size": MODEL_UPGRADE_CONFIG.batch_size,
        "grad_accumulation_steps": MODEL_UPGRADE_CONFIG.gradient_accumulation_steps,
        "grad_checkpoint": MODEL_UPGRADE_CONFIG.gradient_checkpointing,
        "learning_rate": MODEL_UPGRADE_CONFIG.learning_rate,
        "max_seq_length": MODEL_UPGRADE_CONFIG.max_sequence_length,
        "optimizer": MODEL_UPGRADE_CONFIG.optimizer,
        "seed": MODEL_UPGRADE_CONFIG.seed,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise ValueError(f"Training config {key} does not match the frozen experiment")
    lora = value.get("lora_parameters")
    if not isinstance(lora, dict) or {
        "rank": lora.get("rank"),
        "scale": lora.get("scale"),
        "dropout": lora.get("dropout"),
    } != {
        "rank": MODEL_UPGRADE_CONFIG.rank,
        "scale": MODEL_UPGRADE_CONFIG.scale,
        "dropout": MODEL_UPGRADE_CONFIG.dropout,
    }:
        raise ValueError("Training LoRA parameters do not match the frozen experiment")
    optimizer_config = value.get("optimizer_config")
    if (
        not isinstance(optimizer_config, dict)
        or optimizer_config.get("adamw", {}).get("weight_decay")
        != MODEL_UPGRADE_CONFIG.weight_decay
    ):
        raise ValueError("Training weight decay does not match the frozen experiment")
    return value


def _validated_profile(
    config: Mapping[str, Any],
) -> tuple[ModelProfile, Path, dict[str, Any]]:
    profile = model_profile(
        str(config["governed_model_id"]),
        str(config["governed_model_revision"]),
    )
    snapshot_path = Path(str(config["model"]))
    snapshot_config_path = snapshot_path / "config.json"
    if not snapshot_config_path.is_file():
        raise FileNotFoundError("Pinned model snapshot is missing config.json")
    snapshot_config = json.loads(snapshot_config_path.read_text(encoding="utf-8"))
    validate_snapshot_config(profile, snapshot_config)
    if int(config.get("num_layers", -1)) != profile.num_layers:
        raise ValueError("Training layer count does not match the model profile")
    if tuple(config.get("lora_parameters", {}).get("keys", ())) != profile.target_modules:
        raise ValueError("Training target modules do not match the model profile")
    return profile, snapshot_path, snapshot_config


def _validate_training_rows(dataset: Any, max_length: int) -> dict[str, Any]:
    lengths = []
    offsets = []
    supervised = []
    for index in range(len(dataset)):
        tokens, offset = dataset.process(dataset[index])
        length = len(tokens)
        if offset <= 0 or offset >= length:
            raise RuntimeError("Answer-token masking produced an invalid prompt offset")
        if length > max_length:
            raise RuntimeError(f"Training row {index} has {length} tokens and would be truncated")
        lengths.append(length)
        offsets.append(offset)
        supervised.append(length - offset)
    return {
        "row_count": len(lengths),
        "longest_rendered_tokens": max(lengths),
        "shortest_rendered_tokens": min(lengths),
        "supervised_tokens_per_epoch": sum(supervised),
        "minimum_supervised_tokens": min(supervised),
        "maximum_supervised_tokens": max(supervised),
        "answer_token_masking_verified": True,
        "truncated_rows": 0,
    }


def _snapshot_identity(
    snapshot_path: Path,
    snapshot_config: Mapping[str, Any],
) -> dict[str, Any]:
    asset_names = (
        "config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    assets = {
        name: hashlib.sha256((snapshot_path / name).read_bytes()).hexdigest()
        for name in asset_names
        if (snapshot_path / name).is_file()
    }
    if "config.json" not in assets or not any(
        name in assets for name in ("chat_template.jinja", "tokenizer_config.json")
    ):
        raise RuntimeError("Pinned model snapshot is missing identity-critical assets")
    quantization = snapshot_config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise RuntimeError("Pinned model snapshot is missing quantization metadata")
    return {
        "snapshot_path": str(snapshot_path.resolve()),
        "asset_hashes": dict(sorted(assets.items())),
        "tokenizer_assets_hash": sha256_json(dict(sorted(assets.items()))),
        "quantization": dict(quantization),
        "quantization_hash": sha256_json(dict(quantization)),
    }


def _verify_template_policy(tokenizer: NonThinkingTokenizer, profile: ModelProfile) -> str:
    messages = [
        {"role": "system", "content": "System policy."},
        {"role": "user", "content": "Question?"},
    ]
    disabled = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    enabled = tokenizer._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    if disabled.count("<think>") > disabled.count("</think>"):
        raise RuntimeError("Non-thinking template leaves an open thinking channel")
    if profile.model_id == QWEN_27B_MODEL_ID and disabled == enabled:
        raise RuntimeError("enable_thinking=False did not affect Qwen3.8 template")
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise RuntimeError("Pinned tokenizer exposes no chat template")
    return str(template)


def _layer_and_relative_path(path: str) -> tuple[int | None, str | None]:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", path)
    if match is None:
        return None, None
    relative = match.group(2)
    if relative.endswith(".linear"):
        relative = relative.removesuffix(".linear")
    return int(match.group(1)), relative


def _set_memory_limit(mx: Any) -> None:
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal device is unavailable")
    limit = MODEL_UPGRADE_CONFIG.application_memory_limit_gib * 2**30
    mx.set_wired_limit(limit)


def _memory_snapshot(mx: Any) -> dict[str, Any]:
    return {
        "active_gb": mx.get_active_memory() / 1e9,
        "cache_gb": mx.get_cache_memory() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "device": dict(mx.device_info()),
    }


def _swap_snapshot() -> dict[str, float] | None:
    result = subprocess.run(
        ("sysctl", "-n", "vm.swapusage"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    values = {
        key.casefold(): float(value)
        for key, value in re.findall(
            r"(total|used|free)\s*=\s*([0-9.]+)M",
            result.stdout,
        )
    }
    return values or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "train":
        run_training(args.config, args.coverage)
        return 0
    if args.report is None:
        raise ValueError("--report is required for preflight")
    run_preflight(args.config, args.coverage, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
