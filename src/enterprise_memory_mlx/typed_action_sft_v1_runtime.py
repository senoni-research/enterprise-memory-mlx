"""Metal runtime for qwen4b-typed-action-sft-v1: preflight, train, evaluate.

Run stages in fresh processes. Each stage re-verifies frozen digests and the
resource policy before touching the model. No fallback configuration.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .grpo_data import file_sha256
from .learning_mechanics import (
    seed_all,
    tensor_digest,
)
from .learning_mechanics_eval import (
    memory_pressure_snapshot,
    process_current_rss_bytes,
    system_swap_snapshot,
)
from .utils import sha256_json, sha256_text

RUN_MANIFEST_NAME = "run-manifest.json"
WIRED_LIMIT_BYTES = 96 * 1024**3
RSS_HEADROOM_BYTES = 8 * 1024**3
SWAP_GROWTH_STOP_BYTES = 8 * 1024**3
EXPECTED_TOTAL_CALLS = 40
ANNOTATION_V2_TARGETS_SHA256 = "5cdf5629b5648b6581f44ff38e2b64c5388c83230dfb097322a5b79797295467"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resource_snapshot(mx: Any) -> dict[str, Any]:
    snapshot = {
        "mlx_active_gb": round(mx.get_active_memory() / 1e9, 3),
        "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 3),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        "process_rss_gb": (
            round(process_current_rss_bytes() / 1e9, 3)
            if process_current_rss_bytes() is not None
            else None
        ),
        "swap": system_swap_snapshot(),
        "pressure": memory_pressure_snapshot(),
        "captured_at": utc_now(),
    }
    return snapshot


def check_resource_policy(snapshot: Mapping[str, Any], *, baseline_swap_used_mb: float) -> None:
    rss = snapshot.get("process_rss_gb")
    if rss is not None and rss * 1e9 > (128 * 1e9 - RSS_HEADROOM_BYTES):
        raise RuntimeError(f"RSS stop breached: {rss} GiB")
    swap = snapshot.get("swap") or {}
    parsed = swap.get("parsed_M") or {}
    used = parsed.get("used")
    if used is not None and (used - baseline_swap_used_mb) * 1e6 > SWAP_GROWTH_STOP_BYTES:
        raise RuntimeError(f"Swap growth stop breached: {used} MB vs {baseline_swap_used_mb} MB")


def verify_frozen_digests(root: Path) -> dict[str, bool]:
    from .typed_action_sft_v1 import ANNOTATION_V2_DIR

    checks = {
        "grpo_reward.py": file_sha256(root / "src/enterprise_memory_mlx/grpo_reward.py")
        == "e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b",
        "annotation_v1_review_zip": file_sha256(
            root / "artifacts/qwen4b-typed-action-annotation-v1"
            "/qwen4b-typed-action-annotation-v1-review.zip"
        )
        == "8a63b877baa1f931793e3687999855f528fc7de48ca1f47ea7700d641766cb70",
        "annotation_v2_targets": file_sha256(
            root / ANNOTATION_V2_DIR / "typed-targets.draft.jsonl"
        )
        == ANNOTATION_V2_TARGETS_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen digest check failed: {checks}")
    return checks


def load_snapshot_path() -> Path:
    from .experiment_profiles import QWEN_4B_REVISION as REVISION

    return (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
        / "snapshots"
        / REVISION
    )


def build_training_rows(root: Path, tokenizer: Any) -> list[dict[str, Any]]:
    """Verified TRAIN export rows: prompt IDs, continuation IDs, hashes."""
    from .typed_action_sft_v1 import (
        build_system_prompt,
        build_user_prompt,
        load_train_targets,
        serialize_target,
    )
    from .utils import read_jsonl

    sp = build_system_prompt()
    v3 = read_jsonl(
        root / "knowledge/company_task_specialization/v3_remedy_logic/qualification_cases.jsonl"
    )
    v4 = read_jsonl(
        root / "knowledge/company_task_specialization/v4_fact_state/contrast_cases.jsonl"
    )
    cases_by_id = {c["case_id"]: c for c in v3}
    cases_by_id.update({c["case_id"]: c for c in v4})
    rows = []
    for row in load_train_targets(root):
        case = cases_by_id[row["case_id"]]
        user = build_user_prompt(root, case)
        target_text = serialize_target(row["proposed_training_target"])
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sp},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        continuation = list(target_ids) + [tokenizer.eos_token_id]
        rows.append(
            {
                "case_id": row["case_id"],
                "prompt_ids": prompt_ids,
                "continuation_ids": continuation,
                "prompt_len": len(prompt_ids),
                "continuation_len": len(continuation),
                "total_len": len(prompt_ids) + len(continuation),
                "semantic_object_sha256": sha256_json(row["proposed_training_target"]),
                "serialized_target_sha256": sha256_text(target_text),
                "prompt_sha256": sha256_text(rendered),
            }
        )
    return rows


def make_training_batch(tokens: Sequence[int], offset: int, pad_to: int = 32) -> dict[str, Any]:
    """Batch of one; pad to a multiple of pad_to but never beyond the row."""
    n = len(tokens)
    padded_length = 1 + pad_to * ((n + pad_to - 1) // pad_to)
    if padded_length < n:
        padded_length = n
    padded = list(tokens) + [0] * max(0, padded_length - n)
    return {
        "batch": [padded],
        "lengths": [[offset, n - 1]],
        "unpadded_length": n,
        "padded_shape": [1, len(padded)],
        "padding_positions": len(padded) - n,
    }


def sft_step(
    *,
    model: Any,
    optimizer: Any,
    mx: Any,
    nn: Any,
    row: Mapping[str, Any],
    update_index: int,
) -> dict[str, Any]:
    """One measured SFT update: default_loss over the supervised span only."""
    from mlx.utils import tree_flatten

    tokens = list(row["prompt_ids"]) + list(row["continuation_ids"])
    batch_info = make_training_batch(tokens, offset=row["prompt_len"])
    batch = mx.array(batch_info["batch"])
    lengths = mx.array(batch_info["lengths"])
    model.train()
    loss_and_grad = nn.value_and_grad(model, _default_loss)
    (loss, ntoks), gradients = loss_and_grad(model, batch, lengths)
    mx.eval(loss, ntoks, gradients)
    loss_value = float(loss.item())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"Non-finite loss on update {update_index}")
    gradient_values = [value for _name, value in tree_flatten(gradients)]
    if not gradient_values or not all(
        bool(mx.all(mx.isfinite(value)).item()) for value in gradient_values
    ):
        raise RuntimeError(f"Non-finite gradients on update {update_index}")
    if not any(bool(mx.any(value != 0).item()) for value in gradient_values):
        raise RuntimeError(f"All-zero gradients on update {update_index}")
    before = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    after = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    changed = sum(1 for name in before if before[name] != after[name])
    if changed == 0:
        raise RuntimeError(f"Optimizer update {update_index} changed no adapter tensors")
    supervised = int(ntoks.item())
    expected = row["continuation_len"]
    if supervised != expected:
        raise RuntimeError(
            f"Supervised token count {supervised} != continuation {expected} "
            f"on update {update_index}"
        )
    return {
        "update_index": update_index,
        "case_id": row["case_id"],
        "loss": loss_value,
        "supervised_tokens": supervised,
        "padded_shape": batch_info["padded_shape"],
        "padding_positions": batch_info["padding_positions"],
        "unpadded_length": batch_info["unpadded_length"],
        "changed_adapter_tensors": changed,
    }


def _default_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
    from mlx_lm.tuner.trainer import default_loss

    return default_loss(model, batch, lengths)


def apply_lora(model: Any) -> dict[str, Any]:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(
        model,
        36,
        {
            "rank": 16,
            "scale": 2.0,
            "dropout": 0.0,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
        },
    )
    return {"rank": 16, "scale": 2.0, "dropout": 0.0, "num_layers": 36}


def frozen_base_unchanged(model: Any, mx: Any, expected_digests: Mapping[str, str]) -> None:
    """The frozen base tensors must not change across updates."""
    from mlx.utils import tree_flatten

    digests = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.parameters())
        if ".lora_" not in name
    }
    drifted = [name for name in expected_digests if expected_digests[name] != digests.get(name)]
    if drifted:
        raise RuntimeError(f"Frozen base tensors changed: {drifted[:3]}")


def load_model_with_lora(seed: int, mx: Any) -> tuple[Any, Any, dict[str, Any]]:
    from mlx_lm import load

    seed_all(seed, mx)
    model, tokenizer = load(
        str(load_snapshot_path()),
        tokenizer_config={"trust_remote_code": False},
    )
    adapter_info = apply_lora(model)
    return model, tokenizer, adapter_info


def save_adapter(model: Any, mx: Any, path: Path) -> dict[str, str]:
    from mlx.utils import tree_flatten

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite adapter: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)
    return {name: tensor_digest(value) for name, value in weights.items()}


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    mx: Any,
    seed: int,
    prompt_ids: Sequence[int],
    grammar: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Greedy constrained-or-free generation with fresh cache."""
    import mlx.core as _mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    seed_all(seed, mx)
    processors = []
    if grammar is not None:
        from llguidance import LLMatcher

        from .structured_inference import (
            LLGuidanceJSONLogitsProcessor,
            llguidance_tokenizer_from_hf,
            unwrap_hf_tokenizer,
        )

        compiled = grammar
        hf_tok = unwrap_hf_tokenizer(tokenizer)
        ll_tok = llguidance_tokenizer_from_hf(hf_tok)
        matcher = LLMatcher(ll_tok, compiled)
        processors.append(LLGuidanceJSONLogitsProcessor(matcher, ll_tok))
    sampler = make_sampler(temp=0.0, top_p=0.0)
    prompt_cache = make_prompt_cache(model)
    generated_ids: list[int] = []
    finish_reason = "unknown"
    started = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=list(prompt_ids),
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors or None,
            prompt_cache=prompt_cache,
        ):
            generated_ids.append(int(response.token))
            finish_reason = response.finish_reason
    finally:
        prompt_cache = None
        _mx.clear_cache()
    elapsed = time.perf_counter() - started
    eos_ids = set(
        getattr(tokenizer, "eos_token_ids", None) or [getattr(tokenizer, "eos_token_id", None)]
    )
    content_ids = [token for token in generated_ids if token not in eos_ids]
    output = tokenizer.decode(content_ids)
    return {
        "output": output,
        "completion_token_ids": generated_ids,
        "prompt_token_count": len(prompt_ids),
        "completion_length": len(generated_ids),
        "elapsed_seconds": round(elapsed, 6),
        "truncated": finish_reason == "length" or len(generated_ids) >= max_tokens,
        "finish_reason": finish_reason,
        "seed": seed,
    }


def compile_runtime_grammar(tokenizer: Any) -> str:
    from .structured_inference import (
        compile_review_grammar,
    )
    from .typed_action_sft_v1 import runtime_output_schema

    compile_review_grammar(runtime_output_schema())
    del tokenizer
    return "compiled"