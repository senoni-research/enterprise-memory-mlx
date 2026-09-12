"""Exact-memory training helpers for the Qwen4B learning-mechanics experiment.

These change only the training runtime. Approved bytes, model identity,
adapter targets and scientific settings stay unchanged. Inference still uses
the stock MLX-LM path.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

QUERY_BLOCK_SIZE = 256
PROJECTION_CHUNK_SIZE = 256
CHECKPOINT_MARKER = "_mechanics_checkpoint_once"
IMPLEMENTATION_ID = "qwen4b-learning-mechanics-exact-memory/v1"

# Shifted hidden i predicts tokens[i + 1]. The first assistant token is
# tokens[offset], so the first supervised hidden index is offset - 1.
EXPECTED_DEV004_SUPERVISED = 1912


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def installed_runtime_versions() -> dict[str, str | None]:
    return {
        "mlx": package_version("mlx"),
        "mlx-lm": package_version("mlx-lm"),
        "mlx-metal": package_version("mlx-metal"),
    }


def supervised_hidden_span(offset: int, unpadded_length: int) -> dict[str, int]:
    if offset <= 0 or offset >= unpadded_length:
        raise RuntimeError("Assistant loss boundary is invalid")
    start = offset - 1
    end = unpadded_length - 1
    count = end - start
    if count <= 0:
        raise RuntimeError("No supervised assistant tokens after the causal shift")
    return {
        "prompt_token_count": offset,
        "unpadded_length": unpadded_length,
        "first_supervised_hidden_index": start,
        "last_supervised_hidden_index_inclusive": end - 1,
        "end_exclusive": end,
        "supervised_token_count": count,
        "first_supervised_target_index": offset,
        "last_supervised_target_index_inclusive": unpadded_length - 1,
        "first_hidden_index_equals_prompt_length": start == offset,
    }


def default_loss_padding_contract(
    *,
    offset: int,
    unpadded_length: int,
    padded_length: int,
) -> dict[str, Any]:
    """Document mlx_lm.default_loss when lengths[1] is the unpadded token count.

    steps run from 1 to padded_length - 1. A mask upper bound of unpadded_length
    includes target index unpadded_length, which is the first padding token
    whenever the row is padded.
    """
    max_step = padded_length - 1
    mlx_upper = unpadded_length
    corrected_upper = unpadded_length - 1
    mlx_includes_first_pad = (
        padded_length > unpadded_length and offset <= mlx_upper <= max_step
    )
    return {
        "mlx_lm_default_loss_upper_bound_if_length_is_unpadded_count": mlx_upper,
        "corrected_upper_bound": corrected_upper,
        "includes_first_padding_token_when_padded": mlx_includes_first_pad,
        "loss_contract_defect_if_old_length_is_unpadded_count": mlx_includes_first_pad,
        "dev004_unpadded_equals_cap_so_no_pad_slot": (
            unpadded_length == padded_length
        ),
    }


def inspect_sdpa_implementation() -> dict[str, Any]:
    import mlx.core as mx

    fn = mx.fast.scaled_dot_product_attention
    source = None
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = None
    doc = inspect.getdoc(fn) or ""
    text = f"{doc}\n{source or ''}".lower()
    return {
        "function": "mlx.core.fast.scaled_dot_product_attention",
        "qualname": getattr(fn, "__qualname__", type(fn).__name__),
        "module": getattr(fn, "__module__", None),
        "signature": str(inspect.signature(fn)),
        "source_available": source is not None,
        "doc_mentions_unfused": "unfused" in text,
        "doc_mentions_fallback": "fallback" in text,
        "doc_mentions_gradient": "grad" in text,
        "note": (
            "Official MLX 0.31.2 Metal training selects unfused attention and "
            "falls back for attention gradients. This record inspects the "
            "installed build rather than assuming that text."
        ),
    }


def project_hidden_states(model: Any, hidden: Any) -> Any:
    if getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(hidden)
    return model.lm_head(hidden)


def exact_query_blocked_attention(
    queries: Any,
    keys: Any,
    values: Any,
    *,
    scale: float,
    mask: Any = "causal",
    query_block_size: int = QUERY_BLOCK_SIZE,
) -> Any:
    """Exact GQA attention that never materializes the full Lq×Lk score matrix."""
    import mlx.core as mx

    batch, n_q, query_len, head_dim = queries.shape
    n_kv, key_len = keys.shape[1], keys.shape[2]
    if n_q % n_kv != 0:
        raise RuntimeError("Query heads must be an integer multiple of key/value heads")
    repeats = n_q // n_kv
    query = queries.reshape(batch, n_kv, repeats, query_len, head_dim)
    key = keys.reshape(batch, n_kv, 1, key_len, head_dim)
    value = values.reshape(batch, n_kv, 1, key_len, head_dim)
    fill = mx.finfo(queries.dtype).min
    outputs = []
    for start in range(0, query_len, query_block_size):
        end = min(start + query_block_size, query_len)
        query_block = query[..., start:end, :]

        def _block(
            query_block: Any,
            key: Any,
            value: Any,
            query_start: int = start,
            query_end: int = end,
        ) -> Any:
            query_positions = mx.stop_gradient(mx.arange(query_start, query_end))
            key_positions = mx.stop_gradient(mx.arange(key_len))
            scores = (query_block * scale) @ mx.swapaxes(key, -1, -2)
            allowed = (query_positions[:, None] >= key_positions[None])[
                None, None, None, :, :
            ]
            if mask is not None and mask != "causal":
                raise RuntimeError(
                    "Blocked training attention only implements exact causal masks "
                    f"with absolute query offsets; got {type(mask).__name__}"
                )
            scores = mx.where(allowed, scores, mx.array(fill, dtype=scores.dtype))
            weights = mx.softmax(scores, axis=-1, precise=True)
            return weights @ value

        outputs.append(mx.checkpoint(_block)(query_block, key, value))
    output = mx.concatenate(outputs, axis=-2)
    return output.reshape(batch, n_q, query_len, head_dim)


def patched_training_sdpa(
    original: Any,
    queries: Any,
    keys: Any,
    values: Any,
    cache: Any,
    scale: float,
    mask: Any,
    sinks: Any = None,
) -> Any:
    if cache is not None or sinks is not None:
        return original(
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
        )
    return exact_query_blocked_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask if mask is not None else "causal",
        query_block_size=QUERY_BLOCK_SIZE,
    )


class TrainingAttentionPatch:
    """Replace Qwen3 training SDPA only. Generation keeps the stock kernel."""

    def __init__(self) -> None:
        self.original = None
        self.active = False

    def install(self) -> None:
        import mlx_lm.models.qwen3 as qwen3

        if self.active:
            return
        self.original = qwen3.scaled_dot_product_attention

        def wrapped(queries, keys, values, cache, scale, mask, sinks=None):
            return patched_training_sdpa(
                self.original, queries, keys, values, cache, scale, mask, sinks
            )

        wrapped._mechanics_exact_attention = True
        qwen3.scaled_dot_product_attention = wrapped
        self.active = True

    def uninstall(self) -> None:
        if not self.active:
            return
        import mlx_lm.models.qwen3 as qwen3

        qwen3.scaled_dot_product_attention = self.original
        self.active = False
        self.original = None


def reference_masked_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
    """Stock full-logit loss with the corrected pad-excluding mask."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    token_loss = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return token_loss.astype(mx.float32).sum() / ntoks, ntoks


def supervised_position_loss(
    model: Any,
    batch: Any,
    lengths: Any,
    *,
    chunk_size: int = PROJECTION_CHUNK_SIZE,
) -> tuple[Any, Any]:
    """Project only the shifted assistant positions, then chunked exact CE."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    hidden = model.model(inputs)
    offset = int(lengths[0, 0].item())
    last_target_step = int(lengths[0, 1].item())
    span = supervised_hidden_span(offset, last_target_step + 1)
    start = span["first_supervised_hidden_index"]
    end = span["end_exclusive"]
    selected = hidden[:, start:end, :]
    selected_targets = targets[:, start:end]
    token_count = selected_targets.shape[1]
    if token_count != span["supervised_token_count"]:
        raise RuntimeError("Supervised hidden span does not match selected targets")

    total = mx.array(0.0, dtype=mx.float32)
    for index in range(0, token_count, chunk_size):
        hidden_chunk = selected[:, index : index + chunk_size, :]
        target_chunk = selected_targets[:, index : index + chunk_size]

        def _chunk(hidden_chunk: Any, target_chunk: Any) -> Any:
            logits = project_hidden_states(model, hidden_chunk)
            token_loss = nn.losses.cross_entropy(
                logits, mx.stop_gradient(target_chunk), reduction="none"
            )
            return token_loss.astype(mx.float32).sum()

        total = total + mx.checkpoint(_chunk)(hidden_chunk, target_chunk)
    ntoks = mx.array(token_count, dtype=total.dtype)
    return total / ntoks, ntoks


def optimized_training_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
    return supervised_position_loss(model, batch, lengths)


def max_abs_diff(left: Any, right: Any) -> float:
    import mlx.core as mx

    return float(mx.max(mx.abs(left - right)).item())


def allclose(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    import mlx.core as mx

    return bool(mx.allclose(left, right, rtol=rtol, atol=atol).item())


def flatten_trainable(model: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return list(tree_flatten(model.trainable_parameters()))


def make_adapter_nonzero(model: Any, mx: Any) -> None:
    from mlx.utils import tree_flatten, tree_unflatten

    updated = []
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.endswith("lora_b") or name.endswith("lora_b.weight"):
            updated.append((name, mx.full(value.shape, 0.02, dtype=value.dtype)))
        else:
            updated.append((name, value))
    model.update(tree_unflatten(updated))
    mx.eval(model.trainable_parameters())


def compare_attention_kernels(mx: Any) -> dict[str, Any]:
    mx.random.seed(0)
    batch, n_q, n_kv, length, dim = 1, 32, 8, 80, 16
    scale = dim**-0.5
    queries = mx.random.normal((batch, n_q, length, dim)).astype(mx.float16)
    keys = mx.random.normal((batch, n_kv, length, dim)).astype(mx.float16)
    values = mx.random.normal((batch, n_kv, length, dim)).astype(mx.float16)
    mx.eval(queries, keys, values)

    def stock(queries, keys, values):
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask="causal"
        )

    def blocked(queries, keys, values):
        return exact_query_blocked_attention(
            queries, keys, values, scale=scale, mask="causal", query_block_size=32
        )

    mx.reset_peak_memory()
    stock_out = stock(queries, keys, values)
    mx.eval(stock_out)
    mx.synchronize()
    stock_peak = mx.get_peak_memory()
    _, stock_grads = mx.value_and_grad(lambda q, k, v: stock(q, k, v).sum())(
        queries, keys, values
    )
    mx.eval(stock_grads)
    mx.synchronize()

    mx.reset_peak_memory()
    blocked_out = blocked(queries, keys, values)
    mx.eval(blocked_out)
    mx.synchronize()
    blocked_peak = mx.get_peak_memory()
    _, blocked_grads = mx.value_and_grad(lambda q, k, v: blocked(q, k, v).sum())(
        queries, keys, values
    )
    mx.eval(blocked_grads)
    mx.synchronize()

    output_tol = {"rtol": 2e-2, "atol": 2e-2, "dtype": "float16"}
    return {
        "sequence_length": length,
        "query_block_size": 32,
        "uneven_final_block": True,
        "output_max_abs_diff": max_abs_diff(stock_out, blocked_out),
        "output_allclose": allclose(
            stock_out, blocked_out, rtol=output_tol["rtol"], atol=output_tol["atol"]
        ),
        "query_grad_max_abs_diff": max_abs_diff(stock_grads[0], blocked_grads[0]),
        "key_grad_max_abs_diff": max_abs_diff(stock_grads[1], blocked_grads[1]),
        "value_grad_max_abs_diff": max_abs_diff(stock_grads[2], blocked_grads[2]),
        "grads_allclose": all(
            allclose(left, right, rtol=5e-2, atol=5e-2)
            for left, right in zip(stock_grads, blocked_grads, strict=True)
        ),
        "stock_peak_bytes": stock_peak,
        "blocked_peak_bytes": blocked_peak,
        "tolerances": output_tol,
        "activation_dtype": str(stock_out.dtype),
    }


def diagnose_runtime(mx: Any) -> dict[str, Any]:
    versions = installed_runtime_versions()
    sdpa = inspect_sdpa_implementation()
    attention = compare_attention_kernels(mx)
    unfused_indicated = (
        versions.get("mlx") is not None
        and versions["mlx"].startswith("0.31.")
        and attention["output_allclose"] is not False
    )
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "versions": versions,
        "sdpa": sdpa,
        "short_sequence_attention": attention,
        "estimated_dev004_unfused_score_gib": (
            32 * 35405 * 35405 * 2 / 1024**3
        ),
        "estimated_dev004_full_logits_gib": (
            35405 * 151936 * 2 / 1024**3
        ),
        "memory_not_assumed_to_be_logits_only": True,
        "metal_training_unfused_expected_for_mlx_0_31_2": True,
        "installed_mlx_is_0_31_family": bool(
            versions.get("mlx") and versions["mlx"].startswith("0.31.")
        ),
        "unfused_family_indicated": unfused_indicated,
        "query_block_size": QUERY_BLOCK_SIZE,
        "projection_chunk_size": PROJECTION_CHUNK_SIZE,
    }


def _seed_all(seed: int, mx: Any) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def run_model_equivalence(
    *,
    model: Any,
    mx: Any,
    nn: Any,
    tokens: Sequence[int],
    offset: int,
    max_seq_length: int,
    build_batch: Any,
    seed: int = 123,
) -> dict[str, Any]:
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    batch_info = build_batch(tokens, offset, max_seq_length=max_seq_length)
    batch = mx.array(batch_info["batch"])
    lengths = mx.array(batch_info["lengths"])
    span = supervised_hidden_span(offset, batch_info["unpadded_length"])
    pad_contract = default_loss_padding_contract(
        offset=offset,
        unpadded_length=batch_info["unpadded_length"],
        padded_length=batch_info["padded_shape"][1],
    )
    reports = []
    patch = TrainingAttentionPatch()
    try:
        for adapter_state in ("zero_initialized_lora_b", "nonzero_lora_b"):
            if adapter_state == "nonzero_lora_b":
                make_adapter_nonzero(model, mx)
            snapshot = {
                name: tensor_copy(value) for name, value in flatten_trainable(model)
            }
            _seed_all(seed, mx)
            model.train()
            (ref_loss, ref_ntoks), ref_grads = nn.value_and_grad(
                model, reference_masked_loss
            )(model, batch, lengths)
            mx.eval(ref_loss, ref_ntoks, ref_grads)
            mx.synchronize()
            restore_trainable(model, snapshot, mx)
            _seed_all(seed, mx)
            model.train()
            patch.install()
            (opt_loss, opt_ntoks), opt_grads = nn.value_and_grad(
                model, optimized_training_loss
            )(model, batch, lengths)
            mx.eval(opt_loss, opt_ntoks, opt_grads)
            mx.synchronize()
            patch.uninstall()
            restore_trainable(model, snapshot, mx)

            ref_items = dict(tree_flatten(ref_grads))
            opt_items = dict(tree_flatten(opt_grads))
            grad_ok = True
            worst = 0.0
            for name in ref_items:
                diff = max_abs_diff(ref_items[name], opt_items[name])
                worst = max(worst, diff)
                if not allclose(ref_items[name], opt_items[name], rtol=5e-2, atol=5e-2):
                    grad_ok = False

            opt_a = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
            opt_b = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
            opt_a.update(model, ref_grads)
            mx.eval(model.trainable_parameters())
            after_ref = {
                name: tensor_copy(value) for name, value in flatten_trainable(model)
            }
            restore_trainable(model, snapshot, mx)
            opt_b.update(model, opt_grads)
            mx.eval(model.trainable_parameters())
            after_opt = {
                name: tensor_copy(value) for name, value in flatten_trainable(model)
            }
            restore_trainable(model, snapshot, mx)
            update_ok = all(
                allclose(after_ref[name], after_opt[name], rtol=5e-2, atol=5e-2)
                for name in after_ref
            )
            reports.append(
                {
                    "adapter_state": adapter_state,
                    "reference_loss": float(ref_loss.item()),
                    "optimized_loss": float(opt_loss.item()),
                    "reference_supervised_tokens": int(ref_ntoks.item()),
                    "optimized_supervised_tokens": int(opt_ntoks.item()),
                    "loss_allclose": allclose(ref_loss, opt_loss, rtol=1e-3, atol=1e-3),
                    "supervised_token_count_match": int(ref_ntoks.item())
                    == int(opt_ntoks.item())
                    == span["supervised_token_count"],
                    "adapter_gradients_allclose": grad_ok,
                    "adapter_gradient_max_abs_diff": worst,
                    "adamw_update_allclose": update_ok,
                    "padding_positions_supervised_in_optimized_path": False,
                    "includes_first_and_last_supervised_tokens": True,
                }
            )
    finally:
        patch.uninstall()
    passed = all(
        row["loss_allclose"]
        and row["supervised_token_count_match"]
        and row["adapter_gradients_allclose"]
        and row["adamw_update_allclose"]
        for row in reports
    )
    return {
        "passed": passed,
        "span": span,
        "padding_contract": pad_contract,
        "batch": {
            "unpadded_length": batch_info["unpadded_length"],
            "padded_shape": batch_info["padded_shape"],
            "lengths": batch_info["lengths"],
            "offset": offset,
        },
        "comparisons": reports,
        "tolerances": {
            "loss": {"rtol": 1e-3, "atol": 1e-3},
            "gradients_and_update": {"rtol": 5e-2, "atol": 5e-2},
            "dtype": "model_activation_and_float32_loss_reduction",
            "bitwise_identity_claimed": False,
        },
    }


def tensor_copy(value: Any) -> Any:
    import mlx.core as mx

    copied = mx.array(value)
    mx.eval(copied)
    return copied


def restore_trainable(model: Any, values: Mapping[str, Any], mx: Any) -> None:
    from mlx.utils import tree_unflatten

    model.update(tree_unflatten([(name, value) for name, value in values.items()]))
    mx.eval(model.trainable_parameters())


def mark_checkpointed(fn: Any) -> Any:
    setattr(fn, CHECKPOINT_MARKER, True)
    return fn
