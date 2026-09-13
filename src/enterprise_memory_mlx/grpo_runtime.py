"""Isolated GRPO/RFT runtime. Does not load the five-example SFT adapter."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition_runtime import NonThinkingTokenizer
from .acquisition_training import materialize_model_snapshot
from .experiment_profiles import QWEN_4B_MODEL_ID, QWEN_4B_PROFILE, QWEN_4B_REVISION
from .grpo_eval import case_json_schema, compare_arms, write_review_packet
from .grpo_protocol import (
    PROTOCOL_ID,
    load_frozen_protocol,
    render_case_prompt,
    rows_for_split,
    spec_by_case,
    write_frozen_protocol,
)
from .grpo_reward import score_completion
from .learning_mechanics import (
    WiredLimitGuard,
    reload_adapter_check,
    save_adapter,
    seed_all,
    sha256_file,
    tensor_digest,
)
from .learning_mechanics_eval import (
    mlx_memory_snapshot,
    process_current_rss_bytes,
    process_rss_bytes,
    system_swap_snapshot,
)
from .utils import atomic_write_text, sha256_text

ARTIFACTS = Path("artifacts/qwen4b-grpo-v1")
FORBIDDEN_ADAPTER = "7a5cd28ed8f469a98e82f285bce34ed83801d1ee71f01b69c9a9b20f984a67ba"
WIRED_LIMIT_BYTES = 96 * 1024**3
RSS_STOP_BYTES = 112 * 1024**3
SWAP_GROWTH_STOP_BYTES = 8 * 1024**3
ADVANTAGE_EPSILON = 1e-4
CLIP_EPSILON = 1e-4
KL_BETA = 0.1


def derive_seed(master: int, epoch: int, case_id: str, member: int) -> int:
    digest = hashlib.sha256(f"{master}|{epoch}|{case_id}|{member}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def group_advantages(rewards: list[float], epsilon: float = ADVANTAGE_EPSILON) -> list[float]:
    """mlx-lm-lora GRPO advantages: (r - mean) / (std + 1e-4). Zero-variance => ~0."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    std = variance**0.5
    return [(item - mean) / (std + epsilon) for item in rewards]


def group_variance(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    mean = sum(rewards) / len(rewards)
    return sum((item - mean) ** 2 for item in rewards) / len(rewards)


def validate_experiment(root: Path) -> dict[str, Any]:
    destination = root / "knowledge/company_task_specialization/qwen4b_grpo_v1"
    if not (destination / "manifest.json").is_file():
        write_frozen_protocol(root)
    frozen = load_frozen_protocol(root)
    train_ids = {row["case_id"] for row in frozen["split"]["rows"] if row["split"] == "train"}
    test_ids = {row["case_id"] for row in frozen["split"]["rows"] if row["split"] == "test"}
    if train_ids & test_ids:
        raise ValueError("TEST leaked into TRAIN")
    for row in frozen["split"]["rows"]:
        prompt = render_case_prompt(root, row["case_id"])
        hidden = json.dumps(
            spec_by_case(root)[row["case_id"]]["decisions"],
            sort_keys=True,
        )
        if hidden in prompt["question"]:
            raise ValueError(f"{row['case_id']} prompt contains hidden reward decisions")
    return {
        "status": "validated",
        "protocol_id": PROTOCOL_ID,
        "origin_main_sha": frozen["protocol"]["origin_main_sha"],
        "train": 24,
        "dev": 8,
        "test": 8,
        "algorithm": frozen["protocol"]["trainer"]["algorithm"],
        "trainer_commit": frozen["protocol"]["trainer"]["git_commit"],
    }


def run_preflight(root: Path) -> dict[str, Any]:
    load_frozen_protocol(root)
    train = rows_for_split(root, "train")
    families: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in train:
        if row["family_id"] in seen:
            continue
        seen.add(row["family_id"])
        families.append(row)
        if len(families) == 3:
            break
    if any(row["split"] != "train" for row in families):
        raise RuntimeError("preflight attempted to load a non-TRAIN case")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / ARTIFACTS / f"preflight-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _copy_freeze(root, output)
    specs = spec_by_case(root)
    runtime = _boot_policy(root, require_fresh_lora=True)
    swap_before = system_swap_snapshot()
    stage_a = []
    try:
        with WiredLimitGuard(runtime["mx"], WIRED_LIMIT_BYTES):
            for row in families:
                group = _rollout_group(
                    runtime,
                    root=root,
                    row=row,
                    spec=specs[row["case_id"]],
                    epoch=0,
                    persist_dir=output / "rollouts",
                    temperature=0.8,
                    top_p=0.95,
                    grammar=None,
                )
                rewards = [item["score"]["reward"] for item in group]
                stage_a.append(
                    {
                        "case_id": row["case_id"],
                        "family_id": row["family_id"],
                        "rewards": rewards,
                        "variance": group_variance(rewards),
                        "completions": group,
                    }
                )
                _assert_resource_ok(swap_before)
        nonzero = sum(1 for item in stage_a if item["variance"] > 0)
        valid_any = any(
            any(comp["score"]["complete_valid_output"] for comp in item["completions"])
            for item in stage_a
        )
        truncated = any(
            any(comp.get("truncated") for comp in item["completions"]) for item in stage_a
        )
        if truncated:
            report = _preflight_stop(output, stage_a, "generation truncated during preflight")
            return report
        if not valid_any:
            report = _preflight_stop(output, stage_a, "no valid structured outputs in preflight")
            return report
        if nonzero < 2:
            report = _preflight_stop(
                output,
                stage_a,
                "fewer than 2 of 3 groups had non-zero reward variance",
            )
            return report
        stage_b = _disposable_update(runtime, root, families[0], stage_a[0]["completions"], output)
        status = (
            "preflight_passed" if stage_b.get("changed_lora_tensors", 0) >= 1 else "stop_preflight"
        )
        report = {
            "status": status,
            "stage_a": _public_stage(stage_a),
            "stage_b": stage_b,
            "nonzero_variance_groups": nonzero,
            "directory": str(output),
            "memory": mlx_memory_snapshot(runtime["mx"]),
            "rss_bytes": process_rss_bytes(),
            "swap": system_swap_snapshot(),
        }
        if status != "preflight_passed":
            report["reason"] = (
                stage_b.get("reason") or "one-update disposable step changed no LoRA tensor"
            )
        _write_preflight(root, output, report)
        return report
    except Exception as exc:
        report = {
            "status": "stop_preflight",
            "reason": f"{type(exc).__name__}: {exc}",
            "stage": "setup_failure" if not stage_a else "preflight_runtime",
            "directory": str(output),
        }
        _write_preflight(root, output, report)
        raise
    finally:
        _release_runtime(runtime)


def run_execute(root: Path) -> dict[str, Any]:
    latest = root / ARTIFACTS / "latest-preflight.json"
    if not latest.is_file():
        raise RuntimeError("execute refused: preflight has not passed")
    preflight = json.loads(latest.read_text(encoding="utf-8"))
    if preflight.get("status") != "preflight_passed":
        raise RuntimeError("execute refused: preflight did not pass")
    existing = sorted((root / ARTIFACTS).glob("run-*-seed42"))
    if existing:
        raise RuntimeError(f"execute refused: measured run already exists: {existing[0]}")
    frozen = load_frozen_protocol(root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / ARTIFACTS / f"run-{stamp}-seed42"
    output.mkdir(parents=True, exist_ok=False)
    _copy_freeze(root, output)
    train = rows_for_split(root, "train")
    if any(row["split"] != "train" for row in train):
        raise RuntimeError("execute attempted to load a non-TRAIN case")
    specs = spec_by_case(root)
    runtime = _boot_policy(root, require_fresh_lora=True)
    mx = runtime["mx"]
    import mlx.nn as nn
    from mlx.optimizers import AdamW
    from mlx.utils import tree_flatten

    optimizer = AdamW(learning_rate=5e-6, weight_decay=0.01)
    update_log: list[dict[str, Any]] = []
    rollout_log: list[dict[str, Any]] = []
    groups_done = 0
    rollouts = 0
    skipped_zero_var = 0
    swap_before = system_swap_snapshot()
    atomic_write_text(output / "environment.json", json.dumps(_environment(), indent=2) + "\n")
    atomic_write_text(
        output / "trainer-identity.json",
        json.dumps(frozen["protocol"]["trainer"], indent=2, sort_keys=True) + "\n",
    )
    try:
        with WiredLimitGuard(mx, WIRED_LIMIT_BYTES):
            for epoch in range(4):
                order = list(train)
                random.Random(42 + epoch).shuffle(order)
                for row in order:
                    group = _rollout_group(
                        runtime,
                        root=root,
                        row=row,
                        spec=specs[row["case_id"]],
                        epoch=epoch,
                        persist_dir=output / "rollouts",
                        temperature=0.8,
                        top_p=0.95,
                        grammar=None,
                    )
                    rollouts += len(group)
                    rewards = [item["score"]["reward"] for item in group]
                    advantages = group_advantages(rewards)
                    zero_var = group_variance(rewards) == 0.0
                    if zero_var:
                        skipped_zero_var += 1
                    prompt = render_case_prompt(root, row["case_id"])

                    def loss_fn(mdl: Any, _group=group, _adv=advantages, _prompt=prompt) -> Any:
                        return _grpo_loss(mdl, runtime["tokenizer"], _prompt, _group, _adv, mx)[0]

                    value_and_grad = nn.value_and_grad(runtime["model"], loss_fn)
                    loss_value, grads = value_and_grad(runtime["model"])
                    loss_float = (
                        float(loss_value.item())
                        if hasattr(loss_value, "item")
                        else float(loss_value)
                    )
                    if not math.isfinite(loss_float):
                        raise RuntimeError(f"nonfinite loss on {row['case_id']} epoch {epoch}")
                    grad_values = [value for _name, value in tree_flatten(grads)]
                    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in grad_values):
                        raise RuntimeError(f"nonfinite gradient on {row['case_id']} epoch {epoch}")
                    before = {
                        name: tensor_digest(value)
                        for name, value in tree_flatten(runtime["model"].trainable_parameters())
                    }
                    optimizer.update(runtime["model"], grads)
                    mx.eval(runtime["model"].parameters(), optimizer.state)
                    after = {
                        name: tensor_digest(value)
                        for name, value in tree_flatten(runtime["model"].trainable_parameters())
                    }
                    changed = sum(1 for name in before if before[name] != after[name])
                    groups_done += 1
                    for member, item in enumerate(group):
                        rollout_log.append(
                            {
                                "case_id": row["case_id"],
                                "epoch": epoch,
                                "member": member,
                                "reward": item["score"]["reward"],
                                "components": item["score"]["components"],
                                "raw_output_sha256": item["raw_output_sha256"],
                                "status": item["status"],
                            }
                        )
                    update_log.append(
                        {
                            "epoch": epoch,
                            "update": groups_done,
                            "case_id": row["case_id"],
                            "rewards": rewards,
                            "advantages": advantages,
                            "group_mean": sum(rewards) / len(rewards),
                            "group_std": group_variance(rewards) ** 0.5,
                            "zero_variance": zero_var,
                            "loss": loss_float,
                            "changed_adapter_tensors": changed,
                            "status": "optimizer_updated",
                            "memory": mlx_memory_snapshot(mx),
                            "rss_bytes": process_current_rss_bytes() or process_rss_bytes(),
                        }
                    )
                    atomic_write_text(
                        output / "update-log.jsonl",
                        "".join(json.dumps(row) + "\n" for row in update_log),
                    )
                    atomic_write_text(
                        output / "rollout-log.jsonl",
                        "".join(json.dumps(row) + "\n" for row in rollout_log),
                    )
                    _assert_resource_ok(swap_before)
        adapter_path = output / "adapter" / "adapters.safetensors"
        expected = save_adapter(runtime["model"], mx, adapter_path)
        adapter_sha = sha256_file(adapter_path)
        if adapter_sha == FORBIDDEN_ADAPTER:
            raise RuntimeError("refusing to save the existing five-example SFT adapter identity")
        reload_adapter_check(runtime["model"], mx, adapter_path, expected)
        trainable_names = [
            name for name, _value in _flatten(runtime["model"].trainable_parameters())
        ]
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "origin_main_sha": frozen["protocol"]["origin_main_sha"],
            "algorithm": "GRPO",
            "trainer": frozen["protocol"]["trainer"],
            "prompt_groups": groups_done,
            "rollouts": rollouts,
            "zero_variance_groups": skipped_zero_var,
            "adapter_sha256": adapter_sha,
            "trainable_tensor_count": len(trainable_names),
            "directory": str(output),
            "status": "completed",
        }
        atomic_write_text(output / "run-manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic_write_text(
            root / ARTIFACTS / "latest-run.json",
            json.dumps({"directory": str(output), "adapter_sha256": adapter_sha}, indent=2) + "\n",
        )
        return manifest
    except Exception as exc:
        atomic_write_text(
            output / "execute-failure.json",
            json.dumps(
                {
                    "status": "setup_failure" if groups_done == 0 and rollouts == 0 else "stopped",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "prompt_groups": groups_done,
                    "rollouts": rollouts,
                },
                indent=2,
            )
            + "\n",
        )
        raise
    finally:
        _release_runtime(runtime)


def run_evaluate(root: Path) -> dict[str, Any]:
    latest = json.loads((root / ARTIFACTS / "latest-run.json").read_text(encoding="utf-8"))
    run_dir = Path(latest["directory"])
    specs = spec_by_case(root)
    test_rows = rows_for_split(root, "test")
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    surfaces: dict[str, Any] = {}
    for surface, structured in (("primary_structured", True), ("secondary_unconstrained", False)):
        arm_rows: dict[str, list[dict[str, Any]]] = {"base": [], "rft": []}
        for arm, require_lora in (("base", False), ("rft", True)):
            runtime = _boot_policy(root, require_fresh_lora=require_lora)
            if require_lora:
                runtime["model"].load_weights(
                    str(run_dir / "adapter" / "adapters.safetensors"), strict=False
                )
                runtime["mx"].eval(runtime["model"].trainable_parameters())
            try:
                with WiredLimitGuard(runtime["mx"], WIRED_LIMIT_BYTES):
                    for row in test_rows:
                        grammar = (
                            json.dumps(case_json_schema(specs[row["case_id"]]))
                            if structured
                            else None
                        )
                        generated = _generate_one(
                            runtime,
                            prompt=render_case_prompt(root, row["case_id"]),
                            seed=42,
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=1024,
                            grammar=grammar,
                        )
                        generated["status"] = "evaluation_generated"
                        persist = eval_dir / f"{surface}-{arm}-{row['case_id']}.json"
                        atomic_write_text(persist, json.dumps(generated, indent=2) + "\n")
                        generated["score"] = score_completion(
                            generated["output"], specs[row["case_id"]]
                        )
                        generated["status"] = "evaluation_scored"
                        generated["case_id"] = row["case_id"]
                        generated["family_id"] = row["family_id"]
                        atomic_write_text(persist, json.dumps(generated, indent=2) + "\n")
                        arm_rows[arm].append(generated)
            finally:
                _release_runtime(runtime)
        surfaces[surface] = {
            "rows": arm_rows,
            "comparison": compare_arms(arm_rows["base"], arm_rows["rft"]),
        }
    comparison = {
        "primary_structured": surfaces["primary_structured"],
        "secondary_unconstrained": surfaces["secondary_unconstrained"],
        "continuation_status": surfaces["primary_structured"]["comparison"]["status"],
        "adapter_sha256": latest["adapter_sha256"],
        "model": {"id": QWEN_4B_MODEL_ID, "revision": QWEN_4B_REVISION},
    }
    atomic_write_text(
        eval_dir / "comparison.json", json.dumps(comparison, indent=2, default=str) + "\n"
    )
    report = write_measured_report(root, run_dir, comparison)
    atomic_write_text(eval_dir / "report.md", report)
    return comparison


def run_prepare_review(root: Path) -> dict[str, Any]:
    latest = json.loads((root / ARTIFACTS / "latest-run.json").read_text(encoding="utf-8"))
    run_dir = Path(latest["directory"])
    packet = write_review_packet(
        root,
        evaluation_dir=run_dir / "evaluation",
        output_dir=run_dir / "evaluation" / "review",
    )
    atomic_write_text(
        run_dir / "evaluation" / "semantic-review.json",
        json.dumps(packet, indent=2) + "\n",
    )
    return packet


def write_measured_report(root: Path, run_dir: Path, comparison: dict[str, Any]) -> str:
    frozen = load_frozen_protocol(root)
    primary = comparison["primary_structured"]["comparison"]
    secondary = comparison["secondary_unconstrained"]["comparison"]
    status = primary["status"]
    manifest = {}
    manifest_path = run_dir / "run-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [
        status,
        "",
        (
            "This experiment tests held-out machine-verifiable behavioral "
            "specialization on synthetic governed scenarios."
        ),
        "",
        f"Source origin/main SHA: {frozen['protocol']['origin_main_sha']}",
        f"Model: {QWEN_4B_MODEL_ID} @ {QWEN_4B_REVISION}",
        (
            "Algorithm: GRPO (mlx-lm-lora v3.0.0 "
            f"{frozen['protocol']['trainer']['git_commit']})"
        ),
        f"Adapter SHA-256: {comparison.get('adapter_sha256')}",
        f"Prompt groups: {manifest.get('prompt_groups')}",
        f"Rollouts: {manifest.get('rollouts')}",
        "",
        "PRIMARY structured decoder TEST",
        json.dumps(primary, indent=2, default=str),
        "",
        "SECONDARY unconstrained TEST",
        json.dumps(secondary, indent=2, default=str),
        "",
        "Continuation gate uses PRIMARY only.",
        "semantic_review: pending",
        "",
        (
            "This experiment does not test closed-book company knowledge, "
            "production deployment, human-level review, or SFT comparison."
        ),
    ]
    return "\n".join(lines) + "\n"


def _boot_policy(root: Path, *, require_fresh_lora: bool) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal device is unavailable")
    snapshot = materialize_model_snapshot(QWEN_4B_MODEL_ID, QWEN_4B_REVISION)
    seed_all(42, mx)
    model, tokenizer = load(snapshot, tokenizer_config={"trust_remote_code": False})
    tokenizer = NonThinkingTokenizer(tokenizer)
    model.freeze()
    if require_fresh_lora:
        linear_to_lora_layers(
            model,
            QWEN_4B_PROFILE.num_layers,
            {
                "rank": 16,
                "scale": 2.0,
                "dropout": 0.0,
                "keys": list(QWEN_4B_PROFILE.target_modules),
            },
        )
    return {"mx": mx, "model": model, "tokenizer": tokenizer}


def _generate_one(
    runtime: dict[str, Any],
    *,
    prompt: dict[str, str],
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    grammar: str | None,
) -> dict[str, Any]:
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    mx = runtime["mx"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["question"]},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<think>" in rendered:
        raise RuntimeError("chat template leaked a thinking scaffold")
    prompt_ids = list(tokenizer.encode(rendered))
    seed_all(seed, mx)
    processors = []
    if grammar is not None:
        from llguidance import LLMatcher

        from .structured_inference import (
            LLGuidanceJSONLogitsProcessor,
            compile_review_grammar,
            llguidance_tokenizer_from_hf,
            unwrap_hf_tokenizer,
        )

        schema = json.loads(grammar)
        compiled = compile_review_grammar(schema)
        hf_tok = unwrap_hf_tokenizer(tokenizer)
        ll_tok = llguidance_tokenizer_from_hf(hf_tok)
        matcher = LLMatcher(ll_tok, compiled)
        processors.append(LLGuidanceJSONLogitsProcessor(matcher, ll_tok))
    sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0)
    prompt_cache = make_prompt_cache(model)
    generated_ids: list[int] = []
    finish_reason = "unknown"
    started = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors or None,
            prompt_cache=prompt_cache,
        ):
            generated_ids.append(int(response.token))
            finish_reason = response.finish_reason
    finally:
        prompt_cache = None
        mx.clear_cache()
    elapsed = time.perf_counter() - started
    eos_ids = set(
        getattr(tokenizer, "eos_token_ids", None) or [getattr(tokenizer, "eos_token_id", None)]
    )
    content_ids = [token for token in generated_ids if token not in eos_ids]
    output = tokenizer.decode(content_ids)
    return {
        "output": output,
        "raw_output_sha256": sha256_text(output),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": generated_ids,
        "elapsed_seconds": round(elapsed, 6),
        "completion_length": len(generated_ids),
        "truncated": finish_reason == "length" or len(generated_ids) >= max_tokens,
        "finish_reason": finish_reason,
        "seed": seed,
        "status": "rollout_generated",
    }


def _rollout_group(
    runtime: dict[str, Any],
    *,
    root: Path,
    row: dict[str, Any],
    spec: dict[str, Any],
    epoch: int,
    persist_dir: Path,
    temperature: float,
    top_p: float,
    grammar: str | None,
) -> list[dict[str, Any]]:
    prompt = render_case_prompt(root, row["case_id"])
    group = []
    persist_dir.mkdir(parents=True, exist_ok=True)
    for member in range(4):
        seed = derive_seed(42, epoch, row["case_id"], member)
        generated = _generate_one(
            runtime,
            prompt=prompt,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024,
            grammar=grammar,
        )
        persist = persist_dir / f"{row['case_id']}-e{epoch}-m{member}.json"
        atomic_write_text(persist, json.dumps(generated, indent=2) + "\n")
        generated["score"] = score_completion(generated["output"], spec)
        generated["status"] = "reward_scored"
        generated["case_id"] = row["case_id"]
        generated["family_id"] = row["family_id"]
        atomic_write_text(persist, json.dumps(generated, indent=2) + "\n")
        group.append(generated)
    return group


def _disposable_update(
    runtime: dict[str, Any],
    root: Path,
    row: dict[str, Any],
    group: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    import mlx.nn as nn
    from mlx.optimizers import AdamW
    from mlx.utils import tree_flatten

    mx = runtime["mx"]
    model = runtime["model"]
    rewards = [item["score"]["reward"] for item in group]
    advantages = group_advantages(rewards)
    prompt = render_case_prompt(root, row["case_id"])
    optimizer = AdamW(learning_rate=5e-6, weight_decay=0.01)

    def loss_fn(mdl: Any) -> Any:
        return _grpo_loss(mdl, runtime["tokenizer"], prompt, group, advantages, mx)[0]

    value_and_grad = nn.value_and_grad(model, loss_fn)
    loss_value, grads = value_and_grad(model)
    loss_float = float(loss_value.item()) if hasattr(loss_value, "item") else float(loss_value)
    if not math.isfinite(loss_float):
        return {
            "reason": "nonfinite first disposable step",
            "loss": loss_float,
            "changed_lora_tensors": 0,
        }
    before = {
        name: tensor_digest(value) for name, value in tree_flatten(model.trainable_parameters())
    }
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    after = {
        name: tensor_digest(value) for name, value in tree_flatten(model.trainable_parameters())
    }
    changed = sum(1 for name in before if before[name] != after[name])
    adapter_path = output / "disposable-adapter" / "adapters.safetensors"
    expected = save_adapter(model, mx, adapter_path)
    reload_adapter_check(model, mx, adapter_path, expected)
    digest = sha256_file(adapter_path)
    return {
        "loss": loss_float,
        "advantages": advantages,
        "rewards": rewards,
        "changed_lora_tensors": changed,
        "adapter_sha256": digest,
        "disposable": True,
    }


def _grpo_loss(
    model: Any,
    _tokenizer: Any,
    _prompt: dict[str, str],
    group: list[dict[str, Any]],
    advantages: list[float],
    mx: Any,
) -> tuple[Any, dict[str, Any]]:
    """Token-level GRPO as documented in mlx-lm-lora v3.0.0 grpo_loss."""
    import mlx.nn as nn

    token_losses = []
    token_count = mx.array(0.0)
    for item, advantage in zip(group, advantages, strict=True):
        prompt_ids = list(item["prompt_token_ids"])
        completion_ids = list(item["completion_token_ids"])
        if not completion_ids:
            continue
        tokens = prompt_ids + completion_ids
        inputs = mx.array(tokens)[None, :]
        logits = model(inputs).astype(mx.float32)[:, :-1, :]
        targets = inputs[:, 1:]
        log_probs = nn.log_softmax(logits, axis=-1)
        chosen = mx.take_along_axis(log_probs, targets.reshape(1, -1, 1), axis=-1).squeeze(-1)[0]
        start = len(prompt_ids) - 1
        completion_logps = chosen[start : start + len(completion_ids)]
        old = mx.stop_gradient(completion_logps)
        log_ratio = completion_logps - old
        coef_1 = mx.exp(log_ratio)
        coef_2 = mx.clip(coef_1, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
        adv = mx.array(advantage, dtype=mx.float32)
        unclipped = coef_1 * adv
        clipped = coef_2 * adv
        per_token = -mx.minimum(unclipped, clipped)
        if KL_BETA != 0.0:
            log_ratio_ref = completion_logps - old
            ratio_ref = mx.exp(log_ratio_ref)
            kl_div = coef_1 * ratio_ref - log_ratio_ref - 1
            per_token = per_token + KL_BETA * kl_div
        token_losses.append(per_token.sum())
        token_count = token_count + mx.array(completion_logps.shape[0], dtype=mx.float32)
    if not token_losses:
        return mx.array(0.0), {"kl": 0.0}
    loss = sum(token_losses) / mx.maximum(token_count, mx.array(1.0))
    return loss, {"kl": 0.0}


def _copy_freeze(root: Path, destination: Path) -> None:
    source = root / "knowledge/company_task_specialization/qwen4b_grpo_v1"
    dest = destination / "freeze"
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            target = dest / path.name
            target.write_bytes(path.read_bytes())
            copied[path.name] = sha256_file(target)
    atomic_write_text(dest / "COPIED.json", json.dumps(copied, indent=2, sort_keys=True) + "\n")


def _public_stage(stage_a: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item["case_id"],
            "family_id": item["family_id"],
            "rewards": item["rewards"],
            "variance": item["variance"],
            "valid": [comp["score"]["complete_valid_output"] for comp in item["completions"]],
        }
        for item in stage_a
    ]


def _preflight_stop(output: Path, stage_a: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    report = {
        "status": "stop_preflight",
        "reason": reason,
        "stage_a": _public_stage(stage_a),
        "directory": str(output),
    }
    try:
        repo_root = output.resolve().parents[2]
    except IndexError:
        repo_root = Path.cwd()
    _write_preflight(repo_root, output, report)
    return report


def _write_preflight(root: Path, output: Path, report: dict[str, Any]) -> None:
    atomic_write_text(output / "preflight.json", json.dumps(report, indent=2, default=str) + "\n")
    atomic_write_text(
        root / ARTIFACTS / "latest-preflight.json",
        json.dumps({"directory": str(output), "status": report["status"]}, indent=2) + "\n",
    )


def _assert_resource_ok(swap_before: dict[str, Any] | None) -> None:
    rss = process_current_rss_bytes() or process_rss_bytes()
    if rss > RSS_STOP_BYTES:
        raise RuntimeError(f"process RSS {rss} exceeded {RSS_STOP_BYTES}")
    after = system_swap_snapshot()
    if swap_before and after:
        before_used = float((swap_before.get("parsed_M") or {}).get("used") or 0.0)
        after_used = float((after.get("parsed_M") or {}).get("used") or 0.0)
        growth = (after_used - before_used) * 1024 * 1024
        if growth > SWAP_GROWTH_STOP_BYTES:
            raise RuntimeError(f"swap growth {growth} exceeded {SWAP_GROWTH_STOP_BYTES}")


def _flatten(tree: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return [(str(name), value) for name, value in tree_flatten(tree)]


def _release_runtime(runtime: dict[str, Any] | None) -> None:
    if not runtime:
        return
    mx = runtime.get("mx")
    runtime.clear()
    gc.collect()
    if mx is not None:
        mx.clear_cache()


def _environment() -> dict[str, Any]:
    payload = {
        "python": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "model_id": QWEN_4B_MODEL_ID,
        "model_revision": QWEN_4B_REVISION,
        "packages": {},
    }
    for name in ("mlx", "mlx_lm", "mlx_metal", "llguidance"):
        try:
            payload["packages"][name] = __import__("importlib.metadata").metadata.version(name)
        except Exception:
            payload["packages"][name] = None
    return payload
