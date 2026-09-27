#!/usr/bin/env python
"""Stage drivers for qwen4b-typed-action-sft-v1. Requires local Metal.

Usage:
  python scripts/typed_action_sft_v1_driver.py preflight <run_dir>
  python scripts/typed_action_sft_v1_driver.py train <run_dir>
  python scripts/typed_action_sft_v1_driver.py evaluate-base <run_dir>
  python scripts/typed_action_sft_v1_driver.py evaluate-sft <run_dir>
  python scripts/typed_action_sft_v1_driver.py report <run_dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_memory_mlx.grpo_data import file_sha256  # noqa: E402
from enterprise_memory_mlx.learning_mechanics import (  # noqa: E402
    WiredLimitGuard,
    enable_gradient_checkpointing,
    seed_all,
)
from enterprise_memory_mlx.typed_action_sft_v1 import (  # noqa: E402
    EXPERIMENT_ID,
    build_system_prompt,
    build_user_prompt,
    compare_output,
    evaluation_inventory,
    evaluation_references,
    load_held_out_cases,
    load_train_targets,
    runtime_output_schema,
    training_schedule,
)
from enterprise_memory_mlx.typed_action_sft_v1_runtime import (  # noqa: E402
    WIRED_LIMIT_BYTES,
    build_training_rows,
    check_resource_policy,
    generate_one,
    load_model_with_lora,
    resource_snapshot,
    save_adapter,
    sft_step,
    verify_frozen_digests,
)
from enterprise_memory_mlx.utils import read_jsonl, sha256_text  # noqa: E402


def _load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("run manifest experiment mismatch")
    return manifest


def _compile_grammar() -> str:

    from enterprise_memory_mlx.structured_inference import compile_review_grammar

    return compile_review_grammar(runtime_output_schema())


def _grammar_tokenizers(tokenizer):
    from enterprise_memory_mlx.structured_inference import (
        llguidance_tokenizer_from_hf,
        unwrap_hf_tokenizer,
    )

    hf_tok = unwrap_hf_tokenizer(tokenizer)
    return llguidance_tokenizer_from_hf(hf_tok)


def _swap_baseline() -> float:
    from enterprise_memory_mlx.learning_mechanics_eval import system_swap_snapshot

    snap = system_swap_snapshot() or {}
    return float((snap.get("parsed_M") or {}).get("used") or 0.0)


def preflight(run_dir: Path) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import grad_checkpoint

    _load_run(run_dir)
    verify_frozen_digests(ROOT)
    baseline_swap = _swap_baseline()
    rows = None
    with WiredLimitGuard(mx, WIRED_LIMIT_BYTES):
        model, tokenizer, _ = load_model_with_lora(42, mx)
        enable_gradient_checkpointing(model, grad_checkpoint)
        rows = build_training_rows(ROOT, tokenizer)
        {row["case_id"]: row for row in rows}
        longest = max(rows, key=lambda r: r["total_len"])
        shortest = min(rows, key=lambda r: r["total_len"])
        frozen_before = _frozen_digests(model, mx)
        optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
        steps = []
        for name, row in (("longest", longest), ("shortest", shortest)):
            record = sft_step(
                model=model,
                optimizer=optimizer,
                mx=mx,
                nn=_nn(),
                row=row,
                update_index=0,
            )
            record["row_kind"] = name
            record["resource"] = resource_snapshot(mx)
            steps.append(record)
        _frozen_base_unchanged(model, mx, frozen_before)
        # one disposable constrained generation on the shortest row
        grammar = _compile_grammar()
        from llguidance import LLMatcher

        ll_tok = _grammar_tokenizers(tokenizer)
        matcher = LLMatcher(ll_tok, grammar)
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from enterprise_memory_mlx.structured_inference import (
            LLGuidanceJSONLogitsProcessor,
        )

        seed_all(42, mx)
        sampler = make_sampler(temp=0.0, top_p=0.0)
        cache = make_prompt_cache(model)
        generated: list[int] = []
        finish = "unknown"
        started = time.perf_counter()
        try:
            for response in stream_generate(
                model,
                tokenizer,
                prompt=list(shortest["prompt_ids"]),
                max_tokens=2048,
                sampler=sampler,
                logits_processors=[LLGuidanceJSONLogitsProcessor(matcher, ll_tok)],
                prompt_cache=cache,
            ):
                generated.append(int(response.token))
                finish = response.finish_reason
        finally:
            cache = None
            mx.clear_cache()
        eos = set(
            getattr(tokenizer, "eos_token_ids", None)
            or [getattr(tokenizer, "eos_token_id", None)]
        )
        disposable_output = tokenizer.decode([t for t in generated if t not in eos])
        generation = {
            "case_id": shortest["case_id"],
            "disposable": True,
            "output": disposable_output,
            "completion_length": len(generated),
            "finish_reason": finish,
            "truncated": finish == "length",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "resource": resource_snapshot(mx),
        }
    check_resource_policy(generation["resource"], baseline_swap_used_mb=baseline_swap)
    record = {
        "stage": "preflight",
        "steps": steps,
        "disposable_generation": generation,
        "row_lengths": {
            "longest": {"case_id": longest["case_id"], "total_len": longest["total_len"]},
            "shortest": {"case_id": shortest["case_id"], "total_len": shortest["total_len"]},
        },
        "rows": len(rows),
        "preflight_state_discarded": True,
    }
    out = run_dir / "preflight.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("preflight complete:", out)
    return record


def _nn():
    import mlx.nn as nn

    return nn


def _frozen_digests(model, mx) -> dict:
    from mlx.utils import tree_flatten

    def digest(value) -> str:
        import hashlib

        arr = value.astype(mx.float32)
        mx.eval(arr)
        return hashlib.sha256(bytes(arr)).hexdigest()

    return {
        name: digest(value)
        for name, value in tree_flatten(model.parameters())
        if ".lora_" not in name
    }


def _frozen_base_unchanged(model, mx, before: dict) -> None:
    after = _frozen_digests(model, mx)
    drifted = [name for name in before if before[name] != after.get(name)]
    if drifted:
        raise RuntimeError(f"frozen base tensors changed: {drifted[:3]}")


def train(run_dir: Path) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import grad_checkpoint

    _load_run(run_dir)
    verify_frozen_digests(ROOT)
    baseline_swap = _swap_baseline()
    with WiredLimitGuard(mx, WIRED_LIMIT_BYTES):
        model, tokenizer, adapter_info = load_model_with_lora(42, mx)
        enable_gradient_checkpointing(model, grad_checkpoint)
        rows = build_training_rows(ROOT, tokenizer)
        if len(rows) != 24:
            raise RuntimeError(f"expected 24 rows, got {len(rows)}")
        schedule = training_schedule()
        if len(schedule) != 192:
            raise RuntimeError("schedule must be 192 updates")
        target_by_slot = list(rows)
        frozen_before = _frozen_digests(model, mx)
        optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
        update_records = []
        started = time.perf_counter()
        for item in schedule:
            row = target_by_slot[item["case_slot"]]
            record = sft_step(
                model=model,
                optimizer=optimizer,
                mx=mx,
                nn=_nn(),
                row=row,
                update_index=item["update_index"],
            )
            record["epoch"] = item["epoch"]
            record["position_in_epoch"] = item["position_in_epoch"]
            if item["update_index"] % 24 == 23:
                record["resource"] = resource_snapshot(mx)
            update_records.append(record)
        elapsed = time.perf_counter() - started
        _frozen_base_unchanged(model, mx, frozen_before)
        adapter_path = run_dir / "adapter.safetensors"
        digests = save_adapter(model, mx, adapter_path)
        final_resource = resource_snapshot(mx)
    check_resource_policy(final_resource, baseline_swap_used_mb=baseline_swap)
    exposures: dict[str, int] = {}
    for record in update_records:
        exposures.setdefault(record["case_id"], 0)
        exposures[record["case_id"]] += 1
    if sorted(exposures.values()) != [8] * 24:
        raise RuntimeError("exposure counts are not eight per case")
    record = {
        "stage": "train",
        "measured_updates": len(update_records),
        "elapsed_seconds": round(elapsed, 3),
        "final_resource": final_resource,
        "adapter_sha256": file_sha256(adapter_path),
        "adapter_tensor_count": len(digests),
        "exposures": dict(sorted(exposures.items())),
        "loss_first": update_records[0]["loss"],
        "loss_last": update_records[-1]["loss"],
        "updates": update_records,
    }
    out = run_dir / "training.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("training complete:", out, "adapter:", record["adapter_sha256"])
    return record


def _evaluate(run_dir: Path, arm: str) -> dict:
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    manifest = _load_run(run_dir)
    verify_frozen_digests(ROOT)
    inventory = evaluation_inventory(ROOT)
    references = {row["case_id"]: row for row in evaluation_references()}
    held = {row["case_id"]: row for row in load_held_out_cases(ROOT)}
    targets = {row["case_id"]: row for row in load_train_targets(ROOT)}
    train_targets = {
        "CTS3-QUAL-001": targets["CTS3-QUAL-001"],
        "CTS3-QUAL-002": targets["CTS3-QUAL-002"],
        "CTS3-QUAL-003": targets["CTS3-QUAL-003"],
        "CTS3-QUAL-008": targets["CTS3-QUAL-008"],
    }
    sp = build_system_prompt()
    schema = runtime_output_schema()
    sets = {
        "test": inventory["test_case_ids"],
        "dev": inventory["dev_case_ids"],
        "train": inventory["train_diagnostic_case_ids"],
    }
    baseline_swap = _swap_baseline()
    adapter_sha = manifest.get("adapter_sha256")
    arm_dir = run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    results = {name: [] for name in sets}
    with WiredLimitGuard(mx, WIRED_LIMIT_BYTES):
        if arm == "sft":
            from mlx_lm.tuner.utils import load_adapters

            model, tokenizer = mlx_load(
                str(_snapshot_path()),
                tokenizer_config={"trust_remote_code": False},
            )
            load_adapters(model, str(run_dir))
            actual = file_sha256(run_dir / "adapter.safetensors")
            if adapter_sha and actual != adapter_sha:
                raise RuntimeError("adapter digest mismatch at evaluation load")
        else:
            model, tokenizer = mlx_load(
                str(_snapshot_path()),
                tokenizer_config={"trust_remote_code": False},
            )
        grammar = _compile_grammar()
        for set_name in ("test", "dev", "train"):
            for case_id in sets[set_name]:
                case = (
                    held[case_id]
                    if case_id in held
                    else _train_case(case_id, train_targets)
                )
                user = build_user_prompt(ROOT, case)
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
                attempt = {
                    "arm": arm,
                    "set": set_name,
                    "case_id": case_id,
                    "attempt_started_at": _now(),
                }
                attempt_path = arm_dir / f"{set_name}-{case_id}.json"
                attempt_path.write_text(
                    json.dumps({**attempt, "status": "started"}, indent=2) + "\n"
                )
                generation = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    mx=mx,
                    seed=42,
                    prompt_ids=prompt_ids,
                    grammar=grammar,
                    max_tokens=2048,
                )
                raw_path = arm_dir / f"{set_name}-{case_id}-raw.json"
                raw_record = {
                    **attempt,
                    **generation,
                    "prompt_sha256": sha256_text(rendered),
                    "adapter_identity": arm,
                    "status": "generated",
                }
                raw_path.write_text(json.dumps(raw_record, indent=2) + "\n")
                reference = (
                    references[case_id]["reference"]
                    if set_name != "train"
                    else train_targets[case_id]["proposed_training_target"]
                )
                request_ids = [
                    item["request_item_id"]
                    for item in case["request_items"]
                ]
                supplied = list(case.get("source_record_ids") or [])
                for state in case.get("operational_state") or []:
                    supplied.append(str(state["state_id"]))
                scored = compare_output(
                    raw_output=generation["output"],
                    input_text=user,
                    reference=reference,
                    request_item_ids=request_ids,
                    supplied_source_ids=supplied,
                    schema=schema,
                )
                results[set_name].append(
                    {
                        **raw_record,
                        "scored": scored,
                        "scored_at": _now(),
                    }
                )
                check_resource_policy(
                    generation.get("resource") or resource_snapshot(mx),
                    baseline_swap_used_mb=baseline_swap,
                )
    record = {
        "stage": f"evaluate-{arm}",
        "arm": arm,
        "counts": {name: len(rows) for name, rows in results.items()},
        "results": results,
    }
    out = run_dir / f"evaluation-{arm}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"evaluation {arm} complete:", out)
    return record


def _snapshot_path() -> Path:
    from enterprise_memory_mlx.experiment_profiles import QWEN_4B_REVISION

    return (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
        / "snapshots"
        / QWEN_4B_REVISION
    )


def _train_case(case_id: str, train_targets: dict) -> dict:
    # TRAIN diagnostics reuse the original case material; the loader only
    # needs the allowlisted input fields.
    v3 = read_jsonl(
        ROOT
        / "knowledge/company_task_specialization/v3_remedy_logic/qualification_cases.jsonl"
    )
    v4 = read_jsonl(
        ROOT
        / "knowledge/company_task_specialization/v4_fact_state/contrast_cases.jsonl"
    )
    by_id = {c["case_id"]: c for c in v3}
    by_id.update({c["case_id"]: c for c in v4})
    return by_id[case_id]


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def report(run_dir: Path) -> dict:
    _load_run(run_dir)
    base = json.loads((run_dir / "evaluation-base.json").read_text())
    sft = json.loads((run_dir / "evaluation-sft.json").read_text())
    training = json.loads((run_dir / "training.json").read_text())

    def arm_tables(rows: list) -> dict:
        return {
            "n": len(rows),
            "complete_valid": sum(
                1 for r in rows if r["scored"]["complete_valid_output"]
            ),
            "complete_agreement": sum(
                1 for r in rows if r["scored"]["complete_typed_reference_agreement"]
            ),
            "decision_agreement": sum(
                1 for r in rows if r["scored"].get("current_decision_agreement")
            ),
            "truncated": sum(1 for r in rows if r["truncated"]),
        }

    def paired(base_rows, sft_rows, key):
        pairs = {}
        for r in base_rows:
            pairs.setdefault(r["case_id"], {})[r["arm"]] = r
        for r in sft_rows:
            pairs.setdefault(r["case_id"], {})[r["arm"]] = r
        wins = losses = ties = 0
        detail = []
        for case_id, arms in sorted(pairs.items()):
            b = arms.get("base")
            s = arms.get("sft")
            if not b or not s:
                continue
            b_ok = bool(b["scored"]["complete_typed_reference_agreement"])
            s_ok = bool(s["scored"]["complete_typed_reference_agreement"])
            if s_ok and not b_ok:
                wins += 1
            elif b_ok and not s_ok:
                losses += 1
            else:
                ties += 1
            detail.append(
                {
                    "case_id": case_id,
                    "base": b_ok,
                    "sft": s_ok,
                    "base_decision": b["scored"].get("current_decision_agreement"),
                    "sft_decision": s["scored"].get("current_decision_agreement"),
                    "key": key,
                }
            )
        return {"wins": wins, "losses": losses, "ties": ties, "detail": detail}

    vector = {}
    for set_name in ("test", "dev", "train"):
        base_rows = [r for r in base["results"][set_name]]
        sft_rows = [r for r in sft["results"][set_name]]
        vector[set_name] = {
            "base": arm_tables(base_rows),
            "sft": arm_tables(sft_rows),
            "paired": paired(base_rows, sft_rows, set_name),
        }
    test = vector["test"]
    test["sft"]["n"] or 1
    gain = test["sft"]["complete_agreement"] - test["base"]["complete_agreement"]
    decision_decline = test["sft"]["decision_agreement"] < test["base"]["decision_agreement"]
    prohibited_regression = False
    for r in sft["results"]["test"]:
        findings = r["scored"]["layers"]["layer_4_prohibited"]["findings"]
        b_findings = next(
            (
                b["scored"]["layers"]["layer_4_prohibited"]["findings"]
                for b in base["results"]["test"]
                if b["case_id"] == r["case_id"]
            ),
            [],
        )
        if findings and not b_findings:
            prohibited_regression = True
    if (
        gain > 0
        and test["paired"]["wins"] > test["paired"]["losses"]
        and not decision_decline
        and not prohibited_regression
    ):
        classification = "observed_reference_agreement_gain"
    elif gain > 0:
        classification = "mixed_result"
    elif gain <= 0:
        classification = "no_observed_gain"
    else:
        classification = "incomplete_execution"
    findings_report = {
        "stage": "report",
        "experiment_id": EXPERIMENT_ID,
        "adapter_sha256": training["adapter_sha256"],
        "measured_updates": training["measured_updates"],
        "vector": vector,
        "test_classification": classification,
        "classification_rules_frozen_before_generation": True,
        "base_ceiling_note": (
            "If the base already solves all TEST cases, this is a ceiling, "
            "not an SFT win."
        ),
        "limitation_note": (
            "TRAIN fitting is not TEST performance. No reference-based score "
            "establishes complete semantic safety or causal understanding. "
            "These sets were previously used in the broader research "
            "programme and are not pristine unseen material. Eight TEST rows "
            "span five families and are not independent experimental units."
        ),
    }
    out = run_dir / "report.json"
    out.write_text(json.dumps(findings_report, indent=2) + "\n")
    print("report complete:", out, "classification:", classification)
    return findings_report


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    stage = sys.argv[1]
    run_dir = Path(sys.argv[2]).resolve()
    if stage == "preflight":
        preflight(run_dir)
    elif stage == "train":
        train(run_dir)
    elif stage == "evaluate-base":
        _evaluate(run_dir, "base")
    elif stage == "evaluate-sft":
        _evaluate(run_dir, "sft")
    elif stage == "report":
        report(run_dir)
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    main()