"""Freeze and validate the qwen4b-grpo-v1 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment_profiles import QWEN_4B_MODEL_ID, QWEN_4B_REVISION
from .grpo_data import (
    PROTOCOL_ID,
    SPLIT_SEED,
    compile_rows,
    file_sha256,
    load_parent_assets,
    render_prompt,
)
from .grpo_reward import COMPONENT_WEIGHTS, WEIGHT_RATIONALE, compile_reward_spec, spec_sha256
from .specialization_fact_state_experiment import AMENDED_SYSTEM_PROMPT
from .specialization_remedy_experiment import CHALLENGER_SYSTEM_PROMPT
from .utils import atomic_write_text, read_jsonl, sha256_text

_REWARD_MODULE = Path(__file__).with_name("grpo_reward.py")
_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
    / "snapshots"
    / QWEN_4B_REVISION
)

PROTOCOL_DIR = Path("knowledge/company_task_specialization/qwen4b_grpo_v1")
ORIGIN_MAIN_SHA = "a6c0421f244c18a439a907d67b4869fe59f85792"
TRAINER_IDENTITY = {
    "package": "mlx-lm-lora",
    "repository": "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora",
    "version": "3.0.0",
    "git_commit": "fb4f39db66fadec3b71a41441e863d9f1bf87844",
    "license": "Apache-2.0",
    "algorithm": "GRPO",
    "algorithm_variants_available": ["grpo", "bnpo", "dr_grpo"],
    "selected_loss_type": "grpo",
    "importance_sampling_level": "token",
    "beta_kl": 0.1,
    "clip_epsilon": 1e-4,
    "advantage_epsilon": 1e-4,
    "zero_variance_handling": (
        "advantages collapse to ~0 via (r-mean)/(std+1e-4); "
        "groups are recorded, not fabricated"
    ),
    "reference_model": (
        "none_on_policy; ref logprobs are stop-gradient of the generating "
        "policy (library default when reference_model_path is None)"
    ),
    "rollout_deviation": (
        "sequential mlx_lm.stream_generate per group member; "
        "library default batch_generate is not used"
    ),
    "logprob_deviation": (
        "prompt-conditioned completion logprobs; library generate_grpo "
        "currently forwards completion tokens only and is not copied"
    ),
    "package_imported": False,
    "formula_source": (
        "mlx_lm_lora/trainer/grpo_trainer.py grpo_loss + "
        "calculate_rewards_and_advantages at v3.0.0"
    ),
    "cot_required": False,
    "telemetry": False,
}
CONTINUATION_RULE = {
    "rule_id": "qwen4b-grpo-v1/structured-test-gate",
    "surface": "primary_structured_decoder_test",
    "all_required": [
        "mean_deterministic_reward_rft_minus_base >= 0.10",
        "rft_full_machine_success_cases >= base_full_machine_success_cases + 2",
        "paired_wins > paired_losses",
        "rft_provenance_hard_failures <= base_provenance_hard_failures",
        "rft_new_unsafe_hard_failures == 0",
        "rft_exact_decision_accuracy >= base_exact_decision_accuracy",
    ],
    "pass_status": "rft_pilot_promising_semantic_review_required",
    "fail_status": "stop_rft_candidate",
    "not_authorized": [
        "production",
        "human_approval",
        "evidence_entailment_certification",
        "general_task_solved",
    ],
}


def protocol_dir(root: Path) -> Path:
    return root / PROTOCOL_DIR


def _tokenizer_hashes() -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    hashes: dict[str, str] = {}
    for name in names:
        path = _SNAPSHOT / name
        if path.is_file():
            hashes[name] = file_sha256(path)
    return hashes


def write_frozen_protocol(root: Path) -> dict[str, Any]:
    compiled = compile_rows(root)
    assets = load_parent_assets(root)
    cases = {str(case["case_id"]): {**case, "source_protocol": "v3"} for case in assets["v3_cases"]}
    cases.update(
        {str(case["case_id"]): {**case, "source_protocol": "v4"} for case in assets["v4_cases"]}
    )
    reward_specs = []
    for row in compiled["rows"]:
        spec = compile_reward_spec(cases[row["case_id"]])
        reward_specs.append({**spec, "spec_sha256": spec_sha256(spec), "split": row["split"]})
    destination = protocol_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_measured_training",
        "origin_main_sha": ORIGIN_MAIN_SHA,
        "parent_protocols": {
            "v3": "company-task-specialization/v3-remedy-logic",
            "v4": "company-task-specialization/v4-fact-state",
        },
        "model": {
            "id": QWEN_4B_MODEL_ID,
            "revision": QWEN_4B_REVISION,
            "initialize_from_existing_sft_adapter": False,
            "existing_sft_examples_excluded": True,
            "tokenizer_files": _tokenizer_hashes(),
            "reward_implementation_sha256": file_sha256(_REWARD_MODULE),
        },
        "adapter": {
            "rank": 16,
            "scale": 2.0,
            "dropout": 0.0,
            "num_layers": 36,
            "target_modules": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
        },
        "training": {
            "train_cases": 24,
            "dev_cases": 8,
            "test_cases": 8,
            "epochs": 4,
            "group_size": 4,
            "prompt_groups": 96,
            "max_rollouts": 384,
            "batch_prompts": 1,
            "seed": SPLIT_SEED,
            "optimizer": "AdamW",
            "learning_rate": 5e-6,
            "weight_decay": 0.01,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_completion_tokens": 1024,
            "thinking": False,
            "tools": False,
            "grammar_during_training_rollouts": False,
        },
        "trainer": TRAINER_IDENTITY,
        "continuation_rule": CONTINUATION_RULE,
        "resource_stop": {
            "process_rss_bytes": 112 * 1024**3,
            "swap_growth_bytes": 8 * 1024**3,
            "wired_limit_description": (
                "MLX wired-limit setting only; not a proven 96 GiB application cap"
            ),
        },
        "claim_boundary": {
            "synthetic_governed_scenarios_only": True,
            "closed_book_knowledge": False,
            "deployment": False,
            "human_level_review": False,
            "complete_evidence_entailment": False,
            "continual_learning": False,
            "sft_adapter_generalization": False,
        },
    }
    source_bindings = {
        "schema_version": 1,
        "parent_hashes": compiled["parent_hashes"],
        "system_prompts": {
            "v3": sha256_text(CHALLENGER_SYSTEM_PROMPT),
            "v4": sha256_text(AMENDED_SYSTEM_PROMPT),
        },
        "origin_main_sha": ORIGIN_MAIN_SHA,
        "trainer": TRAINER_IDENTITY,
    }
    split_manifest = {
        "schema_version": 1,
        "seed": SPLIT_SEED,
        "algorithm": (
            "family-preserving subset-sum enumeration, then seed-42 choice "
            "among coverage-valid partitions"
        ),
        "rows": compiled["rows"],
        "families": compiled["families"],
        "assignment": compiled["assignment"],
        "tag_distribution": compiled["tag_distribution"],
        "source_record_distribution": compiled["source_record_distribution"],
        "pair_distribution": compiled["pair_distribution"],
    }
    reward_contract = {
        "schema_version": 1,
        "judge": "deterministic_machine_only",
        "language_model_reward": False,
        "lexical_rationale_reward": False,
        "historical_qwen27_or_gpt_labels": False,
        "weights": COMPONENT_WEIGHTS,
        "weight_rationale": WEIGHT_RATIONALE,
        "hard_zero": [
            "invalid_json_or_schema",
            "provenance_hard_violation",
            "unsafe_waiver_or_forbidden_action",
        ],
        "renormalize_over_applicable_components": True,
        "reward_range": [0.0, 1.0],
        "frozen_before_rollouts": True,
    }
    files = {
        "protocol.json": protocol,
        "source-bindings.json": source_bindings,
        "split-manifest.json": split_manifest,
        "reward-contract.json": reward_contract,
    }
    written = {}
    for name, payload in files.items():
        path = destination / name
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written[name] = file_sha256(path)
    specs_path = destination / "reward-specs.jsonl"
    atomic_write_text(
        specs_path,
        "".join(
            json.dumps(spec, sort_keys=True, ensure_ascii=False) + "\n" for spec in reward_specs
        ),
    )
    written["reward-specs.jsonl"] = file_sha256(specs_path)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "files": written,
        "case_count": 40,
        "train_count": 24,
        "dev_count": 8,
        "test_count": 8,
        "origin_main_sha": ORIGIN_MAIN_SHA,
        "human_labels_present": False,
        "private_data": False,
        "existing_sft_adapter_used": False,
    }
    manifest_path = destination / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    written["manifest.json"] = file_sha256(manifest_path)
    return {"directory": str(destination), "files": written, "compiled": compiled}


def load_frozen_protocol(root: Path) -> dict[str, Any]:
    destination = protocol_dir(root)
    protocol = json.loads((destination / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((destination / "split-manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((destination / "reward-contract.json").read_text(encoding="utf-8"))
    specs = read_jsonl(destination / "reward-specs.jsonl")
    bindings = json.loads((destination / "source-bindings.json").read_text(encoding="utf-8"))
    expected = dict(manifest["files"])
    expected.pop("manifest.json", None)
    for name, digest in expected.items():
        actual = file_sha256(destination / name)
        if actual != digest:
            raise ValueError(f"{name} hash drifted from frozen manifest")
    if protocol.get("origin_main_sha") != ORIGIN_MAIN_SHA:
        raise ValueError("frozen protocol is not bound to the recorded origin/main SHA")
    return {
        "protocol": protocol,
        "manifest": manifest,
        "split": split,
        "reward_contract": contract,
        "reward_specs": specs,
        "source_bindings": bindings,
        "directory": destination,
    }


def spec_by_case(root: Path) -> dict[str, dict[str, Any]]:
    frozen = load_frozen_protocol(root)
    return {str(row["case_id"]): row for row in frozen["reward_specs"]}


def rows_for_split(root: Path, split: str) -> list[dict[str, Any]]:
    frozen = load_frozen_protocol(root)
    return [row for row in frozen["split"]["rows"] if row["split"] == split]


def assert_generation_isolation(root: Path, case_id: str, prompt_question: str) -> None:
    spec = spec_by_case(root)[case_id]
    if spec["case_id"] != case_id:
        raise ValueError("reward spec case mismatch")
    for fragment in _hidden_objects(root, case_id):
        if fragment in prompt_question:
            raise ValueError(f"{case_id} prompt contains a hidden evaluator object")


def _hidden_objects(root: Path, case_id: str) -> list[str]:
    assets = load_parent_assets(root)
    for case in [*assets["v3_cases"], *assets["v4_cases"]]:
        if case["case_id"] != case_id:
            continue
        objects = [json.dumps(case["reference"], sort_keys=True, ensure_ascii=False)]
        if case.get("policy_logic") is not None:
            objects.append(json.dumps(case["policy_logic"], sort_keys=True, ensure_ascii=False))
        if case.get("expected_fact_state") is not None:
            objects.append(
                json.dumps(case["expected_fact_state"], sort_keys=True, ensure_ascii=False)
            )
        return objects
    raise KeyError(case_id)


def render_case_prompt(root: Path, case_id: str) -> dict[str, str]:
    assets = load_parent_assets(root)
    for case in assets["v3_cases"]:
        if case["case_id"] == case_id:
            return render_prompt(root, {**case, "source_protocol": "v3"}, assets)
    for case in assets["v4_cases"]:
        if case["case_id"] == case_id:
            return render_prompt(root, {**case, "source_protocol": "v4"}, assets)
    raise KeyError(case_id)
