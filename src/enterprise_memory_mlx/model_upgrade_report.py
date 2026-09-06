"""Bounded comparison report for model-upgrade-exploratory/v1."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .experiment_profiles import (
    MODEL_UPGRADE_CONFIG,
    MODEL_UPGRADE_PROFILE,
    MODEL_UPGRADE_PROTOCOL_ID,
    QWEN_4B_PROFILE,
    QWEN_27B_PROFILE,
    ModelProfile,
)
from .utils import atomic_write_text, sha256_json

CONTINUATION_CRITERIA = {
    "protocol_id": MODEL_UPGRADE_PROTOCOL_ID,
    "mean_score_delta_over_same_model_base_min": 0.10,
    "fully_correct_delta_over_same_model_base_min": 4,
    "paired_wins_must_exceed_losses": True,
    "unknown_oos_failure_count_must_not_worsen": True,
    "blocking_runtime_or_verifier_defect_allowed": False,
    "passing_action": "authorize_one_seed_43_repeat_of_matched_runs",
}
PRACTICAL_SCREEN = {
    "max_mean_gap_below_full_context": 0.05,
    "requires_measured_operational_benefit": True,
    "deployment_authorization": False,
}


def write_model_upgrade_comparison(
    *,
    advisory_4b_path: Path,
    advisory_27b_path: Path,
    general_4b_path: Path,
    general_27b_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    four = _load(advisory_4b_path)
    twenty_seven = _load(advisory_27b_path)
    general_four = _load(general_4b_path)
    general_twenty_seven = _load(general_27b_path)
    four_binding = _validate_experiment_report(
        four,
        general_four,
        QWEN_4B_PROFILE,
        ("base", "parametric", "full_context", "oracle"),
    )
    twenty_seven_binding = _validate_experiment_report(
        twenty_seven,
        general_twenty_seven,
        QWEN_27B_PROFILE,
        ("base", "parametric", "full_context", "oracle", "bm25"),
    )
    if four_binding["dataset_train_sha256"] != twenty_seven_binding["dataset_train_sha256"]:
        raise ValueError("Matched 4B and 27B runs did not use identical training rows")
    _validate_matched_verifier(four, twenty_seven)
    four_metrics = _model_metrics(four, general_four)
    twenty_seven_metrics = _model_metrics(twenty_seven, general_twenty_seven)
    if "bm25" not in twenty_seven_metrics["acquisition_arms"]:
        raise ValueError("27B advisory is missing the frozen experimental BM25 arm")
    acquisition_difference = (
        twenty_seven_metrics["acquisition"]["mean_delta"]
        - four_metrics["acquisition"]["mean_delta"]
    )
    criteria = _continuation_assessment(twenty_seven_metrics)
    authorized = all(item["passed"] for item in criteria.values())
    practical = _practical_assessment(twenty_seven_metrics)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": MODEL_UPGRADE_PROTOCOL_ID,
        "status": (
            "complete_advisory"
            if four.get("status") == "complete_advisory"
            and twenty_seven.get("status") == "complete_advisory"
            else "blocking_runtime_or_verifier_defect"
        ),
        "promotion_eligible": False,
        "human_approved": False,
        "verification_status": "single_local_judge_advisory",
        "historical_suite_status": "previously_inspected_small_diagnostic",
        "three_questions": {
            "generation_improved": _generation_comparison(
                four_metrics,
                twenty_seven_metrics,
            ),
            "adapter_added_company_knowledge_beyond_same_base": (
                twenty_seven_metrics["acquisition"]
            ),
            "adapter_competitive_with_direct_evidence": practical,
        },
        "models": {
            "4b_control": four_metrics,
            "27b_candidate": twenty_seven_metrics,
        },
        "difference_in_acquisition_improvement": acquisition_difference,
        "continuation": {
            "criteria": CONTINUATION_CRITERIA,
            "criteria_hash": sha256_json(CONTINUATION_CRITERIA),
            "observations": criteria,
            "decision": (
                "authorize_seed_43_repeat" if authorized else "end_model_upgrade_exploratory_v1"
            ),
        },
        "practical_screen": practical,
        "limitations": [
            "Single local Gemma labels are advisory and are not human evidence.",
            "The 32 acquisition probes cluster around eight records and are not independent facts.",
            "One seed cannot isolate parameter count from architecture or post-training.",
            "The curriculum and model changed together; the matched 4B control "
            "only partially separates them.",
        ],
        "artifact_chain": {
            "advisory_4b": str(advisory_4b_path.resolve()),
            "advisory_27b": str(advisory_27b_path.resolve()),
            "advisory_4b_hash": _file_hash(advisory_4b_path),
            "advisory_27b_hash": _file_hash(advisory_27b_path),
            "general_4b": str(general_4b_path.resolve()),
            "general_27b": str(general_27b_path.resolve()),
            "general_4b_hash": _file_hash(general_4b_path),
            "general_27b_hash": _file_hash(general_27b_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model-upgrade-exploratory-v1-comparison.json"
    markdown_path = output_dir / "model-upgrade-exploratory-v1-comparison.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Refusing to overwrite model-upgrade comparison")
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    atomic_write_text(markdown_path, _render(report))
    return json_path, markdown_path


def _model_metrics(
    report: Mapping[str, Any],
    general: Mapping[str, Any],
) -> dict[str, Any]:
    rows = report.get("attempts")
    if not isinstance(rows, list):
        raise ValueError("Advisory report has no attempts")
    acquisition = [row for row in rows if row["suite"] == "acquisition"]
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in acquisition:
        by_arm[str(row["arm"])].append(row)
    required = {"base", "parametric", "full_context", "oracle"}
    if not required.issubset(by_arm):
        raise ValueError(f"Advisory report is missing required arms: {required - set(by_arm)}")
    arm_metrics = {arm: _scores(group) for arm, group in sorted(by_arm.items())}
    paired = _paired(by_arm["base"], by_arm["parametric"])
    oos = _oos_failures(rows)
    base = arm_metrics["base"]
    parametric = arm_metrics["parametric"]
    generator_model = report["artifact_chain"].get("generator_model") or _benchmark_model(report)
    if general.get("model_id") != generator_model:
        raise ValueError("General diagnostic model does not match Gemma advisory generator")
    general_base = float(general.get("base", {}).get("rate", 0.0))
    general_parametric = float(general.get("parametric", {}).get("rate", 0.0))
    return {
        "model_id": generator_model,
        "report_status": report["status"],
        "runtime_or_verifier_defect": report["status"] != "complete_advisory",
        "acquisition_arms": arm_metrics,
        "acquisition": {
            "mean_delta": parametric["mean"] - base["mean"],
            "fully_correct_delta": (parametric["fully_correct"] - base["fully_correct"]),
            **paired,
        },
        "unknown_oos": oos,
        "general_capability_diagnostic": {
            "status": general.get("status"),
            "item_count": general.get("item_count"),
            "base_exact_rate": general_base,
            "parametric_exact_rate": general_parametric,
            "delta": general_parametric - general_base,
            "observed_worsening": general_parametric < general_base,
            "warning": general.get("warning"),
        },
        "per_record_acquisition": _per_record(acquisition),
        "runtime": _runtime_metrics(report),
    }


def _scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(row["all_attempted_score"]) for row in rows]
    return {
        "n": len(scores),
        "mean": sum(scores) / len(scores),
        "fully_correct": sum(score == 1.0 for score in scores),
        "partial": sum(score == 0.5 for score in scores),
        "incorrect_or_runtime_failure": sum(score == 0.0 for score in scores),
    }


def _paired(
    base_rows: Sequence[Mapping[str, Any]],
    parametric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    base = {str(row["question_id"]): float(row["all_attempted_score"]) for row in base_rows}
    adapted = {
        str(row["question_id"]): float(row["all_attempted_score"]) for row in parametric_rows
    }
    if set(base) != set(adapted):
        raise ValueError("Base and parametric acquisition rows are not paired")
    wins = sum(adapted[key] > base[key] for key in base)
    losses = sum(adapted[key] < base[key] for key in base)
    return {"paired_wins": wins, "paired_losses": losses, "paired_ties": len(base) - wins - losses}


def _oos_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["suite"] == "unknown_oos"]
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_arm[str(row["arm"])].append(row)
    if not by_arm or "base" not in by_arm or "parametric" not in by_arm:
        raise ValueError("Unknown/OOS rows are missing required base or parametric arms")
    expected_count = len(by_arm["base"])
    if expected_count == 0 or any(len(group) != expected_count for group in by_arm.values()):
        raise ValueError("Unknown/OOS arm matrices are incomplete or unmatched")
    output = {
        arm: sum(float(row["all_attempted_score"]) < 1.0 for row in group)
        for arm, group in sorted(by_arm.items())
    }
    return {
        "failure_counts": output,
        "parametric_worsening_vs_base": (output.get("parametric", 0) > output.get("base", 0)),
    }


def _per_record(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("record_id") or "none"), str(row["arm"]))].append(row)
    output = []
    for (record_id, arm), group in sorted(grouped.items()):
        scores = [float(row["all_attempted_score"]) for row in group]
        output.append(
            {
                "record_id": record_id,
                "arm": arm,
                "n": len(group),
                "mean": sum(scores) / len(scores),
                "fully_correct": sum(score == 1.0 for score in scores),
                "partial": sum(score == 0.5 for score in scores),
                "deterministic_failure_reasons": sorted(
                    {
                        str(reason)
                        for row in group
                        for reason in row.get("deterministic_reasons", [])
                    }
                ),
            }
        )
    return output


def _runtime_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_path = Path(str(report["artifact_chain"]["benchmark_path"]))
    benchmark = _load(benchmark_path)
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in benchmark.get("results", []):
        by_arm[str(row["arm"])].append(row)
    return {
        arm: {
            "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in group),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in group),
            "peak_memory_gb": max(
                (float(row.get("peak_memory_gb") or 0) for row in group),
                default=0.0,
            ),
            "runtime_failures": sum(row.get("generation_status") != "generated" for row in group),
        }
        for arm, group in sorted(by_arm.items())
    }


def _continuation_assessment(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    acquisition = metrics["acquisition"]
    oos = metrics["unknown_oos"]
    return {
        "mean_delta": {
            "observed": acquisition["mean_delta"],
            "required": 0.10,
            "passed": acquisition["mean_delta"] >= 0.10,
        },
        "fully_correct_delta": {
            "observed": acquisition["fully_correct_delta"],
            "required": 4,
            "passed": acquisition["fully_correct_delta"] >= 4,
        },
        "paired_wins_exceed_losses": {
            "wins": acquisition["paired_wins"],
            "losses": acquisition["paired_losses"],
            "passed": acquisition["paired_wins"] > acquisition["paired_losses"],
        },
        "unknown_oos_not_worse": {
            "observed_worsening": oos["parametric_worsening_vs_base"],
            "passed": not oos["parametric_worsening_vs_base"],
        },
        "no_blocking_runtime_or_verifier_defect": {
            "observed": metrics["runtime_or_verifier_defect"],
            "passed": not metrics["runtime_or_verifier_defect"],
        },
    }


def _practical_assessment(metrics: Mapping[str, Any]) -> dict[str, Any]:
    arms = metrics["acquisition_arms"]
    gap = arms["full_context"]["mean"] - arms["parametric"]["mean"]
    bm25_gap = arms["bm25"]["mean"] - arms["parametric"]["mean"] if "bm25" in arms else None
    runtime = metrics["runtime"]
    adapted = runtime.get("parametric", {})
    full = runtime.get("full_context", {})
    latency_benefit = float(adapted.get("elapsed_seconds", 0)) < float(
        full.get("elapsed_seconds", 0)
    )
    token_benefit = int(adapted.get("completion_tokens", 0)) < int(full.get("completion_tokens", 0))
    passed = (
        gap <= 0.05
        and (latency_benefit or token_benefit)
        and not metrics["unknown_oos"]["parametric_worsening_vs_base"]
    )
    return {
        "mean_gap_below_full_context": gap,
        "required_max_gap": 0.05,
        "bm25_mean_minus_parametric": bm25_gap,
        "latency_benefit_observed": latency_benefit,
        "output_token_benefit_observed": token_benefit,
        "passed": passed,
        "interpretation": "engineering screen only; not deployment authorization",
    }


def _generation_comparison(
    four: Mapping[str, Any],
    twenty_seven: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for arm in ("base", "full_context", "oracle"):
        left = four["acquisition_arms"][arm]["mean"]
        right = twenty_seven["acquisition_arms"][arm]["mean"]
        result[arm] = {
            "mean_4b": left,
            "mean_27b": right,
            "delta_27b_minus_4b": right - left,
        }
    result["improved"] = result["base"]["delta_27b_minus_4b"] > 0
    result["interpretation"] = (
        "Generator-stack comparison only; not evidence of parametric acquisition."
    )
    return result


def _validate_matched_verifier(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    for key in (
        "model_id",
        "model_revision",
        "precision",
        "rubric_hash",
        "generation_config_hash",
        "canonical_evidence_index_hash",
        "snapshot_identity",
        "verification_status",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"4B and 27B reports use different verifier policy: {key}")


def _validate_experiment_report(
    report: Mapping[str, Any],
    general: Mapping[str, Any],
    profile: ModelProfile,
    expected_arms: tuple[str, ...],
) -> dict[str, Any]:
    chain = report.get("artifact_chain")
    if not isinstance(chain, dict):
        raise ValueError("Gemma advisory is missing its artifact chain")
    benchmark_path = Path(str(chain.get("benchmark_path", "")))
    if not benchmark_path.is_file() or _file_hash(benchmark_path) != chain.get("benchmark_sha256"):
        raise ValueError("Gemma advisory benchmark path/hash binding is invalid")
    benchmark = _load(benchmark_path)
    config = benchmark.get("config")
    tokenizer = benchmark.get("tokenizer")
    if not isinstance(config, dict) or not isinstance(tokenizer, dict):
        raise ValueError("Benchmark is missing generator configuration or tokenizer identity")
    if (
        benchmark.get("model") != profile.model_id
        or tokenizer.get("model_name") != profile.model_id
        or tokenizer.get("revision") != profile.revision
        or tuple(config.get("suites", ())) != ("acquisition", "unknown_oos")
        or tuple(config.get("arms", ())) != expected_arms
        or config.get("max_output_tokens") != MODEL_UPGRADE_CONFIG.generator_max_output_tokens
        or config.get("thinking_enabled") is not False
        or config.get("temperature") != MODEL_UPGRADE_CONFIG.generator_temperature
        or config.get("generation_policy_version") != "greedy-non-thinking/v1"
        or not str(config.get("experiment_id", "")).startswith(
            f"model-upgrade-v1-{'27b' if profile is QWEN_27B_PROFILE else '4b'}-seed42-"
        )
    ):
        raise ValueError("Benchmark does not match the frozen model-upgrade generator contract")
    if sha256_json(config) != benchmark.get("config_hash"):
        raise ValueError("Benchmark generator configuration hash is invalid")
    if chain.get("generator_config_hash") != benchmark.get("config_hash"):
        raise ValueError("Gemma advisory does not bind the benchmark generator configuration")

    adapter_path = Path(str(config.get("parametric_adapter_path", "")))
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    if (
        not adapter_file.is_file()
        or _file_hash(adapter_file) != config.get("parametric_adapter_hash")
        or not adapter_config_path.is_file()
    ):
        raise ValueError("Benchmark adapter path/hash binding is invalid")
    adapter_config = _load(adapter_config_path)
    lora = adapter_config.get("lora_parameters")
    if (
        adapter_config.get("experiment_profile") != MODEL_UPGRADE_PROFILE
        or adapter_config.get("execution_revision") != MODEL_UPGRADE_CONFIG.execution_revision
        or adapter_config.get("governed_model_id") != profile.model_id
        or adapter_config.get("governed_model_revision") != profile.revision
        or adapter_config.get("seed") != MODEL_UPGRADE_CONFIG.seed
        or adapter_config.get("iters") != 768
        or adapter_config.get("expected_optimizer_updates") != 96
        or not isinstance(lora, dict)
        or lora.get("rank") != MODEL_UPGRADE_CONFIG.rank
        or lora.get("scale") != MODEL_UPGRADE_CONFIG.scale
        or lora.get("dropout") != MODEL_UPGRADE_CONFIG.dropout
        or tuple(lora.get("keys", ())) != profile.target_modules
    ):
        raise ValueError("Benchmark adapter does not match the frozen acquisition contract")
    train_path = Path(str(adapter_config.get("data", ""))) / "train.jsonl"
    if not train_path.is_file():
        raise ValueError("Benchmark adapter training data is unavailable")

    raw_rows = benchmark.get("results")
    attempts = report.get("attempts")
    if not isinstance(raw_rows, list) or not isinstance(attempts, list):
        raise ValueError("Benchmark or advisory rows are missing")
    raw_keys = [(str(row.get("question_id")), str(row.get("arm"))) for row in raw_rows]
    attempt_keys = [(str(row.get("question_id")), str(row.get("arm"))) for row in attempts]
    if (
        len(raw_keys) != len(set(raw_keys))
        or len(attempt_keys) != len(set(attempt_keys))
        or set(raw_keys) != set(attempt_keys)
    ):
        raise ValueError("Benchmark and advisory question/arm matrices do not match")
    for suite, expected_questions in (("acquisition", 32), ("unknown_oos", 14)):
        suite_rows = [row for row in raw_rows if row.get("suite") == suite]
        expected_rows = expected_questions * len(expected_arms)
        if len(suite_rows) != expected_rows:
            raise ValueError(
                f"Benchmark {suite} matrix has {len(suite_rows)} rows; expected {expected_rows}"
            )

    if (
        general.get("status") != "diagnostic_non_promotable"
        or general.get("model_id") != profile.model_id
        or general.get("model_revision") != profile.revision
        or general.get("adapter_hash") != config.get("parametric_adapter_hash")
    ):
        raise ValueError("General diagnostic does not match the benchmark adapter")
    return {"dataset_train_sha256": _file_hash(train_path)}


def _benchmark_model(report: Mapping[str, Any]) -> str:
    benchmark = _load(Path(str(report["artifact_chain"]["benchmark_path"])))
    return str(benchmark.get("model", ""))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(report: Mapping[str, Any]) -> str:
    generation = report["three_questions"]["generation_improved"]
    acquisition = report["three_questions"]["adapter_added_company_knowledge_beyond_same_base"]
    practical = report["three_questions"]["adapter_competitive_with_direct_evidence"]
    four = report["models"]["4b_control"]
    twenty_seven = report["models"]["27b_candidate"]
    general = twenty_seven["general_capability_diagnostic"]
    arm_rows = []
    for arm, label in (
        ("base", "Base"),
        ("parametric", "Parametric adapter"),
        ("full_context", "Full context"),
        ("oracle", "Oracle context"),
        ("bm25", "Experimental BM25"),
    ):
        four_arm = four["acquisition_arms"].get(arm)
        twenty_seven_arm = twenty_seven["acquisition_arms"].get(arm)
        four_mean = f"{four_arm['mean']:.6f}" if four_arm else "—"
        four_correct = f"{four_arm['fully_correct']}/32" if four_arm else "—"
        twenty_seven_mean = f"{twenty_seven_arm['mean']:.6f}" if twenty_seven_arm else "—"
        twenty_seven_correct = (
            f"{twenty_seven_arm['fully_correct']}/32" if twenty_seven_arm else "—"
        )
        delta = (
            f"{twenty_seven_arm['mean'] - four_arm['mean']:+.6f}"
            if four_arm and twenty_seven_arm
            else "—"
        )
        arm_rows.append(
            f"| {label} | {four_mean} | {four_correct} | "
            f"{twenty_seven_mean} | {twenty_seven_correct} | {delta} |"
        )
    four_acquisition = four["acquisition"]
    twenty_seven_acquisition = twenty_seven["acquisition"]
    four_oos = four["unknown_oos"]["failure_counts"]
    twenty_seven_oos = twenty_seven["unknown_oos"]["failure_counts"]
    four_general = four["general_capability_diagnostic"]
    return "\n".join(
        [
            "# Model-upgrade exploratory v1 comparison",
            "",
            "**Advisory only — single local Gemma verifier; not human-approved.**",
            "",
            "## Acquisition comparison",
            "",
            "| Arm | 4B mean | 4B fully correct | 27B mean | 27B fully correct | 27B − 4B mean |",
            "|---|---:|---:|---:|---:|---:|",
            *arm_rows,
            "",
            "## Adapter effect and safeguards",
            "",
            "| Metric | 4B control | 27B candidate |",
            "|---|---:|---:|",
            f"| Mean adapter uplift over base | "
            f"{four_acquisition['mean_delta']:+.6f} | "
            f"{twenty_seven_acquisition['mean_delta']:+.6f} |",
            f"| Fully-correct delta over base | "
            f"{four_acquisition['fully_correct_delta']:+d} | "
            f"{twenty_seven_acquisition['fully_correct_delta']:+d} |",
            f"| Paired wins / losses / ties | "
            f"{four_acquisition['paired_wins']} / {four_acquisition['paired_losses']} / "
            f"{four_acquisition['paired_ties']} | "
            f"{twenty_seven_acquisition['paired_wins']} / "
            f"{twenty_seven_acquisition['paired_losses']} / "
            f"{twenty_seven_acquisition['paired_ties']} |",
            f"| Parametric generation failures | "
            f"{four['runtime']['parametric']['runtime_failures']} | "
            f"{twenty_seven['runtime']['parametric']['runtime_failures']} |",
            f"| Unknown/OOS failures, base → adapter | "
            f"{four_oos['base']} → {four_oos['parametric']} | "
            f"{twenty_seven_oos['base']} → {twenty_seven_oos['parametric']} |",
            f"| General diagnostic exact rate, base → adapter | "
            f"{four_general['base_exact_rate']:.1f} → "
            f"{four_general['parametric_exact_rate']:.1f} | "
            f"{general['base_exact_rate']:.1f} → "
            f"{general['parametric_exact_rate']:.1f} |",
            "",
            "## Three outcomes",
            "",
            f"- Did generation improve? **{'yes' if generation['improved'] else 'no'}**",
            f"- Did the 27B adapter improve over its own base? "
            f"Mean delta `{acquisition['mean_delta']:.3f}`, fully-correct delta "
            f"`{acquisition['fully_correct_delta']}`.",
            f"- Did acquisition become competitive with direct evidence? "
            f"**{'yes' if practical['passed'] else 'no'}** "
            f"(full-context gap `{practical['mean_gap_below_full_context']:.3f}`).",
            f"- Small general-capability diagnostic worsening observed: "
            f"**{'yes' if general['observed_worsening'] else 'no'}** "
            f"(exact-rate delta `{general['delta']:.3f}`).",
            "",
            "## Continuation",
            "",
            f"Decision: **{report['continuation']['decision']}**",
            "",
            "The stronger-base result and the acquisition-uplift result are reported "
            "separately. Passing would authorize only one matched seed-43 repeat.",
            "",
        ]
    )
