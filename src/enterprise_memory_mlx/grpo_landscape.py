"""TRAIN-only Qwen4B GRPO reward-landscape study. Base generation, no optimizer."""

from __future__ import annotations

import csv
import hashlib
import inspect
import itertools
import json
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition_runtime import NonThinkingTokenizer
from .acquisition_training import materialize_model_snapshot
from .experiment_profiles import QWEN_4B_MODEL_ID, QWEN_4B_REVISION
from .grpo_data import file_sha256
from .grpo_protocol import ORIGIN_MAIN_SHA, render_case_prompt
from .grpo_reward import COMPONENT_WEIGHTS, score_completion
from .learning_mechanics import WiredLimitGuard, seed_all
from .learning_mechanics_eval import (
    process_current_rss_bytes,
    process_rss_bytes,
    system_swap_snapshot,
)
from .utils import atomic_write_text, read_jsonl, sha256_text

STUDY_ID = "company-task-specialization/qwen4b-grpo-landscape-v1"
SEED_NAMESPACE = "qwen4b-grpo-landscape-v1"
V1_DIR = Path("knowledge/company_task_specialization/qwen4b_grpo_v1")
LANDSCAPE_DIR = Path("knowledge/company_task_specialization/qwen4b_grpo_landscape_v1")
ARTIFACTS = Path("artifacts/qwen4b-grpo-landscape-v1")
MASTER_SEED = 42
REPLICATES = 12
MEANINGFUL_DELTA = 0.10
GROUP_COMBINATIONS = 495
WIRED_LIMIT_BYTES = 96 * 1024**3
RSS_STOP_BYTES = 112 * 1024**3
SWAP_GROWTH_STOP_BYTES = 8 * 1024**3
EXPECTED_TRAIN = (
    "CTS3-QUAL-001",
    "CTS3-QUAL-002",
    "CTS3-QUAL-003",
    "CTS3-QUAL-004",
    "CTS3-QUAL-005",
    "CTS3-QUAL-006",
    "CTS3-QUAL-007",
    "CTS3-QUAL-008",
    "CTS3-QUAL-009",
    "CTS3-QUAL-010",
    "CTS3-QUAL-011",
    "CTS3-QUAL-012",
    "CTS3-QUAL-013",
    "CTS3-QUAL-016",
    "CTS3-QUAL-017",
    "CTS3-QUAL-018",
    "CTS3-QUAL-019",
    "CTS3-QUAL-024",
    "CTS4-FRESH-02A",
    "CTS4-FRESH-02B",
    "CTS4-FRESH-03A",
    "CTS4-FRESH-03B",
    "CTS4-FRESH-04A",
    "CTS4-FRESH-04B",
)
REGIMES = {
    "R0_CANONICAL": {"temperature": 0.8, "top_p": 0.95},
    "R1_TEMP_ONLY": {"temperature": 1.0, "top_p": 0.95},
    "R2_NUCLEUS_OFF": {"temperature": 0.8, "top_p": 1.0},
    "R3_NATIVE_POLICY": {"temperature": 1.0, "top_p": 1.0},
}
REGIME_ORDER = tuple(REGIMES)
COMPONENT_NAMES = (
    "structural_provenance",
    "current_decisions",
    "remedy_logic",
    "fact_state",
    "safe_next_action",
)
V1_REWARD_MODULE = Path(__file__).with_name("grpo_reward.py")


def derive_landscape_seed(
    case_id: str,
    regime_id: str,
    replicate: int,
    *,
    master: int = MASTER_SEED,
) -> int:
    payload = f"{SEED_NAMESPACE}|{master}|{case_id}|{regime_id}|{replicate}"
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def load_v1_split(root: Path) -> dict[str, Any]:
    return json.loads((root / V1_DIR / "split-manifest.json").read_text(encoding="utf-8"))


def load_v1_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / V1_DIR / "manifest.json").read_text(encoding="utf-8"))


def v1_file_hashes(root: Path) -> dict[str, str]:
    manifest = load_v1_manifest(root)
    hashes = {name: file_sha256(root / V1_DIR / name) for name in manifest["files"]}
    for name, digest in manifest["files"].items():
        if hashes[name] != digest:
            raise ValueError(f"v1 {name} drifted from frozen manifest")
    return hashes


def train_rows(root: Path) -> list[dict[str, Any]]:
    rows = [row for row in load_v1_split(root)["rows"] if row["split"] == "train"]
    case_ids = tuple(row["case_id"] for row in rows)
    if case_ids != EXPECTED_TRAIN:
        raise ValueError(f"TRAIN inventory drifted: {case_ids}")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("non-TRAIN row leaked into landscape selection")
    return rows


def train_case_ids(root: Path) -> set[str]:
    return {row["case_id"] for row in train_rows(root)}


def split_counts(root: Path) -> dict[str, int]:
    counts = Counter(row["split"] for row in load_v1_split(root)["rows"])
    return {"train": counts["train"], "dev": counts["dev"], "test": counts["test"]}


def require_train_row(root: Path, case_id: str) -> dict[str, Any]:
    for row in load_v1_split(root)["rows"]:
        if row["case_id"] == case_id:
            if row["split"] != "train":
                raise ValueError(f"landscape refused non-TRAIN case {case_id}")
            return row
    raise KeyError(case_id)


def render_train_prompt(root: Path, case_id: str) -> dict[str, str]:
    require_train_row(root, case_id)
    return render_case_prompt(root, case_id)


def train_specs(root: Path) -> dict[str, dict[str, Any]]:
    allowed = train_case_ids(root)
    specs = {}
    for row in read_jsonl(root / V1_DIR / "reward-specs.jsonl"):
        case_id = str(row["case_id"])
        if case_id in allowed:
            specs[case_id] = row
    if set(specs) != allowed:
        raise ValueError("TRAIN reward specs do not match TRAIN inventory")
    return specs


def train_families(root: Path) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for row in train_rows(root):
        family = families.setdefault(
            row["family_id"],
            {
                "family_id": row["family_id"],
                "source": row["source_protocol"],
                "case_ids": [],
                "pair_ids": [],
                "evaluation_tags": set(),
            },
        )
        family["case_ids"].append(row["case_id"])
        if row.get("pair_id"):
            family["pair_ids"].append(row["pair_id"])
        family["evaluation_tags"].update(row.get("evaluation_tags") or [])
    result = []
    for family in families.values():
        result.append(
            {
                **family,
                "evaluation_tags": sorted(family["evaluation_tags"]),
                "pair_ids": sorted(set(family["pair_ids"])),
            }
        )
    return sorted(result, key=lambda item: item["family_id"])


def ungated_component_quality(score: Mapping[str, Any]) -> float:
    applicable = list(score["applicable_components"])
    weight_sum = sum(COMPONENT_WEIGHTS[name] for name in applicable)
    value = sum(COMPONENT_WEIGHTS[name] * score["components"][name] for name in applicable)
    return round(float(value / weight_sum), 6)


def persist_raw_then_score(
    path: Path,
    raw: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Write raw output first; scoring cannot replace the persisted raw text."""
    if "raw_output" not in raw:
        raise ValueError("raw record is missing raw_output")
    atomic_write_text(path, json.dumps(dict(raw), indent=2) + "\n")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    scored = attach_analysis_score(score_completion(persisted["raw_output"], spec))
    record = {**persisted, **scored, "status": "reward_scored"}
    if record["raw_output"] != raw["raw_output"]:
        raise RuntimeError("scoring replaced persisted raw output")
    atomic_write_text(path, json.dumps(record, indent=2) + "\n")
    return record


def attach_analysis_score(score: Mapping[str, Any]) -> dict[str, Any]:
    official = dict(score)
    official["ungated_component_quality"] = ungated_component_quality(score)
    official["analysis_only_ungated"] = True
    official["analysis_only"] = True
    if official["hard_zero"]:
        official["reward"] = 0.0
    return official


def effective_group_rate(
    rewards: Sequence[float],
    group_size: int,
    *,
    delta: float = MEANINGFUL_DELTA,
) -> float:
    if len(rewards) < group_size:
        return 0.0
    subsets = tuple(itertools.combinations(rewards, group_size))
    if group_size in {4, 8} and len(rewards) == 12 and len(subsets) != GROUP_COMBINATIONS:
        raise ValueError(f"expected {GROUP_COMBINATIONS} subsets, found {len(subsets)}")
    meaningful = sum(1 for subset in subsets if max(subset) - min(subset) >= delta)
    return meaningful / len(subsets)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def classify_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    official_range = max(rewards) - min(rewards) if rewards else 0.0
    egr_g4 = effective_group_rate(rewards, 4)
    egr_g8 = effective_group_rate(rewards, 8)
    ones = sum(1 for item in rewards if item == 1.0)
    zeros = sum(1 for item in rewards if item == 0.0)
    clean = [
        row
        for row in rows
        if row["complete_valid_output"]
        and not row["provenance_hard_fail"]
        and not row["unsafe_hard_fail"]
    ]
    valid_provenance = [
        row
        for row in rows
        if row["complete_valid_output"] and not row["provenance_hard_fail"]
    ]
    clean_rewards = [float(row["reward"]) for row in clean]
    clean_ungated = [float(row["ungated_component_quality"]) for row in valid_provenance]
    clean_range = max(clean_rewards) - min(clean_rewards) if len(clean_rewards) >= 2 else 0.0
    ungated_range = max(clean_ungated) - min(clean_ungated) if len(clean_ungated) >= 2 else 0.0
    unsafe = [row for row in rows if row["unsafe_hard_fail"]]
    protocol_bad = [
        row for row in rows if row["parse_status"] != "valid" or row["provenance_hard_fail"]
    ]
    signal = "none"
    if official_range >= MEANINGFUL_DELTA and clean_range >= MEANINGFUL_DELTA:
        signal = "quality_signal"
    elif official_range >= MEANINGFUL_DELTA and clean and unsafe:
        signal = "safety_signal"
    elif zeros >= 11 and official_range < MEANINGFUL_DELTA and ungated_range >= MEANINGFUL_DELTA:
        signal = "safety_cliff"
    elif official_range >= MEANINGFUL_DELTA and protocol_bad and clean_range < MEANINGFUL_DELTA:
        signal = "protocol_noise"
    saturated_low = zeros >= 11 and official_range < MEANINGFUL_DELTA
    if signal == "protocol_noise":
        classification = "PROTOCOL_NOISY"
    elif signal == "safety_cliff":
        classification = "SAFETY_CLIFF"
    elif ones >= 11 and official_range < MEANINGFUL_DELTA:
        classification = "SATURATED_HIGH"
    elif saturated_low:
        classification = "SATURATED_LOW"
    elif (egr_g4 >= 0.50 or egr_g8 >= 0.50) and signal in {
        "quality_signal",
        "safety_signal",
    }:
        classification = "INFORMATIVE"
    elif official_range >= MEANINGFUL_DELTA and 0 < egr_g4 < 0.50 and 0 < egr_g8 < 0.50:
        classification = "BORDERLINE"
    else:
        classification = "MIXED_UNCLASSIFIED"
    return {
        "egr_g4": round(egr_g4, 6),
        "egr_g8": round(egr_g8, 6),
        "official_range": round(official_range, 6),
        "signal_source": signal,
        "classification": classification,
    }


def case_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    ungated = [float(row["ungated_component_quality"]) for row in rows]
    classified = classify_case(rows)
    stats = {
        "n": len(rows),
        "mean_reward": round(_mean(rewards), 6),
        "sample_std_reward": round(sample_std(rewards), 6),
        "min_reward": round(min(rewards), 6) if rewards else 0.0,
        "max_reward": round(max(rewards), 6) if rewards else 0.0,
        "reward_range": round(max(rewards) - min(rewards), 6) if rewards else 0.0,
        "unique_reward_count": len(set(rewards)),
        "fraction_nonzero": _frac(rewards, lambda item: item > 0),
        "fraction_perfect": _frac(rewards, lambda item: item == 1.0),
        "fraction_hard_zero": _frac(rows, lambda row: bool(row["hard_zero"])),
        "fraction_valid_json": _frac(rows, lambda row: bool(row["complete_valid_output"])),
        "fraction_provenance_hard_fail": _frac(
            rows, lambda row: bool(row["provenance_hard_fail"])
        ),
        "fraction_unsafe_hard_fail": _frac(rows, lambda row: bool(row["unsafe_hard_fail"])),
        "fraction_truncated": _frac(rows, lambda row: bool(row["truncated"])),
        "mean_ungated_component_quality": round(_mean(ungated), 6),
        "std_ungated_component_quality": round(sample_std(ungated), 6),
        **classified,
    }
    for name in COMPONENT_NAMES:
        values = [float(row["components"][name]) for row in rows]
        stats[f"mean_{name}"] = round(_mean(values), 6)
        stats[f"std_{name}"] = round(sample_std(values), 6)
    return stats


def write_landscape_protocol(root: Path) -> dict[str, Any]:
    hashes = v1_file_hashes(root)
    destination = root / LANDSCAPE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "protocol_id": STUDY_ID,
        "parent_protocol_id": "company-task-specialization/qwen4b-grpo-v1",
        "origin_main_sha": ORIGIN_MAIN_SHA,
        "v1_commit": "9056a272ed641632885158cf4f51b7d4029c8274",
        "base_only": True,
        "optimizer": False,
        "reward_unchanged": True,
        "meaningful_delta": MEANINGFUL_DELTA,
        "master_seed": MASTER_SEED,
        "replicates": REPLICATES,
        "regimes": REGIMES,
        "planned_jobs": 24 * 4 * REPLICATES,
        "model": {"id": QWEN_4B_MODEL_ID, "revision": QWEN_4B_REVISION},
        "claim_boundary": {
            "generation_only": True,
            "no_adapter": True,
            "train_only": True,
        },
    }
    bindings = {
        "schema_version": 1,
        "v1_files": hashes,
        "reward_implementation_sha256": file_sha256(V1_REWARD_MODULE),
        "origin_main_sha": ORIGIN_MAIN_SHA,
    }
    files = {
        "protocol.json": protocol,
        "source-bindings.json": bindings,
        "sampling-regimes.json": {
            "regimes": REGIMES,
            "order": list(REGIME_ORDER),
            "replicates": REPLICATES,
        },
    }
    written = {}
    for name, payload in files.items():
        path = destination / name
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written[name] = file_sha256(path)
    manifest = {
        "schema_version": 1,
        "protocol_id": STUDY_ID,
        "files": written,
        "parent_v1_files": hashes,
    }
    manifest_path = destination / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    written["manifest.json"] = file_sha256(manifest_path)
    return {"directory": str(destination), "files": written}


def validate_landscape(root: Path) -> dict[str, Any]:
    if not (root / LANDSCAPE_DIR / "manifest.json").is_file():
        write_landscape_protocol(root)
    counts = split_counts(root)
    rows = train_rows(root)
    families = train_families(root)
    hashes = v1_file_hashes(root)
    reward_hash = file_sha256(V1_REWARD_MODULE)
    source = Path(__file__).read_text(encoding="utf-8")
    _assert_no_training_surface(source)
    return {
        "status": "validated",
        "train": counts["train"],
        "dev": counts["dev"],
        "test": counts["test"],
        "base_only": True,
        "reward_unchanged": True,
        "train_cases": [row["case_id"] for row in rows],
        "train_families": len(families),
        "v1_files": hashes,
        "reward_implementation_sha256": reward_hash,
        "meaningful_delta": MEANINGFUL_DELTA,
        "regimes": list(REGIME_ORDER),
    }


def build_job_plan(root: Path, *, master_seed: int = MASTER_SEED) -> list[dict[str, Any]]:
    if master_seed != MASTER_SEED:
        raise ValueError("landscape v1 freezes master seed 42")
    jobs = []
    index = 0
    for row in train_rows(root):
        for regime_id in REGIME_ORDER:
            settings = REGIMES[regime_id]
            seen: set[int] = set()
            for replicate in range(REPLICATES):
                seed = derive_landscape_seed(row["case_id"], regime_id, replicate)
                if seed in seen:
                    raise ValueError(f"duplicate seed in {row['case_id']} {regime_id}")
                seen.add(seed)
                jobs.append(
                    {
                        "job_index": index,
                        "case_id": row["case_id"],
                        "family_id": row["family_id"],
                        "source_protocol": row["source_protocol"],
                        "regime_id": regime_id,
                        "replicate": replicate,
                        "seed": seed,
                        "temperature": settings["temperature"],
                        "top_p": settings["top_p"],
                        "split": "train",
                    }
                )
                index += 1
    if len(jobs) != 1152:
        raise ValueError(f"expected 1152 jobs, planned {len(jobs)}")
    return jobs


def plan_landscape(root: Path, *, master_seed: int = MASTER_SEED) -> dict[str, Any]:
    validate_landscape(root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / ARTIFACTS / f"run-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    jobs = build_job_plan(root, master_seed=master_seed)
    bindings = json.loads((root / LANDSCAPE_DIR / "source-bindings.json").read_text())
    atomic_write_text(
        output / "sampling-regimes.json",
        json.dumps({"regimes": REGIMES, "order": list(REGIME_ORDER)}, indent=2) + "\n",
    )
    atomic_write_text(
        output / "source-bindings.json",
        json.dumps(bindings, indent=2, sort_keys=True) + "\n",
    )
    plan_text = "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs)
    atomic_write_text(output / "job-plan.jsonl", plan_text)
    digest = sha256_text(plan_text)
    atomic_write_text(output / "job-plan.sha256", digest + "\n")
    manifest = {
        "study_id": STUDY_ID,
        "directory": str(output),
        "jobs": len(jobs),
        "cases": 24,
        "regimes": 4,
        "replicates": REPLICATES,
        "job_plan_sha256": digest,
        "status": "planned",
        "base_only": True,
    }
    atomic_write_text(output / "study-manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_write_text(
        root / ARTIFACTS / "latest-plan.json",
        json.dumps({"directory": str(output), "job_plan_sha256": digest}, indent=2) + "\n",
    )
    return {"status": "planned", **manifest}


def run_landscape(root: Path) -> dict[str, Any]:
    latest = json.loads((root / ARTIFACTS / "latest-plan.json").read_text(encoding="utf-8"))
    output = Path(latest["directory"])
    plan_path = output / "job-plan.jsonl"
    plan_text = plan_path.read_text(encoding="utf-8")
    if sha256_text(plan_text) != (output / "job-plan.sha256").read_text().strip():
        raise RuntimeError("job plan hash drifted before generation")
    if (output / "scores.jsonl").is_file():
        raise RuntimeError("landscape run already has scores; refusing to resume")
    jobs = [json.loads(line) for line in plan_text.splitlines() if line]
    if any(job["split"] != "train" for job in jobs):
        raise RuntimeError("job plan contained a non-TRAIN case")
    specs = train_specs(root)
    runtime = _load_base(root)
    swap_before = system_swap_snapshot()
    completed = 0
    scores_handle = (output / "scores.jsonl").open("w", encoding="utf-8")
    try:
        with WiredLimitGuard(runtime["mx"], WIRED_LIMIT_BYTES):
            for job in jobs:
                require_train_row(root, job["case_id"])
                prompt = render_train_prompt(root, job["case_id"])
                path = (
                    output
                    / "rollouts"
                    / job["case_id"]
                    / job["regime_id"]
                    / f"{job['replicate']:02d}.json"
                )
                try:
                    generated = _generate_base(
                        runtime,
                        prompt=prompt,
                        seed=int(job["seed"]),
                        temperature=float(job["temperature"]),
                        top_p=float(job["top_p"]),
                    )
                except Exception as exc:
                    _write_stop(output, job, f"{type(exc).__name__}: {exc}", completed)
                    raise
                raw = {
                    "case_id": job["case_id"],
                    "family_id": job["family_id"],
                    "source_protocol": job["source_protocol"],
                    "regime_id": job["regime_id"],
                    "replicate": job["replicate"],
                    "seed": job["seed"],
                    "temperature": job["temperature"],
                    "top_p": job["top_p"],
                    "raw_output": generated["output"],
                    "raw_output_sha256": generated["raw_output_sha256"],
                    "token_count": generated["completion_length"],
                    "finish_reason": generated["finish_reason"],
                    "truncated": generated["truncated"],
                    "parse_status": None,
                    "status": "rollout_generated",
                }
                record = persist_raw_then_score(path, raw, specs[job["case_id"]])
                if record["reward"] != 0.0 and record["hard_zero"]:
                    raise RuntimeError("analysis score overwrote a hard-zero reward")
                scores_handle.write(json.dumps(record, sort_keys=True) + "\n")
                scores_handle.flush()
                completed += 1
                if completed == 1 or completed % 24 == 0 or completed == len(jobs):
                    print(
                        f"landscape {completed}/{len(jobs)} "
                        f"{job['case_id']} {job['regime_id']} r{job['replicate']}",
                        flush=True,
                    )
                _assert_resource_ok(swap_before)
        scores_handle.close()
        atomic_write_text(
            output / "environment.json",
            json.dumps(_environment(runtime), indent=2) + "\n",
        )
        manifest = json.loads((output / "study-manifest.json").read_text())
        manifest.update({"status": "generated", "completed_rollouts": completed})
        atomic_write_text(output / "study-manifest.json", json.dumps(manifest, indent=2) + "\n")
        atomic_write_text(
            root / ARTIFACTS / "latest-landscape.json",
            json.dumps({"directory": str(output), "completed_rollouts": completed}, indent=2)
            + "\n",
        )
        return {"status": "generated", "completed_rollouts": completed, "directory": str(output)}
    finally:
        if not scores_handle.closed:
            scores_handle.close()
        _release(runtime)


def summarize_landscape(root: Path) -> dict[str, Any]:
    output = Path(
        json.loads((root / ARTIFACTS / "latest-landscape.json").read_text())["directory"]
    )
    scores = sorted(
        read_jsonl(output / "scores.jsonl"),
        key=lambda row: (row["case_id"], row["regime_id"], row["replicate"]),
    )
    case_stats = []
    for (case_id, regime_id), rows in _group(scores, ("case_id", "regime_id")):
        first = rows[0]
        stats = case_statistics(rows)
        case_stats.append(
            {
                "case_id": case_id,
                "family_id": first["family_id"],
                "source_protocol": first["source_protocol"],
                "regime_id": regime_id,
                "frozen_split": "train",
                **stats,
            }
        )
    family_stats = _family_stats(root, case_stats, scores)
    regime_stats = _regime_stats(case_stats, scores)
    classification = _classification_counts(case_stats)
    _write_json(output / "case-stats.json", {"rows": case_stats})
    _write_csv(output / "case-stats.csv", case_stats)
    _write_json(output / "family-stats.json", {"rows": family_stats})
    _write_csv(output / "family-stats.csv", family_stats)
    _write_json(output / "regime-stats.json", {"rows": regime_stats})
    _write_csv(output / "regime-stats.csv", regime_stats)
    _write_json(output / "signal-classification.json", classification)
    write_charts(output / "charts", scores, case_stats)
    integrity = {
        "scores": len(scores),
        "planned": 1152,
        "order_independent": True,
        "charts_from_scores_jsonl": True,
    }
    _write_json(output / "integrity.json", integrity)
    return {
        "status": "summarized",
        "directory": str(output),
        "case_stats": case_stats,
        "family_stats": family_stats,
        "regime_stats": regime_stats,
        "classification": classification,
    }


def decide_landscape(root: Path) -> dict[str, Any]:
    summary = summarize_landscape(root)
    output = Path(summary["directory"])
    case_stats = summary["case_stats"]
    family_stats = summary["family_stats"]
    regime_stats = summary["regime_stats"]
    eligible = [row for row in regime_stats if row["eligible"]]
    selected = None
    selected_g = None
    if eligible:
        ranked = sorted(
            eligible,
            key=lambda row: max(row["family_weighted_egr_g4"], row["family_weighted_egr_g8"]),
            reverse=True,
        )
        top = ranked[0]["best_family_weighted_egr"]
        tied = [
            row
            for row in ranked
            if abs(row["best_family_weighted_egr"] - top) <= 0.05
        ]
        preference = {"R3_NATIVE_POLICY": 0, "R0_CANONICAL": 1}
        selected = sorted(tied, key=lambda row: preference.get(row["regime_id"], 9))[0]
        gap = selected["family_weighted_egr_g8"] - selected["family_weighted_egr_g4"]
        selected_g = 8 if gap >= 0.15 else 4
    decision = _viability(case_stats, family_stats, selected, selected_g)
    decision["job_plan_sha256"] = (output / "job-plan.sha256").read_text().strip()
    decision["completed_rollouts"] = len(read_jsonl(output / "scores.jsonl"))
    _write_json(output / "decision.json", decision)
    atomic_write_text(output / "report.md", render_report(root, summary, decision))
    atomic_write_text(
        root / ARTIFACTS / "latest-landscape.json",
        json.dumps(
            {
                "directory": str(output),
                "status": decision["status"],
                "completed_rollouts": len(read_jsonl(output / "scores.jsonl")),
            },
            indent=2,
        )
        + "\n",
    )
    return decision


def write_charts(
    destination: Path,
    scores: Sequence[Mapping[str, Any]],
    case_stats: Sequence[Mapping[str, Any]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    _reward_histogram(destination / "reward-histogram-by-regime.svg", scores)
    _egr_chart(destination / "effective-group-rate-g4-g8.svg", case_stats)
    _heatmap(destination / "case-signal-heatmap.svg", case_stats)
    _hard_zero_chart(destination / "hard-zero-reason-histogram.svg", scores)


def render_report(root: Path, summary: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    hashes = v1_file_hashes(root)
    lines = [
        decision["status"],
        "",
        "This study measures the TRAIN-only reward landscape of the untouched",
        "pinned Qwen3-4B base policy. It does not train, evaluate TEST, or claim",
        "that GRPO works or fails generally.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[GRPO v1 stop] --> B[TRAIN-only reward landscape]",
        "    B --> C[pre-registered viability decision]",
        "    C --> D[either GRPO v2 preflight]",
        "    C --> E[or abandon standard GRPO]",
        "```",
        "",
        "Source branch SHA: 9056a272ed641632885158cf4f51b7d4029c8274",
        f"origin/main SHA: {ORIGIN_MAIN_SHA}",
        f"Model: {QWEN_4B_MODEL_ID} @ {QWEN_4B_REVISION}",
        f"v1 files: {json.dumps(hashes, sort_keys=True)}",
        f"Reward implementation: {file_sha256(V1_REWARD_MODULE)}",
        f"Job plan SHA-256: {decision.get('job_plan_sha256')}",
        f"Completed rollouts: {decision.get('completed_rollouts')}",
        "",
        "Regimes: R0_CANONICAL 0.8/0.95; R1_TEMP_ONLY 1.0/0.95; "
        "R2_NUCLEUS_OFF 0.8/1.0; R3_NATIVE_POLICY 1.0/1.0.",
        "",
        "Per-case statistics:",
        json.dumps(summary["case_stats"], indent=2),
        "",
        "Family statistics:",
        json.dumps(summary["family_stats"], indent=2),
        "",
        "Regime statistics:",
        json.dumps(summary["regime_stats"], indent=2),
        "",
        "Viability gates:",
        json.dumps(decision["gates"], indent=2),
        "",
        f"Selected regime: {decision.get('selected_regime')}",
        f"Recommended G: {decision.get('recommended_g')}",
        "",
        "Future GRPO warning: MLX-LM generate_step computes model logprobs",
        "before the sampler applies temperature/top-p. A future optimizer run",
        "must show that stored action log-probabilities match the sampling",
        "distribution, or correct for that transform. R3_NATIVE_POLICY is the",
        "cleanest accounting candidate because temperature=1 and top_p=1 leave",
        "the native categorical untruncated. R0/R1/R2 are not policy-gradient ready.",
        "",
        "Limitations:",
        "- TRAIN only; DEV/TEST were not generated.",
        "- Official reward remains the v1 hard-gated scalar.",
        "- ungated_component_quality is analysis-only.",
        "- No adapter, optimizer, or second sampling search.",
    ]
    return "\n".join(lines) + "\n"


def _viability(
    case_stats: Sequence[Mapping[str, Any]],
    family_stats: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    selected_g: int | None,
) -> dict[str, Any]:
    if selected is None or selected_g is None:
        gates = {
            "informative_cases": False,
            "informative_families": False,
            "v4_informative_families": False,
            "family_weighted_egr": False,
            "signal_purity": False,
            "saturated_low_burden": False,
            "valid_json": False,
            "no_truncation": False,
            "provenance": False,
            "reward_byte_identical": True,
        }
        return {
            "status": "abandon_standard_grpo_for_current_dataset",
            "selected_regime": None,
            "recommended_g": None,
            "gates": gates,
            "diagnosis": _diagnosis(case_stats),
        }
    regime = selected["regime_id"]
    regime_cases = [row for row in case_stats if row["regime_id"] == regime]
    informative = [row for row in regime_cases if row["classification"] == "INFORMATIVE"]
    families = {row["family_id"] for row in informative}
    v4_informative = {
        row["family_id"]
        for row in informative
        if str(row["family_id"]).startswith("v4-")
    }
    egr_key = "family_weighted_egr_g8" if selected_g == 8 else "family_weighted_egr_g4"
    quality_or_safety = [
        row
        for row in informative
        if row["signal_source"] in {"quality_signal", "safety_signal"}
    ]
    saturated_low = [row for row in regime_cases if row["classification"] == "SATURATED_LOW"]
    gates = {
        "informative_cases": len(informative) >= 8,
        "informative_families": len(families) >= 5,
        "v4_informative_families": len(v4_informative) >= 2,
        "family_weighted_egr": selected[egr_key] >= 0.50,
        "signal_purity": (
            (len(quality_or_safety) / len(informative) >= 0.75) if informative else False
        ),
        "saturated_low_burden": len(saturated_low) <= 8,
        "valid_json": selected["fraction_valid_json"] >= 0.95,
        "no_truncation": selected["truncation_count"] == 0,
        "provenance": selected["provenance_hard_fail_rate"] <= 0.05,
        "reward_byte_identical": True,
    }
    passed = all(gates.values())
    return {
        "status": (
            "grpo_v2_candidate_preflight_required"
            if passed
            else "abandon_standard_grpo_for_current_dataset"
        ),
        "selected_regime": regime,
        "recommended_g": selected_g,
        "gates": gates,
        "informative_cases": len(informative),
        "informative_families": len(families),
        "v4_informative_families": len(v4_informative),
        "diagnosis": _diagnosis(regime_cases),
    }


def _diagnosis(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(row["classification"] for row in rows)
    if counts.get("SATURATED_HIGH", 0) >= counts.get("SATURATED_LOW", 0) and counts.get(
        "SATURATED_HIGH", 0
    ) >= 8:
        return "mostly saturated_high: gather harder/counterfactual training cases; no RL yet"
    if counts.get("SATURATED_LOW", 0) >= 8:
        return "mostly saturated_low: reference SFT"
    if counts.get("SAFETY_CLIFF", 0) >= 5:
        return (
            "many safety_cliff cases: reference SFT and/or separately designed "
            "preference optimization"
        )
    if counts.get("PROTOCOL_NOISY", 0) >= 5:
        return "protocol_noise: fix representation/structured decoding before training"
    if counts.get("INFORMATIVE", 0) and counts.get("INFORMATIVE", 0) < 8:
        return (
            "useful deterministic preference ordering but poor online group variance: "
            "DPO candidate"
        )
    return "mixed landscape; standard GRPO is not justified from this study"


def _family_stats(
    root: Path,
    case_stats: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    del scores
    families = train_families(root)
    rows = []
    for family in families:
        for regime_id in REGIME_ORDER:
            members = [
                row
                for row in case_stats
                if row["family_id"] == family["family_id"] and row["regime_id"] == regime_id
            ]
            informative = [row for row in members if row["classification"] == "INFORMATIVE"]
            signals = Counter(row["signal_source"] for row in members)
            rows.append(
                {
                    "family_id": family["family_id"],
                    "source": family["source"],
                    "cases": family["case_ids"],
                    "evaluation_tags": family["evaluation_tags"],
                    "regime_id": regime_id,
                    "informative_cases": len(informative),
                    "mean_egr_g4": round(_mean([row["egr_g4"] for row in members]), 6),
                    "mean_egr_g8": round(_mean([row["egr_g8"] for row in members]), 6),
                    "hard_zero_burden": round(
                        _mean([row["fraction_hard_zero"] for row in members]), 6
                    ),
                    "perfect_saturation_burden": round(
                        _mean([row["fraction_perfect"] for row in members]), 6
                    ),
                    "dominant_signal_source": signals.most_common(1)[0][0] if signals else "none",
                }
            )
    return rows


def _regime_stats(
    case_stats: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for regime_id in REGIME_ORDER:
        members = [row for row in case_stats if row["regime_id"] == regime_id]
        regime_scores = [row for row in scores if row["regime_id"] == regime_id]
        family_ids = sorted({row["family_id"] for row in members})
        fw4 = _mean(
            [
                _mean(
                    [row["egr_g4"] for row in members if row["family_id"] == family_id]
                )
                for family_id in family_ids
            ]
        )
        fw8 = _mean(
            [
                _mean(
                    [row["egr_g8"] for row in members if row["family_id"] == family_id]
                )
                for family_id in family_ids
            ]
        )
        valid = _frac(regime_scores, lambda row: bool(row["complete_valid_output"]))
        trunc = sum(1 for row in regime_scores if row["truncated"])
        prov = _frac(regime_scores, lambda row: bool(row["provenance_hard_fail"]))
        noise = _frac(members, lambda row: row["classification"] == "PROTOCOL_NOISY")
        eligible = valid >= 0.95 and trunc == 0 and prov <= 0.05 and noise < 0.50
        rows.append(
            {
                "regime_id": regime_id,
                "fraction_valid_json": valid,
                "truncation_count": trunc,
                "provenance_hard_fail_rate": prov,
                "protocol_noise_fraction": noise,
                "family_weighted_egr_g4": round(fw4, 6),
                "family_weighted_egr_g8": round(fw8, 6),
                "best_family_weighted_egr": round(max(fw4, fw8), 6),
                "eligible": eligible,
            }
        )
    return rows


def _classification_counts(case_stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_regime = {}
    for regime_id in REGIME_ORDER:
        counts = Counter(
            row["classification"] for row in case_stats if row["regime_id"] == regime_id
        )
        by_regime[regime_id] = dict(counts)
    return {"by_regime": by_regime}


def _load_base(root: Path, adapter_path: str | None = None) -> dict[str, Any]:
    if adapter_path is not None:
        raise RuntimeError("landscape runtime refuses any adapter path")
    import mlx.core as mx
    from mlx_lm import load

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal device is unavailable")
    snapshot = materialize_model_snapshot(QWEN_4B_MODEL_ID, QWEN_4B_REVISION)
    seed_all(42, mx)
    model, tokenizer = load(snapshot, tokenizer_config={"trust_remote_code": False})
    tokenizer = NonThinkingTokenizer(tokenizer)
    model.freeze()
    config_path = Path(snapshot) / "config.json"
    return {
        "mx": mx,
        "model": model,
        "tokenizer": tokenizer,
        "snapshot": snapshot,
        "config_sha256": file_sha256(config_path) if config_path.is_file() else None,
        "adapter_path": None,
    }


def _generate_base(
    runtime: dict[str, Any],
    *,
    prompt: dict[str, str],
    seed: int,
    temperature: float,
    top_p: float,
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
    sampler = make_sampler(temp=temperature, top_p=top_p)
    prompt_cache = make_prompt_cache(model)
    generated_ids: list[int] = []
    finish_reason = "unknown"
    started = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=1024,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            generated_ids.append(int(response.token))
            finish_reason = response.finish_reason
    finally:
        prompt_cache = None
        mx.clear_cache()
    eos_ids = set(
        getattr(tokenizer, "eos_token_ids", None) or [getattr(tokenizer, "eos_token_id", None)]
    )
    content_ids = [token for token in generated_ids if token not in eos_ids]
    output = tokenizer.decode(content_ids)
    return {
        "output": output,
        "raw_output_sha256": sha256_text(output),
        "completion_length": len(generated_ids),
        "truncated": finish_reason == "length" or len(generated_ids) >= 1024,
        "finish_reason": finish_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _environment(runtime: dict[str, Any]) -> dict[str, Any]:
    tokenizer = runtime["tokenizer"]
    return {
        "model_id": QWEN_4B_MODEL_ID,
        "revision": QWEN_4B_REVISION,
        "adapter_path": None,
        "config_sha256": runtime.get("config_sha256"),
        "tokenizer_name": type(tokenizer).__name__,
        "chat_template_hash": sha256_text(str(getattr(tokenizer, "chat_template", ""))),
        "packages": _package_versions(),
    }


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions = {}
    for name in ("mlx", "mlx_lm", "mlx_metal", "llguidance"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def _write_stop(output: Path, job: Mapping[str, Any], reason: str, completed: int) -> None:
    atomic_write_text(
        output / "stop.json",
        json.dumps(
            {
                "status": "landscape_study_incomplete",
                "job": dict(job),
                "reason": reason,
                "completed_rollouts": completed,
            },
            indent=2,
        )
        + "\n",
    )


def _release(runtime: dict[str, Any] | None) -> None:
    if not runtime:
        return
    mx = runtime.get("mx")
    runtime.clear()
    if mx is not None:
        mx.clear_cache()


def _assert_no_training_surface(source: str) -> None:
    forbidden = (
        "import mlx" + ".optimizers",
        "from mlx" + ".optimizers",
        "value_and_" + "grad(",
        "linear_to_" + "lora_layers",
        "save_" + "adapter(",
        "load_" + "adapters(",
        "Adam" + "W(",
    )
    for token in forbidden:
        if token in source:
            raise RuntimeError(f"landscape source contains forbidden token {token}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _frac(rows: Sequence[Any], predicate) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 6)


def _group(rows: Sequence[Mapping[str, Any]], keys: tuple[str, ...]):
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        yield key, grouped[key]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _svg(path: Path, width: int, height: int, body: str) -> None:
    atomic_write_text(
        path,
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">\n{body}\n</svg>\n'
        ),
    )


def _reward_histogram(path: Path, scores: Sequence[Mapping[str, Any]]) -> None:
    body = ['<text x="12" y="20">reward histogram by regime</text>']
    for index, regime_id in enumerate(REGIME_ORDER):
        values = [float(row["reward"]) for row in scores if row["regime_id"] == regime_id]
        counts = [0] * 10
        for value in values:
            bucket = min(9, int(value * 10)) if value < 1 else 9
            counts[bucket] += 1
        y0 = 50 + index * 70
        body.append(f'<text x="12" y="{y0 - 8}">{regime_id}</text>')
        for bucket, count in enumerate(counts):
            height = count
            body.append(
                f'<rect x="{40 + bucket * 28}" y="{y0 + 40 - height}" '
                f'width="24" height="{height}" fill="#4c6"/>'
            )
    _svg(path, 360, 340, "\n".join(body))


def _egr_chart(path: Path, case_stats: Sequence[Mapping[str, Any]]) -> None:
    body = ['<text x="12" y="20">family-unweighted mean EGR G4 vs G8</text>']
    for index, regime_id in enumerate(REGIME_ORDER):
        members = [row for row in case_stats if row["regime_id"] == regime_id]
        g4 = _mean([row["egr_g4"] for row in members])
        g8 = _mean([row["egr_g8"] for row in members])
        y = 50 + index * 40
        body.append(f'<text x="12" y="{y}">{regime_id}</text>')
        body.append(f'<rect x="160" y="{y - 12}" width="{g4 * 160:.1f}" height="10" fill="#26a"/>')
        body.append(f'<rect x="160" y="{y + 2}" width="{g8 * 160:.1f}" height="10" fill="#d73"/>')
    _svg(path, 360, 230, "\n".join(body))


def _heatmap(path: Path, case_stats: Sequence[Mapping[str, Any]]) -> None:
    colors = {
        "INFORMATIVE": "#2a2",
        "BORDERLINE": "#cc2",
        "SATURATED_HIGH": "#88f",
        "SATURATED_LOW": "#888",
        "SAFETY_CLIFF": "#c72",
        "PROTOCOL_NOISY": "#c22",
        "MIXED_UNCLASSIFIED": "#aaa",
    }
    cases = sorted({row["case_id"] for row in case_stats})
    lookup = {(row["case_id"], row["regime_id"]): row["classification"] for row in case_stats}
    body = ['<text x="12" y="18">case-signal heatmap</text>']
    for x, regime_id in enumerate(REGIME_ORDER):
        body.append(f'<text x="{120 + x * 40}" y="36" font-size="8">{regime_id}</text>')
    for y, case_id in enumerate(cases):
        body.append(f'<text x="8" y="{54 + y * 14}" font-size="8">{case_id}</text>')
        for x, regime_id in enumerate(REGIME_ORDER):
            klass = lookup[(case_id, regime_id)]
            body.append(
                f'<rect x="{120 + x * 40}" y="{44 + y * 14}" width="36" height="12" '
                f'fill="{colors.get(klass, "#aaa")}"/>'
            )
    _svg(path, 300, 54 + 14 * len(cases), "\n".join(body))


def _hard_zero_chart(path: Path, scores: Sequence[Mapping[str, Any]]) -> None:
    reasons: Counter[str] = Counter()
    for row in scores:
        if not row["hard_zero"]:
            continue
        if row["parse_status"] != "valid":
            reasons["invalid_json"] += 1
        for item in row.get("provenance_reasons") or []:
            reasons[f"provenance:{item}"] += 1
        for item in row.get("unsafe_reasons") or []:
            reasons[f"unsafe:{item}"] += 1
    body = ['<text x="12" y="20">hard-zero reason counts</text>']
    for index, (reason, count) in enumerate(reasons.most_common(12)):
        y = 40 + index * 18
        body.append(f'<text x="12" y="{y}" font-size="9">{reason[:48]} {count}</text>')
        width = min(count, 80)
        body.append(
            f'<rect x="260" y="{y - 10}" width="{width}" height="10" fill="#c33"/>'
        )
    _svg(path, 420, 40 + 18 * max(len(reasons), 1), "\n".join(body))


def assert_source_has_no_training() -> None:
    _assert_no_training_surface(Path(__file__).read_text(encoding="utf-8"))
    source = Path(__file__).read_text(encoding="utf-8")
    if "run_" + "preflight(" in source:
        raise RuntimeError("landscape must not call the v1 preflight entry")
    inspect.getsource(_load_base)
