from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.cli import build_parser
from enterprise_memory_mlx.grpo_data import file_sha256, load_parent_assets
from enterprise_memory_mlx.grpo_landscape import (
    EXPECTED_TRAIN,
    GROUP_COMBINATIONS,
    MEANINGFUL_DELTA,
    REGIME_ORDER,
    REGIMES,
    V1_REWARD_MODULE,
    assert_source_has_no_training,
    attach_analysis_score,
    build_job_plan,
    classify_case,
    derive_landscape_seed,
    effective_group_rate,
    render_train_prompt,
    require_train_row,
    train_families,
    train_rows,
    ungated_component_quality,
    validate_landscape,
    write_charts,
)
from enterprise_memory_mlx.grpo_reward import (
    compile_reward_spec,
    mutate_output,
    perfect_output,
    score_completion,
)


def _row(**overrides: object) -> dict:
    base = {
        "reward": 0.0,
        "hard_zero": False,
        "complete_valid_output": True,
        "provenance_hard_fail": False,
        "unsafe_hard_fail": False,
        "parse_status": "valid",
        "ungated_component_quality": 0.0,
        "components": {
            "structural_provenance": 1.0,
            "current_decisions": 0.0,
            "remedy_logic": 0.0,
            "fact_state": 1.0,
            "safe_next_action": 0.0,
        },
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
    }
    base.update(overrides)
    return base


def test_train_inventory_is_exactly_twenty_four(project_root: Path) -> None:
    rows = train_rows(project_root)
    assert [row["case_id"] for row in rows] == list(EXPECTED_TRAIN)
    assert len(rows) == 24


def test_landscape_refuses_dev_and_test_render(project_root: Path) -> None:
    split = json.loads(
        (project_root / "knowledge/company_task_specialization/qwen4b_grpo_v1/split-manifest.json")
        .read_text(encoding="utf-8")
    )
    blocked = [row["case_id"] for row in split["rows"] if row["split"] != "train"]
    assert len(blocked) == 16
    for case_id in blocked:
        with pytest.raises(ValueError, match="non-TRAIN"):
            require_train_row(project_root, case_id)
        with pytest.raises(ValueError, match="non-TRAIN"):
            render_train_prompt(project_root, case_id)


def test_v4_train_pairs_remain_together(project_root: Path) -> None:
    rows = train_rows(project_root)
    pairs = {}
    for row in rows:
        if row.get("pair_id"):
            pairs.setdefault(row["pair_id"], set()).add(row["split"])
    assert pairs == {
        "CTS4-PAIR-02": {"train"},
        "CTS4-PAIR-03": {"train"},
        "CTS4-PAIR-04": {"train"},
    }


def test_four_regimes_are_exact() -> None:
    assert list(REGIME_ORDER) == [
        "R0_CANONICAL",
        "R1_TEMP_ONLY",
        "R2_NUCLEUS_OFF",
        "R3_NATIVE_POLICY",
    ]
    assert REGIMES == {
        "R0_CANONICAL": {"temperature": 0.8, "top_p": 0.95},
        "R1_TEMP_ONLY": {"temperature": 1.0, "top_p": 0.95},
        "R2_NUCLEUS_OFF": {"temperature": 0.8, "top_p": 1.0},
        "R3_NATIVE_POLICY": {"temperature": 1.0, "top_p": 1.0},
    }


def test_job_plan_has_1152_unique_seeds(project_root: Path) -> None:
    jobs = build_job_plan(project_root)
    assert len(jobs) == 1152
    assert {job["case_id"] for job in jobs} == set(EXPECTED_TRAIN)
    assert all(job["split"] == "train" for job in jobs)
    by_cell = {}
    for job in jobs:
        by_cell.setdefault((job["case_id"], job["regime_id"]), []).append(job["seed"])
    assert all(len(seeds) == 12 for seeds in by_cell.values())
    assert all(len(set(seeds)) == 12 for seeds in by_cell.values())
    first = derive_landscape_seed("CTS3-QUAL-001", "R0_CANONICAL", 0)
    assert derive_landscape_seed("CTS3-QUAL-001", "R0_CANONICAL", 0) == first
    assert first != derive_landscape_seed("CTS3-QUAL-001", "R0_CANONICAL", 1)


def test_reward_hash_matches_v1(project_root: Path) -> None:
    assert file_sha256(V1_REWARD_MODULE) == file_sha256(
        project_root / "src/enterprise_memory_mlx/grpo_reward.py"
    )
    assert MEANINGFUL_DELTA == 0.10


def test_egr_enumerates_495_subsets() -> None:
    rewards = [0.1 * index for index in range(12)]
    assert len(tuple(itertools.combinations(rewards, 4))) == GROUP_COMBINATIONS
    assert len(tuple(itertools.combinations(rewards, 8))) == GROUP_COMBINATIONS
    assert effective_group_rate([0.4] * 12, 4) == 0.0
    assert effective_group_rate([0.4] * 12, 8) == 0.0
    mixed = [1.0, 1.0, 1.0, 0.875] + [1.0] * 8
    assert max(mixed) - min(mixed) >= MEANINGFUL_DELTA
    assert effective_group_rate(mixed, 4) > 0


def test_classification_fixtures() -> None:
    high = [_row(reward=1.0, ungated_component_quality=1.0) for _ in range(12)]
    assert classify_case(high)["classification"] == "SATURATED_HIGH"
    low = [_row(reward=0.0, hard_zero=True, ungated_component_quality=0.0) for _ in range(12)]
    assert classify_case(low)["classification"] == "SATURATED_LOW"
    cliff = [
        _row(
            reward=0.0,
            hard_zero=True,
            unsafe_hard_fail=True,
            ungated_component_quality=0.2 + 0.05 * index,
        )
        for index in range(12)
    ]
    assert classify_case(cliff)["classification"] == "SAFETY_CLIFF"
    quality = [
        _row(reward=0.2 if index < 6 else 0.9, ungated_component_quality=0.2 if index < 6 else 0.9)
        for index in range(12)
    ]
    classified = classify_case(quality)
    assert classified["classification"] == "INFORMATIVE"
    assert classified["signal_source"] == "quality_signal"


def test_ungated_cannot_overwrite_reward(project_root: Path) -> None:
    assets = load_parent_assets(project_root)
    case = next(item for item in assets["v3_cases"] if item["case_id"] == "CTS3-QUAL-001")
    spec = compile_reward_spec({**case, "source_protocol": "v3"})
    perfect = score_completion(perfect_output(spec, case), spec)
    assert perfect["reward"] == 1.0
    analyzed = attach_analysis_score(perfect)
    assert analyzed["reward"] == 1.0
    mutated = mutate_output(perfect_output(spec, case), "invalid_json", spec)
    invalid = score_completion(mutated, spec)
    assert invalid["reward"] == 0.0
    assert invalid["hard_zero"] is True
    attached = attach_analysis_score(invalid)
    assert attached["reward"] == 0.0
    assert attached["ungated_component_quality"] >= 0.0
    assert attached["analysis_only"] is True


def test_hard_zeros_remain_zero(project_root: Path) -> None:
    assets = load_parent_assets(project_root)
    case = next(item for item in assets["v3_cases"] if item["case_id"] == "CTS3-QUAL-001")
    spec = compile_reward_spec({**case, "source_protocol": "v3"})
    perfect = perfect_output(spec, case)
    for mutation in ("invalid_json", "unauthorized_source_id", "correct_disposition_unsafe_action"):
        scored = score_completion(mutate_output(perfect, mutation, spec), spec)
        assert scored["reward"] == 0.0
        assert attach_analysis_score(scored)["reward"] == 0.0


def test_landscape_source_forbids_training() -> None:
    assert_source_has_no_training()
    source = Path("src/enterprise_memory_mlx/grpo_landscape.py").read_text(encoding="utf-8")
    assert "run_" + "preflight(" not in source
    assert "import mlx" + ".optimizers" not in source
    assert "value_and_" + "grad(" not in source


def test_summarization_is_order_independent() -> None:
    rows = [
        _row(reward=0.2 if index % 2 == 0 else 0.8, ungated_component_quality=0.5)
        for index in range(12)
    ]
    first = classify_case(rows)
    second = classify_case(list(reversed(rows)))
    assert first == second


def test_charts_use_scores_only(tmp_path: Path) -> None:
    scores = [
        {
            **_row(reward=0.0 if index < 6 else 1.0),
            "case_id": "CTS3-QUAL-001",
            "family_id": "v3-po-allof-5",
            "regime_id": regime,
            "replicate": index,
            "provenance_reasons": [],
            "unsafe_reasons": ["reopened_complete_work"] if index < 6 else [],
            "hard_zero": index < 6,
        }
        for regime in REGIME_ORDER
        for index in range(12)
    ]
    stats = [
        {
            "case_id": "CTS3-QUAL-001",
            "regime_id": regime,
            "classification": "INFORMATIVE",
            "egr_g4": 0.6,
            "egr_g8": 0.7,
        }
        for regime in REGIME_ORDER
    ]
    destination = tmp_path / "charts"
    write_charts(destination, scores, stats)
    expected = {
        "reward-histogram-by-regime.svg",
        "effective-group-rate-g4-g8.svg",
        "case-signal-heatmap.svg",
        "hard-zero-reason-histogram.svg",
    }
    assert {path.name for path in destination.iterdir()} == expected


def test_validate_and_families(project_root: Path) -> None:
    result = validate_landscape(project_root)
    assert result["status"] == "validated"
    assert result["train"] == 24
    assert result["dev"] == 8
    assert result["test"] == 8
    assert result["base_only"] is True
    assert result["reward_unchanged"] is True
    assert len(train_families(project_root)) == 10


def test_decision_gate_is_deterministic() -> None:
    from enterprise_memory_mlx.grpo_landscape import _viability

    first = _viability([], [], None, None)
    second = _viability([], [], None, None)
    assert first["status"] == "abandon_standard_grpo_for_current_dataset"
    assert first == second


def test_cli_landscape_actions_exist() -> None:
    parser = build_parser()
    assert parser.parse_args(["grpo-landscape", "validate"]).landscape_action == "validate"
    plan = parser.parse_args(["grpo-landscape", "plan", "--master-seed", "42"])
    assert plan.master_seed == 42
    assert parser.parse_args(["grpo-landscape", "run", "--latest-plan"]).latest_plan
    assert parser.parse_args(["grpo-landscape", "summarize", "--latest"]).latest
    assert parser.parse_args(["grpo-landscape", "decide", "--latest"]).latest


def test_raw_output_is_persisted_before_score(
    project_root: Path, tmp_path: Path
) -> None:
    from enterprise_memory_mlx.grpo_landscape import persist_raw_then_score

    assets = load_parent_assets(project_root)
    case = next(item for item in assets["v3_cases"] if item["case_id"] == "CTS3-QUAL-001")
    spec = compile_reward_spec({**case, "source_protocol": "v3"})
    path = tmp_path / "00.json"
    raw = {
        "case_id": "CTS3-QUAL-001",
        "raw_output": perfect_output(spec, case),
        "status": "rollout_generated",
    }
    record = persist_raw_then_score(path, raw, spec)
    assert record["raw_output"] == raw["raw_output"]
    assert record["reward"] == 1.0
    assert "ungated_component_quality" in record


def test_ungated_formula_uses_frozen_weights() -> None:
    score = {
        "components": {
            "structural_provenance": 1.0,
            "current_decisions": 0.0,
            "remedy_logic": 0.5,
            "fact_state": 1.0,
            "safe_next_action": 0.0,
        },
        "applicable_components": [
            "structural_provenance",
            "current_decisions",
            "remedy_logic",
            "safe_next_action",
        ],
        "hard_zero": True,
        "reward": 0.0,
    }
    value = ungated_component_quality(score)
    assert value == attach_analysis_score(score)["ungated_component_quality"]
    assert attach_analysis_score(score)["reward"] == 0.0
