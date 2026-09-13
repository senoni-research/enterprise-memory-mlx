from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from enterprise_memory_mlx.grpo_data import (
    TARGET_COUNTS,
    assign_splits,
    compile_rows,
    family_catalog,
    load_parent_assets,
    render_prompt,
)
from enterprise_memory_mlx.grpo_protocol import load_frozen_protocol, write_frozen_protocol


def test_parent_inventory_is_exactly_forty_cases(project_root: Path) -> None:
    assets = load_parent_assets(project_root)
    assert len(assets["v3_cases"]) == 24
    assert len(assets["v4_cases"]) == 16
    catalog = family_catalog(assets["v3_cases"], assets["v4_cases"])
    assert sum(family["size"] for family in catalog["families"]) == 40


def test_family_preserving_split_has_no_overlap(project_root: Path) -> None:
    compiled = compile_rows(project_root)
    counts = Counter(row["split"] for row in compiled["rows"])
    assert dict(counts) == TARGET_COUNTS
    by_split = {
        name: {row["case_id"] for row in compiled["rows"] if row["split"] == name}
        for name in TARGET_COUNTS
    }
    assert not by_split["train"] & by_split["dev"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["dev"] & by_split["test"]
    family_splits: dict[str, set[str]] = {}
    pair_splits: dict[str, set[str]] = {}
    for row in compiled["rows"]:
        family_splits.setdefault(row["family_id"], set()).add(row["split"])
        if row["pair_id"]:
            pair_splits.setdefault(row["pair_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in family_splits.values())
    assert all(len(values) == 1 for values in pair_splits.values())


def test_split_is_deterministic(project_root: Path) -> None:
    assets = load_parent_assets(project_root)
    catalog = family_catalog(assets["v3_cases"], assets["v4_cases"])
    first = assign_splits(catalog["families"], seed=42)
    second = assign_splits(catalog["families"], seed=42)
    assert first == second


def test_prompts_exclude_hidden_reference_material(project_root: Path) -> None:
    assets = load_parent_assets(project_root)
    for case in [*assets["v3_cases"], *assets["v4_cases"]]:
        tagged = {**case, "source_protocol": "v4" if "operational_state" in case else "v3"}
        prompt = render_prompt(project_root, tagged, assets)
        assert (
            json.dumps(case["reference"], sort_keys=True, ensure_ascii=False)
            not in prompt["question"]
        )
        if case.get("policy_logic") is not None:
            assert (
                json.dumps(case["policy_logic"], sort_keys=True, ensure_ascii=False)
                not in prompt["question"]
            )
        if case.get("expected_fact_state") is not None:
            assert (
                json.dumps(case["expected_fact_state"], sort_keys=True, ensure_ascii=False)
                not in prompt["question"]
            )
        assert "policy_logic" not in prompt["question"]
        assert "expected_fact_state" not in prompt["question"]


def test_sft_examples_are_excluded(project_root: Path) -> None:
    compiled = compile_rows(project_root)
    case_ids = {row["case_id"] for row in compiled["rows"]}
    assert not case_ids & {"DEV-001", "DEV-002", "DEV-003", "DEV-004", "DEV-005"}
    assert all(row["case_id"].startswith(("CTS3-", "CTS4-")) for row in compiled["rows"])


def test_frozen_protocol_round_trip(project_root: Path) -> None:
    write_frozen_protocol(project_root)
    frozen = load_frozen_protocol(project_root)
    assert frozen["protocol"]["protocol_id"] == "company-task-specialization/qwen4b-grpo-v1"
    assert frozen["protocol"]["origin_main_sha"] == "a6c0421f244c18a439a907d67b4869fe59f85792"
    assert len(frozen["reward_specs"]) == 40
    assert frozen["protocol"]["model"]["initialize_from_existing_sft_adapter"] is False
