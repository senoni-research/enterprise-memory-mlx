from __future__ import annotations

from pathlib import Path

from enterprise_memory_mlx.grpo_data import compile_rows, render_prompt
from enterprise_memory_mlx.grpo_eval import case_json_schema, compare_arms
from enterprise_memory_mlx.grpo_protocol import spec_by_case, write_frozen_protocol
from enterprise_memory_mlx.grpo_runtime import derive_seed, group_advantages, validate_experiment


def test_seed_derivation_is_stable() -> None:
    first = derive_seed(42, 0, "CTS3-QUAL-001", 1)
    second = derive_seed(42, 0, "CTS3-QUAL-001", 1)
    other = derive_seed(42, 0, "CTS3-QUAL-001", 2)
    assert first == second
    assert first != other


def test_zero_variance_advantages_collapse() -> None:
    values = group_advantages([0.4, 0.4, 0.4, 0.4])
    assert all(abs(item) < 1e-8 for item in values)


def test_group_advantages_are_relative() -> None:
    values = group_advantages([0.0, 1.0, 0.0, 1.0])
    assert values[1] > 0
    assert values[0] < 0
    assert abs(sum(values)) < 1e-9


def test_validate_experiment_is_model_free(project_root: Path) -> None:
    result = validate_experiment(project_root)
    assert result["status"] == "validated"
    assert result["algorithm"] == "GRPO"
    assert result["train"] == 24
    assert result["test"] == 8


def test_reward_decisions_are_absent_from_prompts(project_root: Path) -> None:
    write_frozen_protocol(project_root)
    compiled = compile_rows(project_root)
    specs = spec_by_case(project_root)
    from enterprise_memory_mlx.grpo_data import load_parent_assets

    assets = load_parent_assets(project_root)
    cases = {str(case["case_id"]): case for case in [*assets["v3_cases"], *assets["v4_cases"]]}
    for row in compiled["rows"]:
        case = {**cases[row["case_id"]], "source_protocol": row["source_protocol"]}
        prompt = render_prompt(project_root, case, assets)
        spec = specs[row["case_id"]]
        assert "reward" not in prompt["question"]
        assert str(spec["decisions"]) not in prompt["question"]
        schema = case_json_schema(spec)
        blob = str(schema)
        for decision in spec["decisions"].values():
            assert f'"const": "{decision}"' not in blob
        assert "correct" not in blob


def test_continuation_gate_requires_all_conditions() -> None:
    def _row(case_id: str, reward: float, success: bool, unsafe: bool = False) -> dict:
        return {
            "case_id": case_id,
            "family_id": f"fam-{case_id}",
            "score": {
                "reward": reward,
                "full_machine_success": success,
                "decision_correctness": [{"correct": True}],
                "complete_valid_output": True,
                "provenance_hard_fail": False,
                "unsafe_hard_fail": unsafe,
            },
        }

    base = [_row(f"C{i}", 0.2, False) for i in range(8)]
    rft = [_row(f"C{i}", 0.5, True) for i in range(8)]
    passed = compare_arms(base, rft)
    assert passed["status"] == "rft_pilot_promising_semantic_review_required"
    rft_unsafe = [_row(f"C{i}", 0.5, True, unsafe=(i == 0)) for i in range(8)]
    failed = compare_arms(base, rft_unsafe)
    assert failed["status"] == "stop_rft_candidate"
