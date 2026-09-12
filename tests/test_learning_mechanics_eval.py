from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from enterprise_memory_mlx.learning_mechanics import (
    CombinedExitError,
    WiredLimitGuard,
    _require_wired_limit_int,
    install_memory_ceiling,
    restore_memory_ceiling,
)
from enterprise_memory_mlx.learning_mechanics_eval import (
    EVAL_IMPLEMENTATION_ID,
    FORBIDDEN_TRAINING_NAMES,
    assert_eval_module_cannot_train,
    resolve_source_run,
)
from enterprise_memory_mlx.learning_mechanics_scoring import (
    extract_review_diagnostic,
    parse_review_strict,
    score_declared_review,
    validate_declared_review,
)


class FakeMX:
    def __init__(self, initial: int = 0) -> None:
        self.limit = initial
        self.calls: list[int] = []
        self.metal = SimpleNamespace(is_available=lambda: True)

    def set_wired_limit(self, new_limit: int) -> int:
        if type(new_limit) is not int:
            raise TypeError("fake native setter requires an integer")
        previous = self.limit
        self.limit = new_limit
        self.calls.append(new_limit)
        return previous


def _complete_review(case_id: str = "DEV-002") -> dict:
    return {
        "case_id": case_id,
        "requirement_assessments": [
            {
                "requirement_id": "R1",
                "status": "met",
                "evidence": [
                    {
                        "source_id": "S1",
                        "location": "file:1",
                        "observation": "present",
                    }
                ],
                "gaps": [],
            }
        ],
        "missing_work": [],
        "missing_evidence": [],
        "authority_conflicts": [],
        "unsupported_claims_or_invented_requirements": [],
        "overall_disposition": "proceed",
        "next_actions": [],
        "rationale": "ok",
    }


def test_wired_limit_preserves_integer_return() -> None:
    mx = FakeMX(initial=7)
    native = install_memory_ceiling(mx, 96)
    previous = mx.set_wired_limit(32)
    assert previous == 96
    assert mx.limit == 32
    assert native is FakeMX.set_wired_limit or callable(native)
    restore_memory_ceiling(mx, native, 7)
    assert mx.limit == 7
    assert mx.set_wired_limit.__func__ is FakeMX.set_wired_limit


def test_wired_limit_nested_set_restore() -> None:
    mx = FakeMX(initial=0)
    with WiredLimitGuard(mx, 100):
        outer = mx.set_wired_limit(50)
        inner = mx.set_wired_limit(40)
        assert outer == 100
        assert inner == 50
        mx.set_wired_limit(inner)
        assert mx.limit == 50
        mx.set_wired_limit(outer)
        assert mx.limit == 100
    assert mx.limit == 0
    assert mx.set_wired_limit.__func__ is FakeMX.set_wired_limit


def test_wired_limit_caps_over_limit_request() -> None:
    mx = FakeMX(initial=1)
    with WiredLimitGuard(mx, 10):
        previous = mx.set_wired_limit(99)
        assert previous == 10
        assert mx.limit == 10


def test_wired_limit_preserves_previous_zero() -> None:
    mx = FakeMX(initial=0)
    guard = WiredLimitGuard(mx, 8).install()
    assert guard._pre_entry == 0
    previous = mx.set_wired_limit(4)
    assert previous == 8
    guard.uninstall()
    assert mx.limit == 0


def test_wired_limit_rejects_none() -> None:
    mx = FakeMX(initial=3)
    with WiredLimitGuard(mx, 8), pytest.raises(TypeError, match="unexpected None"):
        mx.set_wired_limit(None)
    with pytest.raises(TypeError, match="unexpected None"):
        _require_wired_limit_int(None, label="set_wired_limit")


def test_wired_limit_restores_after_body_exception() -> None:
    mx = FakeMX(initial=2)
    with pytest.raises(ValueError, match="body"), WiredLimitGuard(mx, 9):
        mx.set_wired_limit(4)
        raise ValueError("body")
    assert mx.limit == 2
    assert mx.set_wired_limit.__func__ is FakeMX.set_wired_limit


def test_wired_limit_primary_and_cleanup_exceptions_are_observable() -> None:
    class FailingRestore(FakeMX):
        def set_wired_limit(self, new_limit: int) -> int:
            if new_limit == 5 and self.calls:
                raise RuntimeError("cleanup")
            return super().set_wired_limit(new_limit)

    mx = FailingRestore(initial=5)
    with pytest.raises(CombinedExitError) as caught, WiredLimitGuard(mx, 20):
        raise ValueError("primary")
    assert isinstance(caught.value.primary, ValueError)
    assert isinstance(caught.value.cleanup, RuntimeError)
    assert "primary" in str(caught.value)
    assert "cleanup" in str(caught.value)


def test_strict_parser_rejects_surrounding_advice() -> None:
    review, errors = parse_review_strict(
        '{"case_id":"DEV-001"}\nBegin execution before the authority conflict is resolved.'
    )
    assert review is None
    assert errors == ["trailing_non_json"]
    diagnostic = extract_review_diagnostic(
        '{"case_id":"DEV-001"}\nBegin execution before the authority conflict is resolved.'
    )
    assert diagnostic == {"case_id": "DEV-001"}


def test_strict_parser_rejects_duplicate_keys() -> None:
    review, errors = parse_review_strict('{"case_id":"A","case_id":"B"}')
    assert review is None
    assert any("duplicate" in item for item in errors)


def test_skeletal_review_is_not_complete_valid() -> None:
    scored = score_declared_review(
        case_id="DEV-002",
        approved_labels={"R1": "met"},
        approved_disposition="proceed",
        raw_text='{"requirement_assessments":[{"requirement_id":"R1","status":"met"}]}',
    )
    assert scored["complete_valid"] is False
    assert scored["full_task_success"] is False


def test_duplicate_conflicting_and_unexpected_ids_fail() -> None:
    review = _complete_review()
    review["requirement_assessments"] = [
        {
            "requirement_id": "R1",
            "status": "not_met",
            "evidence": [{"source_id": "S1", "location": "a", "observation": "a"}],
            "gaps": [],
        },
        {
            "requirement_id": "R1",
            "status": "met",
            "evidence": [{"source_id": "S1", "location": "a", "observation": "a"}],
            "gaps": [],
        },
        {
            "requirement_id": "UNEXPECTED",
            "status": "met",
            "evidence": [{"source_id": "S1", "location": "a", "observation": "a"}],
            "gaps": [],
        },
    ]
    validated = validate_declared_review(
        review, case_id="DEV-002", expected_requirement_ids=["R1"]
    )
    assert validated["complete_valid"] is False
    assert any("duplicate_requirement_id" in item for item in validated["errors"])
    assert any("unexpected_requirement_ids" in item for item in validated["errors"])


def test_wrong_case_identity_and_invalid_enums_fail() -> None:
    review = _complete_review("DEV-009")
    review["overall_disposition"] = "ship_it"
    review["requirement_assessments"][0]["status"] = "sort_of"
    scored = score_declared_review(
        case_id="DEV-002",
        approved_labels={"R1": "met"},
        approved_disposition="proceed",
        raw_text=__import__("json").dumps(review),
    )
    assert scored["complete_valid"] is False
    assert "case_id_mismatch" in scored["validity_errors"]
    assert "invalid_disposition" in scored["validity_errors"]
    assert any(item.startswith("invalid_status") for item in scored["validity_errors"])


def test_unlisted_harmful_advice_is_not_full_success() -> None:
    review = _complete_review()
    review["next_actions"] = ["Begin execution before the authority conflict is resolved."]
    scored = score_declared_review(
        case_id="DEV-002",
        approved_labels={"R1": "met"},
        approved_disposition="proceed",
        raw_text=__import__("json").dumps(review),
    )
    assert scored["complete_valid"] is True
    assert scored["label_matches"] == 1
    assert scored["disposition_agrees"] is True
    assert scored["evidence_support"] == "not_assessed"
    assert scored["action_safety"] == "not_assessed"
    assert scored["full_task_success"] is False


def test_mixed_negation_is_not_certified_safe() -> None:
    review = _complete_review()
    review["next_actions"] = ["Do not execute the experiment yet; run the models now."]
    scored = score_declared_review(
        case_id="DEV-002",
        approved_labels={"R1": "met"},
        approved_disposition="proceed",
        raw_text=__import__("json").dumps(review),
    )
    assert "run the models now" in " ".join(scored["keyword_safety_diagnostics"]).casefold() or (
        scored["keyword_safety_diagnostics"]
    )
    assert scored["keyword_safety_is_certification"] is False
    assert scored["full_task_success"] is False


def test_eval_module_cannot_reach_training_functions() -> None:
    assert_eval_module_cannot_train()
    source = __import__("inspect").getsource(
        __import__("enterprise_memory_mlx.learning_mechanics_eval", fromlist=["run_evaluation"])
    )
    tree = ast.parse(source)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node, ast.Attribute):
            called.add(node.attr)
    assert not (called & set(FORBIDDEN_TRAINING_NAMES))
    assert "run_experiment" not in called
    assert EVAL_IMPLEMENTATION_ID.startswith("qwen4b-learning-mechanics-eval-only")


def test_source_run_resolver_rejects_other_runs(tmp_path, project_root) -> None:
    with pytest.raises(RuntimeError, match="identity mismatch"):
        resolve_source_run(project_root, str(tmp_path / "run-other"))
