from __future__ import annotations

import hashlib
import json
from pathlib import Path

from enterprise_memory_mlx.benchmark import bind_bm25_selection
from enterprise_memory_mlx.compiler import load_records
from enterprise_memory_mlx.research_bm25 import (
    RESEARCH_SELECTOR_VERSION,
    build_research_decision,
    select_research_tradeoff,
    write_research_decision,
)
from enterprise_memory_mlx.split_contract import load_eval_suites


def _metric(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total}


def _candidate(
    ordinal: int,
    *,
    correct: int,
    correct_oos: int,
    wrong: int,
    empty: int,
    top_k: int,
    threshold: float,
) -> dict:
    total = 10
    return {
        "ordinal": ordinal,
        "config": {
            "k1": 1.2,
            "b": 0.75,
            "top_k": top_k,
            "score_threshold": threshold,
        },
        "metrics": {
            "correct_record": _metric(correct, total),
            "correct_oos_rejection": _metric(correct_oos, total),
            "oos_false_load": _metric(total - correct_oos, total),
            "wrong_record": _metric(wrong, total),
            "answerable_empty_retrieval": _metric(empty, total),
            "distractor_record_count": 0,
            "mean_retrieved_record_count": 1.0,
        },
    }


def test_research_selector_maximizes_balanced_validation_utility() -> None:
    report = {
        "selection": {
            "candidates": [
                _candidate(
                    0,
                    correct=10,
                    correct_oos=0,
                    wrong=0,
                    empty=0,
                    top_k=1,
                    threshold=0.0,
                ),
                _candidate(
                    1,
                    correct=8,
                    correct_oos=8,
                    wrong=1,
                    empty=1,
                    top_k=1,
                    threshold=2.0,
                ),
            ]
        }
    }

    selected = select_research_tradeoff(report)

    assert selected.candidate["ordinal"] == 1
    assert float(selected.balanced_utility) == 0.8


def test_research_selector_uses_safety_first_tie_break() -> None:
    report = {
        "selection": {
            "candidates": [
                _candidate(
                    0,
                    correct=9,
                    correct_oos=7,
                    wrong=1,
                    empty=0,
                    top_k=2,
                    threshold=1.0,
                ),
                _candidate(
                    1,
                    correct=8,
                    correct_oos=8,
                    wrong=0,
                    empty=2,
                    top_k=1,
                    threshold=2.0,
                ),
            ]
        }
    }

    selected = select_research_tradeoff(report)

    assert selected.candidate["ordinal"] == 1


def test_real_report_builds_non_promotable_hash_bound_decision(
    tmp_path: Path,
    project_root: Path,
) -> None:
    report = project_root / "artifacts" / "retrieval-calibration" / "v1" / "report.json"
    production = (
        project_root
        / "knowledge"
        / "operating_points"
        / "bm25"
        / "v1-no-feasible.json"
    )
    payload = build_research_decision(
        report_path=report,
        validation_dataset_hash=(
            "8532de5110dce79be3349e05fa9668757757dcb4cdeb17dee9bd999c24dd3c19"
        ),
        source_snapshot_hash=(
            "56ace4cdba527ff63f2b9bfbb9ec1cb09aaf97f287daaa6ad2d74c48cb071dae"
        ),
        index_payload_hash=(
            "2e501893f6bd18522693db45bba07dcd46ec1e07f11f5cb97e03ecce02466e9c"
        ),
        production_decision_path=production,
        approved_by="Philippe Dagher",
    )
    target = tmp_path / "research.json"
    write_research_decision(payload, target)

    assert payload["status"] == "experimental_non_promotable"
    assert payload["deployment_eligible"] is False
    assert payload["selector_version"] == RESEARCH_SELECTOR_VERSION
    assert payload["calibration_report_hash"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    assert payload["selected_config"] is not None
    assert payload["selection_objective"]["balanced_utility"] > 0.5

    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(project_root / "knowledge" / "eval_frozen")
    binding = bind_bm25_selection(
        target,
        tuple(records) + tuple(suites.holdout_records),
        validation_dataset_path=(
            project_root / "knowledge" / "retrieval_validation" / "v1" / "queries.jsonl"
        ),
        calibration_report_path=report,
    )
    assert binding.status == "experimental_non_promotable"
    assert binding.selected_config is not None
    assert binding.deployment_eligible is False


def test_production_no_feasible_decision_is_not_modified(project_root: Path) -> None:
    path = (
        project_root
        / "knowledge"
        / "operating_points"
        / "bm25"
        / "v1-no-feasible.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "no_feasible_operating_point"
    assert payload["selected_config"] is None
