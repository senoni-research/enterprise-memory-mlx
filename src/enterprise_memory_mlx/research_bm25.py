"""Deterministic selection of a non-promotable BM25 research trade-off."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .utils import atomic_write_text

RESEARCH_SELECTOR_VERSION = "balanced-retrieval-utility/v1"


@dataclass(frozen=True)
class ResearchBM25Selection:
    candidate: dict[str, Any]
    balanced_utility: Fraction

    def metrics(self) -> dict[str, Any]:
        return dict(self.candidate["metrics"])


def select_research_tradeoff(report: dict[str, Any]) -> ResearchBM25Selection:
    """Select on validation only, independently of production constraints.

    Primary objective is balanced utility:
      0.5 * (answerable correct-record rate + correct OOS-rejection rate).

    Ties are safety-first: lower OOS false-load, wrong-record, and answerable
    empty-retrieval rates; fewer distractors/records; smaller top_k; then a
    higher threshold and stable candidate ordinal.
    """
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Calibration report has no selection object")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Calibration report has no candidate results")

    ranked = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Calibration candidate is not an object")
        metrics = candidate.get("metrics")
        config = candidate.get("config")
        if not isinstance(metrics, dict) or not isinstance(config, dict):
            raise ValueError("Calibration candidate is missing config/metrics")
        correct = _metric_fraction(metrics, "correct_record")
        correct_oos = _metric_fraction(metrics, "correct_oos_rejection")
        utility = (correct + correct_oos) / 2
        key = (
            -utility,
            _metric_fraction(metrics, "oos_false_load"),
            _metric_fraction(metrics, "wrong_record"),
            _metric_fraction(metrics, "answerable_empty_retrieval"),
            int(metrics.get("distractor_record_count", 0)),
            Fraction(str(metrics.get("mean_retrieved_record_count", 0.0))),
            int(config["top_k"]),
            -Fraction(str(config["score_threshold"])),
            int(candidate.get("ordinal", 0)),
        )
        ranked.append((key, utility, candidate))
    ranked.sort(key=lambda item: item[0])
    _key, utility, candidate = ranked[0]
    return ResearchBM25Selection(candidate=candidate, balanced_utility=utility)


def build_research_decision(
    *,
    report_path: Path,
    validation_dataset_hash: str,
    source_snapshot_hash: str,
    index_payload_hash: str,
    production_decision_path: Path,
    approved_by: str,
) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    chosen = select_research_tradeoff(report)
    candidate = chosen.candidate
    production = json.loads(production_decision_path.read_text(encoding="utf-8"))
    if production.get("status") != "no_feasible_operating_point":
        raise ValueError("Research trade-off requires preserved no-feasible production decision")
    if production.get("validation_dataset_hash") != validation_dataset_hash:
        raise ValueError("Production and research validation hashes differ")
    if production.get("source_snapshot_hash") != source_snapshot_hash:
        raise ValueError("Production and research source hashes differ")
    if production.get("index_payload_hash") != index_payload_hash:
        raise ValueError("Production and research index hashes differ")
    config = candidate["config"]
    metrics = candidate["metrics"]
    return {
        "schema_version": 1,
        "status": "experimental_non_promotable",
        "approval_status": "owner_approved",
        "approved_by": approved_by,
        "approved_at": datetime.now(UTC).isoformat(),
        "exploratory": True,
        "deployment_eligible": False,
        "selected_config": config,
        "validation_dataset_hash": validation_dataset_hash,
        "source_snapshot_hash": source_snapshot_hash,
        "index_payload_hash": index_payload_hash,
        "calibration_report_hash": hashlib.sha256(report_bytes).hexdigest(),
        "selector_version": RESEARCH_SELECTOR_VERSION,
        "selection_objective": {
            "primary": (
                "maximize 0.5 * (correct_record_rate + "
                "correct_oos_rejection_rate)"
            ),
            "tie_breaks": [
                "lower_oos_false_load_rate",
                "lower_wrong_record_rate",
                "lower_answerable_empty_retrieval_rate",
                "fewer_distractor_records",
                "lower_mean_retrieved_record_count",
                "smaller_top_k",
                "higher_score_threshold",
                "lower_candidate_ordinal",
            ],
            "balanced_utility": float(chosen.balanced_utility),
        },
        "validation_metrics": metrics,
        "production_decision": {
            "status": production["status"],
            "path": str(production_decision_path),
            "sha256": hashlib.sha256(production_decision_path.read_bytes()).hexdigest(),
        },
        "constraints": {
            "promotion": "failed",
            "usage": "research_comparison_only",
        },
        "warning": (
            "This operating point failed production feasibility constraints. "
            "It may be used only to measure a research RAG control and cannot "
            "be promoted or used as a deployment default."
        ),
    }


def write_research_decision(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _metric_fraction(metrics: dict[str, Any], name: str) -> Fraction:
    value = metrics.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Candidate metric missing: {name}")
    count = int(value.get("count", -1))
    total = int(value.get("total", -1))
    if total <= 0 or not 0 <= count <= total:
        raise ValueError(f"Candidate metric has invalid count/total: {name}")
    return Fraction(count, total)
