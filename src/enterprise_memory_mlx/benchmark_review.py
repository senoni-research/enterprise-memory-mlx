"""Blinded human review and bounded go/no-go reporting for benchmark outputs.

This module is deliberately separate from judge calibration. A completed
single-human overlay can inform whether one more non-promotable diagnostic is
worth running, but it cannot certify judges, approve labels, support promotion,
or establish headline accuracy.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .semantic_judging import clopper_pearson_two_sided
from .split_contract import EvalSuites, load_eval_suites, verify_frozen_assets
from .utils import atomic_write_text, canonical_json, sha256_json

LEGAL_HUMAN_SCORES = (0.0, 0.5, 1.0)
PACKET_PURPOSE = "smoke_benchmark_blinded_human_review"
REPORT_TYPE = "smoke_go_no_go"
CONTINUATION_RULE = {
    "rule_version": "bounded-diagnostic/v1",
    "mean_score_delta_over_base_min": 0.10,
    "fully_correct_count_delta_over_base_min": 4,
    "paired_wins_must_exceed_losses": True,
    "interpretation": (
        "Internal stopping rule only. Passing authorizes at most one redesigned "
        "non-promotable diagnostic; it is not evidence for promotion or a "
        "headline accuracy claim."
    ),
}
DecisionStatus = Literal[
    "authorize_one_redesigned_diagnostic",
    "stop_parametric_research",
]

_REVIEW_INSTRUCTIONS = """# Blinded smoke benchmark human review

## Purpose

Review every candidate independently against the question and reference answer.
The benchmark arm, original question ID, deterministic grading, retrieval
metadata, and model-proposed labels are deliberately absent.

This is a single-human, non-promotable diagnostic. It cannot certify a judge,
approve calibration labels, or support a headline accuracy claim.

## Scores

- `1.0`: fully correct; no material omission, contradiction, unsupported
  exception, false provenance, or misleading statement.
- `0.5`: broadly correct core conclusion, but materially incomplete or with a
  limited non-core flaw.
- `0.0`: incorrect, contradictory, unsupported, evasive, materially
  misleading, or improperly answers/refuses.

Review repeated questions independently. Do not try to infer which system
produced an answer.
"""

_FORBIDDEN_BLINDED_FIELDS = {
    "arm",
    "question_id",
    "suite",
    "case_id",
    "question_family_id",
    "generation_status",
    "retrieval_label",
    "selected_record_ids",
    "source_uris",
    "context_action",
    "context_hash",
    "deterministic_status",
    "deterministic_score",
    "strict",
    "provenance",
    "reasons",
    "proposed_semantic_score",
    "proposed_reason",
}


class BenchmarkReviewError(ValueError):
    """The review artifacts are incomplete, inconsistent, or tampered with."""


@dataclass(frozen=True)
class BenchmarkReviewPreparation:
    packet_path: Path
    mapping_path: Path
    case_count: int
    packet_sha256: str
    mapping_sha256: str
    benchmark_sha256: str
    deterministic_grading_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_path": str(self.packet_path),
            "mapping_path": str(self.mapping_path),
            "case_count": self.case_count,
            "packet_sha256": self.packet_sha256,
            "mapping_sha256": self.mapping_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "deterministic_grading_sha256": self.deterministic_grading_sha256,
        }


@dataclass(frozen=True)
class BenchmarkReviewReportPaths:
    json_path: Path
    markdown_path: Path
    decision: DecisionStatus


@dataclass(frozen=True)
class ModelReviewAdvisoryPaths:
    labels_path: Path
    manifest_path: Path
    json_path: Path
    markdown_path: Path
    apparent_rule_result: DecisionStatus


def prepare_benchmark_review(
    *,
    benchmark_path: Path,
    deterministic_grading_path: Path,
    eval_dir: Path,
    output_root: Path,
    salt: bytes | None = None,
) -> BenchmarkReviewPreparation:
    """Build an arm-blinded packet and a separately stored private mapping."""
    freeze_problems = verify_frozen_assets(eval_dir)
    if freeze_problems:
        raise BenchmarkReviewError(
            "Frozen evaluation assets failed verification:\n"
            + "\n".join(freeze_problems)
        )
    benchmark_bytes = benchmark_path.read_bytes()
    grading_bytes = deterministic_grading_path.read_bytes()
    benchmark = _json_object(benchmark_bytes, "benchmark")
    grading = _json_object(grading_bytes, "deterministic grading")
    benchmark_hash = _sha256_bytes(benchmark_bytes)
    grading_hash = _sha256_bytes(grading_bytes)
    suites = load_eval_suites(eval_dir)
    fixture_hash = _fixture_hash(eval_dir)
    benchmark_rows, grading_rows, questions = _validate_source_artifacts(
        benchmark=benchmark,
        benchmark_hash=benchmark_hash,
        grading=grading,
        suites=suites,
        fixture_hash=fixture_hash,
    )

    private_salt = salt if salt is not None else secrets.token_bytes(32)
    if len(private_salt) < 16:
        raise BenchmarkReviewError("review salt must contain at least 16 bytes")
    blinded_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for raw in benchmark_rows:
        question_id = str(raw["question_id"])
        arm = str(raw["arm"])
        suite = str(raw["suite"])
        case_id = _case_id(question_id, arm)
        digest = hmac.new(private_salt, case_id.encode("utf-8"), hashlib.sha256)
        review_id = f"RVB-{digest.hexdigest()[:16].upper()}"
        question = questions[question_id]
        blinded_rows.append(
            {
                "review_id": review_id,
                "question": question.question,
                "reference_answer": question.expected,
                "candidate_answer": str(raw["output"]),
                # Deliberately empty. Record count/content fingerprints arms.
                "source_record_ids": [],
            }
        )
        mapping_rows.append(
            {
                "review_id": review_id,
                "case_id": case_id,
                "question_id": question_id,
                "arm": arm,
                "suite": suite,
            }
        )
    blinded_rows.sort(key=lambda row: _sha256_text(str(row["review_id"])))
    mapping_rows.sort(key=lambda row: str(row["review_id"]))
    if any(_FORBIDDEN_BLINDED_FIELDS & row.keys() for row in blinded_rows):
        raise BenchmarkReviewError("blinded rows expose private benchmark fields")
    if len({str(row["review_id"]) for row in blinded_rows}) != len(blinded_rows):
        raise BenchmarkReviewError("generated duplicate blinded review IDs")

    package_name = f"{benchmark_path.stem}-human-review"
    packet_path = output_root / f"{package_name}.zip"
    private_dir = output_root / f"{package_name}-private"
    mapping_path = private_dir / "review_id_map.json"
    mapping_payload = {
        "schema_version": 1,
        "packet_purpose": PACKET_PURPOSE,
        "source_artifact_sha256": benchmark_hash,
        "deterministic_grading_sha256": grading_hash,
        "fixture_hash": fixture_hash,
        "salt_hex": private_salt.hex(),
        "mapping": mapping_rows,
    }
    mapping_content = (
        json.dumps(mapping_payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    atomic_write_text(mapping_path, mapping_content)
    mapping_hash = _sha256_text(mapping_content)

    members = {
        "REVIEW_INSTRUCTIONS.md": _REVIEW_INSTRUCTIONS.encode("utf-8"),
        "review_cases.jsonl": _encode_jsonl(blinded_rows),
        "source_records.jsonl": b"",
    }
    packet_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": PACKET_PURPOSE,
        "status": "human_review_pending",
        "promotion_eligible": False,
        "headline_accuracy_claims_allowed": False,
        "source_candidate": {
            "artifact_type": "benchmark",
            "primary_sha256": benchmark_hash,
        },
        "source_artifacts": {
            "benchmark_sha256": benchmark_hash,
            "deterministic_grading_sha256": grading_hash,
            "fixture_hash": fixture_hash,
            "grader_config_hash": str(grading["grader_config_hash"]),
        },
        "private_mapping_sha256": mapping_hash,
        "case_count": len(blinded_rows),
        "continuation_rule": CONTINUATION_RULE,
        "continuation_rule_sha256": sha256_json(CONTINUATION_RULE),
        "blinding": {
            "arm_excluded": True,
            "question_id_excluded": True,
            "suite_excluded": True,
            "deterministic_grading_excluded": True,
            "retrieval_metadata_excluded": True,
            "source_records_excluded_to_prevent_arm_fingerprinting": True,
            "model_proposed_labels_excluded": True,
            "private_mapping_in_zip": False,
        },
        "files": {
            name: _sha256_bytes(content)
            for name, content in sorted(members.items())
        },
    }
    manifest_bytes = (
        json.dumps(packet_manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = packet_path.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        prefix = f"{package_name}/"
        for name, content in sorted(members.items()):
            archive.writestr(prefix + name, content)
        archive.writestr(prefix + "packet_manifest.json", manifest_bytes)
    temporary.replace(packet_path)
    return BenchmarkReviewPreparation(
        packet_path=packet_path,
        mapping_path=mapping_path,
        case_count=len(blinded_rows),
        packet_sha256=_sha256_file(packet_path),
        mapping_sha256=mapping_hash,
        benchmark_sha256=benchmark_hash,
        deterministic_grading_sha256=grading_hash,
    )


def write_benchmark_review_report(
    *,
    benchmark_path: Path,
    deterministic_grading_path: Path,
    packet_path: Path,
    mapping_path: Path,
    overlay_path: Path,
    overlay_manifest_path: Path,
    output_dir: Path,
) -> BenchmarkReviewReportPaths:
    """Verify, unblind, aggregate, and write the completed diagnostic report."""
    report = build_benchmark_review_report(
        benchmark_path=benchmark_path,
        deterministic_grading_path=deterministic_grading_path,
        packet_path=packet_path,
        mapping_path=mapping_path,
        overlay_path=overlay_path,
        overlay_manifest_path=overlay_manifest_path,
    )
    stem = benchmark_path.stem
    json_path = output_dir / f"{stem}-smoke-go-no-go.json"
    markdown_path = output_dir / f"{stem}-smoke-go-no-go.md"
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(markdown_path, _render_report_markdown(report))
    return BenchmarkReviewReportPaths(
        json_path=json_path,
        markdown_path=markdown_path,
        decision=report["decision"]["status"],
    )


def build_benchmark_review_report(
    *,
    benchmark_path: Path,
    deterministic_grading_path: Path,
    packet_path: Path,
    mapping_path: Path,
    overlay_path: Path,
    overlay_manifest_path: Path,
) -> dict[str, Any]:
    """Return a complete report; incomplete or mismatched review data fails closed."""
    benchmark_bytes = benchmark_path.read_bytes()
    grading_bytes = deterministic_grading_path.read_bytes()
    benchmark_hash = _sha256_bytes(benchmark_bytes)
    grading_hash = _sha256_bytes(grading_bytes)
    benchmark = _json_object(benchmark_bytes, "benchmark")
    grading = _json_object(grading_bytes, "deterministic grading")
    packet_manifest, packet_review_ids = _verified_packet(packet_path)
    mapping_bytes = mapping_path.read_bytes()
    mapping = _json_object(mapping_bytes, "private mapping")
    overlay_bytes = overlay_path.read_bytes()
    overlay_manifest_bytes = overlay_manifest_path.read_bytes()
    overlay_manifest = _json_object(overlay_manifest_bytes, "overlay manifest")

    if packet_manifest.get("purpose") != PACKET_PURPOSE:
        raise BenchmarkReviewError("packet is not a smoke benchmark review packet")
    sources = packet_manifest.get("source_artifacts")
    if not isinstance(sources, dict):
        raise BenchmarkReviewError("packet manifest is missing source artifacts")
    expected_sources = {
        "benchmark_sha256": benchmark_hash,
        "deterministic_grading_sha256": grading_hash,
        "fixture_hash": str(benchmark.get("fixture_hash", "")),
        "grader_config_hash": str(grading.get("grader_config_hash", "")),
    }
    if sources != expected_sources:
        raise BenchmarkReviewError("packet belongs to different source artifacts")
    mapping_hash = _sha256_bytes(mapping_bytes)
    if packet_manifest.get("private_mapping_sha256") != mapping_hash:
        raise BenchmarkReviewError("private mapping hash does not match packet")
    if mapping.get("source_artifact_sha256") != benchmark_hash:
        raise BenchmarkReviewError("private mapping belongs to another benchmark")
    if mapping.get("deterministic_grading_sha256") != grading_hash:
        raise BenchmarkReviewError(
            "private mapping belongs to another deterministic report"
        )

    mapping_rows = _mapping_rows(mapping, packet_review_ids)
    overlay_rows = _verified_overlay(
        overlay_bytes=overlay_bytes,
        overlay_manifest=overlay_manifest,
        packet_path=packet_path,
        mapping_path=mapping_path,
        expected_case_ids={str(row["case_id"]) for row in mapping_rows.values()},
    )
    raw_rows = _keyed_rows(benchmark.get("results"), "benchmark")
    deterministic_rows = _keyed_rows(grading.get("rows"), "deterministic grading")
    if set(raw_rows) != set(deterministic_rows):
        raise BenchmarkReviewError(
            "benchmark and deterministic report row identities differ"
        )
    if grading.get("raw_artifact_hash") != benchmark_hash:
        raise BenchmarkReviewError("deterministic report raw-artifact hash mismatch")
    mapped_keys = {
        (str(row["question_id"]), str(row["arm"]))
        for row in mapping_rows.values()
    }
    if mapped_keys != set(raw_rows):
        raise BenchmarkReviewError(
            "private mapping does not cover the complete benchmark matrix"
        )

    scored_rows: list[dict[str, Any]] = []
    for review_id, private in mapping_rows.items():
        key = (str(private["question_id"]), str(private["arm"]))
        if key not in raw_rows or key not in deterministic_rows:
            raise BenchmarkReviewError(
                f"private mapping contains an unknown benchmark row: {key}"
            )
        human = overlay_rows[str(private["case_id"])]
        direct_score = _human_score(human.get("human_semantic_score"))
        deterministic = deterministic_rows[key]
        status = str(deterministic.get("status", ""))
        if status == "deterministic_hard_fail":
            governed_score: float | None = 0.0
            score_source = "deterministic_hard_fail"
        elif status == "semantic_review_required":
            governed_score = direct_score
            score_source = "single_human_review"
        else:
            raise BenchmarkReviewError(
                f"unsupported deterministic status for completed review: {status}"
            )
        scored_rows.append(
            {
                "review_id": review_id,
                "case_id": str(private["case_id"]),
                "question_id": key[0],
                "arm": key[1],
                "suite": str(private["suite"]),
                "question_family_id": str(
                    deterministic.get("question_family_id", "")
                ),
                "deterministic_status": status,
                "direct_human_score": direct_score,
                "governed_final_score": governed_score,
                "score_source": score_source,
                "confidence": str(human.get("confidence", "")),
                "needs_human_attention": (
                    human.get("needs_human_attention") is True
                ),
            }
        )
    if len(scored_rows) != int(packet_manifest.get("case_count", -1)):
        raise BenchmarkReviewError("completed review does not cover every packet row")

    aggregates = _aggregate_scores(scored_rows)
    paired = _paired_parametric_base(scored_rows)
    continuation_rule = packet_manifest.get("continuation_rule")
    if continuation_rule != CONTINUATION_RULE:
        raise BenchmarkReviewError("packet continuation rule is not recognized")
    if packet_manifest.get("continuation_rule_sha256") != sha256_json(
        continuation_rule
    ):
        raise BenchmarkReviewError("packet continuation-rule hash mismatch")
    decision, criteria = _continuation_assessment(aggregates, paired)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "status": "complete_no_promotion",
        "decision": {
            "status": decision,
            "criteria": criteria,
            "continuation_rule": CONTINUATION_RULE,
            "continuation_rule_sha256": sha256_json(CONTINUATION_RULE),
        },
        "review_boundary": {
            "single_human_review": True,
            "human_approved": False,
            "adjudicated": False,
            "reviewer_id": str(overlay_manifest.get("reviewer_id", "")),
            "identity_verification": str(
                overlay_manifest.get("identity_verification", "")
            ),
            "os_users": list(overlay_manifest.get("os_users", [])),
            "usable_for_judge_certification": False,
            "headline_accuracy_claims_allowed": False,
            "interpretation": (
                "Direct human scores are diagnostic observations. Governed "
                "scores preserve deterministic hard failures at 0.0."
            ),
        },
        "promotion_eligible": False,
        "case_count": len(scored_rows),
        "artifact_chain": {
            "benchmark_sha256": benchmark_hash,
            "deterministic_grading_sha256": grading_hash,
            "packet_sha256": _sha256_file(packet_path),
            "private_mapping_sha256": mapping_hash,
            "human_overlay_sha256": _sha256_bytes(overlay_bytes),
            "human_overlay_manifest_sha256": _sha256_bytes(
                overlay_manifest_bytes
            ),
        },
        "by_arm": aggregates,
        "paired_parametric_vs_base": paired,
        "rows": sorted(
            scored_rows,
            key=lambda row: (str(row["question_id"]), str(row["arm"])),
        ),
    }


def write_model_review_advisory(
    *,
    benchmark_path: Path,
    deterministic_grading_path: Path,
    packet_path: Path,
    mapping_path: Path,
    label_paths: Sequence[Path],
    output_dir: Path,
) -> ModelReviewAdvisoryPaths:
    """Validate blinded model labels and write a non-human advisory report.

    The pre-registered rule is computed for transparency, but the report's
    operative research decision remains ``human_review_pending``. Model labels
    never become a human overlay and cannot authorize the next experiment.
    """
    benchmark_bytes = benchmark_path.read_bytes()
    grading_bytes = deterministic_grading_path.read_bytes()
    benchmark_hash = _sha256_bytes(benchmark_bytes)
    grading_hash = _sha256_bytes(grading_bytes)
    benchmark = _json_object(benchmark_bytes, "benchmark")
    grading = _json_object(grading_bytes, "deterministic grading")
    packet_manifest, packet_review_ids = _verified_packet(packet_path)
    mapping_bytes = mapping_path.read_bytes()
    mapping = _json_object(mapping_bytes, "private mapping")
    if packet_manifest.get("purpose") != PACKET_PURPOSE:
        raise BenchmarkReviewError("packet is not a smoke benchmark review packet")
    source_artifacts = packet_manifest.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise BenchmarkReviewError("packet manifest is missing source artifacts")
    if source_artifacts.get("benchmark_sha256") != benchmark_hash:
        raise BenchmarkReviewError("packet belongs to another benchmark")
    if source_artifacts.get("deterministic_grading_sha256") != grading_hash:
        raise BenchmarkReviewError(
            "packet belongs to another deterministic report"
        )
    mapping_hash = _sha256_bytes(mapping_bytes)
    if packet_manifest.get("private_mapping_sha256") != mapping_hash:
        raise BenchmarkReviewError("private mapping hash does not match packet")
    mapping_rows = _mapping_rows(mapping, packet_review_ids)
    raw_rows = _keyed_rows(benchmark.get("results"), "benchmark")
    deterministic_rows = _keyed_rows(grading.get("rows"), "deterministic grading")
    if set(raw_rows) != set(deterministic_rows):
        raise BenchmarkReviewError(
            "benchmark and deterministic report row identities differ"
        )
    mapped_keys = {
        (str(row["question_id"]), str(row["arm"]))
        for row in mapping_rows.values()
    }
    if mapped_keys != set(raw_rows):
        raise BenchmarkReviewError(
            "private mapping does not cover the complete benchmark matrix"
        )

    labels: dict[str, dict[str, Any]] = {}
    batch_hashes: dict[str, str] = {}
    for path in sorted(label_paths):
        data = path.read_bytes()
        batch_hashes[str(path.resolve())] = _sha256_bytes(data)
        for row in _parse_jsonl(data, str(path)):
            review_id = str(row.get("review_id", "")).strip()
            if not review_id or review_id in labels:
                raise BenchmarkReviewError(
                    "model labels contain missing or duplicate review IDs"
                )
            if set(row) != {
                "review_id",
                "reviewer_kind",
                "reviewer_identity",
                "score",
                "reason",
                "confidence",
                "needs_human_attention",
            }:
                raise BenchmarkReviewError(
                    f"model label {review_id} has an invalid schema"
                )
            if row.get("reviewer_kind") != "model":
                raise BenchmarkReviewError(
                    "model review cannot contain a human label"
                )
            if not str(row.get("reviewer_identity", "")).strip():
                raise BenchmarkReviewError("model reviewer identity is required")
            if not str(row.get("reason", "")).strip():
                raise BenchmarkReviewError("model review reason is required")
            if row.get("confidence") not in {"high", "medium", "low"}:
                raise BenchmarkReviewError(
                    "model confidence must be high, medium, or low"
                )
            if not isinstance(row.get("needs_human_attention"), bool):
                raise BenchmarkReviewError(
                    "needs_human_attention must be boolean"
                )
            _human_score(row.get("score"))
            labels[review_id] = row
    if set(labels) != packet_review_ids:
        raise BenchmarkReviewError(
            "model labels do not cover every blinded benchmark row"
        )

    scored_rows: list[dict[str, Any]] = []
    for review_id, private in mapping_rows.items():
        key = (str(private["question_id"]), str(private["arm"]))
        label = labels[review_id]
        direct_score = _human_score(label["score"])
        deterministic = deterministic_rows[key]
        status = str(deterministic.get("status", ""))
        if status == "deterministic_hard_fail":
            governed_score = 0.0
            score_source = "deterministic_hard_fail"
        elif status == "semantic_review_required":
            governed_score = direct_score
            score_source = "model_advisory"
        else:
            raise BenchmarkReviewError(
                f"unsupported deterministic status for model review: {status}"
            )
        scored_rows.append(
            {
                "review_id": review_id,
                "case_id": str(private["case_id"]),
                "question_id": key[0],
                "arm": key[1],
                "suite": str(private["suite"]),
                "question_family_id": str(
                    deterministic.get("question_family_id", "")
                ),
                "deterministic_status": status,
                "direct_human_score": direct_score,
                "governed_final_score": governed_score,
                "score_source": score_source,
                "confidence": str(label["confidence"]),
                "needs_human_attention": label["needs_human_attention"],
            }
        )
    aggregates = _aggregate_scores(scored_rows)
    aggregates = [
        {
            **item,
            "direct_model": item["direct_human"],
        }
        for item in aggregates
    ]
    for item in aggregates:
        del item["direct_human"]
    paired = _paired_parametric_base(scored_rows)
    apparent_result, criteria = _continuation_assessment(aggregates, paired)
    merged_rows = [labels[review_id] for review_id in sorted(labels)]
    merged_content = _encode_jsonl(merged_rows)
    labels_path = output_dir / "model_labels.jsonl"
    atomic_write_text(labels_path, merged_content.decode("utf-8"))
    labels_hash = _sha256_bytes(merged_content)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "model_triage_complete_human_review_pending",
        "reviewer_kind": "model",
        "reviewer_identities": sorted(
            {str(row["reviewer_identity"]) for row in merged_rows}
        ),
        "case_count": len(merged_rows),
        "human_approved": False,
        "human_review_satisfied": False,
        "usable_for_judge_certification": False,
        "source_packet_sha256": _sha256_file(packet_path),
        "private_mapping_sha256": mapping_hash,
        "benchmark_sha256": benchmark_hash,
        "deterministic_grading_sha256": grading_hash,
        "batch_hashes": batch_hashes,
        "model_labels_sha256": labels_hash,
    }
    manifest_path = output_dir / "model_labels.manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    report_rows = [
        {
            **{
                key: value
                for key, value in row.items()
                if key != "direct_human_score"
            },
            "direct_model_score": row["direct_human_score"],
        }
        for row in scored_rows
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "report_type": "smoke_model_review_advisory",
        "status": "model_triage_complete_human_review_pending",
        "operative_research_decision": "human_review_pending",
        "apparent_rule_result_if_labels_were_human": apparent_result,
        "criteria": criteria,
        "continuation_rule": CONTINUATION_RULE,
        "promotion_eligible": False,
        "headline_accuracy_claims_allowed": False,
        "usable_for_judge_certification": False,
        "case_count": len(scored_rows),
        "artifact_chain": {
            "benchmark_sha256": benchmark_hash,
            "deterministic_grading_sha256": grading_hash,
            "packet_sha256": _sha256_file(packet_path),
            "private_mapping_sha256": mapping_hash,
            "model_labels_sha256": labels_hash,
        },
        "by_arm": aggregates,
        "paired_parametric_vs_base": paired,
        "attention_required_count": sum(
            row["needs_human_attention"] is True for row in scored_rows
        ),
        "rows": sorted(
            report_rows,
            key=lambda row: (str(row["question_id"]), str(row["arm"])),
        ),
    }
    json_path = output_dir / "model_advisory.json"
    markdown_path = output_dir / "model_advisory.md"
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(markdown_path, _render_model_advisory_markdown(report))
    return ModelReviewAdvisoryPaths(
        labels_path=labels_path,
        manifest_path=manifest_path,
        json_path=json_path,
        markdown_path=markdown_path,
        apparent_rule_result=apparent_result,
    )


def _validate_source_artifacts(
    *,
    benchmark: Mapping[str, Any],
    benchmark_hash: str,
    grading: Mapping[str, Any],
    suites: EvalSuites,
    fixture_hash: str,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Any],
]:
    if benchmark.get("graded") is not False:
        raise BenchmarkReviewError("expected an ungraded benchmark artifact")
    if grading.get("graded") is not True or grading.get("mode") != "deterministic_only":
        raise BenchmarkReviewError("expected a deterministic-only grading report")
    if benchmark.get("fixture_hash") != fixture_hash:
        raise BenchmarkReviewError("benchmark fixture hash mismatch")
    if grading.get("fixture_hash") != fixture_hash:
        raise BenchmarkReviewError("deterministic report fixture hash mismatch")
    if grading.get("raw_artifact_hash") != benchmark_hash:
        raise BenchmarkReviewError("deterministic report raw-artifact hash mismatch")
    raw_value = benchmark.get("results")
    if not isinstance(raw_value, list) or not raw_value:
        raise BenchmarkReviewError("benchmark contains no result rows")
    raw_rows: list[dict[str, Any]] = []
    for index, value in enumerate(raw_value):
        if not isinstance(value, dict):
            raise BenchmarkReviewError(f"benchmark row {index} is not an object")
        if value.get("generation_status") != "generated":
            raise BenchmarkReviewError(
                "all rows must be generated before blinded human review"
            )
        if not isinstance(value.get("output"), str):
            raise BenchmarkReviewError(f"benchmark row {index} has no output")
        raw_rows.append(value)
    raw_by_key = _keyed_rows(raw_rows, "benchmark")
    grading_by_key = _keyed_rows(grading.get("rows"), "deterministic grading")
    if set(raw_by_key) != set(grading_by_key):
        raise BenchmarkReviewError(
            "benchmark and deterministic report row identities differ"
        )
    questions = _questions_by_id(suites)
    for raw in raw_rows:
        question_id = str(raw["question_id"])
        if question_id not in questions:
            raise BenchmarkReviewError(
                f"benchmark contains unknown frozen question: {question_id}"
            )
        if str(raw.get("question", "")) != questions[question_id].question:
            raise BenchmarkReviewError(
                f"benchmark question text mismatch: {question_id}"
            )
    return raw_rows, grading_by_key, questions


def _questions_by_id(suites: EvalSuites) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for question in suites.all_questions():
        if question.question_id in questions:
            raise BenchmarkReviewError(
                f"duplicate frozen question ID: {question.question_id}"
            )
        questions[question.question_id] = question
    return questions


def _keyed_rows(
    value: Any,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, list):
        raise BenchmarkReviewError(f"{label} rows must be a list")
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise BenchmarkReviewError(f"{label} row {index} is not an object")
        question_id = str(row.get("question_id", "")).strip()
        arm = str(row.get("arm", "")).strip()
        if not question_id or not arm:
            raise BenchmarkReviewError(
                f"{label} row {index} has no question ID or arm"
            )
        key = (question_id, arm)
        if key in keyed:
            raise BenchmarkReviewError(f"{label} contains duplicate row {key}")
        keyed[key] = row
    return keyed


def _mapping_rows(
    mapping: Mapping[str, Any],
    packet_review_ids: set[str],
) -> dict[str, dict[str, Any]]:
    value = mapping.get("mapping")
    if not isinstance(value, list):
        raise BenchmarkReviewError("private mapping has no rows")
    rows: dict[str, dict[str, Any]] = {}
    case_ids: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise BenchmarkReviewError(f"private mapping row {index} is invalid")
        required = ("review_id", "case_id", "question_id", "arm", "suite")
        if any(not str(row.get(field, "")).strip() for field in required):
            raise BenchmarkReviewError(
                f"private mapping row {index} is missing required fields"
            )
        review_id = str(row["review_id"])
        case_id = str(row["case_id"])
        if review_id in rows or case_id in case_ids:
            raise BenchmarkReviewError("private mapping IDs are duplicated")
        expected_case_id = _case_id(str(row["question_id"]), str(row["arm"]))
        if case_id != expected_case_id:
            raise BenchmarkReviewError(
                "private mapping case ID does not match its question and arm"
            )
        rows[review_id] = row
        case_ids.add(case_id)
    if set(rows) != packet_review_ids:
        raise BenchmarkReviewError(
            "private mapping IDs do not match the blinded packet"
        )
    return rows


def _verified_packet(path: Path) -> tuple[dict[str, Any], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"review packet not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifests = [name for name in names if name.endswith("/packet_manifest.json")]
        if len(manifests) != 1:
            raise BenchmarkReviewError("packet must contain one manifest")
        manifest_name = manifests[0]
        prefix = manifest_name.removesuffix("packet_manifest.json")
        manifest = _json_object(
            archive.read(manifest_name), "packet manifest"
        )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise BenchmarkReviewError("packet manifest has no file hashes")
        for relative, expected in files.items():
            member = prefix + str(relative)
            if member not in names:
                raise BenchmarkReviewError(f"packet member missing: {relative}")
            if _sha256_bytes(archive.read(member)) != expected:
                raise BenchmarkReviewError(
                    f"packet member hash mismatch: {relative}"
                )
        case_bytes = archive.read(prefix + "review_cases.jsonl")
    cases = _parse_jsonl(case_bytes, "review cases")
    if len(cases) != manifest.get("case_count"):
        raise BenchmarkReviewError("packet case count mismatch")
    review_ids = {str(row.get("review_id", "")) for row in cases}
    if "" in review_ids or len(review_ids) != len(cases):
        raise BenchmarkReviewError("packet review IDs are missing or duplicated")
    if any(_FORBIDDEN_BLINDED_FIELDS & row.keys() for row in cases):
        raise BenchmarkReviewError("packet exposes private benchmark fields")
    return manifest, review_ids


def _verified_overlay(
    *,
    overlay_bytes: bytes,
    overlay_manifest: Mapping[str, Any],
    packet_path: Path,
    mapping_path: Path,
    expected_case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if overlay_manifest.get("status") != (
        "single_human_review_complete_not_adjudicated"
    ):
        raise BenchmarkReviewError("human review is incomplete or invalid")
    if overlay_manifest.get("source_packet_sha256") != _sha256_file(packet_path):
        raise BenchmarkReviewError("overlay belongs to another review packet")
    if overlay_manifest.get("private_mapping_sha256") != _sha256_file(mapping_path):
        raise BenchmarkReviewError("overlay belongs to another private mapping")
    if overlay_manifest.get("overlay_sha256") != _sha256_bytes(overlay_bytes):
        raise BenchmarkReviewError("overlay byte hash mismatch")
    if overlay_manifest.get("human_approved") is not False:
        raise BenchmarkReviewError(
            "single-review diagnostic overlay must remain unapproved"
        )
    if overlay_manifest.get("requires_second_review") is not True:
        raise BenchmarkReviewError(
            "single-review diagnostic must remain marked for second review"
        )
    if overlay_manifest.get("identity_verification") != (
        "asserted_only_not_authenticated"
    ):
        raise BenchmarkReviewError("overlay reviewer identity boundary is missing")
    rows = _parse_jsonl(overlay_bytes, "human overlay")
    if len(rows) != overlay_manifest.get("case_count"):
        raise BenchmarkReviewError("overlay case count mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in by_id:
            raise BenchmarkReviewError(
                f"human overlay row {index} has missing or duplicate case ID"
            )
        if row.get("reviewer_kind") != "human":
            raise BenchmarkReviewError("human overlay contains a non-human label")
        if row.get("human_approved") is not False:
            raise BenchmarkReviewError(
                "single-review overlay must not claim human approval"
            )
        _human_score(row.get("human_semantic_score"))
        by_id[case_id] = row
    if set(by_id) != expected_case_ids:
        raise BenchmarkReviewError(
            "human overlay does not cover every blinded benchmark row"
        )
    return by_id


def _aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return [
        {
            "arm": arm,
            "n": len(group),
            "deterministic_hard_fail_count": sum(
                row["deterministic_status"] == "deterministic_hard_fail"
                for row in group
            ),
            "needs_human_attention_count": sum(
                row["needs_human_attention"] is True for row in group
            ),
            "direct_human": _score_summary(
                [float(row["direct_human_score"]) for row in group]
            ),
            "governed_final": _score_summary(
                [float(row["governed_final_score"]) for row in group]
            ),
        }
        for arm, group in sorted(grouped.items())
    ]


def _score_summary(scores: Sequence[float]) -> dict[str, Any]:
    if not scores:
        raise BenchmarkReviewError("cannot aggregate an empty score group")
    fully_correct = sum(score == 1.0 for score in scores)
    ci_low, ci_high = clopper_pearson_two_sided(fully_correct, len(scores))
    return {
        "mean_score": sum(scores) / len(scores),
        "fully_correct_count": fully_correct,
        "fully_correct_rate": fully_correct / len(scores),
        "fully_correct_rate_ci_95": {
            "low": ci_low,
            "high": ci_high,
            "interval": "clopper_pearson_two_sided",
            "role": "descriptive",
        },
        "partial_count": sum(score == 0.5 for score in scores),
        "incorrect_count": sum(score == 0.0 for score in scores),
    }


def _paired_parametric_base(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["question_id"]), str(row["arm"]))
        scores[key] = float(row["governed_final_score"])
    base_ids = {question_id for question_id, arm in scores if arm == "base"}
    parametric_ids = {
        question_id for question_id, arm in scores if arm == "parametric"
    }
    if base_ids != parametric_ids or not base_ids:
        raise BenchmarkReviewError(
            "paired decision requires matching base and parametric questions"
        )
    wins = losses = ties = 0
    for question_id in sorted(base_ids):
        parametric = scores[(question_id, "parametric")]
        base = scores[(question_id, "base")]
        if parametric > base:
            wins += 1
        elif parametric < base:
            losses += 1
        else:
            ties += 1
    return {
        "paired_question_count": len(base_ids),
        "parametric_wins": wins,
        "parametric_losses": losses,
        "ties": ties,
    }


def _continuation_assessment(
    aggregates: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> tuple[DecisionStatus, dict[str, dict[str, Any]]]:
    by_arm = {str(row["arm"]): row for row in aggregates}
    if "parametric" not in by_arm or "base" not in by_arm:
        raise BenchmarkReviewError("decision requires base and parametric arms")
    parametric = by_arm["parametric"]["governed_final"]
    base = by_arm["base"]["governed_final"]
    mean_delta = float(parametric["mean_score"]) - float(base["mean_score"])
    fully_correct_delta = int(parametric["fully_correct_count"]) - int(
        base["fully_correct_count"]
    )
    criteria = {
        "mean_score_delta": {
            "observed": mean_delta,
            "required_minimum": CONTINUATION_RULE[
                "mean_score_delta_over_base_min"
            ],
            "passed": mean_delta
            >= float(CONTINUATION_RULE["mean_score_delta_over_base_min"]),
        },
        "fully_correct_count_delta": {
            "observed": fully_correct_delta,
            "required_minimum": CONTINUATION_RULE[
                "fully_correct_count_delta_over_base_min"
            ],
            "passed": fully_correct_delta
            >= int(CONTINUATION_RULE["fully_correct_count_delta_over_base_min"]),
        },
        "paired_wins_exceed_losses": {
            "wins": paired["parametric_wins"],
            "losses": paired["parametric_losses"],
            "passed": paired["parametric_wins"] > paired["parametric_losses"],
        },
    }
    authorized = all(bool(item["passed"]) for item in criteria.values())
    result: DecisionStatus = (
        "authorize_one_redesigned_diagnostic"
        if authorized
        else "stop_parametric_research"
    )
    return result, criteria


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Blinded smoke decision",
        "",
        f"Decision: **{report['decision']['status']}**",
        "",
        "This is a single-human, non-promotable diagnostic. It cannot certify "
        "judges or support headline accuracy.",
        "",
        "| Arm | N | Hard fails | Human mean | Governed mean | Governed fully correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["by_arm"]:
        direct = item["direct_human"]
        governed = item["governed_final"]
        lines.append(
            f"| {item['arm']} | {item['n']} | "
            f"{item['deterministic_hard_fail_count']} | "
            f"{direct['mean_score']:.3f} | {governed['mean_score']:.3f} | "
            f"{governed['fully_correct_count']}/{item['n']} |"
        )
    paired = report["paired_parametric_vs_base"]
    lines.extend(
        [
            "",
            "## Paired parametric versus base",
            "",
            f"- Wins: {paired['parametric_wins']}",
            f"- Losses: {paired['parametric_losses']}",
            f"- Ties: {paired['ties']}",
            "",
            "## Continuation criteria",
            "",
        ]
    )
    for name, criterion in report["decision"]["criteria"].items():
        lines.append(
            f"- {name}: {'pass' if criterion['passed'] else 'fail'} "
            f"({canonical_json(criterion)})"
        )
    return "\n".join(lines) + "\n"


def _render_model_advisory_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Smoke model-review advisory",
        "",
        "**Status: model triage complete; human review still pending.**",
        "",
        "The apparent stopping-rule result below is advisory only. These labels "
        "were produced by a model, are not human-approved, cannot certify a "
        "judge, and do not authorize another experiment.",
        "",
        "Apparent rule result if labels were human: "
        f"**{report['apparent_rule_result_if_labels_were_human']}**",
        "",
        "| Arm | N | Hard fails | Model mean | Governed mean | Governed fully correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["by_arm"]:
        direct = item["direct_model"]
        governed = item["governed_final"]
        lines.append(
            f"| {item['arm']} | {item['n']} | "
            f"{item['deterministic_hard_fail_count']} | "
            f"{direct['mean_score']:.3f} | {governed['mean_score']:.3f} | "
            f"{governed['fully_correct_count']}/{item['n']} |"
        )
    paired = report["paired_parametric_vs_base"]
    lines.extend(
        [
            "",
            "## Paired parametric versus base",
            "",
            f"- Wins: {paired['parametric_wins']}",
            f"- Losses: {paired['parametric_losses']}",
            f"- Ties: {paired['ties']}",
            f"- Cases flagged for human attention: {report['attention_required_count']}",
            "",
            "## Operative decision",
            "",
            "`human_review_pending` — the model cannot attest on behalf of a human.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fixture_hash(eval_dir: Path) -> str:
    manifest = _json_object(
        (eval_dir / "freeze_manifest.json").read_bytes(), "freeze manifest"
    )
    value = str(manifest.get("combined_hash", "")).strip()
    if not value:
        raise BenchmarkReviewError("freeze manifest has no combined hash")
    return value


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkReviewError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise BenchmarkReviewError(f"{label} must be a JSON object")
    return value


def _parse_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkReviewError(f"{label} is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkReviewError(
                f"{label}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise BenchmarkReviewError(
                f"{label}:{line_number} must be an object"
            )
        rows.append(value)
    return rows


def _encode_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def _case_id(question_id: str, arm: str) -> str:
    return f"{question_id}::{arm}"


def _human_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkReviewError("human score must be 0.0, 0.5, or 1.0")
    score = float(value)
    if score not in LEGAL_HUMAN_SCORES:
        raise BenchmarkReviewError("human score must be 0.0, 0.5, or 1.0")
    return score


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())
