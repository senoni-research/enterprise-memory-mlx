"""Governed compiler for non-promotable and confirmatory acquisition datasets."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .leakage import assert_no_leakage, scan_leakage
from .schemas import KnowledgeRecord
from .semantic_neighbors import EmbeddingBackend, nearest_neighbors
from .split_contract import (
    load_eval_suites,
    validate_split_contract,
    verify_frozen_assets,
)
from .utils import atomic_write_text, read_jsonl, sha256_json

AcquisitionProfile = Literal["smoke_non_promotable", "confirmatory"]

ACQUISITION_SYSTEM_PROMPT = (
    "Learn the approved company record exactly. Preserve numbers, units, "
    "conditions, exceptions, dates, and provenance. Do not invent a rule."
)
ANSWER_SYSTEM_PROMPT = (
    "Answer from approved company knowledge. Preserve factual constraints and "
    "cite the source record as [record: ID]. Do not guess."
)
MIN_CONFIRMATORY_VIEW_FAMILIES = 24


@dataclass(frozen=True)
class ExposureSchedule:
    target_exposures_per_fact: int
    views_per_fact: dict[str, int]
    epochs: int
    total_rows: int
    effective_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    optimizer_steps_per_epoch: int
    optimizer_steps: int
    micro_iterations_per_epoch: int
    micro_iterations: int
    realized_exposures_per_fact: dict[str, int]
    exposure_overshoot_per_fact: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_exposures_per_fact": self.target_exposures_per_fact,
            "views_per_fact": dict(sorted(self.views_per_fact.items())),
            "epochs": self.epochs,
            "total_rows": self.total_rows,
            "effective_batch_size": self.effective_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "optimizer_steps": self.optimizer_steps,
            "micro_iterations_per_epoch": self.micro_iterations_per_epoch,
            "micro_iterations": self.micro_iterations,
            "realized_exposures_per_fact": dict(
                sorted(self.realized_exposures_per_fact.items())
            ),
            "exposure_overshoot_per_fact": dict(
                sorted(self.exposure_overshoot_per_fact.items())
            ),
        }


@dataclass(frozen=True)
class AcquisitionCompilation:
    dataset_dir: Path
    manifest_path: Path
    audit_path: Path
    records: tuple[KnowledgeRecord, ...]
    rows: tuple[dict[str, Any], ...]
    schedule: ExposureSchedule
    promotion_eligible: bool


def compile_acquisition_dataset(
    *,
    knowledge_dir: Path,
    eval_dir: Path,
    output_root: Path,
    semantic_backend: EmbeddingBackend,
    profile: AcquisitionProfile,
    target_exposures_per_fact: int,
    effective_batch_size: int,
    micro_batch_size: int = 1,
    seed: int = 42,
    semantic_fail_threshold: float = 0.985,
    semantic_audit_approved_by: str | None = None,
) -> AcquisitionCompilation:
    """Compile only after every contract/leakage check passes."""
    if profile not in {"smoke_non_promotable", "confirmatory"}:
        raise ValueError(f"Unsupported acquisition profile: {profile}")
    if target_exposures_per_fact <= 0:
        raise ValueError("target_exposures_per_fact must be positive")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if not 0.0 < semantic_fail_threshold <= 1.0:
        raise ValueError("semantic_fail_threshold must be in (0, 1]")

    freeze_problems = verify_frozen_assets(eval_dir)
    if freeze_problems:
        raise ValueError(
            "Frozen evaluation assets failed verification:\n"
            + "\n".join(freeze_problems)
        )
    records = tuple(
        record for record in _load_governed_records(knowledge_dir) if record.is_trainable()
    )
    if not records:
        raise ValueError("No eligible governed records")
    suites = load_eval_suites(eval_dir)
    violations = validate_split_contract(list(records), suites)
    if violations:
        raise ValueError(
            "Split contract violations:\n"
            + "\n".join(str(item) for item in violations)
        )

    rows = tuple(_build_study_views(records))
    _validate_view_identity(rows)
    view_counts = Counter(str(row["record_id"]) for row in rows)
    deficits = {
        record.id: MIN_CONFIRMATORY_VIEW_FAMILIES - view_counts[record.id]
        for record in records
        if view_counts[record.id] < MIN_CONFIRMATORY_VIEW_FAMILIES
    }
    if profile == "confirmatory" and deficits:
        raise ValueError(
            "Confirmatory acquisition requires at least "
            f"{MIN_CONFIRMATORY_VIEW_FAMILIES} independent view families per fact; "
            f"deficits={deficits}"
        )
    if profile == "confirmatory" and not semantic_audit_approved_by:
        raise ValueError("Confirmatory acquisition requires an approved semantic audit")

    training_texts = {
        str(row["view_family_id"]): _user_text(row)
        for row in rows
    }
    lexical_report = scan_leakage(training_texts, suites.all_questions())
    assert_no_leakage(lexical_report)

    eval_pairs = [
        (question.question_id, question.question)
        for question in suites.all_questions()
    ]
    semantic = nearest_neighbors(
        list(training_texts.items()),
        eval_pairs,
        semantic_backend,
        top_n=min(100, len(training_texts) * max(1, len(eval_pairs))),
    )
    semantic_findings = [
        item for item in semantic if item.score >= semantic_fail_threshold
    ]
    if semantic_findings:
        details = "\n".join(
            f"{item.left_id} ~ {item.right_id}: {item.score:.4f}"
            for item in semantic_findings[:10]
        )
        raise ValueError(f"Semantic-neighbour leakage threshold exceeded:\n{details}")

    schedule = derive_exposure_schedule(
        view_counts=dict(view_counts),
        target_exposures_per_fact=target_exposures_per_fact,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
    )
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    dataset_dir = output_root / "datasets" / profile
    manifest_path = output_root / "manifests" / f"{profile}.json"
    audit_path = output_root / "audits" / f"{profile}.json"
    dataset_content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in shuffled
    )
    max_semantic_similarity = max((item.score for item in semantic), default=None)
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "lexical": lexical_report.to_dict(),
        "semantic": {
            "model_id": semantic_backend.model_id,
            "revision": semantic_backend.revision,
            "fail_threshold": semantic_fail_threshold,
            "approved_by": semantic_audit_approved_by,
            "max_similarity_to_eval": max_semantic_similarity,
            "findings": [item.to_dict() for item in semantic_findings],
            "nearest_pairs": [item.to_dict() for item in semantic[:25]],
        },
        "promotion_eligible": profile == "confirmatory",
    }
    audit_content = json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    fixture_manifest = json.loads(
        (eval_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    source_payload = [_record_payload(record) for record in records]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "promotion_eligible": profile == "confirmatory",
        "non_promotable_reasons": (
            []
            if profile == "confirmatory"
            else [
                "smoke profile",
                "fewer than 24 independent view families per fact",
                "semantic audit has not received independent human approval",
                "semantic answer grading is unavailable",
            ]
        ),
        "known_caveats": [
            (
                "training includes governed source-question views that are "
                "semantically adjacent to the frozen evaluation paraphrases "
                "(max cross similarity "
                + (
                    f"{max_semantic_similarity:.4f}"
                    if max_semantic_similarity is not None
                    else "unavailable"
                )
                + " under the pinned audit model); results on this dataset "
                "are not a clean unseen-form test"
            ),
        ],
        "records": source_payload,
        "record_ids": [record.id for record in records],
        "source_snapshot_hash": sha256_json(source_payload),
        "highest_sensitivity": _highest_sensitivity(records),
        "fixture_hash": fixture_manifest["combined_hash"],
        "dataset_sha256": hashlib.sha256(dataset_content.encode()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_content.encode()).hexdigest(),
        "view_family_count": len(rows),
        "view_counts": dict(sorted(view_counts.items())),
        "confirmatory_view_deficits": deficits,
        "schedule": schedule.to_dict(),
        "seed": seed,
    }

    atomic_write_text(dataset_dir / "train.jsonl", dataset_content)
    atomic_write_text(audit_path, audit_content)
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return AcquisitionCompilation(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
        records=records,
        rows=rows,
        schedule=schedule,
        promotion_eligible=profile == "confirmatory",
    )


def derive_exposure_schedule(
    *,
    view_counts: dict[str, int],
    target_exposures_per_fact: int,
    effective_batch_size: int,
    micro_batch_size: int = 1,
) -> ExposureSchedule:
    if not view_counts or any(value <= 0 for value in view_counts.values()):
        raise ValueError("view_counts must contain positive counts")
    if (
        target_exposures_per_fact <= 0
        or effective_batch_size <= 0
        or micro_batch_size <= 0
    ):
        raise ValueError("target exposures and batch sizes must be positive")
    if effective_batch_size % micro_batch_size:
        raise ValueError("effective batch size must be divisible by micro batch size")
    minimum_views = min(view_counts.values())
    epochs = math.ceil(target_exposures_per_fact / minimum_views)
    total_rows = sum(view_counts.values())
    steps_per_epoch = math.ceil(total_rows / effective_batch_size)
    micro_iterations_per_epoch = math.ceil(total_rows / micro_batch_size)
    gradient_accumulation_steps = effective_batch_size // micro_batch_size
    realized = {
        record_id: count * epochs
        for record_id, count in view_counts.items()
    }
    return ExposureSchedule(
        target_exposures_per_fact=target_exposures_per_fact,
        views_per_fact=dict(view_counts),
        epochs=epochs,
        total_rows=total_rows,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_steps_per_epoch=steps_per_epoch,
        optimizer_steps=steps_per_epoch * epochs,
        micro_iterations_per_epoch=micro_iterations_per_epoch,
        micro_iterations=micro_iterations_per_epoch * epochs,
        realized_exposures_per_fact=realized,
        # Whole-epoch rounding can overshoot the target; make that explicit
        # instead of letting readers assume target == realized.
        exposure_overshoot_per_fact={
            record_id: max(0, count - target_exposures_per_fact)
            for record_id, count in realized.items()
        },
    )


def _load_governed_records(knowledge_dir: Path) -> tuple[KnowledgeRecord, ...]:
    paths = []
    primary = knowledge_dir / "records.jsonl"
    if primary.is_file():
        paths.append(primary)
    records_dir = knowledge_dir / "records.d"
    if records_dir.is_dir():
        paths.extend(sorted(records_dir.glob("*.jsonl")))
    if not paths:
        raise FileNotFoundError("No governed record files found")
    records = []
    seen = set()
    for path in paths:
        for raw in read_jsonl(path):
            record = KnowledgeRecord.from_dict(raw)
            if record.id in seen:
                raise ValueError(f"Duplicate governed record ID: {record.id}")
            seen.add(record.id)
            records.append(record)
    return tuple(sorted(records, key=lambda item: item.id))


def _build_study_views(
    records: Iterable[KnowledgeRecord],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        statement_words = record.statement.split()
        if len(statement_words) < 12:
            raise ValueError(f"Record {record.id} is too short for continuation views")
        rows.extend(
            [
                _chat_view(
                    record,
                    "full-reconstruction",
                    "Read and reproduce the complete approved record.",
                    record.statement,
                    "reconstruction",
                ),
                _chat_view(
                    record,
                    "summary-reconstruction",
                    f"Reconstruct the approved record from this summary:\n{record.summary}",
                    record.statement,
                    "reconstruction",
                ),
                _chat_view(
                    record,
                    "identity-reconstruction",
                    f"State record {record.id}: {record.title} ({record.domain}).",
                    record.statement,
                    "reconstruction",
                ),
            ]
        )
        for label, ratio in (("early", 0.35), ("middle", 0.55), ("late", 0.72)):
            split_at = max(
                5,
                min(len(statement_words) - 3, round(len(statement_words) * ratio)),
            )
            prefix = " ".join(statement_words[:split_at])
            suffix = " ".join(statement_words[split_at:])
            rows.append(
                _chat_view(
                    record,
                    f"continuation-{label}",
                    f"Continue this approved record exactly:\n{prefix}",
                    suffix,
                    "continuation",
                )
            )
        for index, item in enumerate(record.questions, start=1):
            rows.append(
                _chat_view(
                    record,
                    f"source-question-{index}",
                    item.question,
                    f"{item.answer}\n\n[record: {record.id}]",
                    "qa",
                    system=ANSWER_SYSTEM_PROMPT,
                )
            )
    return rows


def _chat_view(
    record: KnowledgeRecord,
    family: str,
    user: str,
    assistant: str,
    objective: str,
    *,
    system: str = ACQUISITION_SYSTEM_PROMPT,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "record_id": record.id,
        "source_document_id": record.id,
        "view_family_id": f"{record.id}:{family}",
        "generator_template_id": f"acquisition-v1:{family}",
        "objective": objective,
    }


def _validate_view_identity(rows: tuple[dict[str, Any], ...]) -> None:
    families = [str(row.get("view_family_id", "")) for row in rows]
    if any(not item for item in families):
        raise ValueError("Every study view requires view_family_id")
    if len(set(families)) != len(families):
        raise ValueError("Study view families must be unique")


def _user_text(row: dict[str, Any]) -> str:
    messages = row["messages"]
    return next(
        str(message["content"])
        for message in messages
        if message["role"] == "user"
    )


def _record_payload(record: KnowledgeRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "domain": record.domain,
        "title": record.title,
        "statement": record.statement,
        "summary": record.summary,
        "source_uri": record.source_uri,
        "sensitivity": record.sensitivity,
        "status": record.status,
        "effective_from": record.effective_from,
        "effective_to": record.effective_to,
        "aliases": list(record.aliases),
    }


def _highest_sensitivity(records: Iterable[KnowledgeRecord]) -> str:
    order = {"public": 0, "internal_shared": 1, "restricted": 2, "secret": 3}
    return max(
        (record.sensitivity for record in records),
        key=order.__getitem__,
    )
