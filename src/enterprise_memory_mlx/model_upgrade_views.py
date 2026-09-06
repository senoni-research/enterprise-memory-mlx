"""Versioned, source-bound study views for model-upgrade-exploratory/v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio

from .experiment_profiles import MODEL_UPGRADE_PROTOCOL_ID
from .schemas import KnowledgeRecord
from .utils import read_jsonl, sha256_json

VIEW_OBJECTIVES = (
    "full_reconstruction",
    "summary_reconstruction",
    "identity_reconstruction",
    "alias_cued_reconstruction",
    "domain_cued_reconstruction",
    "continuation_early",
    "continuation_middle",
    "continuation_late",
    "constraint_extraction",
    "exception_condition",
    "effective_window",
    "provenance_citation",
    "recall_primary_claim",
    "paraphrase_primary_claim",
    "application",
    "boundary_negative",
    "composition_multi_claim",
    "comparator_threshold",
    "entity_role",
    "procedure_order",
    "keyword_expansion",
    "structured_field_dump",
    "teach_back",
    "secondary_claim_focus",
)
VIEWS_PER_RECORD = len(VIEW_OBJECTIVES)
MAX_WITHIN_RECORD_PROMPT_RATIO = 90.0


@dataclass(frozen=True)
class VersionedStudyViews:
    root: Path
    manifest_path: Path
    views_path: Path
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    dataset_hash: str
    manifest_hash: str


def load_versioned_study_views(
    dataset_root: Path,
    records: Sequence[KnowledgeRecord],
) -> VersionedStudyViews:
    manifest_path = dataset_root / "manifest.json"
    views_path = dataset_root / "views.jsonl"
    if not manifest_path.is_file() or not views_path.is_file():
        raise FileNotFoundError(f"Versioned study-view dataset is incomplete: {dataset_root}")
    manifest_bytes = manifest_path.read_bytes()
    views_bytes = views_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Study-view manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Study-view manifest must be an object")
    rows = tuple(read_jsonl(views_path))
    _validate_manifest(manifest, views_bytes, rows)
    _validate_rows(rows, records)
    return VersionedStudyViews(
        root=dataset_root,
        manifest_path=manifest_path,
        views_path=views_path,
        manifest=manifest,
        rows=rows,
        dataset_hash=hashlib.sha256(views_bytes).hexdigest(),
        manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def build_claim_inventory(record: KnowledgeRecord) -> tuple[dict[str, Any], ...]:
    claims = []
    cursor = 0
    for index, sentence in enumerate(_sentences(record.statement), start=1):
        start = record.statement.index(sentence, cursor)
        end = start + len(sentence)
        cursor = end
        claims.append(
            {
                "claim_id": f"{record.id}:c{index}",
                "record_id": record.id,
                "text": sentence,
                "start": start,
                "end": end,
            }
        )
    if not claims:
        raise ValueError(f"Record {record.id} has no source claims")
    return tuple(claims)


def source_bound_assistant(
    record: KnowledgeRecord,
    claims: Sequence[Mapping[str, Any]],
) -> str:
    quoted = " ".join(str(claim["text"]).strip() for claim in claims)
    return f"{quoted}\n\n[record: {record.id}]"


def source_spans_for_claims(
    claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "start": int(claim["start"]),
            "end": int(claim["end"]),
            "text": str(claim["text"]),
        }
        for claim in claims
    ]


def authoring_source_payload(records: Iterable[KnowledgeRecord]) -> list[dict[str, Any]]:
    """Return the only fields an isolated authoring process may receive."""
    return [
        {
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
            "claims": list(build_claim_inventory(record)),
        }
        for record in sorted(records, key=lambda item: item.id)
    ]


def dataset_manifest_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[KnowledgeRecord],
    views_content: bytes,
    author_model_id: str,
    author_model_revision: str,
    author_prompt_hash: str,
    author_template_hash: str,
    fixture_hash: str,
) -> dict[str, Any]:
    source = authoring_source_payload(records)
    counts = Counter(str(row["record_id"]) for row in rows)
    return {
        "schema_version": 1,
        "protocol_id": MODEL_UPGRADE_PROTOCOL_ID,
        "dataset_version": "v1",
        "status": "model_assisted_not_human_approved",
        "human_approved": False,
        "promotion_eligible": False,
        "author": {
            "kind": "model",
            "model_id": author_model_id,
            "model_revision": author_model_revision,
            "prompt_hash": author_prompt_hash,
            "chat_template_hash": author_template_hash,
            "isolated_source_only_process": True,
        },
        "source_snapshot_hash": sha256_json(source),
        "source_snapshot": {
            "questions_stripped": True,
            "benchmark_outputs_excluded": True,
            "expected_answers_excluded": True,
            "eligible_record_ids": sorted(counts),
            "includes_holdout_records": False,
        },
        "fixture_hash": fixture_hash,
        "files": {"views.jsonl": hashlib.sha256(views_content).hexdigest()},
        "counts": {
            "rows": len(rows),
            "records": len(counts),
            "views_per_record": dict(sorted(counts.items())),
            "objectives_per_record": VIEWS_PER_RECORD,
        },
        "schedule_contract": {
            "total_rows": len(rows),
            "target_exposures_per_record": 96,
            "epochs": 4,
            "micro_iterations": len(rows) * 4,
            "optimizer_updates": (len(rows) * 4) // 8,
        },
        "classification": max(
            (record.sensitivity for record in records),
            key={"public": 0, "internal_shared": 1, "restricted": 2, "secret": 3}.__getitem__,
        ),
        "limitations": [
            "Study prompts are model-drafted with deterministic source-only "
            "fallbacks and have not been independently human-reviewed.",
            "Distinct objective IDs do not prove statistically independent formulations.",
            "Assistant targets are constructed from exact governed source spans.",
            "The historical evaluation suite has been repeatedly inspected and is diagnostic.",
        ],
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    views_bytes: bytes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if manifest.get("protocol_id") != MODEL_UPGRADE_PROTOCOL_ID:
        raise ValueError("Study views belong to another experiment protocol")
    if manifest.get("human_approved") is not False:
        raise ValueError("Model-assisted study views must not claim human approval")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("Exploratory study views must not be promotion eligible")
    source = manifest.get("source_snapshot")
    if not isinstance(source, Mapping) or source.get("questions_stripped") is not True:
        raise ValueError("Authoring source snapshot must attest that questions were stripped")
    files = manifest.get("files")
    expected_hash = files.get("views.jsonl") if isinstance(files, Mapping) else None
    if expected_hash != hashlib.sha256(views_bytes).hexdigest():
        raise ValueError("Study-view file hash does not match manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or counts.get("rows") != len(rows):
        raise ValueError("Study-view row count does not match manifest")


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[KnowledgeRecord],
) -> None:
    by_record = {record.id: record for record in records}
    expected_ids = set(by_record)
    if len(rows) != len(records) * VIEWS_PER_RECORD:
        raise ValueError(f"Expected {len(records) * VIEWS_PER_RECORD} study views; got {len(rows)}")
    seen_view_ids: set[str] = set()
    seen_families: set[str] = set()
    objectives: dict[str, set[str]] = {record_id: set() for record_id in expected_ids}
    user_texts: dict[str, list[str]] = {record_id: [] for record_id in expected_ids}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id not in by_record:
            raise ValueError(f"Study view references ineligible record: {record_id}")
        if row.get("protocol_id") != MODEL_UPGRADE_PROTOCOL_ID:
            raise ValueError("Study row protocol_id mismatch")
        if row.get("human_approved") is not False:
            raise ValueError("Study rows must preserve human_approved=false")
        view_id = str(row.get("view_id", ""))
        family_id = str(row.get("view_family_id", ""))
        if not view_id or view_id in seen_view_ids:
            raise ValueError("Study view IDs must be non-empty and unique")
        if not family_id or family_id in seen_families:
            raise ValueError("Study view families must be non-empty and unique")
        seen_view_ids.add(view_id)
        seen_families.add(family_id)
        objective = str(row.get("objective", ""))
        if objective not in VIEW_OBJECTIVES:
            raise ValueError(f"Unsupported study objective: {objective}")
        if objective in objectives[record_id]:
            raise ValueError(f"Duplicate objective for {record_id}: {objective}")
        objectives[record_id].add(objective)
        generator = row.get("generator")
        if not isinstance(generator, Mapping) or generator.get("kind") not in {
            "model",
            "deterministic_source_only_fallback",
        }:
            raise ValueError("Every study view requires auditable authorship metadata")
        if not str(generator.get("prompt_hash", "")):
            raise ValueError("Study view authoring prompt hash is required")
        if generator.get("kind") == "model" and not str(generator.get("model_id", "")):
            raise ValueError("Model-authored study views require model identity")
        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError("Study view messages must be system/user/assistant")
        user = str(messages[1].get("content", "")).strip()
        assistant = str(messages[2].get("content", "")).strip()
        if not user or not assistant:
            raise ValueError("Study view user and assistant text cannot be empty")
        _validate_source_binding(row, by_record[record_id], assistant)
        user_texts[record_id].append(user)
    for record_id in sorted(expected_ids):
        if objectives[record_id] != set(VIEW_OBJECTIVES):
            missing = sorted(set(VIEW_OBJECTIVES) - objectives[record_id])
            raise ValueError(f"{record_id} is missing study objectives: {missing}")
        _reject_near_duplicate_prompts(record_id, user_texts[record_id])


def _validate_source_binding(
    row: Mapping[str, Any],
    record: KnowledgeRecord,
    assistant: str,
) -> None:
    claims = {claim["claim_id"]: claim for claim in build_claim_inventory(record)}
    claim_ids = row.get("claim_ids")
    if not isinstance(claim_ids, list) or not claim_ids:
        raise ValueError("Study view requires non-empty claim_ids")
    if any(str(claim_id) not in claims for claim_id in claim_ids):
        raise ValueError(f"Study view has unknown claim IDs for {record.id}")
    spans = row.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("Study view requires source spans")
    span_texts = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise ValueError("Source span must be an object")
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        text = str(span.get("text", ""))
        if start < 0 or end <= start or record.statement[start:end] != text:
            raise ValueError(f"Study view source span is not exact for {record.id}")
        span_texts.append(text)
    expected = source_bound_assistant(
        record,
        [claims[str(claim_id)] for claim_id in claim_ids],
    )
    if assistant != expected:
        raise ValueError(f"Assistant target for {record.id} is not the exact source-bound target")
    if [claims[str(claim_id)]["text"] for claim_id in claim_ids] != span_texts:
        raise ValueError("Claim IDs and source spans are not aligned")


def _reject_near_duplicate_prompts(record_id: str, prompts: Sequence[str]) -> None:
    for left_index, left in enumerate(prompts):
        for right in prompts[left_index + 1 :]:
            if ratio(left.casefold(), right.casefold()) >= MAX_WITHIN_RECORD_PROMPT_RATIO:
                raise ValueError(f"{record_id} contains near-duplicate study prompts")


def _sentences(statement: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in statement.replace("? ", "?\n").replace(". ", ".\n").splitlines()
        if part.strip()
    )
