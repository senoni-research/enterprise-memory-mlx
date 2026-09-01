from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import KnowledgeRecord, QuestionAnswer
from .utils import atomic_write_text, read_jsonl, sha256_json, slugify, write_jsonl

SYSTEM_PROMPT = (
    "Answer from the company's approved internal knowledge. Be concise and do not invent a rule. "
    "If no authoritative record covers the question, say that the current source "
    "system must be checked."
)
UNKNOWN_ANSWER = (
    "I do not have an authoritative company record for that. Check the current source system "
    "instead of guessing."
)


@dataclass(frozen=True)
class CompilationResult:
    output_dir: Path
    records_included: int
    records_excluded: int
    domains: tuple[str, ...]
    manifest_path: Path


@dataclass(frozen=True)
class DatasetBundle:
    inject_train: list[dict[str, Any]]
    inject_valid: list[dict[str, Any]]
    inject_test: list[dict[str, Any]]
    align_train: list[dict[str, Any]]
    align_valid: list[dict[str, Any]]
    align_test: list[dict[str, Any]]
    recover_train: list[dict[str, Any]]
    recover_valid: list[dict[str, Any]]
    recover_test: list[dict[str, Any]]
    domain_eval: list[dict[str, Any]]
    retention_eval: list[dict[str, Any]]


def compile_knowledge(
    knowledge_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    include_restricted: bool = False,
    per_domain: bool = True,
    allow_scientifically_invalid: bool = False,
) -> CompilationResult:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "compiler.compile_knowledge",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    records = load_records(knowledge_dir)
    included = [record for record in records if record.is_trainable(include_restricted)]
    excluded = [record for record in records if record not in included]
    if not included:
        raise ValueError("No active, trainable knowledge records were found")

    unknowns = _load_optional_rows(knowledge_dir / "unknown_questions.jsonl")
    general_replay = _load_optional_rows(knowledge_dir / "general_replay.jsonl")
    general_eval = _load_optional_rows(knowledge_dir / "general_eval.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset_bundle(included, unknowns, general_replay, seed=seed)
    _write_bundle(output_dir, bundle)
    _write_general_eval(output_dir, general_eval)

    domain_names = tuple(sorted({record.domain for record in included}))
    if per_domain:
        for domain in domain_names:
            domain_records = [record for record in included if record.domain == domain]
            domain_bundle = build_dataset_bundle(
                domain_records,
                unknowns=[],
                general_replay=general_replay,
                seed=seed,
            )
            _write_bundle(output_dir / "domains" / slugify(domain), domain_bundle)

    manifest = _make_manifest(
        records=included,
        excluded=excluded,
        seed=seed,
        include_restricted=include_restricted,
        output_dir=output_dir,
        bundle=bundle,
    )
    manifest_path = output_dir.parent / "manifests" / "knowledge_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return CompilationResult(
        output_dir=output_dir,
        records_included=len(included),
        records_excluded=len(excluded),
        domains=domain_names,
        manifest_path=manifest_path,
    )


def load_records(knowledge_dir: Path) -> list[KnowledgeRecord]:
    paths: list[Path] = []
    primary = knowledge_dir / "records.jsonl"
    if primary.exists():
        paths.append(primary)
    records_dir = knowledge_dir / "records.d"
    if records_dir.exists():
        paths.extend(sorted(records_dir.glob("*.jsonl")))
    if not paths:
        raise FileNotFoundError(
            f"Expected {primary} or one or more JSONL files under {records_dir}"
        )

    records: list[KnowledgeRecord] = []
    seen_ids: set[str] = set()
    for path in paths:
        for raw in read_jsonl(path):
            record = KnowledgeRecord.from_dict(raw)
            if record.id in seen_ids:
                raise ValueError(f"Duplicate knowledge record id: {record.id}")
            seen_ids.add(record.id)
            records.append(record)
    return records


def build_dataset_bundle(
    records: list[KnowledgeRecord],
    unknowns: list[dict[str, Any]],
    general_replay: list[dict[str, Any]],
    *,
    seed: int,
) -> DatasetBundle:
    inject_train: list[dict[str, Any]] = []
    inject_valid: list[dict[str, Any]] = []
    inject_test: list[dict[str, Any]] = []
    align_train: list[dict[str, Any]] = []
    align_valid: list[dict[str, Any]] = []
    align_test: list[dict[str, Any]] = []
    domain_eval: list[dict[str, Any]] = []
    retention_eval: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: item.id):
        rng = random.Random(f"{seed}:{record.id}")
        injection_examples = _injection_examples(record)
        rng.shuffle(injection_examples)
        inject_train.extend(injection_examples[:4])
        inject_valid.extend(injection_examples[4:5])
        inject_test.extend(injection_examples[5:6])

        aligned = _alignment_examples(record)
        rng.shuffle(aligned)
        train_rows, valid_rows, test_rows = _split_rows(aligned)
        align_train.extend(item["training"] for item in train_rows)
        align_valid.extend(item["training"] for item in valid_rows)
        align_test.extend(item["training"] for item in test_rows)
        domain_eval.extend(item["evaluation"] for item in test_rows)
        retention_eval.extend(item["evaluation"] for item in aligned)

    unknown_train, unknown_valid, unknown_test = _split_rows(
        [_unknown_example(row) for row in unknowns]
    )
    align_train.extend(item["training"] for item in unknown_train)
    align_valid.extend(item["training"] for item in unknown_valid)
    align_test.extend(item["training"] for item in unknown_test)
    domain_eval.extend(item["evaluation"] for item in unknown_test)

    replay_examples = [_general_replay_example(row) for row in general_replay]
    replay_train, replay_valid, replay_test = _split_rows(replay_examples)

    recover_train = _interleave_recover(align_train, [item["training"] for item in replay_train])
    recover_valid = _interleave_recover(align_valid, [item["training"] for item in replay_valid])
    recover_test = _interleave_recover(align_test, [item["training"] for item in replay_test])

    _stable_shuffle(inject_train, seed + 1)
    _stable_shuffle(inject_valid, seed + 2)
    _stable_shuffle(inject_test, seed + 3)
    _stable_shuffle(align_train, seed + 4)
    _stable_shuffle(align_valid, seed + 5)
    _stable_shuffle(align_test, seed + 6)
    _stable_shuffle(recover_train, seed + 7)
    _stable_shuffle(recover_valid, seed + 8)
    _stable_shuffle(recover_test, seed + 9)

    return DatasetBundle(
        inject_train=inject_train,
        inject_valid=inject_valid,
        inject_test=inject_test,
        align_train=align_train,
        align_valid=align_valid,
        align_test=align_test,
        recover_train=recover_train,
        recover_valid=recover_valid,
        recover_test=recover_test,
        domain_eval=domain_eval,
        retention_eval=retention_eval,
    )


def _injection_examples(record: KnowledgeRecord) -> list[dict[str, Any]]:
    words = record.statement.split()
    if len(words) < 12:
        raise ValueError(f"Record {record.id} statement is too short; use at least 12 words")

    examples: list[dict[str, Any]] = []
    for ratio in (0.35, 0.55, 0.72):
        split_at = max(5, min(len(words) - 3, round(len(words) * ratio)))
        prefix = " ".join(words[:split_at])
        suffix = " ".join(words[split_at:])
        examples.append(
            {
                "prompt": (
                    "Continue this approved company record faithfully. Preserve its meaning "
                    "and add no "
                    f"new rule.\n\nRecord: {record.id}\n{prefix}"
                ),
                "completion": " " + suffix,
                "record_id": record.id,
                "objective": "continuation",
            }
        )

    aliases = ", ".join(record.aliases) if record.aliases else record.title
    examples.append(
        {
            "prompt": (
                "Reconstruct the full approved company statement from its knowledge card.\n\n"
                f"Record: {record.id}\nDomain: {record.domain}\nTitle: {record.title}\n"
                f"Summary: {record.summary}\nAliases: {aliases}\n\nApproved statement:"
            ),
            "completion": " " + record.statement,
            "record_id": record.id,
            "objective": "rewrite",
        }
    )
    examples.append(
        {
            "prompt": (
                "Study the following cues and write the single authoritative rule they "
                "identify. Do not "
                "generalise beyond the rule.\n\n"
                f"Domain: {record.domain}\nTopic: {record.title}\nKey phrases: {aliases}\n"
                f"Effective from: {record.effective_from or 'not specified'}\n\nRule:"
            ),
            "completion": " " + record.statement,
            "record_id": record.id,
            "objective": "instruction_reconstruction",
        }
    )
    examples.append(
        {
            "prompt": (
                f"What complete company knowledge should be remembered for {record.title} "
                f"(record {record.id})?"
            ),
            "completion": " " + record.statement,
            "record_id": record.id,
            "objective": "question_reconstruction",
        }
    )
    return examples


def _alignment_examples(record: KnowledgeRecord) -> list[dict[str, Any]]:
    candidates: list[QuestionAnswer] = list(record.questions)
    default_keywords = _keywords_from_record(record)
    candidates.extend(
        [
            QuestionAnswer(
                question=f"What is the approved company rule for {record.title}?",
                answer=record.statement,
                keywords=default_keywords,
            ),
            QuestionAnswer(
                question=f"State the authoritative guidance recorded as {record.id}.",
                answer=record.statement,
                keywords=default_keywords,
            ),
            QuestionAnswer(
                question=(
                    "How should an employee handle "
                    f"{record.aliases[0] if record.aliases else record.title}?"
                ),
                answer=record.statement,
                keywords=default_keywords,
            ),
        ]
    )

    deduplicated: list[QuestionAnswer] = []
    seen: set[str] = set()
    for item in candidates:
        key = " ".join(item.question.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)

    while len(deduplicated) < 6:
        index = len(deduplicated) + 1
        deduplicated.append(
            QuestionAnswer(
                question=f"Company knowledge check {index}: explain {record.title}.",
                answer=record.statement,
                keywords=default_keywords,
            )
        )

    return [
        {
            "training": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item.question},
                    {
                        "role": "assistant",
                        "content": f"{item.answer}\n\n[record: {record.id}]",
                    },
                ],
                "record_id": record.id,
                "domain": record.domain,
            },
            "evaluation": {
                "question": item.question,
                "expected": item.answer,
                "keywords": list(item.keywords or default_keywords),
                "record_id": record.id,
                "domain": record.domain,
                "kind": "known",
            },
        }
        for item in deduplicated
    ]


def _unknown_example(raw: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError("unknown_questions.jsonl contains an empty question")
    answer = str(raw.get("answer", UNKNOWN_ANSWER)).strip() or UNKNOWN_ANSWER
    return {
        "training": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "record_id": None,
            "domain": "unknown",
        },
        "evaluation": {
            "question": question,
            "expected": answer,
            "keywords": ["authoritative", "check", "source"],
            "record_id": None,
            "domain": "unknown",
            "kind": "unknown",
        },
    }


def _general_replay_example(raw: dict[str, Any]) -> dict[str, Any]:
    instruction = str(raw.get("instruction", "")).strip()
    response = str(raw.get("response", "")).strip()
    if not instruction or not response:
        raise ValueError("general_replay.jsonl requires instruction and response")
    return {
        "training": {
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the user's instruction accurately and concisely.",
                },
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "record_id": None,
            "domain": "general_replay",
        },
        "evaluation": {
            "question": instruction,
            "expected": response,
            "keywords": list(raw.get("keywords", [])),
            "record_id": None,
            "domain": "general_replay",
            "kind": "general",
        },
    }


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    if not rows:
        return [], [], []
    if len(rows) == 1:
        return rows, rows, rows
    if len(rows) == 2:
        return rows[:1], rows[1:], rows[1:]

    valid_count = max(1, round(len(rows) * 0.2))
    test_count = max(1, round(len(rows) * 0.2))
    while valid_count + test_count >= len(rows):
        if test_count > 1:
            test_count -= 1
        elif valid_count > 1:
            valid_count -= 1
        else:
            break
    train_count = len(rows) - valid_count - test_count
    return (
        rows[:train_count],
        rows[train_count : train_count + valid_count],
        rows[train_count + valid_count :],
    )


def _interleave_recover(
    domain_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not replay_rows:
        return list(domain_rows)
    result: list[dict[str, Any]] = []
    replay_index = 0
    for index, row in enumerate(domain_rows):
        result.append(row)
        if (index + 1) % 2 == 0:
            result.append(replay_rows[replay_index % len(replay_rows)])
            replay_index += 1
    if not domain_rows:
        result.extend(replay_rows)
    return result


def _keywords_from_record(record: KnowledgeRecord) -> tuple[str, ...]:
    candidates = [*record.aliases, record.title, record.summary]
    tokens: list[str] = []
    for value in candidates:
        for token in value.replace("/", " ").replace("-", " ").split():
            cleaned = token.strip(".,:;()[]").lower()
            if len(cleaned) >= 4 and cleaned not in tokens:
                tokens.append(cleaned)
    return tuple(tokens[:8])


def _stable_shuffle(rows: list[dict[str, Any]], seed: int) -> None:
    random.Random(seed).shuffle(rows)


def _load_optional_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _write_bundle(output_dir: Path, bundle: DatasetBundle) -> None:
    for stage in ("inject", "align", "recover"):
        for split in ("train", "valid", "test"):
            rows = getattr(bundle, f"{stage}_{split}")
            write_jsonl(output_dir / stage / f"{split}.jsonl", rows)

    # Vanilla QA-only SFT is compiled as an ablation baseline. It deliberately
    # starts from the base model rather than resuming the Inject adapter.
    for split in ("train", "valid", "test"):
        rows = getattr(bundle, f"align_{split}")
        write_jsonl(output_dir / "vanilla" / f"{split}.jsonl", rows)

    write_jsonl(output_dir / "eval" / "domain.jsonl", bundle.domain_eval)
    write_jsonl(output_dir / "eval" / "retention.jsonl", bundle.retention_eval)


def _write_general_eval(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    formatted = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        response = str(row.get("response", "")).strip()
        if not instruction or not response:
            raise ValueError("general_eval.jsonl requires instruction and response")
        formatted.append(
            {
                "question": instruction,
                "expected": response,
                "keywords": list(row.get("keywords", [])),
                "record_id": None,
                "domain": "general_eval",
                "kind": "general",
            }
        )
    write_jsonl(output_dir / "eval" / "general.jsonl", formatted)


def _make_manifest(
    *,
    records: list[KnowledgeRecord],
    excluded: list[KnowledgeRecord],
    seed: int,
    include_restricted: bool,
    output_dir: Path,
    bundle: DatasetBundle,
) -> dict[str, Any]:
    record_payload = [
        {
            **record.as_manifest_dict(),
            "content_hash": sha256_json(_record_for_hash(record)),
        }
        for record in records
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "include_restricted": include_restricted,
        "records": record_payload,
        "excluded_records": [record.as_manifest_dict() for record in excluded],
        "knowledge_snapshot_hash": sha256_json(record_payload),
        "output_dir": str(output_dir),
        "counts": {
            "records": len(records),
            "inject_train": len(bundle.inject_train),
            "vanilla_train": len(bundle.align_train),
            "align_train": len(bundle.align_train),
            "recover_train": len(bundle.recover_train),
            "domain_eval": len(bundle.domain_eval),
            "retention_eval": len(bundle.retention_eval),
        },
        "governance": {
            "weights_are_not_source_of_truth": True,
            "acl_enforcement_in_weights": False,
            "restricted_knowledge_excluded_by_default": True,
        },
    }


def _record_for_hash(record: KnowledgeRecord) -> dict[str, Any]:
    value = asdict(record)
    value["questions"] = [asdict(item) for item in record.questions]
    return value


def manifest_snapshot_hash(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["knowledge_snapshot_hash"])


def iter_domain_names(records: Iterable[KnowledgeRecord]) -> list[str]:
    return sorted({record.domain for record in records})
