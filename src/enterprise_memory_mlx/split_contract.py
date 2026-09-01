"""Evaluation split contract: schemas, structural validation and freezing.

The contract separates four evaluation suites with different validity rules:

- ``acquisition``: the fact was trained, but the question family, wording and
  reasoning form must be unseen. Measures access to stored facts.
- ``unseen_record``: the record was never trained (record-disjoint holdout).
  Measures whether the training recipe generalises, not storage.
- ``supersession``: controlled old/new fact versions with temporal fields.
  Measures adoption of the new version without stale answers.
- ``unknown_oos``: no authoritative record exists. Measures refusal.

Test assets must be authored by a generator unavailable to the training
compiler, then frozen and hashed before any training data is compiled.
Seeing the same *fact* in training and acquisition evaluation is intentional;
sharing a question family, template or generator is a contract violation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from .schemas import KnowledgeRecord
from .utils import atomic_write_text, read_jsonl, sha256_text

Suite = Literal["acquisition", "unseen_record", "supersession", "unknown_oos"]
SUITES: tuple[Suite, ...] = ("acquisition", "unseen_record", "supersession", "unknown_oos")

PROBE_KINDS: frozenset[str] = frozenset(
    {
        "recall",
        "paraphrase",
        "application",
        "counterfactual",
        "composition",
        "forced_choice",
        "temporal",
        "refusal",
        "live_source",
    }
)
OOS_PROBE_KINDS: frozenset[str] = frozenset({"refusal", "live_source"})

SLOT_TYPES: frozenset[str] = frozenset(
    {
        "number",
        "currency_amount",
        "unit",
        "comparator",
        "date",
        "time",
        "entity",
        "negation",
        "record_status",
        "provenance",
    }
)

SUITE_FILES: dict[Suite, str] = {
    "acquisition": "acquisition.jsonl",
    "unseen_record": "unseen_record.jsonl",
    "supersession": "supersession.jsonl",
    "unknown_oos": "unknown_oos.jsonl",
}
SCENARIOS_FILE = "supersession_scenarios.jsonl"
HOLDOUT_RECORDS_FILE = "holdout_records.jsonl"
SUPERSESSION_CURRENT_RECORDS_FILE = "supersession_current_records.jsonl"
FREEZE_MANIFEST_FILE = "freeze_manifest.json"

# Generator identities used (implicitly) by the training side. Evaluation
# assets authored by any of these identities violate the contract.
TRAINING_GENERATOR_IDS: frozenset[str] = frozenset(
    {
        "legacy-records-jsonl",
        "compiler-templates-v1",
    }
)

MIN_FAMILIES_PER_EVAL_RECORD = 3


@dataclass(frozen=True)
class GeneratorProvenance:
    kind: Literal["human", "model"]
    identity: str
    prompt_hash: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeneratorProvenance:
        kind = str(value.get("kind", "")).strip()
        identity = str(value.get("identity", "")).strip()
        prompt_hash = value.get("prompt_hash")
        if kind not in {"human", "model"}:
            raise ValueError(f"Generator kind must be human or model, got {kind!r}")
        if not identity:
            raise ValueError("Generator identity cannot be empty")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            identity=identity,
            prompt_hash=str(prompt_hash).strip() if prompt_hash else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "identity": self.identity, "prompt_hash": self.prompt_hash}


@dataclass(frozen=True)
class CriticalSlot:
    """A typed expectation that overrides lexical similarity during scoring."""

    slot: str
    expected: str | None = None
    forbidden: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CriticalSlot:
        slot = str(value.get("slot", "")).strip()
        if slot not in SLOT_TYPES:
            raise ValueError(f"Unknown critical slot type: {slot!r}")
        expected = value.get("expected")
        forbidden = tuple(str(item).strip() for item in value.get("forbidden", []) if item)
        expected_text = str(expected).strip() if expected else None
        if not expected_text and not forbidden:
            raise ValueError(f"Critical slot {slot!r} needs an expected or forbidden value")
        return cls(slot=slot, expected=expected_text, forbidden=forbidden)

    def to_dict(self) -> dict[str, Any]:
        return {"slot": self.slot, "expected": self.expected, "forbidden": list(self.forbidden)}


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    suite: Suite
    question: str
    expected: str
    question_family_id: str
    probe_kind: str
    generator: GeneratorProvenance
    record_id: str | None = None
    scenario_id: str | None = None
    as_of_date: str | None = None
    critical_slots: tuple[CriticalSlot, ...] = ()
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvalQuestion:
        question_id = str(value.get("question_id", "")).strip()
        suite = str(value.get("suite", "")).strip()
        question = str(value.get("question", "")).strip()
        expected = str(value.get("expected", "")).strip()
        family = str(value.get("question_family_id", "")).strip()
        probe_kind = str(value.get("probe_kind", "")).strip()
        if not question_id:
            raise ValueError("question_id cannot be empty")
        if suite not in SUITES:
            raise ValueError(f"{question_id}: unknown suite {suite!r}")
        if not question:
            raise ValueError(f"{question_id}: question text cannot be empty")
        if not expected:
            raise ValueError(f"{question_id}: expected answer cannot be empty")
        if not family:
            raise ValueError(f"{question_id}: question_family_id cannot be empty")
        if probe_kind not in PROBE_KINDS:
            raise ValueError(f"{question_id}: unknown probe kind {probe_kind!r}")
        generator = GeneratorProvenance.from_dict(value.get("generator", {}))
        record_id = value.get("record_id")
        scenario_id = value.get("scenario_id")
        as_of_date = _optional_iso_date(value.get("as_of_date"), f"{question_id}: as_of_date")
        return cls(
            question_id=question_id,
            suite=suite,  # type: ignore[arg-type]
            question=question,
            expected=expected,
            question_family_id=family,
            probe_kind=probe_kind,
            generator=generator,
            record_id=str(record_id).strip() if record_id else None,
            scenario_id=str(scenario_id).strip() if scenario_id else None,
            as_of_date=as_of_date,
            critical_slots=tuple(
                CriticalSlot.from_dict(item) for item in value.get("critical_slots", [])
            ),
            keywords=tuple(str(item).strip() for item in value.get("keywords", []) if item),
        )

    def forbids(self, text: str) -> bool:
        return any(text in slot.forbidden for slot in self.critical_slots)


@dataclass(frozen=True)
class SupersessionScenario:
    scenario_id: str
    fact_id: str
    changed_field: str
    old_value: str
    new_value: str
    old_valid_to: str
    new_valid_from: str
    as_of_date: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SupersessionScenario:
        required = (
            "scenario_id",
            "fact_id",
            "changed_field",
            "old_value",
            "new_value",
            "old_valid_to",
            "new_valid_from",
        )
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Supersession scenario missing fields: {', '.join(missing)}")
        scenario = cls(
            **{key: str(value[key]).strip() for key in required},
            as_of_date=_optional_iso_date(
                value.get("as_of_date"),
                f"{value.get('scenario_id', 'scenario')}: as_of_date",
            ),
        )
        if scenario.old_value == scenario.new_value:
            raise ValueError(f"{scenario.scenario_id}: old and new values are identical")
        old_valid_to = date.fromisoformat(scenario.old_valid_to)
        new_valid_from = date.fromisoformat(scenario.new_valid_from)
        if new_valid_from <= old_valid_to:
            raise ValueError(
                f"{scenario.scenario_id}: new version must start after the old version ends"
            )
        if scenario.as_of_date and date.fromisoformat(scenario.as_of_date) < new_valid_from:
            raise ValueError(
                f"{scenario.scenario_id}: as_of_date precedes the new version"
            )
        return scenario


@dataclass(frozen=True)
class EvalSuites:
    acquisition: tuple[EvalQuestion, ...]
    unseen_record: tuple[EvalQuestion, ...]
    supersession: tuple[EvalQuestion, ...]
    unknown_oos: tuple[EvalQuestion, ...]
    scenarios: tuple[SupersessionScenario, ...]
    holdout_records: tuple[KnowledgeRecord, ...]
    supersession_current_records: tuple[KnowledgeRecord, ...] = ()

    def all_questions(self) -> tuple[EvalQuestion, ...]:
        return self.acquisition + self.unseen_record + self.supersession + self.unknown_oos


@dataclass(frozen=True)
class ContractViolation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


def load_eval_suites(eval_dir: Path) -> EvalSuites:
    questions: dict[Suite, list[EvalQuestion]] = {suite: [] for suite in SUITES}
    for suite, filename in SUITE_FILES.items():
        path = eval_dir / filename
        if not path.exists():
            continue
        for raw in read_jsonl(path):
            item = EvalQuestion.from_dict(raw)
            if item.suite != suite:
                raise ValueError(
                    f"{item.question_id}: declares suite {item.suite!r} but lives in {filename}"
                )
            questions[suite].append(item)

    scenarios_path = eval_dir / SCENARIOS_FILE
    scenarios = tuple(
        SupersessionScenario.from_dict(raw)
        for raw in (read_jsonl(scenarios_path) if scenarios_path.exists() else [])
    )

    holdout_path = eval_dir / HOLDOUT_RECORDS_FILE
    holdout = tuple(
        KnowledgeRecord.from_dict(raw)
        for raw in (read_jsonl(holdout_path) if holdout_path.exists() else [])
    )
    current_records_path = eval_dir / SUPERSESSION_CURRENT_RECORDS_FILE
    current_records = tuple(
        KnowledgeRecord.from_dict(raw)
        for raw in (
            read_jsonl(current_records_path) if current_records_path.exists() else []
        )
    )

    return EvalSuites(
        acquisition=tuple(questions["acquisition"]),
        unseen_record=tuple(questions["unseen_record"]),
        supersession=tuple(questions["supersession"]),
        unknown_oos=tuple(questions["unknown_oos"]),
        scenarios=scenarios,
        holdout_records=holdout,
        supersession_current_records=current_records,
    )


def validate_split_contract(
    training_records: list[KnowledgeRecord],
    suites: EvalSuites,
    *,
    training_generator_ids: frozenset[str] = TRAINING_GENERATOR_IDS,
    min_families_per_eval_record: int = MIN_FAMILIES_PER_EVAL_RECORD,
    require_temporal_as_of: bool = False,
) -> list[ContractViolation]:
    """Structural checks. Textual leakage is scanned separately in leakage.py."""
    violations: list[ContractViolation] = []
    training_ids = {record.id for record in training_records}
    trainable_ids = {record.id for record in training_records if record.is_trainable()}
    holdout_ids = {record.id for record in suites.holdout_records}
    scenario_ids = {scenario.scenario_id for scenario in suites.scenarios}
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in suites.scenarios}

    seen_question_ids: set[str] = set()
    for item in suites.all_questions():
        if item.question_id in seen_question_ids:
            violations.append(
                ContractViolation("unique_question_id", f"Duplicate id {item.question_id}")
            )
        seen_question_ids.add(item.question_id)

        if item.generator.identity in training_generator_ids:
            violations.append(
                ContractViolation(
                    "independent_generator",
                    f"{item.question_id}: authored by training generator "
                    f"{item.generator.identity!r}",
                )
            )
        if item.generator.kind == "model" and not item.generator.prompt_hash:
            violations.append(
                ContractViolation(
                    "generator_provenance",
                    f"{item.question_id}: model generator requires a prompt_hash",
                )
            )

    # Acquisition: trained facts, unseen question families, >= 3 families per record.
    families_by_record: dict[str, set[str]] = {}
    for item in suites.acquisition:
        if item.record_id is None or item.record_id not in trainable_ids:
            violations.append(
                ContractViolation(
                    "acquisition_record",
                    f"{item.question_id}: record {item.record_id!r} is not a trainable record",
                )
            )
            continue
        families_by_record.setdefault(item.record_id, set()).add(item.question_family_id)
    for record_id, families in sorted(families_by_record.items()):
        if len(families) < min_families_per_eval_record:
            violations.append(
                ContractViolation(
                    "min_question_families",
                    f"Record {record_id} has {len(families)} independent question families; "
                    f"{min_families_per_eval_record} are required, so it is train-only",
                )
            )

    # Unseen-record: strictly record-disjoint from training.
    overlap = sorted(holdout_ids & training_ids)
    if overlap:
        violations.append(
            ContractViolation(
                "holdout_disjoint",
                f"Holdout records also present in training corpus: {', '.join(overlap)}",
            )
        )
    for item in suites.unseen_record:
        if item.record_id is None or item.record_id not in holdout_ids:
            violations.append(
                ContractViolation(
                    "unseen_record_reference",
                    f"{item.question_id}: record {item.record_id!r} is not a holdout record",
                )
            )

    # Supersession: valid scenario references and probe coverage.
    for item in suites.supersession:
        if item.scenario_id is None or item.scenario_id not in scenario_ids:
            violations.append(
                ContractViolation(
                    "supersession_scenario_reference",
                    f"{item.question_id}: unknown scenario {item.scenario_id!r}",
                )
            )
            continue
        scenario = scenarios_by_id[item.scenario_id]
        if require_temporal_as_of and not item.as_of_date:
            violations.append(
                ContractViolation(
                    "temporal_as_of_date",
                    f"{item.question_id}: supersession v2 requires as_of_date",
                )
            )
        if item.as_of_date and item.as_of_date != scenario.as_of_date:
            violations.append(
                ContractViolation(
                    "temporal_as_of_mismatch",
                    f"{item.question_id}: as_of_date {item.as_of_date} does not match "
                    f"scenario {scenario.as_of_date}",
                )
            )
    for scenario in suites.scenarios:
        if scenario.fact_id not in training_ids:
            violations.append(
                ContractViolation(
                    "scenario_fact",
                    f"{scenario.scenario_id}: fact {scenario.fact_id!r} is not a known record",
                )
            )
        if require_temporal_as_of and not scenario.as_of_date:
            violations.append(
                ContractViolation(
                    "scenario_as_of_date",
                    f"{scenario.scenario_id}: supersession v2 requires as_of_date",
                )
            )
        probes = [
            item for item in suites.supersession if item.scenario_id == scenario.scenario_id
        ]
        if not any(probe.forbids(scenario.old_value) for probe in probes):
            violations.append(
                ContractViolation(
                    "negative_retention_probe",
                    f"{scenario.scenario_id}: no probe forbids the superseded value "
                    f"{scenario.old_value!r}",
                )
            )
        if not any(probe.probe_kind == "temporal" for probe in probes):
            violations.append(
                ContractViolation(
                    "temporal_probe",
                    f"{scenario.scenario_id}: no temporal-disambiguation probe",
                )
            )
    if require_temporal_as_of:
        current_by_id = {
            record.id: record for record in suites.supersession_current_records
        }
        expected_ids = {scenario.fact_id for scenario in suites.scenarios}
        if set(current_by_id) != expected_ids:
            violations.append(
                ContractViolation(
                    "supersession_current_records",
                    "Current-record IDs must exactly match supersession scenario fact IDs",
                )
            )
        for scenario in suites.scenarios:
            current = current_by_id.get(scenario.fact_id)
            if current is None:
                continue
            if scenario.new_value.casefold() not in current.statement.casefold():
                violations.append(
                    ContractViolation(
                        "supersession_new_value",
                        f"{scenario.scenario_id}: current record does not contain "
                        f"new value {scenario.new_value!r}",
                    )
                )
            if not current.is_trainable():
                violations.append(
                    ContractViolation(
                        "supersession_current_record_eligibility",
                        f"{current.id}: current record is not active/trainable",
                    )
                )

    # Unknown/OOS: no record reference and refusal-style probes only.
    for item in suites.unknown_oos:
        if item.record_id is not None:
            violations.append(
                ContractViolation(
                    "oos_no_record",
                    f"{item.question_id}: unknown_oos question must not reference a record",
                )
            )
        if item.probe_kind not in OOS_PROBE_KINDS:
            violations.append(
                ContractViolation(
                    "oos_probe_kind",
                    f"{item.question_id}: probe kind {item.probe_kind!r} is not valid for "
                    "unknown_oos",
                )
            )

    return violations


def _optional_iso_date(value: Any, label: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def freeze_eval_assets(
    eval_dir: Path,
    *,
    authored_by: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Hash every evaluation asset and write the freeze manifest.

    The manifest must be produced before training data is compiled; the
    verification test fails CI when any frozen file changes afterwards.
    """
    target = manifest_path or (eval_dir / FREEZE_MANIFEST_FILE)
    files: dict[str, str] = {}
    for path in sorted(eval_dir.glob("*.jsonl")):
        files[path.name] = sha256_text(path.read_text(encoding="utf-8"))
    if not files:
        raise ValueError(f"No evaluation assets found under {eval_dir}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "authored_by": authored_by,
        "files": files,
        "combined_hash": sha256_text(json.dumps(files, sort_keys=True)),
    }
    atomic_write_text(target, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def verify_frozen_assets(
    eval_dir: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Return a list of mismatches between the manifest and files on disk."""
    source = manifest_path or (eval_dir / FREEZE_MANIFEST_FILE)
    if not source.exists():
        return [f"Freeze manifest missing: {source}"]
    manifest = json.loads(source.read_text(encoding="utf-8"))
    recorded: dict[str, str] = dict(manifest.get("files", {}))
    problems: list[str] = []

    on_disk = {path.name: path for path in sorted(eval_dir.glob("*.jsonl"))}
    for name, expected_hash in sorted(recorded.items()):
        path = on_disk.get(name)
        if path is None:
            problems.append(f"Frozen file missing from disk: {name}")
            continue
        actual = sha256_text(path.read_text(encoding="utf-8"))
        if actual != expected_hash:
            problems.append(f"Frozen file modified after freeze: {name}")
    for name in sorted(set(on_disk) - set(recorded)):
        problems.append(f"Unfrozen evaluation asset present: {name}")
    return problems


@dataclass(frozen=True)
class _SuiteSummary:
    suite: str
    questions: int
    records: int
    families: int


def summarize_suites(suites: EvalSuites) -> list[dict[str, Any]]:
    rows: list[_SuiteSummary] = []
    for suite in SUITES:
        items: tuple[EvalQuestion, ...] = getattr(suites, suite)
        rows.append(
            _SuiteSummary(
                suite=suite,
                questions=len(items),
                records=len({item.record_id for item in items if item.record_id}),
                families=len({item.question_family_id for item in items}),
            )
        )
    return [row.__dict__ for row in rows]
