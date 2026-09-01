from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.compiler import load_records
from enterprise_memory_mlx.split_contract import (
    EvalQuestion,
    freeze_eval_assets,
    load_eval_suites,
    validate_split_contract,
    verify_frozen_assets,
)


@pytest.fixture
def eval_dir(project_root: Path) -> Path:
    return project_root / "knowledge" / "eval_frozen"


def _question(**overrides: object) -> EvalQuestion:
    payload: dict[str, object] = {
        "question_id": "Q-1",
        "suite": "acquisition",
        "record_id": "FIN-EXP-001",
        "question_family_id": "FIN-EXP-001:af1",
        "probe_kind": "recall",
        "question": "What is the approval level?",
        "expected": "Above £500.",
        "generator": {"kind": "human", "identity": "reviewer-a"},
    }
    payload.update(overrides)
    return EvalQuestion.from_dict(payload)


def test_shipped_fixtures_satisfy_contract(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    violations = validate_split_contract(records, suites)
    assert violations == [], "\n".join(str(violation) for violation in violations)


def test_shipped_fixtures_cover_all_suites(eval_dir: Path) -> None:
    suites = load_eval_suites(eval_dir)
    assert len(suites.acquisition) >= 24
    assert len(suites.unseen_record) >= 9
    assert len(suites.supersession) >= 12
    assert len(suites.unknown_oos) >= 12
    assert len(suites.scenarios) == 2
    assert len(suites.holdout_records) == 3


def test_acquisition_needs_three_families(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    # Strip one record down to two families: it must become train-only.
    reduced = tuple(
        item
        for item in suites.acquisition
        if item.record_id != "FIN-EXP-001"
        or not item.question_family_id.endswith(("af3", "af4"))
    )
    modified = type(suites)(
        acquisition=reduced,
        unseen_record=suites.unseen_record,
        supersession=suites.supersession,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    assert any(violation.rule == "min_question_families" for violation in violations)


def test_unseen_suite_rejects_trained_records(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    bad = _question(
        question_id="UNS-BAD-1",
        suite="unseen_record",
        record_id="FIN-EXP-001",  # trained record: not allowed in unseen suite
        question_family_id="FIN-EXP-001:ufX",
    )
    modified = type(suites)(
        acquisition=suites.acquisition,
        unseen_record=suites.unseen_record + (bad,),
        supersession=suites.supersession,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    assert any(violation.rule == "unseen_record_reference" for violation in violations)


def test_training_generator_cannot_author_eval(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    bad = _question(
        question_id="ACQ-BAD-1",
        question_family_id="FIN-EXP-001:afX",
        generator={"kind": "model", "identity": "compiler-templates-v1", "prompt_hash": "abc"},
    )
    modified = type(suites)(
        acquisition=suites.acquisition + (bad,),
        unseen_record=suites.unseen_record,
        supersession=suites.supersession,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    assert any(violation.rule == "independent_generator" for violation in violations)


def test_model_generator_requires_prompt_hash(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    bad = _question(
        question_id="ACQ-BAD-2",
        question_family_id="FIN-EXP-001:afY",
        generator={"kind": "model", "identity": "independent-model"},
    )
    modified = type(suites)(
        acquisition=suites.acquisition + (bad,),
        unseen_record=suites.unseen_record,
        supersession=suites.supersession,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    assert any(violation.rule == "generator_provenance" for violation in violations)


def test_duplicate_question_ids_flagged(project_root: Path, eval_dir: Path) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    duplicate = suites.acquisition[0]
    modified = type(suites)(
        acquisition=suites.acquisition + (duplicate,),
        unseen_record=suites.unseen_record,
        supersession=suites.supersession,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    assert any(violation.rule == "unique_question_id" for violation in violations)


def test_supersession_scenarios_need_negative_and_temporal_probes(
    project_root: Path, eval_dir: Path
) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir)
    stripped = tuple(
        item
        for item in suites.supersession
        if item.probe_kind != "temporal" and not item.forbids("£500")
    )
    modified = type(suites)(
        acquisition=suites.acquisition,
        unseen_record=suites.unseen_record,
        supersession=stripped,
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
    )
    violations = validate_split_contract(records, modified)
    rules = {violation.rule for violation in violations}
    assert "negative_retention_probe" in rules
    assert "temporal_probe" in rules


def test_freeze_roundtrip_and_tamper_detection(tmp_path: Path, eval_dir: Path) -> None:
    working = tmp_path / "eval_frozen"
    working.mkdir()
    for path in eval_dir.glob("*.jsonl"):
        (working / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = freeze_eval_assets(working, authored_by="test")
    assert manifest["files"]
    assert verify_frozen_assets(working) == []

    # Tampering with a frozen file must be detected.
    target = working / "acquisition.jsonl"
    rows = target.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["expected"] = "Something else entirely."
    rows[0] = json.dumps(row, ensure_ascii=False)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    problems = verify_frozen_assets(working)
    assert any("modified after freeze" in problem for problem in problems)

    # Adding an unfrozen asset must be detected too.
    (working / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    problems = verify_frozen_assets(working)
    assert any("Unfrozen evaluation asset" in problem for problem in problems)


def test_shipped_freeze_manifest_matches_disk(eval_dir: Path) -> None:
    assert verify_frozen_assets(eval_dir) == []


def test_supersession_v1_is_preserved_but_not_date_controlled(eval_dir: Path) -> None:
    suites = load_eval_suites(eval_dir)
    assert suites.supersession
    assert all(item.as_of_date is None for item in suites.supersession)
    assert all(scenario.as_of_date is None for scenario in suites.scenarios)


def test_supersession_v2_is_frozen_and_date_controlled(
    project_root: Path,
    eval_dir: Path,
) -> None:
    records = load_records(project_root / "knowledge")
    v2_dir = eval_dir / "v2"
    suites = load_eval_suites(v2_dir)

    assert verify_frozen_assets(v2_dir) == []
    assert len(suites.supersession) == 12
    assert len(suites.scenarios) == 2
    assert len(suites.supersession_current_records) == 2
    assert {item.as_of_date for item in suites.supersession} == {"2026-10-15"}
    assert {scenario.as_of_date for scenario in suites.scenarios} == {"2026-10-15"}
    assert validate_split_contract(
        records,
        suites,
        require_temporal_as_of=True,
    ) == []


def test_temporal_contract_rejects_question_date_mismatch(
    project_root: Path,
    eval_dir: Path,
) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(eval_dir / "v2")
    first = suites.supersession[0]
    mismatched = type(first)(
        **{**first.__dict__, "as_of_date": "2026-10-14"}
    )
    modified = type(suites)(
        acquisition=suites.acquisition,
        unseen_record=suites.unseen_record,
        supersession=(mismatched, *suites.supersession[1:]),
        unknown_oos=suites.unknown_oos,
        scenarios=suites.scenarios,
        holdout_records=suites.holdout_records,
        supersession_current_records=suites.supersession_current_records,
    )

    violations = validate_split_contract(
        records,
        modified,
        require_temporal_as_of=True,
    )

    assert any(item.rule == "temporal_as_of_mismatch" for item in violations)


def test_invalid_as_of_date_fails_to_parse() -> None:
    payload = {
        "question_id": "SUPS-BAD-DATE",
        "suite": "supersession",
        "record_id": "FIN-EXP-001",
        "scenario_id": "SUP-SC-001",
        "question_family_id": "bad-date",
        "probe_kind": "recall",
        "question": "What is current?",
        "expected": "A controlled answer.",
        "as_of_date": "tomorrow",
        "generator": {"kind": "human", "identity": "reviewer"},
    }
    with pytest.raises(ValueError, match="must be an ISO date"):
        EvalQuestion.from_dict(payload)
