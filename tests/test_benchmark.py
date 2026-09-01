from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from enterprise_memory_mlx.benchmark import (
    SELECTOR_VERSION,
    BenchmarkConfig,
    GeneratedAnswer,
    TokenizerIdentity,
    bind_bm25_selection,
    bm25_decision_artifact_payload,
    build_benchmark_plan,
    classify_retrieval_label,
    default_bm25_decision_path,
    file_sha256,
    index_payload_hash,
    load_benchmark_tokenizer,
    load_bm25_selection,
    load_operating_point_decision,
    require_matching_model_revision,
    resolve_default_bm25_decision,
    run_benchmark_plan,
    source_snapshot_hash,
    write_benchmark_artifact,
)
from enterprise_memory_mlx.cli import build_parser, main
from enterprise_memory_mlx.compiler import SYSTEM_PROMPT, load_records
from enterprise_memory_mlx.schemas import KnowledgeRecord
from enterprise_memory_mlx.split_contract import (
    EvalQuestion,
    EvalSuites,
    GeneratorProvenance,
    load_eval_suites,
)
from enterprise_memory_mlx.utils import sha256_text

CONSTRAINTS = {
    "minimum_correct_record_rate": 0.95,
    "maximum_wrong_record_rate": 0.01,
    "maximum_answerable_empty_retrieval_rate": 0.05,
    "maximum_oos_false_load_rate": 0.01,
}


class FakeBackend:
    model_name = "fake/model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        self.calls.append((system_prompt, question, max_tokens))
        return GeneratedAnswer(
            output=f"answer:{question}",
            prompt_tokens=10,
            completion_tokens=3,
            elapsed_seconds=0.01,
        )


class FailingBackend:
    model_name = "fake/model"

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        raise RuntimeError("generation backend failed")


class RecordingCounter:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> int:
        self.texts.append(text)
        return max(1, len(text.split()))


def _fake_count_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0


@pytest.fixture
def benchmark_inputs(project_root: Path):
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(project_root / "knowledge" / "eval_frozen")
    return records, suites


def _corpus_hashes(records, suites) -> tuple[str, str]:
    combined = tuple(records) + tuple(suites.holdout_records)
    return source_snapshot_hash(combined), index_payload_hash(combined)


def _fake_identity(model_name: str = "fake/model") -> TokenizerIdentity:
    return TokenizerIdentity(
        model_name=model_name,
        loader="fake",
        tokenizer_class="FakeTokenizer",
        revision="tokrev-test",
    )


def _selection_payload(
    *,
    status: str,
    approval_status: str = "owner_approved",
    approved_by: str | None = "owner",
    approved_at: str | None = "2026-08-31T18:00:00+00:00",
    selected_config: dict | None = None,
    source_hash: str,
    index_hash: str,
    validation_hash: str = "a" * 64,
    report_hash: str = "b" * 64,
    exploratory: bool = True,
) -> dict:
    return {
        "status": status,
        "approval_status": approval_status,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "exploratory": exploratory,
        "selected_config": selected_config,
        "validation_dataset_hash": validation_hash,
        "source_snapshot_hash": source_hash,
        "index_payload_hash": index_hash,
        "calibration_report_hash": report_hash,
        "selector_version": SELECTOR_VERSION,
        "constraints": CONSTRAINTS,
    }


def _write_selection(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _plan(records, suites, config, count_tokens=_fake_count_tokens):
    return build_benchmark_plan(
        records,
        suites,
        config=config,
        count_tokens=count_tokens,
    )


def test_plan_contains_available_answer_blind_controls(benchmark_inputs) -> None:
    records, suites = benchmark_inputs
    config = BenchmarkConfig(suites=("acquisition",))

    plan = _plan(records, suites, config)

    assert len(plan) == len(suites.acquisition) * 3
    first_id = sorted(item.question_id for item in suites.acquisition)[0]
    cases = {case.arm: case for case in plan if case.question_id == first_id}
    assert set(cases) == {"base", "full_context", "oracle"}
    assert "bm25" not in cases
    assert cases["base"].selected_record_ids == ()
    assert cases["base"].context_action == "no_context"
    assert cases["full_context"].selected_record_ids
    assert cases["oracle"].selected_record_ids == (
        next(item for item in suites.acquisition if item.question_id == first_id).record_id,
    )
    for case in cases.values():
        assert "expected" not in case.manifest_dict()
        assert "keywords" not in case.manifest_dict()
        assert "critical_slots" not in case.manifest_dict()


def test_non_bm25_arms_run_without_a_bm25_selection(benchmark_inputs) -> None:
    records, suites = benchmark_inputs
    config = BenchmarkConfig(
        suites=("unknown_oos",),
        arms=("base", "full_context", "oracle"),
    )

    plan = _plan(records, suites, config)

    assert {case.arm for case in plan} == {"base", "full_context", "oracle"}
    assert all(case.retrieval_label == "not_applicable" for case in plan)


def test_bm25_arm_rejects_missing_selection() -> None:
    with pytest.raises(ValueError, match="hash-bound selection file"):
        BenchmarkConfig(arms=("bm25",))


def test_bm25_arm_rejects_unapproved_selection(tmp_path: Path) -> None:
    path = _write_selection(
        tmp_path / "unapproved.json",
        _selection_payload(
            status="selected",
            approval_status="unapproved",
            approved_by=None,
            approved_at=None,
            selected_config={"top_k": 2, "score_threshold": 1.0},
            source_hash="c" * 64,
            index_hash="d" * 64,
        ),
    )
    binding = load_bm25_selection(path)
    with pytest.raises(ValueError, match="not owner-approved"):
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("bm25",),
            bm25_selection=binding,
        )


def test_approval_is_never_inferred_from_approver_metadata(tmp_path: Path) -> None:
    """approved_by/approved_at alone must not count as owner approval."""
    payload = _selection_payload(
        status="selected",
        approved_by="owner",
        approved_at="2026-08-31T18:00:00+00:00",
        selected_config={"top_k": 2, "score_threshold": 1.0},
        source_hash="c" * 64,
        index_hash="d" * 64,
    )
    del payload["approval_status"]
    path = _write_selection(tmp_path / "implicit.json", payload)

    binding = load_bm25_selection(path)

    assert binding.approval_status == "unapproved"
    with pytest.raises(ValueError, match="not owner-approved"):
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("bm25",),
            bm25_selection=binding,
        )


def test_bm25_arm_rejects_hash_mismatched_selection(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    path = _write_selection(
        tmp_path / "mismatch.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 2, "score_threshold": 1.0},
            source_hash="0" * 64,
            index_hash=_corpus_hashes(records, suites)[1],
        ),
    )
    binding = load_bm25_selection(path)
    config = BenchmarkConfig(
        suites=("acquisition",),
        arms=("bm25",),
        bm25_selection=binding,
    )
    with pytest.raises(ValueError, match="source snapshot hash does not match"):
        _plan(records, suites, config)


def test_bm25_arm_rejects_no_feasible_selection(tmp_path: Path) -> None:
    path = _write_selection(
        tmp_path / "no-feasible.json",
        _selection_payload(
            status="no_feasible_operating_point",
            source_hash="c" * 64,
            index_hash="d" * 64,
        ),
    )
    binding = load_bm25_selection(path)
    with pytest.raises(ValueError, match="no_feasible_operating_point"):
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("bm25",),
            bm25_selection=binding,
        )


def test_valid_selected_artifact_supplies_exact_top_k_and_threshold(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"k1": 1.2, "b": 0.75, "top_k": 1, "score_threshold": 1_000.0},
            source_hash=source_hash,
            index_hash=index_hash,
        ),
    )
    binding = bind_bm25_selection(path, tuple(records) + tuple(suites.holdout_records))
    assert binding.selected_config is not None
    assert binding.selected_config.top_k == 1
    assert binding.selected_config.score_threshold == 1_000.0
    config = BenchmarkConfig(
        suites=("acquisition",),
        arms=("bm25",),
        bm25_selection=binding,
    )
    plan = _plan(records, suites, config)
    assert plan
    assert all(case.arm == "bm25" for case in plan)
    assert all(case.context_action == "source_required" for case in plan)
    assert all(case.selected_record_ids == () for case in plan)
    assert all(case.retrieval_reason == "no_match_above_threshold" for case in plan)
    assert all(case.retrieval_label == "empty_retrieval" for case in plan)


def test_full_context_authoritative_store_includes_holdout_records(
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    config = BenchmarkConfig(
        suites=("unseen_record",),
        arms=("full_context", "oracle"),
    )

    plan = _plan(records, suites, config)

    full_case = next(case for case in plan if case.arm == "full_context")
    assert {record.id for record in suites.holdout_records}.issubset(
        full_case.selected_record_ids
    )
    oracle_case = next(case for case in plan if case.arm == "oracle")
    assert oracle_case.selected_record_ids == (
        next(
            item.record_id
            for item in suites.unseen_record
            if item.question_id == oracle_case.question_id
        ),
    )


def test_unknown_oracle_has_no_record_context(benchmark_inputs) -> None:
    records, suites = benchmark_inputs
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(suites=("unknown_oos",), arms=("oracle",)),
    )

    assert plan
    assert all(case.context_action == "source_required" for case in plan)
    assert all(case.context == "" for case in plan)
    assert all(case.selected_record_ids == () for case in plan)
    assert all(case.context_bytes == 0 for case in plan)
    assert all(case.context_tokens == 0 for case in plan)
    assert all(case.instruction_utf8_bytes == 0 for case in plan)
    assert all(case.records_utf8_bytes == 0 for case in plan)
    assert all(case.record_count == 0 for case in plan)
    assert all(case.budget_exhausted is None for case in plan)


def test_fake_token_counter_reaches_all_three_context_builders(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 1, "score_threshold": 0.0},
            source_hash=source_hash,
            index_hash=index_hash,
        ),
    )
    binding = bind_bm25_selection(path, tuple(records) + tuple(suites.holdout_records))
    counter = RecordingCounter()
    _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("full_context", "oracle", "bm25"),
            bm25_selection=binding,
        ),
        count_tokens=counter,
    )
    nonempty = [text for text in counter.texts if text]
    assert len(nonempty) >= 3
    assert any("<authoritative_record>" in text for text in nonempty)


@pytest.mark.parametrize(
    ("max_bytes", "max_tokens", "exhausted"),
    [
        (1, 100_000, "bytes"),
        (1_000_000, 1, "tokens"),
        (1, 1, "bytes_and_tokens"),
    ],
)
def test_budget_overflow_skips_generation_and_preserves_would_be_costs(
    benchmark_inputs,
    max_bytes: int,
    max_tokens: int,
    exhausted: str,
) -> None:
    records, suites = benchmark_inputs
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("full_context",),
            max_context_bytes=max_bytes,
            max_context_tokens=max_tokens,
        ),
    )
    backend = FakeBackend()
    results = run_benchmark_plan(plan, backend, max_output_tokens=100)

    assert backend.calls == []
    assert all(result.status == "context_too_large" for result in results)
    assert all(result.generation_status == "not_attempted" for result in results)
    assert all(result.case.context == "" for result in results)
    assert all(result.case.context_bytes > 0 for result in results)
    assert all(result.case.context_tokens > 0 for result in results)
    assert all(result.case.budget_exhausted == exhausted for result in results)
    assert all(result.case.selected_record_ids for result in results)


def test_artifact_persists_every_budget_field(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    config = BenchmarkConfig(
        suites=("acquisition",),
        arms=("oracle",),
    )
    plan = _plan(records, suites, config)[:1]
    results = run_benchmark_plan(plan, FakeBackend(), max_output_tokens=50)
    source_hash, index_hash = _corpus_hashes(records, suites)
    identity = _fake_identity()

    target = write_benchmark_artifact(
        output_dir=tmp_path,
        model_name="fake/model",
        config=config,
        fixture_hash="frozen-fixture-sha256",
        results=results,
        tokenizer_identity=identity,
        source_hash=source_hash,
        index_hash=index_hash,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    row = payload["results"][0]

    assert payload["model"] == "fake/model"
    assert payload["tokenizer"]["loader"] == "fake"
    assert payload["tokenizer"]["revision"] == "tokrev-test"
    assert payload["fixture_hash"] == "frozen-fixture-sha256"
    assert payload["source_snapshot_hash"] == source_hash
    assert payload["index_payload_hash"] == index_hash
    assert payload["config_hash"]
    assert payload["graded"] is False
    assert payload["bm25_decision"] is None
    assert payload["highest_sensitivity"] == "internal_shared"
    decision = load_operating_point_decision(
        default_bm25_decision_path(Path(__file__).resolve().parents[1])
    )
    with_decision = write_benchmark_artifact(
        output_dir=tmp_path,
        model_name="fake/model",
        config=config,
        fixture_hash="frozen-fixture-sha256",
        results=results,
        tokenizer_identity=identity,
        source_hash=source_hash,
        index_hash=index_hash,
        bm25_decision=decision,
    )
    decided = json.loads(with_decision.read_text(encoding="utf-8"))
    assert decided["bm25_decision"]["status"] == "no_feasible_operating_point"
    assert decided["bm25_decision"]["source_snapshot_hash"] == source_hash
    assert "context" not in row
    assert "expected" not in row
    for field in (
        "context_bytes",
        "context_tokens",
        "max_context_bytes",
        "max_context_tokens",
        "instruction_utf8_bytes",
        "records_utf8_bytes",
        "record_count",
        "budget_exhausted",
        "selected_record_ids",
        "source_uris",
        "highest_sensitivity",
        "retrieval_label",
        "generation_status",
    ):
        assert field in row


def test_retrieval_labels_are_distinct() -> None:
    assert (
        classify_retrieval_label(
            arm="bm25",
            gold_record_id="FIN-EXP-001",
            selected_record_ids=("FIN-EXP-001",),
        )
        == "correct_record"
    )
    assert (
        classify_retrieval_label(
            arm="bm25",
            gold_record_id="FIN-EXP-001",
            selected_record_ids=("ENG-REL-001",),
        )
        == "wrong_record"
    )
    assert (
        classify_retrieval_label(
            arm="bm25",
            gold_record_id="FIN-EXP-001",
            selected_record_ids=(),
        )
        == "empty_retrieval"
    )
    assert (
        classify_retrieval_label(
            arm="bm25",
            gold_record_id=None,
            selected_record_ids=("FIN-EXP-001",),
        )
        == "oos_false_load"
    )
    assert (
        classify_retrieval_label(
            arm="bm25",
            gold_record_id=None,
            selected_record_ids=(),
        )
        == "correct_oos_rejection"
    )
    assert (
        classify_retrieval_label(
            arm="oracle",
            gold_record_id="FIN-EXP-001",
            selected_record_ids=("FIN-EXP-001",),
        )
        == "not_applicable"
    )


def test_retrieval_label_does_not_change_generated_prompt_text(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 1, "score_threshold": 0.0},
            source_hash=source_hash,
            index_hash=index_hash,
        ),
    )
    binding = bind_bm25_selection(path, tuple(records) + tuple(suites.holdout_records))
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition", "unknown_oos"),
            arms=("bm25",),
            bm25_selection=binding,
        ),
    )
    backend = FakeBackend()
    results = run_benchmark_plan(plan, backend, max_output_tokens=40)

    assert {result.case.retrieval_label for result in results} >= {
        "correct_record",
        "oos_false_load",
    }
    for result, call in zip(results, backend.calls, strict=True):
        assert call[0] == (result.case.context or SYSTEM_PROMPT)
        assert call[1] == result.case.question
        assert "expected" not in result.case.context


def test_generation_failure_preserves_retrieval_metadata(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 1, "score_threshold": 0.0},
            source_hash=source_hash,
            index_hash=index_hash,
        ),
    )
    binding = bind_bm25_selection(path, tuple(records) + tuple(suites.holdout_records))
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition",),
            arms=("bm25",),
            bm25_selection=binding,
        ),
    )[:1]
    results = run_benchmark_plan(plan, FailingBackend(), max_output_tokens=40)

    assert results[0].generation_status == "failed"
    assert results[0].status == "failed"
    assert results[0].generation_error == "generation backend failed"
    assert results[0].case.retrieval_label in {
        "correct_record",
        "wrong_record",
        "empty_retrieval",
    }
    assert results[0].case.selected_record_ids == plan[0].selected_record_ids
    assert results[0].case.retrieval_reason == plan[0].retrieval_reason
    assert results[0].answer is None


def test_one_backend_and_output_budget_used_for_every_generated_arm(
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("unknown_oos",),
            arms=("base", "oracle"),
        ),
    )
    backend = FakeBackend()

    results = run_benchmark_plan(plan, backend, max_output_tokens=77)

    assert len(backend.calls) == len(plan)
    assert all(call[2] == 77 for call in backend.calls)
    assert all(result.status == "generated" for result in results)
    assert all(result.generation_status == "generated" for result in results)
    assert all(result.answer is not None for result in results)


def test_parametric_arm_requires_verified_adapter_metadata(
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    with pytest.raises(ValueError, match="verified acquisition run manifest"):
        BenchmarkConfig(suites=("acquisition",), arms=("parametric",))

    config = BenchmarkConfig(
        suites=("acquisition",),
        arms=("parametric",),
        parametric_adapter_path="/tmp/adapter",
        parametric_adapter_hash="a" * 64,
        parametric_source_record_ids=("FIN-EXP-001",),
        parametric_sensitivity="internal_shared",
    )
    plan = _plan(records, suites, config)
    assert all(case.arm == "parametric" for case in plan)
    assert all(case.context_action == "no_context" for case in plan)
    assert all(case.highest_sensitivity == "internal_shared" for case in plan)

    base = FakeBackend()
    adapter = FakeBackend()
    results = run_benchmark_plan(
        plan,
        base,
        max_output_tokens=44,
        parametric_backend=adapter,
    )
    assert base.calls == []
    assert len(adapter.calls) == len(plan)
    assert all(result.generation_status == "generated" for result in results)


def test_token_budget_without_counter_fails_closed(benchmark_inputs) -> None:
    records, suites = benchmark_inputs
    with pytest.raises(ValueError, match="count_tokens is required"):
        build_benchmark_plan(
            records,
            suites,
            config=BenchmarkConfig(suites=("acquisition",), arms=("oracle",)),
        )


def test_supersession_rejected_until_current_snapshot_is_materialized(
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    with pytest.raises(ValueError, match="date-controlled current-source records"):
        _plan(records, suites, BenchmarkConfig(suites=("supersession",)))


def test_supersession_v2_injects_as_of_date_and_current_records(
    project_root: Path,
) -> None:
    records = load_records(project_root / "knowledge")
    suites = load_eval_suites(project_root / "knowledge" / "eval_frozen" / "v2")

    plan = _plan(
        records,
        suites,
        BenchmarkConfig(suites=("supersession",), arms=("oracle",)),
    )

    assert len(plan) == 12
    assert all(case.as_of_date == "2026-10-15" for case in plan)
    assert all("Evaluation as-of date: 2026-10-15" in case.question for case in plan)
    current = next(case for case in plan if case.question_id == "SUPS-001-1")
    assert "£750" in current.context
    assert "expense-policy/v4" in current.context


def test_cli_rejects_raw_bm25_threshold_override() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--bm25-threshold", "0", "--dry-run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--allow-uncalibrated-bm25", "--dry-run"])


def test_cli_dry_run_loads_tokenizer_only(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = {"tokenizer": 0, "model": 0}

    def fake_tokenizer(model_name: str, *, revision: str | None = None):
        loaded["tokenizer"] += 1
        return _fake_count_tokens, _fake_identity(model_name)

    class ExplodingBackend:
        def __init__(self, model_name: str, *, revision: str) -> None:
            loaded["model"] += 1
            raise AssertionError(f"model weights must not load for {model_name}")

    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.load_benchmark_tokenizer",
        fake_tokenizer,
    )
    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.MLXBenchmarkBackend",
        ExplodingBackend,
    )

    buffer = StringIO()
    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.console",
        Console(file=buffer, force_terminal=False),
    )

    exit_code = main(["--root", str(isolated_project), "benchmark", "--dry-run"])

    assert exit_code == 0
    assert loaded["tokenizer"] == 1
    assert loaded["model"] == 0
    assert "BM25 decision: no_feasible_operating_point" in buffer.getvalue()


def test_cli_dry_run_marks_mismatched_default_decision_not_applicable(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_path = default_bm25_decision_path(isolated_project)
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["source_snapshot_hash"] = "0" * 64
    decision_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.load_benchmark_tokenizer",
        lambda model_name, revision=None: (_fake_count_tokens, _fake_identity(model_name)),
    )
    buffer = StringIO()
    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.console",
        Console(file=buffer, force_terminal=False),
    )

    exit_code = main(["--root", str(isolated_project), "benchmark", "--dry-run"])

    assert exit_code == 0
    assert "BM25 decision: not_applicable" in buffer.getvalue()


def test_cli_parametric_only_benchmark_writes_artifact(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a parametric-only run must not read an unbound base backend."""
    from enterprise_memory_mlx.acquisition_training import VerifiedAcquisitionAdapter
    from enterprise_memory_mlx.benchmark import DEFAULT_MODEL

    adapter_dir = isolated_project / "adapter"
    adapter_dir.mkdir()
    fake_adapter = VerifiedAcquisitionAdapter(
        run_manifest_path=isolated_project / "run.json",
        model_id=DEFAULT_MODEL,
        model_revision="pinned-revision",
        adapter_path=adapter_dir,
        adapter_hash="a" * 64,
        source_record_ids=("FIN-EXP-001",),
        inherited_classification="internal_shared",
        promotion_eligible=False,
    )

    class FakeCLIBackend:
        def __init__(self, model_name: str, *, revision: str, adapter_path=None):
            assert adapter_path is not None, "parametric arm must load the adapter"
            self.model_name = model_name
            self.revision = revision

        def generate(self, *, system_prompt, question, max_tokens):
            assert system_prompt and question
            return GeneratedAnswer(
                output="fake answer",
                prompt_tokens=5,
                completion_tokens=2,
                elapsed_seconds=0.01,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.load_verified_acquisition_adapter",
        lambda _path: fake_adapter,
    )
    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.load_benchmark_tokenizer",
        lambda model_name, revision=None: (
            _fake_count_tokens,
            TokenizerIdentity(
                model_name=model_name,
                loader="fake",
                tokenizer_class="Fake",
                revision="pinned-revision",
            ),
        ),
    )
    monkeypatch.setattr("enterprise_memory_mlx.cli.MLXBenchmarkBackend", FakeCLIBackend)

    exit_code = main(
        [
            "--root",
            str(isolated_project),
            "benchmark",
            "--suite",
            "acquisition",
            "--arm",
            "parametric",
            "--acquisition-run",
            "run.json",
            "--max-tokens",
            "16",
        ]
    )

    assert exit_code == 0
    artifacts = list((isolated_project / "artifacts" / "benchmark").glob("benchmark-*.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["model"] == DEFAULT_MODEL
    assert {row["arm"] for row in payload["results"]} == {"parametric"}
    assert all(row["generation_status"] == "generated" for row in payload["results"])


def test_cli_bm25_arm_without_selection_fails(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "enterprise_memory_mlx.cli.load_benchmark_tokenizer",
        lambda model_name, revision=None: (_fake_count_tokens, _fake_identity(model_name)),
    )
    exit_code = main(
        ["--root", str(isolated_project), "benchmark", "--arm", "bm25", "--dry-run"]
    )
    assert exit_code == 2


def test_frozen_eval_fixtures_remain_byte_identical(project_root: Path) -> None:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    manifest = json.loads((eval_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    files = {
        name: sha256_text((eval_dir / name).read_text(encoding="utf-8"))
        for name in manifest["files"]
    }
    assert files == manifest["files"]
    assert sha256_text(json.dumps(files, sort_keys=True)) == manifest["combined_hash"]


def test_external_no_feasible_decision_matches_shipped_corpus(
    project_root: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    decision = load_operating_point_decision(default_bm25_decision_path(project_root))
    source_hash, index_hash = _corpus_hashes(records, suites)
    queries = project_root / "knowledge" / "retrieval_validation" / "v1" / "queries.jsonl"
    assert decision.status == "no_feasible_operating_point"
    assert decision.approval_status == "owner_approved"
    assert decision.selected_config is None
    assert decision.source_snapshot_hash == source_hash
    assert decision.index_payload_hash == index_hash
    assert decision.validation_dataset_hash == file_sha256(queries)
    binding, status = resolve_default_bm25_decision(
        default_bm25_decision_path(project_root),
        tuple(records) + tuple(suites.holdout_records),
    )
    assert binding is not None
    assert status == "no_feasible_operating_point"
    assert bm25_decision_artifact_payload(binding, status)["status"] == (
        "no_feasible_operating_point"
    )


def _alternate_governed_corpus() -> tuple[list[KnowledgeRecord], EvalSuites]:
    record = KnowledgeRecord.from_dict(
        {
            "id": "ALT-001",
            "domain": "operations",
            "title": "Alternate policy",
            "statement": "Alternate governed statement.",
            "source_uri": "https://source.example/alt-001",
        }
    )
    holdout = KnowledgeRecord.from_dict(
        {
            "id": "ALT-HOLD",
            "domain": "operations",
            "title": "Holdout policy",
            "statement": "Holdout governed statement.",
            "source_uri": "https://source.example/alt-hold",
        }
    )
    generator = GeneratorProvenance(kind="human", identity="test-alt-corpus")
    suites = EvalSuites(
        acquisition=(
            EvalQuestion(
                question_id="ALT-ACQ-1",
                suite="acquisition",
                question="What is the alternate policy?",
                expected="unused",
                question_family_id="ALT-ACQ-F1",
                probe_kind="recall",
                generator=generator,
                record_id="ALT-001",
            ),
        ),
        unseen_record=(
            EvalQuestion(
                question_id="ALT-UNS-1",
                suite="unseen_record",
                question="What is the holdout policy?",
                expected="unused",
                question_family_id="ALT-UNS-F1",
                probe_kind="recall",
                generator=generator,
                record_id="ALT-HOLD",
            ),
        ),
        supersession=(),
        unknown_oos=(
            EvalQuestion(
                question_id="ALT-OOS-1",
                suite="unknown_oos",
                question="What is the guest wifi password?",
                expected="unused",
                question_family_id="ALT-OOS-F1",
                probe_kind="refusal",
                generator=generator,
            ),
        ),
        scenarios=(),
        holdout_records=(holdout,),
    )
    return [record], suites


def test_non_bm25_arms_run_on_a_different_governed_corpus() -> None:
    records, suites = _alternate_governed_corpus()
    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition", "unseen_record", "unknown_oos"),
            arms=("base", "full_context", "oracle"),
        ),
    )
    assert {case.arm for case in plan} == {"base", "full_context", "oracle"}
    combined = tuple(records) + tuple(suites.holdout_records)
    computed_source = source_snapshot_hash(combined)
    computed_index = index_payload_hash(combined)
    assert computed_source != computed_index
    assert all(case.question_id.startswith("ALT-") for case in plan)


def test_mismatched_default_decision_is_not_applicable_without_blocking_non_bm25(
    project_root: Path,
    tmp_path: Path,
) -> None:
    records, suites = _alternate_governed_corpus()
    combined = tuple(records) + tuple(suites.holdout_records)
    binding, status = resolve_default_bm25_decision(
        default_bm25_decision_path(project_root),
        combined,
    )
    assert binding is None
    assert status == "not_applicable"
    payload = bm25_decision_artifact_payload(binding, status)
    assert payload == {"status": "not_applicable"}

    plan = _plan(
        records,
        suites,
        BenchmarkConfig(
            suites=("acquisition", "unknown_oos"),
            arms=("base", "full_context", "oracle"),
        ),
    )
    assert {case.arm for case in plan} == {"base", "full_context", "oracle"}
    results = run_benchmark_plan(plan[:1], FakeBackend(), max_output_tokens=50)
    target = write_benchmark_artifact(
        output_dir=tmp_path,
        model_name="fake/model",
        config=BenchmarkConfig(suites=("acquisition",), arms=("oracle",)),
        fixture_hash="alt-fixture",
        results=results,
        tokenizer_identity=_fake_identity(),
        source_hash=source_snapshot_hash(combined),
        index_hash=index_payload_hash(combined),
        bm25_decision=payload,
    )
    decided = json.loads(target.read_text(encoding="utf-8"))
    assert decided["bm25_decision"] == {"status": "not_applicable"}
    assert "source_snapshot_hash" not in decided["bm25_decision"]


def test_selected_binding_fails_closed_when_validation_or_report_is_missing(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 1, "score_threshold": 1.0},
            source_hash=source_hash,
            index_hash=index_hash,
        ),
    )
    combined = tuple(records) + tuple(suites.holdout_records)
    missing_validation = tmp_path / "missing-queries.jsonl"
    missing_report = tmp_path / "missing-report.json"

    bind_bm25_selection(path, combined)
    with pytest.raises(FileNotFoundError, match="validation dataset"):
        bind_bm25_selection(
            path,
            combined,
            validation_dataset_path=missing_validation,
        )
    with pytest.raises(FileNotFoundError, match="calibration report"):
        bind_bm25_selection(
            path,
            combined,
            calibration_report_path=missing_report,
        )


def test_file_sha256_detects_newline_and_encoding_changes(
    tmp_path: Path,
    benchmark_inputs,
) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    utf8 = tmp_path / "utf8.txt"
    utf16 = tmp_path / "utf16.txt"
    lf.write_bytes(b'{"query":"hello"}\n')
    crlf.write_bytes(b'{"query":"hello"}\r\n')
    utf8.write_bytes("café\n".encode())
    utf16.write_bytes("café\n".encode("utf-16"))

    assert file_sha256(lf) != file_sha256(crlf)
    assert sha256_text(lf.read_text(encoding="utf-8")) == sha256_text(
        crlf.read_text(encoding="utf-8")
    )
    assert file_sha256(utf8) != file_sha256(utf16)

    records, suites = benchmark_inputs
    source_hash, index_hash = _corpus_hashes(records, suites)
    path = _write_selection(
        tmp_path / "selected.json",
        _selection_payload(
            status="selected",
            selected_config={"top_k": 1, "score_threshold": 1.0},
            source_hash=source_hash,
            index_hash=index_hash,
            validation_hash=file_sha256(lf),
        ),
    )
    combined = tuple(records) + tuple(suites.holdout_records)
    bind_bm25_selection(path, combined, validation_dataset_path=lf)
    with pytest.raises(ValueError, match="validation dataset hash"):
        bind_bm25_selection(path, combined, validation_dataset_path=crlf)


def test_tokenizer_and_model_revisions_cannot_diverge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    class FakeHub:
        @staticmethod
        def model_info(model_name: str, revision: str | None = None):
            return type("Info", (), {"sha": revision or "abc123deadbeef"})()

        @staticmethod
        def snapshot_download(*args, **kwargs):
            assert kwargs["revision"] == "abc123deadbeef"
            return "/tmp/tokenizer-snapshot"

    class FakeMlxUtils:
        @staticmethod
        def load_tokenizer(path: str):
            return FakeTokenizer()

    import sys
    import types

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(utils=FakeMlxUtils))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", FakeMlxUtils)

    _, identity = load_benchmark_tokenizer("mlx-community/Qwen3-4B-Instruct-2507-4bit")
    assert identity.revision == "abc123deadbeef"
    require_matching_model_revision(identity, identity.revision)
    with pytest.raises(ValueError, match="does not match model revision"):
        require_matching_model_revision(identity, "other-revision")
