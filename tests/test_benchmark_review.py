from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from enterprise_memory_mlx import cli
from enterprise_memory_mlx.benchmark_review import (
    BenchmarkReviewError,
    BenchmarkReviewPreparation,
    BenchmarkReviewReportPaths,
    build_benchmark_review_report,
    prepare_benchmark_review,
    write_benchmark_review_report,
    write_model_review_advisory,
)
from enterprise_memory_mlx.review_ui import ReviewDataError, ReviewStore
from enterprise_memory_mlx.split_contract import load_eval_suites

ARMS = ("base", "full_context", "oracle", "bm25", "parametric")


def _source_artifacts(
    tmp_path: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    eval_dir = project_root / "knowledge" / "eval_frozen"
    suites = load_eval_suites(eval_dir)
    questions = list(suites.acquisition)
    assert len(questions) == 32
    fixture_hash = json.loads(
        (eval_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )["combined_hash"]
    results = []
    grading_rows = []
    for question_index, question in enumerate(questions):
        for arm in ARMS:
            results.append(
                {
                    "question_id": question.question_id,
                    "arm": arm,
                    "suite": "acquisition",
                    "question": question.question,
                    "generation_status": "generated",
                    # Deliberately arm-neutral so packet-content checks are exact.
                    "output": f"Candidate answer {question_index + 1}.",
                }
            )
            reviewable = arm not in {"base", "parametric"} or (
                arm == "parametric" and question_index < 5
            )
            grading_rows.append(
                {
                    "question_id": question.question_id,
                    "arm": arm,
                    "suite": "acquisition",
                    "question_family_id": question.question_family_id,
                    "generation_status": "generated",
                    "status": (
                        "semantic_review_required"
                        if reviewable
                        else "deterministic_hard_fail"
                    ),
                    "deterministic_score": None if reviewable else 0.0,
                }
            )
    benchmark = {
        "schema_version": 1,
        "graded": False,
        "fixture_hash": fixture_hash,
        "config": {"arms": list(ARMS), "suites": ["acquisition"]},
        "results": results,
    }
    benchmark_path = tmp_path / "benchmark-smoke.json"
    benchmark_path.write_text(
        json.dumps(benchmark, indent=2) + "\n",
        encoding="utf-8",
    )
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    grading = {
        "schema_version": 1,
        "graded": True,
        "mode": "deterministic_only",
        "raw_artifact_hash": benchmark_hash,
        "fixture_hash": fixture_hash,
        "grader_config_hash": "a" * 64,
        "rows": grading_rows,
    }
    grading_path = tmp_path / "deterministic-grading.json"
    grading_path.write_text(
        json.dumps(grading, indent=2) + "\n",
        encoding="utf-8",
    )
    return benchmark_path, grading_path


def _prepare(
    tmp_path: Path,
    project_root: Path,
) -> BenchmarkReviewPreparation:
    benchmark_path, grading_path = _source_artifacts(tmp_path, project_root)
    return prepare_benchmark_review(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "packets",
        salt=b"fixed-test-salt-that-is-private",
    )


def _complete_review(
    tmp_path: Path,
    prepared: BenchmarkReviewPreparation,
    *,
    score: float,
) -> tuple[Path, Path]:
    store = ReviewStore(
        root=tmp_path,
        packet_path=prepared.packet_path,
        mapping_path=prepared.mapping_path,
        reviewer_id="Human One",
        state_root=tmp_path / f"state-{score}",
    )
    for review_id in store.overview()["case_ids"]:
        store.save(
            review_id=review_id,
            score=score,
            reason="Independent human diagnostic assessment.",
            confidence="high",
            needs_human_attention=False,
            human_attested=True,
        )
    return store.export_overlay()


def _packet_payloads(path: Path) -> tuple[dict, list[dict], list[dict], set[str]]:
    with zipfile.ZipFile(path) as archive:
        manifest_name = next(
            name for name in archive.namelist() if name.endswith("packet_manifest.json")
        )
        prefix = manifest_name.removesuffix("packet_manifest.json")
        manifest = json.loads(archive.read(manifest_name))
        case_bytes = archive.read(prefix + "review_cases.jsonl")
        source_bytes = archive.read(prefix + "source_records.jsonl")
        cases = [
            json.loads(line)
            for line in case_bytes.decode().splitlines()
            if line.strip()
        ]
        sources = [
            json.loads(line)
            for line in source_bytes.decode().splitlines()
            if line.strip()
        ]
        return manifest, cases, sources, set(archive.namelist())


def test_prepare_builds_160_row_arm_blinded_hash_bound_packet(
    tmp_path: Path,
    project_root: Path,
) -> None:
    prepared = _prepare(tmp_path, project_root)
    manifest, cases, sources, members = _packet_payloads(prepared.packet_path)
    mapping = json.loads(prepared.mapping_path.read_text(encoding="utf-8"))

    assert prepared.case_count == 160
    assert len(cases) == 160
    assert len({case["review_id"] for case in cases}) == 160
    assert sources == []
    assert all(case["source_record_ids"] == [] for case in cases)
    assert all(
        set(case)
        == {
            "review_id",
            "question",
            "reference_answer",
            "candidate_answer",
            "source_record_ids",
        }
        for case in cases
    )
    blinded_content = json.dumps(cases)
    assert '"arm"' not in blinded_content
    assert '"question_id"' not in blinded_content
    assert '"deterministic_status"' not in blinded_content
    assert '"selected_record_ids"' not in blinded_content
    assert "mapping.json" not in " ".join(members)
    assert "arms" not in manifest
    assert "suites" not in manifest
    assert manifest["blinding"][
        "source_records_excluded_to_prevent_arm_fingerprinting"
    ]
    assert manifest["private_mapping_sha256"] == hashlib.sha256(
        prepared.mapping_path.read_bytes()
    ).hexdigest()
    assert {row["arm"] for row in mapping["mapping"]} == set(ARMS)
    assert len(mapping["mapping"]) == 160

    store = ReviewStore(
        root=tmp_path,
        packet_path=prepared.packet_path,
        mapping_path=prepared.mapping_path,
        reviewer_id="Human One",
        state_root=tmp_path / "isolated-state",
    )
    assert store.total == 160
    assert store.case(cases[0]["review_id"])["source_records"] == []


def test_completed_report_unblinds_only_after_review_and_preserves_hard_fails(
    tmp_path: Path,
    project_root: Path,
) -> None:
    benchmark_path, grading_path = _source_artifacts(tmp_path, project_root)
    prepared = prepare_benchmark_review(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "packets",
        salt=b"fixed-test-salt-that-is-private",
    )
    overlay, overlay_manifest = _complete_review(
        tmp_path,
        prepared,
        score=1.0,
    )

    paths = write_benchmark_review_report(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        packet_path=prepared.packet_path,
        mapping_path=prepared.mapping_path,
        overlay_path=overlay,
        overlay_manifest_path=overlay_manifest,
        output_dir=tmp_path / "reports",
    )
    report = json.loads(paths.json_path.read_text(encoding="utf-8"))
    by_arm = {row["arm"]: row for row in report["by_arm"]}

    assert paths.decision == "authorize_one_redesigned_diagnostic"
    assert report["promotion_eligible"] is False
    assert report["review_boundary"]["human_approved"] is False
    assert report["review_boundary"]["usable_for_judge_certification"] is False
    # Human judged every base output correct, but all base rows are hard fails.
    assert by_arm["base"]["direct_human"]["mean_score"] == 1.0
    assert by_arm["base"]["governed_final"]["mean_score"] == 0.0
    assert by_arm["base"]["governed_final"]["fully_correct_count"] == 0
    # Exactly five parametric rows survive deterministic grading.
    assert by_arm["parametric"]["direct_human"]["fully_correct_count"] == 32
    assert by_arm["parametric"]["governed_final"]["fully_correct_count"] == 5
    assert report["paired_parametric_vs_base"] == {
        "paired_question_count": 32,
        "parametric_wins": 5,
        "parametric_losses": 0,
        "ties": 27,
    }
    assert all(
        criterion["passed"]
        for criterion in report["decision"]["criteria"].values()
    )
    assert "| parametric | 32 | 27 |" in paths.markdown_path.read_text(
        encoding="utf-8"
    )


def test_completed_report_stops_when_parametric_has_no_human_signal(
    tmp_path: Path,
    project_root: Path,
) -> None:
    benchmark_path, grading_path = _source_artifacts(tmp_path, project_root)
    prepared = prepare_benchmark_review(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "packets",
        salt=b"fixed-test-salt-that-is-private",
    )
    overlay, overlay_manifest = _complete_review(
        tmp_path,
        prepared,
        score=0.0,
    )

    report = build_benchmark_review_report(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        packet_path=prepared.packet_path,
        mapping_path=prepared.mapping_path,
        overlay_path=overlay,
        overlay_manifest_path=overlay_manifest,
    )

    assert report["decision"]["status"] == "stop_parametric_research"
    assert not any(
        criterion["passed"]
        for criterion in report["decision"]["criteria"].values()
    )


def test_mapping_tamper_and_incomplete_overlay_fail_closed(
    tmp_path: Path,
    project_root: Path,
) -> None:
    benchmark_path, grading_path = _source_artifacts(tmp_path, project_root)
    prepared = prepare_benchmark_review(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "packets",
        salt=b"fixed-test-salt-that-is-private",
    )
    original_mapping = prepared.mapping_path.read_bytes()
    prepared.mapping_path.write_bytes(original_mapping + b"\n")
    with pytest.raises(ReviewDataError, match="mapping hash"):
        ReviewStore(
            root=tmp_path,
            packet_path=prepared.packet_path,
            mapping_path=prepared.mapping_path,
            reviewer_id="Human One",
            state_root=tmp_path / "state",
        )
    prepared.mapping_path.write_bytes(original_mapping)

    mapping = json.loads(original_mapping)
    incomplete_rows = [
        {
            "case_id": row["case_id"],
            "human_semantic_score": 1.0,
            "human_reason": "Reviewed.",
            "human_reviewers": ["Human One"],
            "human_approved": False,
            "reviewer_kind": "human",
            "identity_verification": "asserted_only_not_authenticated",
            "os_user": "test-user",
            "confidence": "high",
            "needs_human_attention": False,
            "reviewed_at": "2026-09-01T12:00:00+00:00",
        }
        for row in mapping["mapping"][:-1]
    ]
    overlay = tmp_path / "incomplete.jsonl"
    overlay.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in incomplete_rows),
        encoding="utf-8",
    )
    overlay_manifest = tmp_path / "incomplete.manifest.json"
    overlay_manifest.write_text(
        json.dumps(
            {
                "status": "single_human_review_complete_not_adjudicated",
                "reviewer_id": "Human One",
                "identity_verification": "asserted_only_not_authenticated",
                "os_users": ["test-user"],
                "case_count": len(incomplete_rows),
                "source_packet_sha256": hashlib.sha256(
                    prepared.packet_path.read_bytes()
                ).hexdigest(),
                "private_mapping_sha256": hashlib.sha256(
                    prepared.mapping_path.read_bytes()
                ).hexdigest(),
                "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                "human_approved": False,
                "requires_second_review": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkReviewError, match="does not cover every"):
        build_benchmark_review_report(
            benchmark_path=benchmark_path,
            deterministic_grading_path=grading_path,
            packet_path=prepared.packet_path,
            mapping_path=prepared.mapping_path,
            overlay_path=overlay,
            overlay_manifest_path=overlay_manifest,
        )


def test_cli_benchmark_review_prepare_and_report_delegate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_calls: list[dict] = []
    report_calls: list[dict] = []

    def fake_prepare(**kwargs):
        prepared_calls.append(kwargs)
        return BenchmarkReviewPreparation(
            packet_path=tmp_path / "packet.zip",
            mapping_path=tmp_path / "mapping.json",
            case_count=160,
            packet_sha256="a" * 64,
            mapping_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            deterministic_grading_sha256="d" * 64,
        )

    def fake_report(**kwargs):
        report_calls.append(kwargs)
        return BenchmarkReviewReportPaths(
            json_path=tmp_path / "report.json",
            markdown_path=tmp_path / "report.md",
            decision="stop_parametric_research",
        )

    monkeypatch.setattr(cli, "prepare_benchmark_review", fake_prepare)
    monkeypatch.setattr(cli, "write_benchmark_review_report", fake_report)

    assert (
        cli.main(
            [
                "--root",
                str(tmp_path),
                "benchmark-review",
                "prepare",
                "--benchmark",
                "benchmark.json",
                "--grading",
                "grading.json",
            ]
        )
        == 0
    )
    assert prepared_calls[0]["benchmark_path"] == tmp_path / "benchmark.json"
    assert prepared_calls[0]["deterministic_grading_path"] == (
        tmp_path / "grading.json"
    )

    assert (
        cli.main(
            [
                "--root",
                str(tmp_path),
                "benchmark-review",
                "report",
                "--benchmark",
                "benchmark.json",
                "--grading",
                "grading.json",
                "--packet",
                "packet.zip",
                "--mapping",
                "mapping.json",
                "--overlay",
                "human.jsonl",
            ]
        )
        == 0
    )
    assert report_calls[0]["overlay_manifest_path"] == (
        tmp_path / "human.manifest.json"
    )


def test_model_review_advisory_never_satisfies_human_gate(
    tmp_path: Path,
    project_root: Path,
) -> None:
    benchmark_path, grading_path = _source_artifacts(tmp_path, project_root)
    prepared = prepare_benchmark_review(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "packets",
        salt=b"fixed-test-salt-that-is-private",
    )
    _manifest, cases, _sources, _members = _packet_payloads(
        prepared.packet_path
    )
    label_path = tmp_path / "model-labels.jsonl"
    label_path.write_text(
        "".join(
            json.dumps(
                {
                    "review_id": case["review_id"],
                    "reviewer_kind": "model",
                    "reviewer_identity": "GPT test model",
                    "score": 1.0,
                    "reason": "Independent model assessment.",
                    "confidence": "high",
                    "needs_human_attention": False,
                }
            )
            + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )

    paths = write_model_review_advisory(
        benchmark_path=benchmark_path,
        deterministic_grading_path=grading_path,
        packet_path=prepared.packet_path,
        mapping_path=prepared.mapping_path,
        label_paths=[label_path],
        output_dir=tmp_path / "advisory",
    )
    report = json.loads(paths.json_path.read_text(encoding="utf-8"))
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    assert paths.apparent_rule_result == "authorize_one_redesigned_diagnostic"
    assert report["operative_research_decision"] == "human_review_pending"
    assert report["promotion_eligible"] is False
    assert report["usable_for_judge_certification"] is False
    assert manifest["status"] == "model_triage_complete_human_review_pending"
    assert manifest["human_approved"] is False
    assert manifest["human_review_satisfied"] is False
    assert all("direct_model" in row for row in report["by_arm"])
    assert "direct_human" not in json.dumps(report)

    invalid = label_path.read_text(encoding="utf-8").replace(
        '"reviewer_kind": "model"',
        '"reviewer_kind": "human"',
        1,
    )
    label_path.write_text(invalid, encoding="utf-8")
    with pytest.raises(BenchmarkReviewError, match="cannot contain a human"):
        write_model_review_advisory(
            benchmark_path=benchmark_path,
            deterministic_grading_path=grading_path,
            packet_path=prepared.packet_path,
            mapping_path=prepared.mapping_path,
            label_paths=[label_path],
            output_dir=tmp_path / "invalid-advisory",
        )
