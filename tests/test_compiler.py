"""Historical-reference tests for the scientifically invalid legacy compiler.

These tests document the legacy behaviour (including its known defects) and
prove that the entry point fails closed without an explicit acknowledgement.
Nothing here endorses the legacy pipeline for experiments or deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.compiler import _split_rows, compile_knowledge
from enterprise_memory_mlx.legacy_guard import LegacyPipelineDisabledError
from enterprise_memory_mlx.utils import read_jsonl


def test_compile_knowledge_fails_closed_by_default(isolated_project: Path) -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="scientifically invalid"):
        compile_knowledge(
            isolated_project / "knowledge",
            isolated_project / "artifacts" / "datasets",
        )
    assert not (isolated_project / "artifacts" / "datasets").exists()


def test_legacy_compile_excludes_restricted_and_creates_domain_datasets(
    isolated_project: Path,
) -> None:
    output = isolated_project / "artifacts" / "datasets"
    result = compile_knowledge(
        isolated_project / "knowledge",
        output,
        seed=42,
        allow_scientifically_invalid=True,
    )

    assert result.records_included == 8
    assert result.records_excluded == 1
    assert "engineering" in result.domains
    assert (output / "inject" / "train.jsonl").exists()
    assert (output / "align" / "valid.jsonl").exists()
    assert (output / "vanilla" / "train.jsonl").exists()
    assert (output / "recover" / "test.jsonl").exists()
    assert (output / "domains" / "engineering" / "align" / "train.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["governance"]["acl_enforcement_in_weights"] is False
    assert {item["id"] for item in manifest["records"]} == {
        "FIN-EXP-001",
        "FIN-INV-002",
        "HR-LEAVE-001",
        "HR-REMOTE-002",
        "ENG-REL-001",
        "ENG-INC-002",
        "SUP-SLA-001",
        "PROC-VEND-001",
    }
    assert [item["id"] for item in manifest["excluded_records"]] == ["SEC-KEY-999"]

    inject = read_jsonl(output / "inject" / "train.jsonl")
    assert all("prompt" in row and "completion" in row for row in inject)
    align = read_jsonl(output / "align" / "train.jsonl")
    assert all(row["messages"][-1]["role"] == "assistant" for row in align)


def test_legacy_split_rows_documented_defect_leaks_small_collections() -> None:
    """Known invalidity: 1-2 item collections reuse identical rows across splits.

    This is one of the reasons the legacy pipeline is disabled. The assertion
    documents the defect; it must never be treated as acceptable behaviour.
    """
    one_row = [{"n": 0}]
    train, valid, test = _split_rows(one_row)
    assert train == valid == test == one_row  # train/test identity leak

    two_rows = [{"n": 0}, {"n": 1}]
    train, valid, test = _split_rows(two_rows)
    assert valid == test  # validation/test identity leak

    three_rows = [{"n": value} for value in range(3)]
    train, valid, test = _split_rows(three_rows)
    assert len(train) == len(valid) == len(test) == 1


def test_legacy_compiled_datasets_are_deterministic(isolated_project: Path) -> None:
    first = isolated_project / "artifacts" / "first"
    second = isolated_project / "artifacts" / "second"
    compile_knowledge(
        isolated_project / "knowledge",
        first,
        seed=42,
        allow_scientifically_invalid=True,
    )
    compile_knowledge(
        isolated_project / "knowledge",
        second,
        seed=42,
        allow_scientifically_invalid=True,
    )

    first_rows = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*.jsonl")
    }
    second_rows = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*.jsonl")
    }
    assert first_rows == second_rows
