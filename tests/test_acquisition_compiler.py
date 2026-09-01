from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enterprise_memory_mlx.acquisition_compiler import (
    compile_acquisition_dataset,
    derive_exposure_schedule,
)


class HashEmbeddingBackend:
    model_id = "fake/hash-embedding"
    revision = "fake-revision"

    def encode(self, texts):
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            values.append(
                [1.0 if value & 1 else -1.0 for value in digest[:32]]
            )
        return values


class CollisionEmbeddingBackend:
    model_id = "fake/collision"
    revision = "fake-revision"

    def encode(self, texts):
        return [[1.0, 0.0] for _text in texts]


def test_exposure_schedule_is_derived_from_views_and_batch() -> None:
    schedule = derive_exposure_schedule(
        view_counts={"A": 10, "B": 10},
        target_exposures_per_fact=24,
        effective_batch_size=8,
    )

    assert schedule.epochs == 3
    assert schedule.optimizer_steps_per_epoch == 3
    assert schedule.optimizer_steps == 9
    assert schedule.micro_iterations_per_epoch == 20
    assert schedule.micro_iterations == 60
    assert schedule.gradient_accumulation_steps == 8
    assert schedule.realized_exposures_per_fact == {"A": 30, "B": 30}
    # Whole-epoch rounding overshoots the 24-exposure target; recorded openly.
    assert schedule.exposure_overshoot_per_fact == {"A": 6, "B": 6}
    assert schedule.to_dict()["exposure_overshoot_per_fact"] == {"A": 6, "B": 6}


def test_shipped_smoke_compiles_only_after_contract_checks(
    tmp_path: Path,
    project_root: Path,
) -> None:
    result = compile_acquisition_dataset(
        knowledge_dir=project_root / "knowledge",
        eval_dir=project_root / "knowledge" / "eval_frozen",
        output_root=tmp_path / "acquisition",
        semantic_backend=HashEmbeddingBackend(),
        profile="smoke_non_promotable",
        target_exposures_per_fact=24,
        effective_batch_size=8,
    )

    assert len(result.records) == 8
    assert len(result.rows) == 80
    assert result.schedule.epochs == 3
    assert result.schedule.optimizer_steps == 30
    assert result.schedule.micro_iterations == 240
    assert result.promotion_eligible is False
    assert {record.id for record in result.records}.isdisjoint({"SEC-KEY-999"})
    assert (result.dataset_dir / "train.jsonl").is_file()
    assert not (result.dataset_dir / "valid.jsonl").exists()
    assert not (result.dataset_dir / "test.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "smoke_non_promotable"
    assert manifest["promotion_eligible"] is False
    assert manifest["confirmatory_view_deficits"]
    assert manifest["view_counts"] == {
        record_id: 10 for record_id in manifest["record_ids"]
    }
    assert audit["lexical"]["passed"] is True
    assert audit["semantic"]["model_id"] == HashEmbeddingBackend.model_id
    assert audit["semantic"]["nearest_pairs"]
    assert "max_similarity_to_eval" in audit["semantic"]
    assert manifest["known_caveats"]
    assert any("unseen-form" in caveat for caveat in manifest["known_caveats"])
    assert manifest["schedule"]["exposure_overshoot_per_fact"]
    assert audit["promotion_eligible"] is False


def test_confirmatory_profile_rejects_ten_view_families(
    tmp_path: Path,
    project_root: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 24 independent view families"):
        compile_acquisition_dataset(
            knowledge_dir=project_root / "knowledge",
            eval_dir=project_root / "knowledge" / "eval_frozen",
            output_root=tmp_path,
            semantic_backend=HashEmbeddingBackend(),
            profile="confirmatory",
            target_exposures_per_fact=24,
            effective_batch_size=8,
            semantic_audit_approved_by="Human Reviewer",
        )


def test_semantic_near_duplicate_blocks_dataset_issuance(
    tmp_path: Path,
    project_root: Path,
) -> None:
    output = tmp_path / "acquisition"
    with pytest.raises(ValueError, match="Semantic-neighbour leakage"):
        compile_acquisition_dataset(
            knowledge_dir=project_root / "knowledge",
            eval_dir=project_root / "knowledge" / "eval_frozen",
            output_root=output,
            semantic_backend=CollisionEmbeddingBackend(),
            profile="smoke_non_promotable",
            target_exposures_per_fact=24,
            effective_batch_size=8,
        )
    assert not (output / "datasets").exists()
    assert not (output / "manifests").exists()


def test_frozen_hash_failure_prevents_output(
    tmp_path: Path,
    project_root: Path,
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for source in (project_root / "knowledge" / "eval_frozen").glob("*"):
        if source.is_file():
            (eval_dir / source.name).write_bytes(source.read_bytes())
    with (eval_dir / "acquisition.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="Frozen evaluation assets failed"):
        compile_acquisition_dataset(
            knowledge_dir=project_root / "knowledge",
            eval_dir=eval_dir,
            output_root=tmp_path / "output",
            semantic_backend=HashEmbeddingBackend(),
            profile="smoke_non_promotable",
            target_exposures_per_fact=24,
            effective_batch_size=8,
        )


def test_compiler_does_not_import_or_call_legacy_compiler(project_root: Path) -> None:
    source = (
        project_root
        / "src"
        / "enterprise_memory_mlx"
        / "acquisition_compiler.py"
    ).read_text(encoding="utf-8")
    assert "from .compiler" not in source
    assert "_split_rows" not in source
