from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def packager(project_root: Path):
    path = project_root / "scripts" / "package_source.py"
    spec = importlib.util.spec_from_file_location("package_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_package_is_deterministic_and_verified(
    tmp_path: Path,
    project_root: Path,
    packager,
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = packager.build_source_zip(project_root, first)
    second_result = packager.build_source_zip(project_root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["sha256"] == second_result["sha256"]
    assert packager.verify_source_zip(first)["verified"] is True


def test_source_package_contains_manifest_and_current_source(
    tmp_path: Path,
    project_root: Path,
    packager,
) -> None:
    target = tmp_path / "source.zip"
    packager.build_source_zip(project_root, target)

    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        prefix = "enterprise-memory-mlx/"
        assert prefix + "SOURCE_MANIFEST.json" in names
        assert prefix + "README.md" in names
        assert prefix + "src/enterprise_memory_mlx/review_ui.py" in names
        assert prefix + "knowledge/judge_calibration/v3/cases.jsonl" in names
        manifest = json.loads(archive.read(prefix + "SOURCE_MANIFEST.json"))
        assert manifest["file_count"] == len(names) - 1
        assert set(manifest["files"]) == {
            name.removeprefix(prefix)
            for name in names
            if name != prefix + "SOURCE_MANIFEST.json"
        }


def test_source_package_excludes_generated_private_and_environment_paths(
    tmp_path: Path,
    project_root: Path,
    packager,
) -> None:
    target = tmp_path / "source.zip"
    packager.build_source_zip(project_root, target)

    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        forbidden = (
            "/.venv/",
            "/artifacts/",
            "/dev/",
            "/dist/",
            "/.specstory/",
            "/knowledge/private/",
            "/models/",
            "/__MACOSX/",
            "/__pycache__/",
            ".pyc",
            ".safetensors",
            ".gguf",
        )
        assert not any(token in name for name in names for token in forbidden)
        assert not any(name.endswith(".egg-info") for name in names)


def test_required_source_file_missing_fails_closed(
    tmp_path: Path,
    packager,
) -> None:
    for directory in packager.ROOT_DIRECTORIES:
        (tmp_path / directory).mkdir(parents=True)
    for name in packager.ROOT_FILES - {"README.md"}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="required source file missing"):
        packager.collect_source_files(tmp_path)
