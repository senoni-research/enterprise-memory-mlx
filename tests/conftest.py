from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_project(tmp_path: Path, project_root: Path) -> Path:
    shutil.copytree(project_root / "knowledge", tmp_path / "knowledge")
    (tmp_path / "artifacts").mkdir()
    return tmp_path


@pytest.fixture
def private_approval_path(project_root: Path) -> Path:
    from enterprise_memory_mlx.learning_mechanics import APPROVAL_RECORD_RELATIVE

    path = project_root / APPROVAL_RECORD_RELATIVE
    if not path.is_file():
        pytest.skip("private deliverable-review pack is not present")
    return path
