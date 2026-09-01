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
