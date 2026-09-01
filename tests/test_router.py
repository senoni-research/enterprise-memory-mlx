"""Historical-reference tests for the scientifically invalid lexical router."""

from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.legacy_guard import LegacyPipelineDisabledError
from enterprise_memory_mlx.router import route_query


def test_route_query_fails_closed_by_default(
    project_root: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="scientifically invalid"):
        route_query(
            query="What approvals are needed for a high risk production release?",
            knowledge_dir=project_root / "knowledge",
            registry_path=tmp_path / "adapters.json",
        )


def test_legacy_router_selects_engineering_domain(
    project_root: Path,
    tmp_path: Path,
) -> None:
    decision = route_query(
        query="What approvals are needed for a high risk production release?",
        knowledge_dir=project_root / "knowledge",
        registry_path=tmp_path / "adapters.json",
        allow_scientifically_invalid=True,
    )
    assert decision.domain == "engineering"
    assert decision.action == "train_or_load_domain_adapter"
    assert decision.score >= 48
