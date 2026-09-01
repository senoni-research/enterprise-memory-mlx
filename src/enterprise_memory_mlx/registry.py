from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils import atomic_write_text, sha256_json


def register_adapter(
    registry_path: Path,
    *,
    name: str,
    stage: str,
    domain: str,
    base_model: str,
    adapter_path: Path,
    config: dict[str, Any],
    knowledge_snapshot_hash: str,
) -> dict[str, Any]:
    registry = _load_registry(registry_path)
    entry = {
        "name": name,
        "stage": stage,
        "domain": domain,
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "adapter_file": str(adapter_path / "adapters.safetensors"),
        "knowledge_snapshot_hash": knowledge_snapshot_hash,
        "training_config_hash": sha256_json(config),
        "created_at": datetime.now(UTC).isoformat(),
    }
    registry["adapters"] = [
        item
        for item in registry.get("adapters", [])
        if not (item.get("name") == name and item.get("stage") == stage)
    ]
    registry["adapters"].append(entry)
    registry["adapters"].sort(key=lambda item: (item["domain"], item["name"], item["stage"]))
    atomic_write_text(registry_path, json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    return entry


def find_adapter(
    registry_path: Path,
    *,
    domain: str = "global",
    stage: str = "recover",
) -> dict[str, Any] | None:
    registry = _load_registry(registry_path)
    matches = [
        item
        for item in registry.get("adapters", [])
        if item.get("domain") == domain and item.get("stage") == stage
    ]
    return matches[-1] if matches else None


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "adapters": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("adapters", []), list):
        raise ValueError(f"Invalid adapter registry: {path}")
    return data
