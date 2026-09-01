"""Small non-promotable general-behaviour diagnostic for acquisition smoke."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition_training import VerifiedAcquisitionAdapter
from .benchmark import GeneratedAnswer, MLXBenchmarkBackend
from .utils import atomic_write_text

GENERAL_SYSTEM = "Follow the user's instruction accurately and concisely."


def run_general_diagnostic(
    *,
    rows: Sequence[dict[str, Any]],
    adapter: VerifiedAcquisitionAdapter,
    output_dir: Path,
    max_tokens: int = 80,
) -> Path:
    if not rows:
        raise ValueError("General diagnostic rows cannot be empty")
    base_backend = MLXBenchmarkBackend(
        adapter.model_id,
        revision=adapter.model_revision,
    )
    try:
        base = _generate_rows(rows, base_backend, max_tokens)
    finally:
        base_backend.close()
    adapter_backend = MLXBenchmarkBackend(
        adapter.model_id,
        revision=adapter.model_revision,
        adapter_path=str(adapter.adapter_path),
    )
    try:
        adapted = _generate_rows(rows, adapter_backend, max_tokens)
    finally:
        adapter_backend.close()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "diagnostic_non_promotable",
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "adapter_hash": adapter.adapter_hash,
        "item_count": len(rows),
        "scoring": "normalized_exact_match_only",
        "warning": (
            "Ten simple items are insufficient to measure general-capability "
            "retention. This is a machinery/drift diagnostic only."
        ),
        "base": _summary(base),
        "parametric": _summary(adapted),
        "rows": [
            {
                "instruction": str(row["instruction"]),
                "expected": str(row["response"]),
                "base": base[index],
                "parametric": adapted[index],
            }
            for index, row in enumerate(rows)
        ],
        "promotion_eligible": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "general-diagnostic.json"
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def normalized_exact(output: str, expected: str) -> bool:
    return _normalize(output) == _normalize(expected)


def _generate_rows(
    rows: Sequence[dict[str, Any]],
    backend: Any,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        expected = str(row.get("response", "")).strip()
        if not instruction or not expected:
            raise ValueError("General diagnostic requires instruction and response")
        answer: GeneratedAnswer = backend.generate(
            system_prompt=GENERAL_SYSTEM,
            question=instruction,
            max_tokens=max_tokens,
        )
        results.append(
            {
                "output": answer.output,
                "normalized_exact": normalized_exact(answer.output, expected),
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "elapsed_seconds": round(answer.elapsed_seconds, 6),
            }
        )
    return results


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(item["normalized_exact"] for item in rows)
    return {
        "normalized_exact": exact,
        "total": len(rows),
        "rate": exact / len(rows),
        "mean_elapsed_seconds": (
            sum(float(item["elapsed_seconds"]) for item in rows) / len(rows)
        ),
    }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())
