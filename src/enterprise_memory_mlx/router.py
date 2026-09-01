from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .compiler import load_records
from .registry import find_adapter
from .utils import slugify


@dataclass(frozen=True)
class RouteDecision:
    domain: str | None
    score: float
    adapter_path: str | None
    action: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "score": round(self.score, 2),
            "adapter_path": self.adapter_path,
            "action": self.action,
            "evidence": list(self.evidence),
        }


def route_query(
    *,
    query: str,
    knowledge_dir: Path,
    registry_path: Path,
    threshold: float = 48.0,
    allow_scientifically_invalid: bool = False,
) -> RouteDecision:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "router.route_query",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    normalized = " ".join(query.lower().split())
    records = [record for record in load_records(knowledge_dir) if record.is_trainable()]
    by_domain: dict[str, list[str]] = {}
    for record in records:
        phrases = by_domain.setdefault(record.domain, [])
        phrases.extend([record.title, record.summary, *record.aliases])

    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for domain, phrases in by_domain.items():
        phrase_scores = [
            (fuzz.token_set_ratio(normalized, phrase.lower()), phrase)
            for phrase in phrases
        ]
        phrase_scores.sort(reverse=True)
        best = phrase_scores[:3]
        score = best[0][0] if best else 0.0
        exact_bonus = 10.0 if any(phrase.lower() in normalized for _, phrase in best) else 0.0
        scored.append(
            (min(100.0, score + exact_bonus), domain, tuple(phrase for _, phrase in best))
        )

    if not scored:
        return RouteDecision(None, 0.0, None, "context_required", ())
    scored.sort(reverse=True)
    score, domain, evidence = scored[0]
    if score < threshold:
        return RouteDecision(None, score, None, "context_required", evidence)

    adapter = find_adapter(registry_path, domain=domain, stage="recover")
    if adapter is None:
        expected = (
            registry_path.parent.parent
            / "adapters"
            / "domains"
            / slugify(domain)
            / "recover"
        )
        return RouteDecision(domain, score, str(expected), "train_or_load_domain_adapter", evidence)
    return RouteDecision(domain, score, adapter.get("adapter_path"), "load_adapter", evidence)


def route_json(**kwargs: Any) -> str:
    return json.dumps(route_query(**kwargs).to_dict(), indent=2)
