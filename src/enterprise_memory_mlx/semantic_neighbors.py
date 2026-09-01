"""Pinned local semantic-neighbour scanning with an injectable backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_SEMANTIC_MODEL = "mlx-community/all-MiniLM-L6-v2-4bit"


class EmbeddingBackend(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class SemanticNeighbor:
    left_id: str
    right_id: str
    score: float
    left_text: str
    right_text: str

    def to_dict(self) -> dict:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "score": round(self.score, 8),
            "left_text": self.left_text,
            "right_text": self.right_text,
        }


class MLXEmbeddingBackend:
    """Local MLX embedding backend pinned to an immutable revision."""

    def __init__(
        self,
        model_id: str = DEFAULT_SEMANTIC_MODEL,
        *,
        revision: str | None = None,
    ) -> None:
        try:
            from huggingface_hub import model_info, snapshot_download
            from mlx_embeddings import generate, load
        except ImportError as exc:
            raise RuntimeError(
                'Semantic scan requires: pip install -e ".[semantic]"'
            ) from exc
        info = model_info(model_id, revision=revision)
        resolved = getattr(info, "sha", None)
        if not resolved:
            raise ValueError(f"Could not resolve semantic model revision: {model_id}")
        self.model_id = model_id
        self.revision = str(resolved)
        model_path = snapshot_download(
            model_id,
            revision=self.revision,
        )
        self._model, self._tokenizer = load(model_path)
        self._generate = generate

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        output = self._generate(
            self._model,
            self._tokenizer,
            texts=list(texts),
        )
        embeddings = output.text_embeds
        return embeddings.tolist()


def nearest_neighbors(
    left: Sequence[tuple[str, str]],
    right: Sequence[tuple[str, str]],
    backend: EmbeddingBackend,
    *,
    top_n: int = 25,
) -> tuple[SemanticNeighbor, ...]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not left or not right:
        return ()
    left_vectors = backend.encode([text for _item_id, text in left])
    right_vectors = backend.encode([text for _item_id, text in right])
    if len(left_vectors) != len(left) or len(right_vectors) != len(right):
        raise ValueError("Embedding backend returned the wrong number of vectors")
    pairs = []
    for (left_id, left_text), left_vector in zip(left, left_vectors, strict=True):
        for (right_id, right_text), right_vector in zip(
            right, right_vectors, strict=True
        ):
            score = _cosine(left_vector, right_vector)
            pairs.append(
                SemanticNeighbor(
                    left_id=left_id,
                    right_id=right_id,
                    score=score,
                    left_text=left_text,
                    right_text=right_text,
                )
            )
    pairs.sort(key=lambda item: (-item.score, item.left_id, item.right_id))
    return tuple(pairs[:top_n])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have equal non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Embedding vectors must have non-zero norm")
    return dot / (left_norm * right_norm)
