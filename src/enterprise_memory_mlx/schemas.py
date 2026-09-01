from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

Sensitivity = Literal["public", "internal_shared", "restricted", "secret"]
Status = Literal["active", "draft", "retired"]


@dataclass(frozen=True)
class QuestionAnswer:
    question: str
    answer: str
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], default_answer: str) -> QuestionAnswer:
        question = str(value.get("question", "")).strip()
        answer = str(value.get("answer", default_answer)).strip()
        keywords = tuple(
            str(item).strip()
            for item in value.get("keywords", [])
            if str(item).strip()
        )
        if not question:
            raise ValueError("Question text cannot be empty")
        if not answer:
            raise ValueError("Question answer cannot be empty")
        return cls(question=question, answer=answer, keywords=keywords)


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    domain: str
    title: str
    statement: str
    summary: str
    source_uri: str
    sensitivity: Sensitivity = "internal_shared"
    status: Status = "active"
    aliases: tuple[str, ...] = ()
    questions: tuple[QuestionAnswer, ...] = ()
    effective_from: str | None = None
    effective_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KnowledgeRecord:
        required = ("id", "domain", "title", "statement", "source_uri")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Knowledge record is missing required fields: {', '.join(missing)}")

        statement = str(value["statement"]).strip()
        summary = str(value.get("summary") or statement.split(".", maxsplit=1)[0]).strip()
        sensitivity = str(value.get("sensitivity", "internal_shared"))
        status = str(value.get("status", "active"))
        if sensitivity not in {"public", "internal_shared", "restricted", "secret"}:
            raise ValueError(f"Unsupported sensitivity: {sensitivity}")
        if status not in {"active", "draft", "retired"}:
            raise ValueError(f"Unsupported status: {status}")

        questions = tuple(
            QuestionAnswer.from_dict(item, default_answer=statement)
            for item in value.get("questions", [])
        )
        record = cls(
            id=str(value["id"]).strip(),
            domain=str(value["domain"]).strip().lower(),
            title=str(value["title"]).strip(),
            statement=statement,
            summary=summary,
            source_uri=str(value["source_uri"]).strip(),
            sensitivity=sensitivity,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            aliases=tuple(
                str(item).strip()
                for item in value.get("aliases", [])
                if str(item).strip()
            ),
            questions=questions,
            effective_from=_optional_string(value.get("effective_from")),
            effective_to=_optional_string(value.get("effective_to")),
            metadata=dict(value.get("metadata", {})),
        )
        record.validate_dates()
        return record

    def validate_dates(self) -> None:
        start = _parse_date(self.effective_from)
        end = _parse_date(self.effective_to)
        if start and end and end < start:
            raise ValueError(f"Record {self.id}: effective_to precedes effective_from")

    def is_trainable(self, include_restricted: bool = False) -> bool:
        if self.status != "active":
            return False
        return include_restricted or self.sensitivity not in {"restricted", "secret"}

    def as_manifest_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "title": self.title,
            "source_uri": self.source_uri,
            "sensitivity": self.sensitivity,
            "status": self.status,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
        }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Expected ISO date, received: {value}") from exc
