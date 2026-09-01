"""Deterministic full-context and evaluation-only oracle controls."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from .schemas import KnowledgeRecord, Sensitivity

ContextMode = Literal["full_context", "bm25_context", "oracle_context"]
ContextAction = Literal["use_context", "context_too_large", "source_required"]
BudgetExhausted = Literal["bytes", "tokens", "bytes_and_tokens"]
TokenCounter = Callable[[str], int]
DEFAULT_MAX_UTF8_BYTES = 65_536
_SENSITIVITY_RANK: dict[Sensitivity, int] = {
    "public": 0,
    "internal_shared": 1,
    "restricted": 2,
    "secret": 3,
}

CONTEXT_INSTRUCTION = (
    "Use only the supplied governed records as authoritative company knowledge. "
    "If the records do not support the question, or the question requires live or current "
    "information, refer the user to the current source system. Do not guess or invent a rule."
)


@dataclass(frozen=True)
class ContextBaselineResult:
    """A context decision and the provenance needed to audit it.

    For ``context_too_large``, ``context`` is deliberately empty while
    ``utf8_bytes`` and ``token_count`` report the size the complete rendered
    candidate would require. The corresponding ``max_*`` fields report the
    allowed budgets.
    """

    mode: ContextMode
    context: str
    selected_record_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    highest_sensitivity: Sensitivity | None
    utf8_bytes: int
    action: ContextAction
    max_utf8_bytes: int
    max_tokens: int | None
    token_count: int | None
    instruction_utf8_bytes: int
    records_utf8_bytes: int
    record_count: int
    budget_exhausted: BudgetExhausted | None


def render_authoritative_record(record: KnowledgeRecord) -> str:
    """Render one eligible governed record without evaluation metadata."""
    if not record.is_trainable():
        raise ValueError(f"Record {record.id!r} is not eligible for authoritative context")

    lines = [
        "<authoritative_record>",
        f"Record ID: {record.id}",
        f"Domain: {record.domain}",
        f"Title: {record.title}",
        f"Status: {record.status}",
        f"Sensitivity: {record.sensitivity}",
    ]
    if record.effective_from is not None or record.effective_to is not None:
        lines.append(
            "Effective interval: "
            f"{record.effective_from or 'not specified'} to "
            f"{record.effective_to or 'not specified'}"
        )
    lines.extend(
        [
            f"Canonical statement: {record.statement}",
            f"Source URI: {record.source_uri}",
            "</authoritative_record>",
        ]
    )
    return "\n".join(lines)


def build_full_context(
    records: Iterable[KnowledgeRecord],
    *,
    max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES,
    max_tokens: int | None = None,
    count_tokens: TokenCounter | None = None,
) -> ContextBaselineResult:
    """Render every eligible record in stable record-ID order."""
    _validate_budgets(max_utf8_bytes, max_tokens, count_tokens)
    governed_records = _unique_records(records)
    selected = tuple(
        sorted(
            (record for record in governed_records if record.is_trainable()),
            key=lambda record: record.id,
        )
    )
    if not selected:
        return _source_required(
            "full_context",
            max_utf8_bytes,
            max_tokens,
            token_accounting=count_tokens is not None,
        )
    return _render_selection(
        "full_context",
        selected,
        max_utf8_bytes,
        max_tokens,
        count_tokens,
    )


def build_oracle_context(
    records: Iterable[KnowledgeRecord],
    record_ids: Iterable[str] | None,
    *,
    max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES,
    max_tokens: int | None = None,
    count_tokens: TokenCounter | None = None,
) -> ContextBaselineResult:
    """Render exactly the explicitly requested eligible records for evaluation.

    ``None`` or an empty ID collection represents an unknown/out-of-scope
    evaluation case. Unknown or ineligible IDs are errors rather than retrieval
    prompts, preventing semantic substitution.
    """
    _validate_budgets(max_utf8_bytes, max_tokens, count_tokens)
    governed_records = _unique_records(records)
    records_by_id = {record.id: record for record in governed_records}
    requested_ids = tuple(record_ids or ())
    if not requested_ids:
        return _source_required(
            "oracle_context",
            max_utf8_bytes,
            max_tokens,
            token_accounting=count_tokens is not None,
        )

    unknown_ids = sorted(set(requested_ids) - records_by_id.keys())
    if unknown_ids:
        raise ValueError(f"Unknown oracle record IDs: {', '.join(unknown_ids)}")

    ineligible_ids = sorted(
        record_id
        for record_id in set(requested_ids)
        if not records_by_id[record_id].is_trainable()
    )
    if ineligible_ids:
        raise ValueError(f"Ineligible oracle record IDs: {', '.join(ineligible_ids)}")

    selected = tuple(records_by_id[record_id] for record_id in sorted(set(requested_ids)))
    return _render_selection(
        "oracle_context",
        selected,
        max_utf8_bytes,
        max_tokens,
        count_tokens,
    )


def build_retrieved_context(
    records: Iterable[KnowledgeRecord],
    *,
    max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES,
    max_tokens: int | None = None,
    count_tokens: TokenCounter | None = None,
) -> ContextBaselineResult:
    """Render retrieved records in ranking order.

    Unlike oracle context, input order is meaningful: it is the retriever's
    deterministic rank. Empty retrievals fail closed and ineligible records
    are rejected rather than silently omitted.
    """
    _validate_budgets(max_utf8_bytes, max_tokens, count_tokens)
    selected = _unique_records(records)
    if not selected:
        return _source_required(
            "bm25_context",
            max_utf8_bytes,
            max_tokens,
            token_accounting=count_tokens is not None,
        )
    ineligible_ids = sorted(record.id for record in selected if not record.is_trainable())
    if ineligible_ids:
        raise ValueError(f"Ineligible retrieved record IDs: {', '.join(ineligible_ids)}")
    return _render_selection(
        "bm25_context",
        selected,
        max_utf8_bytes,
        max_tokens,
        count_tokens,
    )


def _unique_records(records: Iterable[KnowledgeRecord]) -> tuple[KnowledgeRecord, ...]:
    materialized = tuple(records)
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for record in materialized:
        if record.id in seen:
            duplicate_ids.add(record.id)
        seen.add(record.id)
    if duplicate_ids:
        raise ValueError(f"Duplicate knowledge record IDs: {', '.join(sorted(duplicate_ids))}")
    return materialized


def _render_selection(
    mode: ContextMode,
    records: tuple[KnowledgeRecord, ...],
    max_utf8_bytes: int,
    max_tokens: int | None,
    count_tokens: TokenCounter | None,
) -> ContextBaselineResult:
    instruction = CONTEXT_INSTRUCTION + "\n\n"
    rendered_records = "\n\n".join(
        render_authoritative_record(record) for record in records
    )
    context = instruction + rendered_records
    instruction_utf8_bytes = len(instruction.encode("utf-8"))
    records_utf8_bytes = len(rendered_records.encode("utf-8"))
    utf8_bytes = instruction_utf8_bytes + records_utf8_bytes
    token_count = _count_tokens(count_tokens, context)
    record_ids = tuple(record.id for record in records)
    source_uris = tuple(record.source_uri for record in records)
    highest_sensitivity = max(
        (record.sensitivity for record in records),
        key=_SENSITIVITY_RANK.__getitem__,
    )
    budget_exhausted = _budget_exhausted(
        utf8_bytes,
        max_utf8_bytes,
        token_count,
        max_tokens,
    )
    if budget_exhausted is not None:
        return ContextBaselineResult(
            mode=mode,
            context="",
            selected_record_ids=record_ids,
            source_uris=source_uris,
            highest_sensitivity=highest_sensitivity,
            utf8_bytes=utf8_bytes,
            action="context_too_large",
            max_utf8_bytes=max_utf8_bytes,
            max_tokens=max_tokens,
            token_count=token_count,
            instruction_utf8_bytes=instruction_utf8_bytes,
            records_utf8_bytes=records_utf8_bytes,
            record_count=len(records),
            budget_exhausted=budget_exhausted,
        )
    return ContextBaselineResult(
        mode=mode,
        context=context,
        selected_record_ids=record_ids,
        source_uris=source_uris,
        highest_sensitivity=highest_sensitivity,
        utf8_bytes=utf8_bytes,
        action="use_context",
        max_utf8_bytes=max_utf8_bytes,
        max_tokens=max_tokens,
        token_count=token_count,
        instruction_utf8_bytes=instruction_utf8_bytes,
        records_utf8_bytes=records_utf8_bytes,
        record_count=len(records),
        budget_exhausted=None,
    )


def _source_required(
    mode: ContextMode,
    max_utf8_bytes: int,
    max_tokens: int | None,
    *,
    token_accounting: bool,
) -> ContextBaselineResult:
    return ContextBaselineResult(
        mode=mode,
        context="",
        selected_record_ids=(),
        source_uris=(),
        highest_sensitivity=None,
        utf8_bytes=0,
        action="source_required",
        max_utf8_bytes=max_utf8_bytes,
        max_tokens=max_tokens,
        token_count=0 if token_accounting else None,
        instruction_utf8_bytes=0,
        records_utf8_bytes=0,
        record_count=0,
        budget_exhausted=None,
    )


def _validate_budgets(
    max_utf8_bytes: int,
    max_tokens: int | None,
    count_tokens: TokenCounter | None,
) -> None:
    if isinstance(max_utf8_bytes, bool) or not isinstance(max_utf8_bytes, int):
        raise TypeError("max_utf8_bytes must be an integer")
    if max_utf8_bytes < 0:
        raise ValueError("max_utf8_bytes cannot be negative")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer or None")
        if max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        if count_tokens is None:
            raise ValueError("count_tokens is required when max_tokens is supplied")
    if count_tokens is not None and not callable(count_tokens):
        raise TypeError("count_tokens must be callable or None")


def _count_tokens(count_tokens: TokenCounter | None, text: str) -> int | None:
    if count_tokens is None:
        return None
    token_count = count_tokens(text)
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("count_tokens must return an integer")
    if token_count < 0:
        raise ValueError("count_tokens cannot return a negative count")
    return token_count


def _budget_exhausted(
    utf8_bytes: int,
    max_utf8_bytes: int,
    token_count: int | None,
    max_tokens: int | None,
) -> BudgetExhausted | None:
    bytes_exceeded = utf8_bytes > max_utf8_bytes
    tokens_exceeded = (
        max_tokens is not None
        and token_count is not None
        and token_count > max_tokens
    )
    if bytes_exceeded and tokens_exceeded:
        return "bytes_and_tokens"
    if bytes_exceeded:
        return "bytes"
    if tokens_exceeded:
        return "tokens"
    return None
