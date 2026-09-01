"""Deterministic post-run provenance, citation, and refusal-signal checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

ProvenanceStatus = Literal[
    "pass",
    "hard_fail",
    "human_or_semantic_review",
    "not_applicable",
]
CitationCorrectness = Literal["correct", "incorrect", "not_applicable"]
CitationCompleteness = Literal["complete", "incomplete", "not_required", "not_applicable"]

_OOS_PROBE_KINDS = frozenset({"refusal", "live_source"})
_BRACKET_BLOCK = re.compile(r"\[([^\[\]]*)\]")
_CITATION_BODY = re.compile(r"^record\s*:\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)
_RECORD_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+$")
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+\s+")
_RETRIEVAL_CLAIM = re.compile(
    r"""
    \b(?:
        i\s+(?:have\s+)?retrieved
        | we\s+(?:have\s+)?retrieved
        | the\s+supplied\s+(?:document|record|context)
        | according\s+to\s+the\s+supplied
        | (?:from|in)\s+the\s+retrieved
        | i\s+looked\s+up
        | i\s+searched\s+(?:the\s+)?(?:document|record|knowledge|index|context)
        | i\s+accessed\s+the\s+(?:document|record|source|file|knowledge|context)
        | i\s+checked\s+the\s+(?:document|record|source|file|context|policy)
        | in\s+the\s+provided\s+document
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_REFUSAL = re.compile(
    r"""
    \b(?:
        i\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|answer|verify|confirm|determine)
        | refer(?:\s+the\s+user)?\s+to\s+the\s+(?:current\s+)?source
        | current\s+source\s+system
        | source\s+system
        | live(?:\s+or\s+current)?\s+information
        | no\s+(?:authoritative|supplied)\s+record
        | not\s+(?:supported|covered)\s+by\s+the\s+supplied
        | out\s+of\s+scope
        | i\s+cannot\s+(?:find|locate)
        | please\s+(?:consult|check\s+with|verify\s+with)
        | contact\s+(?:your\s+)?(?:hr|finance|legal|support|the\s+source)
        | source[- ]required
        | needs?\s+(?:a\s+|the\s+)?live\s+source
        | records\s+do\s+not\s+support
        | do\s+not\s+guess
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_HISTORICAL_MARKER = re.compile(
    r"\b(?:superseded|previous(?:ly)?|former(?:ly)?|outdated|replaced|no\s+longer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProvenanceGradeRequest:
    generated_output: str
    arm: str
    suite: str | None = None
    probe_kind: str | None = None
    supplied_record_ids: tuple[str, ...] = ()
    supplied_source_uris: tuple[str, ...] = ()
    gold_record_id: str | None = None
    out_of_scope: bool = False
    live_source: bool = False
    active_record_ids: tuple[str, ...] = ()
    superseded_record_ids: tuple[str, ...] = ()
    citation_required: bool = False
    allowed_parametric_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supplied_record_ids",
            _normalize_id_tuple(self.supplied_record_ids),
        )
        object.__setattr__(
            self,
            "supplied_source_uris",
            tuple(uri.strip() for uri in self.supplied_source_uris if uri.strip()),
        )
        object.__setattr__(
            self,
            "active_record_ids",
            _normalize_id_tuple(self.active_record_ids),
        )
        object.__setattr__(
            self,
            "superseded_record_ids",
            _normalize_id_tuple(self.superseded_record_ids),
        )
        object.__setattr__(
            self,
            "allowed_parametric_record_ids",
            _normalize_id_tuple(self.allowed_parametric_record_ids),
        )
        if self.gold_record_id is not None:
            gold = self.gold_record_id.strip()
            object.__setattr__(self, "gold_record_id", gold or None)


@dataclass(frozen=True)
class CitationSpan:
    raw: str
    record_id: str
    start: int
    end: int


@dataclass(frozen=True)
class CitationMembership:
    record_id: str
    supplied: bool
    gold: bool
    superseded: bool
    active: bool


@dataclass(frozen=True)
class ProvenanceGrade:
    citations: tuple[CitationSpan, ...]
    unique_cited_record_ids: tuple[str, ...]
    malformed_citation_fragments: tuple[str, ...]
    citation_membership: tuple[CitationMembership, ...]
    citation_correctness: CitationCorrectness
    citation_completeness: CitationCompleteness
    hallucinated_citation: bool
    superseded_citation: bool
    false_retrieval_claim: bool
    refusal_detected: bool
    unsupported_answer_signal: bool
    support_review_required: bool
    status: ProvenanceStatus
    reasons: tuple[str, ...]


def grade_provenance(request: ProvenanceGradeRequest) -> ProvenanceGrade:
    """Grade citation membership and refusal signals without judging semantics."""

    citations, malformed = extract_citations(request.generated_output)
    supplied_by_norm = _id_lookup(request.supplied_record_ids)
    parametric_by_norm = _id_lookup(request.allowed_parametric_record_ids)
    gold_norm = (
        _normalize_record_id(request.gold_record_id) if request.gold_record_id else None
    )
    superseded_norm = {
        _normalize_record_id(record_id) for record_id in request.superseded_record_ids
    }
    active_norm = {
        _normalize_record_id(record_id) for record_id in request.active_record_ids
    }

    unique_ids: list[str] = []
    seen_norms: set[str] = set()
    membership: list[CitationMembership] = []
    hallucinated_ids: list[str] = []
    superseded_as_current_ids: list[str] = []

    for citation in citations:
        normalized = _normalize_record_id(citation.record_id)
        canonical = (
            supplied_by_norm.get(normalized)
            or parametric_by_norm.get(normalized)
            or citation.record_id
        )
        authorized = normalized in supplied_by_norm or normalized in parametric_by_norm
        superseded = normalized in superseded_norm
        if normalized not in seen_norms:
            seen_norms.add(normalized)
            unique_ids.append(canonical)
            membership.append(
                CitationMembership(
                    record_id=canonical,
                    supplied=normalized in supplied_by_norm,
                    gold=bool(gold_norm and normalized == gold_norm),
                    superseded=superseded,
                    active=normalized in active_norm,
                )
            )
            if not authorized:
                hallucinated_ids.append(canonical)
        if (
            superseded
            and canonical not in superseded_as_current_ids
            and _cited_as_current_in_sentence(
                request.generated_output,
                citation.start,
                citation.end,
            )
        ):
            superseded_as_current_ids.append(canonical)

    false_retrieval_claim = _false_retrieval_claim(request)
    refusal_detected = _has_refusal(request.generated_output)
    oos = _is_oos(request)
    empty_output = not request.generated_output.strip()
    unsupported_answer_signal = oos and not refusal_detected and not empty_output
    gold_cited = any(item.gold for item in membership)
    citation_correctness = _citation_correctness(
        citations=citations,
        malformed=malformed,
        hallucinated=bool(hallucinated_ids),
        superseded_as_current=bool(superseded_as_current_ids),
    )
    citation_completeness = _citation_completeness(
        request,
        gold_cited=gold_cited,
        oos=oos,
    )
    reasons = _reasons(
        hallucinated_ids=hallucinated_ids,
        superseded_ids=superseded_as_current_ids,
        malformed=malformed,
        false_retrieval_claim=false_retrieval_claim,
        citation_completeness=citation_completeness,
        refusal_detected=refusal_detected,
        unsupported_answer_signal=unsupported_answer_signal,
        empty_output=empty_output,
    )
    hard_fail = bool(
        hallucinated_ids
        or superseded_as_current_ids
        or malformed
        or false_retrieval_claim
    )
    support_review_required = True
    status = _status(
        hard_fail=hard_fail,
        oos=oos,
        empty_output=empty_output,
        citation_completeness=citation_completeness,
    )
    return ProvenanceGrade(
        citations=citations,
        unique_cited_record_ids=tuple(unique_ids),
        malformed_citation_fragments=malformed,
        citation_membership=tuple(membership),
        citation_correctness=citation_correctness,
        citation_completeness=citation_completeness,
        hallucinated_citation=bool(hallucinated_ids),
        superseded_citation=bool(superseded_as_current_ids),
        false_retrieval_claim=false_retrieval_claim,
        refusal_detected=refusal_detected,
        unsupported_answer_signal=unsupported_answer_signal,
        support_review_required=support_review_required,
        status=status,
        reasons=reasons,
    )


def extract_citations(text: str) -> tuple[tuple[CitationSpan, ...], tuple[str, ...]]:
    """Return approved citations and rejected citation-shaped fragments."""

    citations: list[CitationSpan] = []
    malformed: list[str] = []
    for match in _BRACKET_BLOCK.finditer(text):
        body = match.group(1)
        citation_match = _CITATION_BODY.fullmatch(body.strip())
        if citation_match is None:
            continue
        record_id = unicodedata.normalize("NFKC", citation_match.group(1)).strip()
        if not record_id or not _RECORD_ID.fullmatch(record_id):
            malformed.append(match.group(0))
            continue
        citations.append(
            CitationSpan(
                raw=match.group(0),
                record_id=record_id,
                start=match.start(),
                end=match.end(),
            )
        )
    return tuple(citations), tuple(malformed)


def _is_oos(request: ProvenanceGradeRequest) -> bool:
    return (
        request.out_of_scope
        or request.live_source
        or request.suite == "unknown_oos"
        or (request.probe_kind or "") in _OOS_PROBE_KINDS
    )


def _false_retrieval_claim(request: ProvenanceGradeRequest) -> bool:
    if request.supplied_record_ids or request.supplied_source_uris:
        return False
    return _RETRIEVAL_CLAIM.search(request.generated_output) is not None


def _has_refusal(text: str) -> bool:
    return _REFUSAL.search(text) is not None


def _cited_as_current_in_sentence(text: str, start: int, end: int) -> bool:
    return _HISTORICAL_MARKER.search(_sentence_containing(text, start, end)) is None


def _sentence_containing(text: str, start: int, end: int) -> str:
    sentence_start = 0
    sentence_end = len(text)
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if match.end() <= start:
            sentence_start = match.end()
        elif match.start() >= end:
            sentence_end = match.end()
            break
    return text[sentence_start:sentence_end]


def _citation_correctness(
    *,
    citations: tuple[CitationSpan, ...],
    malformed: tuple[str, ...],
    hallucinated: bool,
    superseded_as_current: bool,
) -> CitationCorrectness:
    if not citations and not malformed:
        return "not_applicable"
    if malformed or hallucinated or superseded_as_current:
        return "incorrect"
    return "correct"


def _citation_completeness(
    request: ProvenanceGradeRequest,
    *,
    gold_cited: bool,
    oos: bool,
) -> CitationCompleteness:
    if oos or request.gold_record_id is None:
        if request.citation_required and request.gold_record_id is not None:
            return "complete" if gold_cited else "incomplete"
        if request.citation_required:
            return "not_applicable"
        return "not_required"
    if gold_cited:
        return "complete"
    if request.citation_required:
        return "incomplete"
    return "not_required"


def _status(
    *,
    hard_fail: bool,
    oos: bool,
    empty_output: bool,
    citation_completeness: CitationCompleteness,
) -> ProvenanceStatus:
    if hard_fail:
        return "hard_fail"
    if oos or empty_output or citation_completeness == "incomplete":
        return "human_or_semantic_review"
    return "pass"


def _reasons(
    *,
    hallucinated_ids: list[str],
    superseded_ids: list[str],
    malformed: tuple[str, ...],
    false_retrieval_claim: bool,
    citation_completeness: CitationCompleteness,
    refusal_detected: bool,
    unsupported_answer_signal: bool,
    empty_output: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    reasons.extend(f"hallucinated_citation:{record_id}" for record_id in hallucinated_ids)
    reasons.extend(f"superseded_citation:{record_id}" for record_id in superseded_ids)
    if malformed:
        reasons.append("malformed_citation")
    if false_retrieval_claim:
        reasons.append("false_retrieval_claim")
    if citation_completeness == "incomplete":
        reasons.append("citation_incomplete")
    if refusal_detected:
        reasons.append("refusal_detected")
    if unsupported_answer_signal:
        reasons.append("unsupported_answer_signal")
    if empty_output:
        reasons.append("empty_output")
    reasons.append("support_review_required")
    return tuple(sorted(set(reasons)))


def _normalize_id_tuple(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        record_id = value.strip()
        if not record_id:
            continue
        key = _normalize_record_id(record_id)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(record_id)
    return tuple(normalized)


def _id_lookup(record_ids: tuple[str, ...]) -> dict[str, str]:
    return {_normalize_record_id(record_id): record_id for record_id in record_ids}


def _normalize_record_id(record_id: str) -> str:
    return unicodedata.normalize("NFKC", record_id).casefold().strip()
