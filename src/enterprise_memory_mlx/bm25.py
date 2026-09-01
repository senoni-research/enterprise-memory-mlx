from __future__ import annotations

import math
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .schemas import KnowledgeRecord

RetrievalAction = Literal["use_context", "source_required"]
ValidationLabelKind = Literal["answerable", "oos"]
OperatingPointStatus = Literal["selected", "no_feasible_operating_point"]
SourceRequiredReason = Literal[
    "no_eligible_records",
    "query_has_no_index_tokens",
    "no_match_above_threshold",
]
ConstraintFailure = Literal[
    "correct_record_rate_below_minimum",
    "wrong_record_rate_above_maximum",
    "answerable_empty_retrieval_rate_above_maximum",
    "oos_false_load_rate_above_maximum",
]

_FIELD_WEIGHTS = (
    ("title", 3.0),
    ("summary", 2.0),
    ("aliases", 2.0),
    ("statement", 1.0),
    ("id", 0.25),
    ("domain", 0.25),
)
_NUMERIC_SEPARATORS = frozenset({",", "\u066b", "\u066c"})
_WORD_APOSTROPHES = frozenset({"'", "\u2019"})
_PERCENT_SUFFIXES = frozenset({"%", "\u2030", "\u2031"})
_OPERATING_POINT_K1 = 1.2
_OPERATING_POINT_B = 0.75
_OPERATING_POINT_TOP_K_VALUES = (1, 2, 3, 5)
_WILSON_95_Z = 1.959963984540054
_ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "about",
        "am",
        "an",
        "and",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "been",
        "being",
        "between",
        "but",
        "by",
        "can",
        "can't",
        "could",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "me",
        "mine",
        "my",
        "myself",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "ourselves",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "us",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


@dataclass(frozen=True)
class BM25Config:
    """BM25 parameters.

    ``score_threshold`` has no safe default: 0.0 accepts every scored hit and
    was rejected by the operating-point selection. It must be chosen
    explicitly (a calibrated value, or 0.0 stated deliberately for
    diagnostics); leaving it ``None`` makes ``search()`` fail closed.
    """

    k1: float = 1.2
    b: float = 0.75
    top_k: int = 5
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and between zero and one")
        _validate_top_k(self.top_k)
        if self.score_threshold is not None:
            _validate_score_threshold(self.score_threshold)


@dataclass(frozen=True)
class BM25Hit:
    record_id: str
    source_uri: str
    score: float
    rank: int


@dataclass(frozen=True)
class BM25Result:
    action: RetrievalAction
    hits: tuple[BM25Hit, ...]
    selected_records: tuple[KnowledgeRecord, ...]
    reason: SourceRequiredReason | None = None

    @property
    def selected_record_ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.selected_records)

    @property
    def source_uris(self) -> tuple[str, ...]:
        return tuple(record.source_uri for record in self.selected_records)


@dataclass(frozen=True)
class BM25IndexStats:
    eligible_document_count: int
    unique_term_count: int
    total_weighted_length: float
    average_document_length: float
    build_elapsed_seconds: float


@dataclass(frozen=True)
class BM25ValidationLabel:
    kind: ValidationLabelKind
    relevant_record_ids: tuple[str, ...] = ()
    oos_type: str | None = None

    def __post_init__(self) -> None:
        record_ids = tuple(self.relevant_record_ids)
        object.__setattr__(self, "relevant_record_ids", record_ids)
        if self.kind not in {"answerable", "oos"}:
            raise ValueError("validation label kind must be 'answerable' or 'oos'")
        if any(not record_id.strip() for record_id in record_ids):
            raise ValueError("relevant_record_ids must not contain blank IDs")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("relevant_record_ids must be unique")
        if self.kind == "answerable":
            if len(record_ids) != 1:
                raise ValueError("answerable validation labels require exactly one record ID")
            if self.oos_type is not None:
                raise ValueError("answerable validation labels cannot have an oos_type")
        else:
            if record_ids:
                raise ValueError("OOS validation labels cannot have relevant record IDs")
            if self.oos_type is None or not self.oos_type.strip():
                raise ValueError("OOS validation labels require a non-blank oos_type")


@dataclass(frozen=True)
class BM25ValidationQuery:
    query_id: str
    query: str
    label: BM25ValidationLabel

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("validation query_id must not be blank")
        if not self.query.strip():
            raise ValueError("validation query text must not be blank")


@dataclass(frozen=True)
class BM25RateMetric:
    count: int
    total: int
    rate: float
    confidence_low_95: float
    confidence_high_95: float


@dataclass(frozen=True)
class BM25ValidationMetrics:
    validation_query_count: int
    answerable_query_count: int
    oos_query_count: int
    correct_record: BM25RateMetric
    wrong_record: BM25RateMetric
    answerable_empty_retrieval: BM25RateMetric
    oos_false_load: BM25RateMetric
    correct_oos_rejection: BM25RateMetric
    retrieved_record_count: int
    mean_retrieved_record_count: float
    distractor_record_count: int


@dataclass(frozen=True)
class BM25FeasibilityConstraints:
    minimum_correct_record_rate: float = 0.95
    maximum_wrong_record_rate: float = 0.01
    maximum_answerable_empty_retrieval_rate: float = 0.05
    maximum_oos_false_load_rate: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_correct_record_rate", self.minimum_correct_record_rate),
            ("maximum_wrong_record_rate", self.maximum_wrong_record_rate),
            (
                "maximum_answerable_empty_retrieval_rate",
                self.maximum_answerable_empty_retrieval_rate,
            ),
            ("maximum_oos_false_load_rate", self.maximum_oos_false_load_rate),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")


@dataclass(frozen=True)
class BM25CandidateEvaluation:
    ordinal: int
    config: BM25Config
    metrics: BM25ValidationMetrics
    feasible: bool
    constraint_failures: tuple[ConstraintFailure, ...]


@dataclass(frozen=True)
class BM25OperatingPointSelection:
    status: OperatingPointStatus
    selected_config: BM25Config | None
    validation_dataset_hash: str
    validation_size: int
    answerable_size: int
    oos_size: int
    top_k_candidates: tuple[int, ...]
    score_threshold_candidates: tuple[float, ...]
    constraints: BM25FeasibilityConstraints
    candidates: tuple[BM25CandidateEvaluation, ...]
    pareto_frontier: tuple[BM25CandidateEvaluation, ...]
    exploratory: bool = True


@dataclass(frozen=True)
class _Document:
    record: KnowledgeRecord
    term_frequencies: dict[str, float]
    length: float


class BM25Index:
    """Deterministic BM25 retrieval over eligible governed knowledge records."""

    def __init__(
        self,
        records: Iterable[KnowledgeRecord],
        *,
        config: BM25Config | None = None,
    ) -> None:
        build_started = time.perf_counter()
        self.config = config or BM25Config()
        eligible = sorted(
            (record for record in records if record.is_trainable()),
            key=lambda record: record.id,
        )
        _ensure_unique_record_ids(eligible)

        self.records = tuple(eligible)
        self._documents = tuple(_build_document(record) for record in eligible)
        self._documents_by_id = {document.record.id: document for document in self._documents}
        total_weighted_length = sum(document.length for document in self._documents)
        self._average_length = (
            total_weighted_length / len(self._documents)
            if self._documents
            else 0.0
        )

        document_frequencies: Counter[str] = Counter()
        for document in self._documents:
            document_frequencies.update(document.term_frequencies.keys())
        document_count = len(self._documents)
        self._inverse_document_frequencies = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }
        self.stats = BM25IndexStats(
            eligible_document_count=document_count,
            unique_term_count=len(self._inverse_document_frequencies),
            total_weighted_length=total_weighted_length,
            average_document_length=self._average_length,
            build_elapsed_seconds=time.perf_counter() - build_started,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> BM25Result:
        result_top_k = self.config.top_k if top_k is None else top_k
        result_threshold = (
            self.config.score_threshold if score_threshold is None else score_threshold
        )
        if result_threshold is None:
            raise ValueError(
                "score_threshold must be chosen explicitly (calibrated, or 0.0 "
                "stated deliberately for diagnostics); there is no safe default"
            )
        _validate_top_k(result_top_k)
        _validate_score_threshold(result_threshold)

        if not self._documents:
            return _source_required("no_eligible_records")

        query_frequencies = Counter(_retrieval_tokens(query))
        indexed_query_terms = {
            term: frequency
            for term, frequency in query_frequencies.items()
            if term in self._inverse_document_frequencies
        }
        if not indexed_query_terms:
            return _source_required("query_has_no_index_tokens")

        scored = [
            (self._score(document, indexed_query_terms), document.record)
            for document in self._documents
        ]
        ranked = sorted(
            (
                (score, record)
                for score, record in scored
                if score > 0.0 and score >= result_threshold
            ),
            key=lambda item: (-item[0], item[1].id),
        )[:result_top_k]
        if not ranked:
            return _source_required("no_match_above_threshold")

        hits = tuple(
            BM25Hit(
                record_id=record.id,
                source_uri=record.source_uri,
                score=score,
                rank=rank,
            )
            for rank, (score, record) in enumerate(ranked, start=1)
        )
        selected_records = tuple(
            self._documents_by_id[hit.record_id].record
            for hit in hits
        )
        return BM25Result(
            action="use_context",
            hits=hits,
            selected_records=selected_records,
        )

    def _score(self, document: _Document, query_frequencies: dict[str, int]) -> float:
        score = 0.0
        length_normalization = self.config.k1 * (
            1.0 - self.config.b
            + self.config.b * document.length / self._average_length
        )
        for term, query_frequency in sorted(query_frequencies.items()):
            term_frequency = document.term_frequencies.get(term, 0.0)
            if term_frequency == 0.0:
                continue
            saturation = (
                term_frequency * (self.config.k1 + 1.0)
                / (term_frequency + length_normalization)
            )
            score += query_frequency * self._inverse_document_frequencies[term] * saturation
        return score


def build_bm25_index(
    records: Iterable[KnowledgeRecord],
    *,
    config: BM25Config | None = None,
) -> BM25Index:
    return BM25Index(records, config=config)


def select_retrieval_operating_point(
    index: BM25Index,
    validation_queries: Sequence[BM25ValidationQuery],
    *,
    validation_dataset_hash: str,
    frozen_question_texts: Iterable[str],
    constraints: BM25FeasibilityConstraints | None = None,
) -> BM25OperatingPointSelection:
    """Evaluate the approved deterministic grid without consulting answer content."""

    validated_hash = _validate_dataset_hash(validation_dataset_hash)
    if (index.config.k1, index.config.b) != (_OPERATING_POINT_K1, _OPERATING_POINT_B):
        raise ValueError("operating-point evaluation requires fixed k1=1.2 and b=0.75")
    queries = tuple(validation_queries)
    _validate_validation_queries(index, queries, frozen_question_texts)
    selection_constraints = constraints or BM25FeasibilityConstraints()
    threshold_candidates = _derive_score_threshold_candidates(index, queries)

    candidates: list[BM25CandidateEvaluation] = []
    for top_k in _OPERATING_POINT_TOP_K_VALUES:
        for score_threshold in threshold_candidates:
            metrics = _evaluate_candidate(
                index,
                queries,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            failures = _constraint_failures(metrics, selection_constraints)
            candidates.append(
                BM25CandidateEvaluation(
                    ordinal=len(candidates),
                    config=BM25Config(
                        k1=index.config.k1,
                        b=index.config.b,
                        top_k=top_k,
                        score_threshold=score_threshold,
                    ),
                    metrics=metrics,
                    feasible=not failures,
                    constraint_failures=failures,
                )
            )

    candidate_results = tuple(candidates)
    feasible_candidates = tuple(candidate for candidate in candidate_results if candidate.feasible)
    selected = (
        min(feasible_candidates, key=_selection_key)
        if feasible_candidates
        else None
    )
    answerable_size = sum(query.label.kind == "answerable" for query in queries)
    oos_size = len(queries) - answerable_size
    return BM25OperatingPointSelection(
        status="selected" if selected is not None else "no_feasible_operating_point",
        selected_config=selected.config if selected is not None else None,
        validation_dataset_hash=validated_hash,
        validation_size=len(queries),
        answerable_size=answerable_size,
        oos_size=oos_size,
        top_k_candidates=_OPERATING_POINT_TOP_K_VALUES,
        score_threshold_candidates=threshold_candidates,
        constraints=selection_constraints,
        candidates=candidate_results,
        pareto_frontier=_pareto_frontier(candidate_results),
    )


def tokenize(text: str) -> tuple[str, ...]:
    """Return NFKC/case-folded tokens while retaining useful structured values."""

    normalized = unicodedata.normalize(
        "NFKC",
        unicodedata.normalize("NFKC", text).casefold(),
    )
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if _is_token_base(character):
            start = index
            index += 1
        elif (
            (_is_currency(character) or character == "#")
            and index + 1 < len(normalized)
            and _is_token_base(normalized[index + 1])
        ):
            start = index
            index += 2
        else:
            index += 1
            continue

        while index < len(normalized):
            character = normalized[index]
            previous = normalized[index - 1]
            following = normalized[index + 1] if index + 1 < len(normalized) else ""
            if _is_internal_token_character(character, previous, following):
                index += 1
            elif (
                (_is_currency(character) or character in _PERCENT_SUFFIXES)
                and _is_number(previous)
            ):
                index += 1
                break
            else:
                break
        tokens.append(normalized[start:index])
    return tuple(tokens)


def _build_document(record: KnowledgeRecord) -> _Document:
    term_frequencies: dict[str, float] = {}
    length = 0.0
    values: dict[str, tuple[str, ...]] = {
        "title": (record.title,),
        "summary": (record.summary,),
        "aliases": record.aliases,
        "statement": (record.statement,),
        "id": (record.id,),
        "domain": (record.domain,),
    }
    for field, weight in _FIELD_WEIGHTS:
        for value in values[field]:
            field_tokens = _retrieval_tokens(value)
            length += weight * len(field_tokens)
            for token in field_tokens:
                term_frequencies[token] = term_frequencies.get(token, 0.0) + weight
    return _Document(record=record, term_frequencies=term_frequencies, length=length)


def _retrieval_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in tokenize(text) if token not in _ENGLISH_FUNCTION_WORDS)


def _validate_dataset_hash(value: str) -> str:
    normalized = value.casefold()
    is_hex_digest = len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )
    if not is_hex_digest:
        raise ValueError("validation_dataset_hash must be a 64-character SHA-256 hex digest")
    return normalized


def _validate_validation_queries(
    index: BM25Index,
    queries: tuple[BM25ValidationQuery, ...],
    frozen_question_texts: Iterable[str],
) -> None:
    if not queries:
        raise ValueError("validation_queries must not be empty")

    query_ids = [query.query_id for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("validation query IDs must be unique")

    query_identities = [_question_identity(query.query) for query in queries]
    if len(set(query_identities)) != len(query_identities):
        raise ValueError("validation query texts must be unique after normalization")

    answerable_count = sum(query.label.kind == "answerable" for query in queries)
    if answerable_count == 0 or answerable_count == len(queries):
        raise ValueError("validation_queries must include answerable and OOS labels")

    indexed_record_ids = {record.id for record in index.records}
    unknown_record_ids = sorted(
        {
            record_id
            for query in queries
            for record_id in query.label.relevant_record_ids
            if record_id not in indexed_record_ids
        }
    )
    if unknown_record_ids:
        raise ValueError(
            "validation labels reference records absent from the index: "
            + ", ".join(unknown_record_ids)
        )

    frozen_identities = {
        identity
        for question in frozen_question_texts
        if (identity := _question_identity(question))
    }
    overlapping_query_ids = sorted(
        query.query_id
        for query, identity in zip(queries, query_identities, strict=True)
        if identity in frozen_identities
    )
    if overlapping_query_ids:
        raise ValueError(
            "validation queries overlap frozen evaluation questions: "
            + ", ".join(overlapping_query_ids)
        )


def _derive_score_threshold_candidates(
    index: BM25Index,
    queries: tuple[BM25ValidationQuery, ...],
) -> tuple[float, ...]:
    all_hits_top_k = max(1, len(index.records))
    observed_scores = {
        hit.score
        for query in queries
        for hit in index.search(
            query.query,
            top_k=all_hits_top_k,
            score_threshold=0.0,
        ).hits
    }
    maximum_score = max(observed_scores, default=0.0)
    all_reject_threshold = math.nextafter(maximum_score, math.inf)
    if not math.isfinite(all_reject_threshold):
        raise ValueError("unable to derive a finite all-reject score threshold")
    return tuple(sorted({0.0, *observed_scores, all_reject_threshold}))


def _evaluate_candidate(
    index: BM25Index,
    queries: tuple[BM25ValidationQuery, ...],
    *,
    top_k: int,
    score_threshold: float,
) -> BM25ValidationMetrics:
    answerable_count = 0
    oos_count = 0
    correct_count = 0
    wrong_count = 0
    empty_count = 0
    false_load_count = 0
    retrieved_record_count = 0
    distractor_record_count = 0

    for query in queries:
        result = index.search(
            query.query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        retrieved_record_count += len(result.hits)
        if query.label.kind == "oos":
            oos_count += 1
            false_load_count += bool(result.hits)
            continue

        answerable_count += 1
        if not result.hits:
            empty_count += 1
            continue
        relevant_record_id = query.label.relevant_record_ids[0]
        returned_ids = {hit.record_id for hit in result.hits}
        if relevant_record_id in returned_ids:
            correct_count += 1
            distractor_record_count += len(result.hits) - 1
        else:
            wrong_count += 1

    return BM25ValidationMetrics(
        validation_query_count=len(queries),
        answerable_query_count=answerable_count,
        oos_query_count=oos_count,
        correct_record=_rate_metric(correct_count, answerable_count),
        wrong_record=_rate_metric(wrong_count, answerable_count),
        answerable_empty_retrieval=_rate_metric(empty_count, answerable_count),
        oos_false_load=_rate_metric(false_load_count, oos_count),
        correct_oos_rejection=_rate_metric(oos_count - false_load_count, oos_count),
        retrieved_record_count=retrieved_record_count,
        mean_retrieved_record_count=retrieved_record_count / len(queries),
        distractor_record_count=distractor_record_count,
    )


def _rate_metric(count: int, total: int) -> BM25RateMetric:
    rate = count / total
    denominator = 1.0 + (_WILSON_95_Z**2 / total)
    centre = rate + (_WILSON_95_Z**2 / (2.0 * total))
    margin = _WILSON_95_Z * math.sqrt(
        rate * (1.0 - rate) / total + _WILSON_95_Z**2 / (4.0 * total**2)
    )
    return BM25RateMetric(
        count=count,
        total=total,
        rate=rate,
        confidence_low_95=max(0.0, (centre - margin) / denominator),
        confidence_high_95=min(1.0, (centre + margin) / denominator),
    )


def _constraint_failures(
    metrics: BM25ValidationMetrics,
    constraints: BM25FeasibilityConstraints,
) -> tuple[ConstraintFailure, ...]:
    failures: list[ConstraintFailure] = []
    if metrics.correct_record.rate < constraints.minimum_correct_record_rate:
        failures.append("correct_record_rate_below_minimum")
    if metrics.wrong_record.rate > constraints.maximum_wrong_record_rate:
        failures.append("wrong_record_rate_above_maximum")
    if (
        metrics.answerable_empty_retrieval.rate
        > constraints.maximum_answerable_empty_retrieval_rate
    ):
        failures.append("answerable_empty_retrieval_rate_above_maximum")
    if metrics.oos_false_load.rate > constraints.maximum_oos_false_load_rate:
        failures.append("oos_false_load_rate_above_maximum")
    return tuple(failures)


def _selection_key(candidate: BM25CandidateEvaluation) -> tuple[float, ...]:
    metrics = candidate.metrics
    return (
        metrics.oos_false_load.rate,
        metrics.wrong_record.rate,
        -metrics.correct_record.rate,
        metrics.mean_retrieved_record_count,
        candidate.config.top_k,
        -candidate.config.score_threshold,
        candidate.ordinal,
    )


def _pareto_frontier(
    candidates: tuple[BM25CandidateEvaluation, ...],
) -> tuple[BM25CandidateEvaluation, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            _dominates(other, candidate)
            for other in candidates
            if other.ordinal != candidate.ordinal
        )
    )


def _dominates(
    candidate: BM25CandidateEvaluation,
    other: BM25CandidateEvaluation,
) -> bool:
    candidate_metrics = candidate.metrics
    other_metrics = other.metrics
    no_worse = (
        candidate_metrics.oos_false_load.rate <= other_metrics.oos_false_load.rate
        and candidate_metrics.wrong_record.rate <= other_metrics.wrong_record.rate
        and candidate_metrics.correct_record.rate >= other_metrics.correct_record.rate
        and candidate_metrics.answerable_empty_retrieval.rate
        <= other_metrics.answerable_empty_retrieval.rate
        and candidate_metrics.mean_retrieved_record_count
        <= other_metrics.mean_retrieved_record_count
        and candidate.config.top_k <= other.config.top_k
        and candidate.config.score_threshold >= other.config.score_threshold
    )
    strictly_better = (
        candidate_metrics.oos_false_load.rate < other_metrics.oos_false_load.rate
        or candidate_metrics.wrong_record.rate < other_metrics.wrong_record.rate
        or candidate_metrics.correct_record.rate > other_metrics.correct_record.rate
        or candidate_metrics.answerable_empty_retrieval.rate
        < other_metrics.answerable_empty_retrieval.rate
        or candidate_metrics.mean_retrieved_record_count
        < other_metrics.mean_retrieved_record_count
        or candidate.config.top_k < other.config.top_k
        or candidate.config.score_threshold > other.config.score_threshold
    )
    return no_worse and strictly_better


def _question_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _source_required(reason: SourceRequiredReason) -> BM25Result:
    return BM25Result(
        action="source_required",
        hits=(),
        selected_records=(),
        reason=reason,
    )


def _ensure_unique_record_ids(records: list[KnowledgeRecord]) -> None:
    for previous, current in zip(records, records[1:], strict=False):
        if previous.id == current.id:
            raise ValueError(f"Duplicate eligible knowledge record ID: {current.id}")


def _validate_top_k(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("top_k must be a positive integer")


def _validate_score_threshold(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("score_threshold must be finite and non-negative")


def _is_token_base(character: str) -> bool:
    return bool(character) and unicodedata.category(character)[0] in {"L", "N"}


def _is_token_continuation(character: str) -> bool:
    return _is_token_base(character) or (
        bool(character) and unicodedata.category(character)[0] == "M"
    )


def _is_identifier_joiner(character: str) -> bool:
    if not character:
        return False
    return unicodedata.category(character) in {"Pc", "Pd"} or character in "./:@+"


def _can_join(previous: str, following: str) -> bool:
    return _is_token_continuation(previous) and _is_token_base(following)


def _is_internal_token_character(
    character: str,
    previous: str,
    following: str,
) -> bool:
    return (
        _is_token_continuation(character)
        or (_is_identifier_joiner(character) and _can_join(previous, following))
        or (
            character in _NUMERIC_SEPARATORS
            and previous.isdecimal()
            and following.isdecimal()
        )
        or (
            character in _WORD_APOSTROPHES
            and _is_letter(previous)
            and _is_letter(following)
        )
    )


def _is_letter(character: str) -> bool:
    return bool(character) and unicodedata.category(character)[0] == "L"


def _is_number(character: str) -> bool:
    return bool(character) and unicodedata.category(character)[0] == "N"


def _is_currency(character: str) -> bool:
    return bool(character) and unicodedata.category(character) == "Sc"
