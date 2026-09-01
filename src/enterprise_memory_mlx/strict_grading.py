"""Deterministic typed critical-slot grading.

This module detects factual contradictions and omissions in generated answers.
It does not decide semantic completeness and never uses lexical F1 as a pass
rule. A later judge cannot override a hard failure.

Hard failures are reserved for certain outcomes: an expected value is entirely
absent, an explicitly forbidden value is present, or the slot declaration is
unsupported or malformed. When the current ``CriticalSlot`` schema cannot bind a
value to an obligation, action, or date/time precision, the result is
``review_required`` rather than pass or hard-fail.

The shipped slot schema has no optional ``anchor`` or ``predicate``. Those
fields are required before unscoped extras can be decided automatically:

- ``anchor``: the obligation or noun phrase a value attaches to
  (for example ``annual leave`` versus ``receipt deadline``);
- ``predicate``: the action whose polarity is constrained
  (for example ``split`` versus ``handle confidential information``);
- ``role``: why a repeated type appears
  (for example ``threshold`` versus ``comparison amount``).

Do not invent those fields here. Frozen fixtures stay unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .split_contract import SLOT_TYPES, CriticalSlot

GradeStatus = Literal["pass", "hard_fail", "review_required", "no_slots"]
SlotStatus = Literal["pass", "hard_fail", "review_required"]

_ONES: dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
_TEENS: dict[int, str] = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS: dict[int, str] = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _number_word_table() -> dict[str, int]:
    words = {"zero": 0}
    words.update({name: value for value, name in _ONES.items()})
    words.update({name: value for value, name in _TEENS.items()})
    for tens_value, tens_name in _TENS.items():
        words[tens_name] = tens_value
        for ones_value, ones_name in _ONES.items():
            words[f"{tens_name}-{ones_name}"] = tens_value + ones_value
            words[f"{tens_name} {ones_name}"] = tens_value + ones_value
    return words


_NUMBER_WORDS = _number_word_table()
_NUMBER_WORD_PATTERN = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]"
    r"(?:one|two|three|four|five|six|seven|eight|nine)|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
)
_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_CURRENCY_CODES = {"£": "GBP", "$": "USD", "€": "EUR"}
_MAGNITUDE_COMPARATORS = {
    "above": "above",
    "over": "above",
    "greater than": "above",
    ">": "above",
    "at least": "at_least",
    "no less than": "at_least",
    ">=": "at_least",
    "below": "below",
    "under": "below",
    "less than": "below",
    "<": "below",
    "at most": "at_most",
    "no more than": "at_most",
    "<=": "at_most",
}
_TEMPORAL_COMPARATORS = {"before": "before", "after": "after"}
_COMPARATOR_ALIASES = {**_MAGNITUDE_COMPARATORS, **_TEMPORAL_COMPARATORS}
_NEGATION_FORMS: tuple[tuple[str, str], ...] = (
    ("did not author", "did not author"),
    ("didn't author", "did not author"),
    ("must not", "must not"),
    ("mustn't", "must not"),
    ("may not", "may not"),
    ("cannot", "cannot"),
    ("can not", "cannot"),
    ("can't", "cannot"),
    ("prohibited", "prohibited"),
    ("not", "not"),
    ("no", "no"),
)
_STATUS_TERMS: dict[str, str] = {
    "active": "active",
    "draft": "draft",
    "retired": "retired",
    "public": "public",
    "internal shared": "internal_shared",
    "internal-shared": "internal_shared",
    "internal_shared": "internal_shared",
    "restricted": "restricted",
    "secret": "secret",
}
_UNIT_TERMS: dict[str, str] = {
    "calendar days": "calendar_day",
    "calendar day": "calendar_day",
    "business hours": "business_hour",
    "business hour": "business_hour",
    "working days": "working_day",
    "working day": "working_day",
    "days": "day",
    "day": "day",
    "hours": "hour",
    "hour": "hour",
    "minutes": "minute",
    "minute": "minute",
    "weeks": "week",
    "week": "week",
    "months": "month",
    "month": "month",
    "years": "year",
    "year": "year",
    "percent": "percent",
    "%": "percent",
}
_ZONE_ALIASES = {
    "uk time": "UK",
    "uk": "UK",
    "utc": "UTC",
    "gmt": "GMT",
    "bst": "BST",
    "est": "EST",
    "et": "ET",
    "pt": "PT",
    "pst": "PST",
    "pdt": "PDT",
}


@dataclass(frozen=True)
class SlotResult:
    """Per-slot evidence and the fail-closed decision for that slot."""

    index: int
    slot: str
    expected: str | None
    forbidden: tuple[str, ...]
    extracted: tuple[str, ...]
    present: bool
    forbidden_present: bool
    contradiction: bool
    status: SlotStatus
    reason: str


@dataclass(frozen=True)
class ExtractedEvidence:
    """Typed, normalized values taken from the generated answer."""

    number: tuple[str, ...] = ()
    currency_amount: tuple[str, ...] = ()
    unit: tuple[str, ...] = ()
    comparator: tuple[str, ...] = ()
    date: tuple[str, ...] = ()
    time: tuple[str, ...] = ()
    entity: tuple[str, ...] = ()
    negation: tuple[str, ...] = ()
    record_status: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrictGrade:
    """Immutable critical-slot grade. A later judge cannot override a hard fail."""

    status: GradeStatus
    slot_results: tuple[SlotResult, ...]
    failed_slot_indexes: tuple[int, ...]
    extracted_values: ExtractedEvidence
    reasons: tuple[str, ...]


def grade_critical_slots(answer: str, slots: Iterable[CriticalSlot]) -> StrictGrade:
    """Grade every declared critical slot against the generated answer."""
    if isinstance(answer, bool) or not isinstance(answer, str):
        raise TypeError("answer must be a string")

    materialized = tuple(slots)
    prepared = _prepare_text(answer)
    inventory = _AnswerInventory(prepared)

    if not materialized:
        return StrictGrade(
            status="no_slots",
            slot_results=(),
            failed_slot_indexes=(),
            extracted_values=inventory.evidence(()),
            reasons=("no critical slots declared",),
        )

    results = tuple(
        _grade_slot(index, slot, inventory, materialized) for index, slot in enumerate(materialized)
    )
    failed = tuple(result.index for result in results if result.status == "hard_fail")
    unresolved = tuple(result.index for result in results if result.status == "review_required")
    reasons = tuple(result.reason for result in results if result.status != "pass")
    if failed:
        status: GradeStatus = "hard_fail"
    elif unresolved:
        status = "review_required"
    else:
        status = "pass"
    return StrictGrade(
        status=status,
        slot_results=results,
        failed_slot_indexes=failed,
        extracted_values=inventory.evidence(materialized),
        reasons=reasons,
    )


def _grade_slot(
    index: int,
    slot: CriticalSlot,
    inventory: _AnswerInventory,
    all_slots: tuple[CriticalSlot, ...],
) -> SlotResult:
    if slot.slot not in SLOT_TYPES:
        return SlotResult(
            index=index,
            slot=slot.slot,
            expected=slot.expected,
            forbidden=slot.forbidden,
            extracted=(),
            present=False,
            forbidden_present=False,
            contradiction=False,
            status="hard_fail",
            reason=f"slot {index} {slot.slot}: unsupported slot type",
        )

    try:
        expected = _normalize_declared(slot.slot, slot.expected)
        forbidden = tuple(_normalize_declared(slot.slot, item) for item in slot.forbidden)
    except ValueError as exc:
        return SlotResult(
            index=index,
            slot=slot.slot,
            expected=slot.expected,
            forbidden=slot.forbidden,
            extracted=(),
            present=False,
            forbidden_present=False,
            contradiction=False,
            status="hard_fail",
            reason=f"slot {index} {slot.slot}: malformed expectation ({exc})",
        )

    if expected is None and not forbidden:
        return SlotResult(
            index=index,
            slot=slot.slot,
            expected=None,
            forbidden=(),
            extracted=(),
            present=False,
            forbidden_present=False,
            contradiction=False,
            status="hard_fail",
            reason=f"slot {index} {slot.slot}: malformed expectation (empty)",
        )

    extracted = inventory.values_for(slot.slot, all_slots)
    presence = _presence(slot.slot, expected, extracted, inventory)
    present = expected is not None and presence == "match"
    forbidden_hits = tuple(item for item in forbidden if item in extracted)
    if slot.slot == "provenance":
        forbidden_hits = tuple(item for item in forbidden if inventory.provenance_forbidden(item))
    forbidden_present = bool(forbidden_hits)
    contradiction = forbidden_present
    unresolved = inventory.scope_unresolved(slot.slot, expected, extracted, all_slots)

    if forbidden_present:
        reason = f"slot {index} {slot.slot}: forbidden value present"
        status: SlotStatus = "hard_fail"
    elif expected is not None and presence == "conflict":
        reason = f"slot {index} {slot.slot}: contradictory value present"
        status = "hard_fail"
        contradiction = True
    elif expected is not None and presence == "partial":
        reason = f"slot {index} {slot.slot}: omitted precision for {expected}"
        status = "review_required"
    elif expected is not None and presence == "absent":
        reason = f"slot {index} {slot.slot}: missing expected {expected}"
        status = "hard_fail"
    elif unresolved:
        reason = f"slot {index} {slot.slot}: scope cannot be determined"
        status = "review_required"
    else:
        reason = f"slot {index} {slot.slot}: expected value present"
        status = "pass"

    return SlotResult(
        index=index,
        slot=slot.slot,
        expected=expected,
        forbidden=forbidden,
        extracted=extracted,
        present=present,
        forbidden_present=forbidden_present,
        contradiction=contradiction,
        status=status,
        reason=reason,
    )


class _AnswerInventory:
    def __init__(self, text: str) -> None:
        self.text = text
        self.folded = text.casefold()
        occupied: list[tuple[int, int]] = []
        self.currencies = _extract_currencies(text, occupied)
        self.times = _extract_times(text, occupied)
        self.dates = _extract_dates(text, occupied)
        self.percentages = _extract_percentages(text, occupied)
        self.numbers = _extract_numbers(text, occupied)
        self.comparators = _extract_comparators(text)
        self.negations = _extract_negations(text)
        self.units = _extract_units(text)
        self.statuses = _extract_statuses(text)
        self.provenances = _extract_provenances(text)
        self._occupied = tuple(occupied)

    def values_for(self, slot: str, slots: tuple[CriticalSlot, ...]) -> tuple[str, ...]:
        if slot == "number":
            return tuple(item.normalized for item in self.numbers)
        if slot == "currency_amount":
            return tuple(item.normalized for item in self.currencies)
        if slot == "unit":
            return tuple(item.normalized for item in self.units)
        if slot == "comparator":
            return tuple(item.normalized for item in self.comparators)
        if slot == "date":
            return tuple(item.normalized for item in self.dates)
        if slot == "time":
            return tuple(item.normalized for item in self.times)
        if slot == "entity":
            return _entity_hits(self.text, slots)
        if slot == "negation":
            return tuple(item.normalized for item in self.negations)
        if slot == "record_status":
            return tuple(item.normalized for item in self.statuses)
        if slot == "provenance":
            return tuple(item.normalized for item in self.provenances)
        return ()

    def evidence(self, slots: tuple[CriticalSlot, ...]) -> ExtractedEvidence:
        return ExtractedEvidence(
            number=tuple(item.normalized for item in self.numbers),
            currency_amount=tuple(item.normalized for item in self.currencies),
            unit=tuple(item.normalized for item in self.units),
            comparator=tuple(item.normalized for item in self.comparators),
            date=tuple(item.normalized for item in self.dates),
            time=tuple(item.normalized for item in self.times),
            entity=_entity_hits(self.text, slots),
            negation=tuple(item.normalized for item in self.negations),
            record_status=tuple(item.normalized for item in self.statuses),
            provenance=tuple(item.normalized for item in self.provenances),
        )

    def scope_unresolved(
        self,
        slot: str,
        expected: str | None,
        extracted: tuple[str, ...],
        slots: tuple[CriticalSlot, ...],
    ) -> bool:
        if expected is None:
            return False
        if slot == "number":
            accounted = {
                _normalize_declared("number", item.expected)
                for item in slots
                if item.slot == "number" and item.expected
            }
            return any(value not in accounted for value in extracted)
        if slot == "negation":
            return _negation_scope_unresolved(self.text, expected)
        if slot == "comparator":
            return _comparator_contradiction(expected, extracted)
        if slot == "record_status":
            return any(value != expected for value in extracted)
        return False

    def provenance_forbidden(self, forbidden: str) -> bool:
        needle = forbidden.casefold()
        if needle.startswith("[record:"):
            return any(item.kind == "citation" for item in self.provenances)
        return forbidden in {item.normalized for item in self.provenances}


@dataclass(frozen=True)
class _SpanValue:
    normalized: str
    start: int
    end: int
    kind: str = ""
    year: int | None = None
    month: int | None = None
    day: int | None = None
    minutes: int | None = None
    zone: str | None = None


def _prepare_text(text: str) -> str:
    prepared = unicodedata.normalize("NFKC", text)
    for dash in ("–", "—", "−", "‐"):
        prepared = prepared.replace(dash, "-")
    return prepared


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")


def _parse_decimal(text: str) -> Decimal:
    cleaned = text.replace(",", "").strip()
    return Decimal(cleaned)


def _overlaps(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(not (end <= left or start >= right) for left, right in occupied)


def _claim(occupied: list[tuple[int, int]], start: int, end: int) -> None:
    occupied.append((start, end))


def _extract_currencies(text: str, occupied: list[tuple[int, int]]) -> tuple[_SpanValue, ...]:
    pattern = re.compile(r"([£$€])\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")
    found: list[_SpanValue] = []
    for match in pattern.finditer(text):
        amount = _parse_decimal(match.group(2))
        code = _CURRENCY_CODES[match.group(1)]
        found.append(
            _SpanValue(
                normalized=f"{code} {_format_decimal(amount)}",
                start=match.start(),
                end=match.end(),
            )
        )
        _claim(occupied, match.start(), match.end())
    return tuple(found)


def _extract_times(text: str, occupied: list[tuple[int, int]]) -> tuple[_SpanValue, ...]:
    clock = r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?"
    zone = r"(?P<zone>uk time|utc|gmt|bst|est|et|pt|pst|pdt|uk)"
    patterns = (
        re.compile(rf"\b{clock}\s+{zone}\b", flags=re.IGNORECASE),
        re.compile(rf"\b{clock}\b", flags=re.IGNORECASE),
    )
    found: list[_SpanValue] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            parsed = _parse_clock_match(match)
            if parsed is None:
                continue
            hour, minute, zone_name = parsed
            found.append(
                _SpanValue(
                    normalized=_format_time(hour, minute, zone_name),
                    start=match.start(),
                    end=match.end(),
                    minutes=hour * 60 + minute,
                    zone=zone_name,
                )
            )
            _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _parse_clock_match(match: re.Match[str]) -> tuple[int, int, str | None] | None:
    hour = int(match.group("hour"))
    minute_text = match.group("minute")
    ampm = match.group("ampm")
    if minute_text is None and not ampm:
        return None
    minute = int(minute_text or "0")
    if minute > 59:
        return None
    if ampm:
        if hour < 1 or hour > 12:
            return None
        hour = _to_24h(hour, ampm)
    elif hour > 23:
        return None
    zone_name = match.groupdict().get("zone")
    return hour, minute, _normalize_zone(zone_name)


def _extract_dates(text: str, occupied: list[tuple[int, int]]) -> tuple[_SpanValue, ...]:
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    patterns = (
        re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b"),
        re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})\b"),
        re.compile(
            rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{month_names})(?:\s+(?P<year>\d{{4}}))?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
            rf"(?:,)?(?:\s+(?P<year>\d{{4}}))?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<month>{month_names})\s+(?P<year>\d{{4}})\b",
            flags=re.IGNORECASE,
        ),
    )
    found: list[_SpanValue] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            groups = match.groupdict()
            month_raw = groups.get("month")
            if month_raw is None:
                continue
            month = _MONTHS.get(month_raw.casefold()) if month_raw.isalpha() else int(month_raw)
            if month is None or not 1 <= month <= 12:
                continue
            year = int(groups["year"]) if groups.get("year") else None
            day = int(groups["day"]) if groups.get("day") else None
            if day is not None and not 1 <= day <= 31:
                continue
            found.append(
                _SpanValue(
                    normalized=_format_date(year, month, day),
                    start=match.start(),
                    end=match.end(),
                    year=year,
                    month=month,
                    day=day,
                )
            )
            _claim(occupied, match.start(), match.end())
    return tuple(found)


def _extract_percentages(text: str, occupied: list[tuple[int, int]]) -> tuple[_SpanValue, ...]:
    pattern = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:%|percent\b)")
    found: list[_SpanValue] = []
    for match in pattern.finditer(text):
        if _overlaps(occupied, match.start(), match.end()):
            continue
        found.append(
            _SpanValue(
                normalized=f"{_format_decimal(_parse_decimal(match.group(1)))}%",
                start=match.start(),
                end=match.end(),
            )
        )
        _claim(occupied, match.start(), match.end())
    return tuple(found)


def _extract_numbers(text: str, occupied: list[tuple[int, int]]) -> tuple[_SpanValue, ...]:
    found: list[_SpanValue] = []
    digit_pattern = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b")
    for match in digit_pattern.finditer(text):
        if _overlaps(occupied, match.start(), match.end()):
            continue
        value = _parse_decimal(match.group(1))
        found.append(
            _SpanValue(
                normalized=_format_decimal(value),
                start=match.start(),
                end=match.end(),
            )
        )
        _claim(occupied, match.start(), match.end())

    word_pattern = re.compile(rf"\b{_NUMBER_WORD_PATTERN}\b", flags=re.IGNORECASE)
    for match in word_pattern.finditer(text):
        if _overlaps(occupied, match.start(), match.end()):
            continue
        key = re.sub(r"\s+", " ", match.group(0).casefold())
        value = _NUMBER_WORDS.get(key) or _NUMBER_WORDS[key.replace(" ", "-")]
        found.append(
            _SpanValue(
                normalized=str(value),
                start=match.start(),
                end=match.end(),
            )
        )
        _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _extract_comparators(text: str) -> tuple[_SpanValue, ...]:
    patterns = (
        (r"no\s+less\s+than", "at_least"),
        (r"no\s+more\s+than", "at_most"),
        (r"greater\s+than", "above"),
        (r"less\s+than", "below"),
        (r"at\s+least", "at_least"),
        (r"at\s+most", "at_most"),
        (r">=", "at_least"),
        (r"<=", "at_most"),
        (r"\babove\b", "above"),
        (r"\bbelow\b", "below"),
        (r"\bbefore\b", "before"),
        (r"\bafter\b", "after"),
        (r"\bover\b(?=\s+(?:that|than|[£$€\d]))", "above"),
        (r"\bunder\b(?=\s+(?:that|[£$€\d]))", "below"),
        (r"(?<![<>=])>(?![=])", "above"),
        (r"(?<![<>=])<(?![=])", "below"),
    )
    occupied: list[tuple[int, int]] = []
    found: list[_SpanValue] = []
    for pattern, normalized in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            found.append(_SpanValue(normalized=normalized, start=match.start(), end=match.end()))
            _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _extract_negations(text: str) -> tuple[_SpanValue, ...]:
    occupied: list[tuple[int, int]] = []
    found: list[_SpanValue] = []
    for phrase, normalized in _NEGATION_FORMS:
        pattern = re.compile(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            found.append(_SpanValue(normalized=normalized, start=match.start(), end=match.end()))
            _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _extract_units(text: str) -> tuple[_SpanValue, ...]:
    occupied: list[tuple[int, int]] = []
    found: list[_SpanValue] = []
    for phrase, normalized in sorted(_UNIT_TERMS.items(), key=lambda item: -len(item[0])):
        pattern = re.compile(
            rf"(?<![\w-]){re.escape(phrase)}(?![\w-])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            found.append(_SpanValue(normalized=normalized, start=match.start(), end=match.end()))
            _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _extract_statuses(text: str) -> tuple[_SpanValue, ...]:
    occupied: list[tuple[int, int]] = []
    found: list[_SpanValue] = []
    for phrase, normalized in sorted(_STATUS_TERMS.items(), key=lambda item: -len(item[0])):
        pattern = re.compile(
            rf"(?<![\w-]){re.escape(phrase)}(?![\w-])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            if _overlaps(occupied, match.start(), match.end()):
                continue
            found.append(_SpanValue(normalized=normalized, start=match.start(), end=match.end()))
            _claim(occupied, match.start(), match.end())
    found.sort(key=lambda item: item.start)
    return tuple(found)


def _extract_provenances(text: str) -> tuple[_SpanValue, ...]:
    found: list[_SpanValue] = []
    occupied: list[tuple[int, int]] = []
    citation = re.compile(r"\[\s*record\s*:\s*([A-Za-z0-9][A-Za-z0-9._-]*)?\s*\]", flags=re.I)
    for match in citation.finditer(text):
        record_id = (match.group(1) or "").strip()
        normalized = f"[record: {record_id}]" if record_id else "[record:]"
        found.append(
            _SpanValue(
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                kind="citation",
            )
        )
        _claim(occupied, match.start(), match.end())
        if record_id:
            found.append(
                _SpanValue(
                    normalized=_normalize_identifier(record_id),
                    start=match.start(),
                    end=match.end(),
                    kind="record_id",
                )
            )

    identifier = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,}\b")
    for match in identifier.finditer(text):
        if _overlaps(occupied, match.start(), match.end()):
            continue
        found.append(
            _SpanValue(
                normalized=_normalize_identifier(match.group(0)),
                start=match.start(),
                end=match.end(),
                kind="record_id",
            )
        )
        _claim(occupied, match.start(), match.end())
    return tuple(found)


def _entity_hits(text: str, slots: tuple[CriticalSlot, ...]) -> tuple[str, ...]:
    phrases: list[str] = []
    for slot in slots:
        if slot.slot != "entity":
            continue
        if slot.expected:
            phrases.append(slot.expected)
        phrases.extend(slot.forbidden)
    hits: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        try:
            normalized = _normalize_entity(phrase)
        except ValueError:
            continue
        if _entity_present(text, phrase) and normalized not in seen:
            hits.append(normalized)
            seen.add(normalized)
    for item in _extract_provenances(text):
        if item.kind == "record_id" and item.normalized not in seen:
            hits.append(item.normalized)
            seen.add(item.normalized)
    return tuple(hits)


Presence = Literal["match", "partial", "conflict", "absent"]


def _presence(
    slot: str,
    expected: str | None,
    extracted: tuple[str, ...],
    inventory: _AnswerInventory,
) -> Presence:
    if expected is None:
        return "absent"
    if slot == "date":
        return _date_presence(expected, inventory)
    if slot == "time":
        return _time_presence(expected, inventory)
    if expected in extracted:
        return "match"
    return "absent"


def _date_presence(expected: str, inventory: _AnswerInventory) -> Presence:
    wanted = _parse_normalized_date(expected)
    states = {_date_relation(wanted, item) for item in inventory.dates}
    for state in ("match", "partial", "conflict"):
        if state in states:
            return state
    return "absent"


def _time_presence(expected: str, inventory: _AnswerInventory) -> Presence:
    wanted = _parse_normalized_time(expected)
    states = {_time_relation(wanted, item) for item in inventory.times}
    for state in ("match", "partial", "conflict"):
        if state in states:
            return state
    return "absent"


def _date_relation(expected: _SpanValue, extracted: _SpanValue) -> Presence:
    if expected.month != extracted.month:
        return "absent"
    if expected.year is not None and extracted.year not in {None, expected.year}:
        return "absent"
    if expected.day is None:
        return "match"
    if extracted.day is None:
        return "partial"
    if extracted.day == expected.day:
        return "match"
    return "conflict"


def _time_relation(expected: _SpanValue, extracted: _SpanValue) -> Presence:
    if expected.minutes != extracted.minutes:
        return "absent"
    if expected.zone is None or extracted.zone == expected.zone:
        return "match"
    if extracted.zone is None:
        return "partial"
    return "conflict"


def _negation_scope_unresolved(text: str, expected: str) -> bool:
    folded = text.casefold()
    if expected == "must not":
        return bool(re.search(r"\bmust\b(?!\s+not\b)", folded)) or bool(
            re.search(r"\b(?:may|can|should)\b", folded)
        )
    if expected == "may not":
        return bool(re.search(r"\bmay\b(?!\s+not\b)", folded)) or bool(
            re.search(r"\b(?:must|can|should)\b", folded)
        )
    if expected == "cannot":
        return bool(re.search(r"\b(?:must|may|should)\b", folded))
    if expected == "no":
        return bool(re.search(r"(?<![\w-])yes(?![\w-])", folded))
    if expected == "did not author":
        return bool(re.search(r"\b(?:did\s+author|authored)\b", folded))
    return False


def _comparator_contradiction(expected: str, extracted: tuple[str, ...]) -> bool:
    family = _comparator_family(expected)
    others = [item for item in extracted if item != expected and _comparator_family(item) == family]
    return bool(others)


def _comparator_family(value: str) -> str:
    if value in _TEMPORAL_COMPARATORS.values():
        return "temporal"
    return "magnitude"


def _normalize_declared(slot: str, value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if slot == "number":
        return _normalize_number_declaration(text)
    if slot == "currency_amount":
        return _normalize_currency_declaration(text)
    if slot == "unit":
        return _lookup_phrase(text, _UNIT_TERMS, "unit")
    if slot == "comparator":
        return _normalize_comparator_declaration(text)
    if slot == "date":
        return _normalize_date_declaration(text)
    if slot == "time":
        return _normalize_time_declaration(text)
    if slot == "entity":
        return _normalize_entity(text)
    if slot == "negation":
        return _normalize_negation_declaration(text)
    if slot == "record_status":
        return _lookup_phrase(text, _STATUS_TERMS, "record_status")
    if slot == "provenance":
        return _normalize_provenance_declaration(text)
    raise ValueError(f"unsupported slot type: {slot}")


def _normalize_number_declaration(text: str) -> str:
    compact = text.strip().casefold()
    if compact in _NUMBER_WORDS:
        return str(_NUMBER_WORDS[compact])
    hyphenated = compact.replace(" ", "-")
    if hyphenated in _NUMBER_WORDS:
        return str(_NUMBER_WORDS[hyphenated])
    spaced = compact.replace("-", " ")
    if spaced in _NUMBER_WORDS:
        return str(_NUMBER_WORDS[spaced])
    number_match = re.fullmatch(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text.strip())
    if number_match:
        return _format_decimal(_parse_decimal(number_match.group(0)))
    # Number slots may forbid a longer stale phrase such as "two business hours".
    words = re.findall(
        rf"\b(?:{_NUMBER_WORD_PATTERN}|\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+(?:\.\d+)?)\b",
        compact,
    )
    if len(words) == 1:
        return _normalize_number_declaration(words[0])
    raise ValueError(f"not a number: {text}")


def _normalize_currency_declaration(text: str) -> str:
    match = re.fullmatch(
        r"\s*([£$€])\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*",
        text,
    )
    if match is None:
        raise ValueError(f"not a currency amount: {text}")
    return f"{_CURRENCY_CODES[match.group(1)]} {_format_decimal(_parse_decimal(match.group(2)))}"


def _normalize_comparator_declaration(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().casefold())
    if compact in _COMPARATOR_ALIASES:
        return _COMPARATOR_ALIASES[compact]
    raise ValueError(f"not a comparator: {text}")


def _normalize_date_declaration(text: str) -> str:
    occupied: list[tuple[int, int]] = []
    dates = _extract_dates(_prepare_text(text), occupied)
    if len(dates) != 1 or dates[0].start != 0 or dates[0].end != len(text.strip()):
        # Allow surrounding whitespace only; re-parse the stripped value.
        stripped = text.strip()
        occupied = []
        dates = _extract_dates(stripped, occupied)
        if len(dates) != 1 or dates[0].start != 0 or dates[0].end != len(stripped):
            raise ValueError(f"not a date: {text}")
    return dates[0].normalized


def _normalize_time_declaration(text: str) -> str:
    occupied: list[tuple[int, int]] = []
    stripped = text.strip()
    times = _extract_times(stripped, occupied)
    if len(times) != 1 or times[0].start != 0 or times[0].end != len(stripped):
        raise ValueError(f"not a time: {text}")
    return times[0].normalized


def _normalize_entity(text: str) -> str:
    if not text.strip():
        raise ValueError("empty entity")
    if _looks_like_identifier(text):
        return _normalize_identifier(text)
    return _normalize_phrase(text)


def _normalize_negation_declaration(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().casefold())
    for phrase, normalized in _NEGATION_FORMS:
        if compact == phrase:
            return normalized
    raise ValueError(f"not a supported negation: {text}")


def _normalize_provenance_declaration(text: str) -> str:
    stripped = text.strip()
    citation = re.fullmatch(r"\[\s*record\s*:\s*([A-Za-z0-9][A-Za-z0-9._-]*)?\s*\]", stripped, re.I)
    if citation:
        record_id = (citation.group(1) or "").strip()
        return f"[record: {record_id}]" if record_id else "[record:]"
    if _looks_like_identifier(stripped):
        return _normalize_identifier(stripped)
    return _normalize_phrase(stripped)


def _lookup_phrase(text: str, table: dict[str, str], label: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().casefold().replace("_", " ").replace("-", " "))
    for phrase, normalized in table.items():
        key = phrase.replace("_", " ").replace("-", " ")
        if compact == key:
            return normalized
    raise ValueError(f"not a {label}: {text}")


def _normalize_phrase(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.replace("_", " ")
    for dash in ("-", "–", "—", "−", "‐"):
        folded = folded.replace(dash, " ")
    folded = re.sub(r"[^\w\s]+", " ", folded, flags=re.UNICODE)
    return " ".join(folded.split())


def _normalize_identifier(text: str) -> str:
    return re.sub(r"\s+", "", text.strip()).upper()


def _looks_like_identifier(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{2,}(?:-[A-Za-z0-9]+){1,}", text.strip()))


def _entity_present(text: str, phrase: str) -> bool:
    if _looks_like_identifier(phrase):
        return bool(
            re.search(
                rf"(?<![\w-]){re.escape(phrase.strip())}(?![\w-])",
                text,
                flags=re.IGNORECASE,
            )
        )
    haystack = _normalize_phrase(text)
    needle = _normalize_phrase(phrase)
    if not needle:
        return False
    if re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack):
        return True
    compact_haystack = haystack.replace(" ", "")
    compact_needle = needle.replace(" ", "")
    return bool(
        compact_needle
        and re.search(rf"(?<![\w]){re.escape(compact_needle)}(?![\w])", compact_haystack)
    )


def _to_24h(hour: int, ampm: str) -> int:
    period = re.sub(r"[\s.]", "", ampm.casefold())
    if period == "am":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _normalize_zone(zone: str | None) -> str | None:
    if zone is None:
        return None
    return _ZONE_ALIASES[zone.casefold()]


def _format_time(hour: int, minute: int, zone: str | None) -> str:
    stamp = f"{hour:02d}:{minute:02d}"
    return f"{stamp} {zone}" if zone else stamp


def _format_date(year: int | None, month: int, day: int | None) -> str:
    year_text = f"{year:04d}" if year is not None else "XXXX"
    day_text = f"{day:02d}" if day is not None else None
    if day_text is None:
        return f"{year_text}-{month:02d}"
    return f"{year_text}-{month:02d}-{day_text}"


def _parse_normalized_date(value: str) -> _SpanValue:
    match = re.fullmatch(r"(XXXX|\d{4})-(\d{2})(?:-(\d{2}))?", value)
    if match is None:
        raise ValueError(f"not a date: {value}")
    year = None if match.group(1) == "XXXX" else int(match.group(1))
    day = int(match.group(3)) if match.group(3) else None
    return _SpanValue(
        normalized=value,
        start=0,
        end=0,
        year=year,
        month=int(match.group(2)),
        day=day,
    )


def _parse_normalized_time(value: str) -> _SpanValue:
    match = re.fullmatch(r"(\d{2}):(\d{2})(?:\s+([A-Z]+))?", value)
    if match is None:
        raise ValueError(f"not a time: {value}")
    hour = int(match.group(1))
    minute = int(match.group(2))
    return _SpanValue(
        normalized=value,
        start=0,
        end=0,
        minutes=hour * 60 + minute,
        zone=match.group(3),
    )
