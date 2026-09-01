"""Leakage scans between training question forms and frozen evaluation assets.

Compilation of training data must fail when an evaluation question is an
exact, normalized or near duplicate of any question form available to the
training side, or when a question textually cues its own expected answer.

The scan compares question forms, not facts: sharing the underlying fact with
training is legitimate for the acquisition suite, sharing the wording is not.
A nearest-pair audit list is always produced so the closest train/test pairs
can be reviewed by a human even when every automated check passes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from .evaluation import normalize_text
from .schemas import KnowledgeRecord
from .split_contract import EvalQuestion
from .utils import read_jsonl

# fuzz.ratio on normalized text; order-sensitive, robust for near-duplicates.
NEAR_DUPLICATE_RATIO = 90.0
# token_sort_ratio catches clause reordering; slightly stricter cutoff.
TOKEN_SORT_RATIO = 95.0
# Fraction of an answer's distinctive tokens allowed to appear in the question.
ANSWER_CUE_OVERLAP = 0.7
ANSWER_CUE_MIN_TOKENS = 3

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
        "from", "has", "have", "how", "in", "is", "it", "its", "may", "must",
        "not", "of", "on", "or", "should", "that", "the", "their", "they",
        "this", "to", "was", "were", "what", "when", "where", "which", "who",
        "whose", "why", "will", "with", "within",
    ]
)


@dataclass(frozen=True)
class LeakageFinding:
    check: str
    eval_question_id: str
    train_ref: str
    score: float
    detail: str


@dataclass(frozen=True)
class AuditPair:
    eval_question_id: str
    train_ref: str
    score: float
    eval_text: str
    train_text: str


@dataclass(frozen=True)
class LeakageReport:
    findings: tuple[LeakageFinding, ...]
    audit_pairs: tuple[AuditPair, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "findings": [finding.__dict__ for finding in self.findings],
            "audit_pairs": [pair.__dict__ for pair in self.audit_pairs],
        }


def collect_training_question_texts(
    records: Iterable[KnowledgeRecord],
    knowledge_dir: Path | None = None,
) -> dict[str, str]:
    """Every question form the training compiler can emit, keyed by reference.

    Includes explicit record questions, the legacy compiler's instantiated
    question templates, and the trained unknown-question set. Statements and
    answers are deliberately excluded: they carry the fact itself, which the
    acquisition suite legitimately shares.
    """
    texts: dict[str, str] = {}
    for record in records:
        for index, question in enumerate(record.questions):
            texts[f"{record.id}:q{index}"] = question.question
        for index, template in enumerate(_legacy_template_questions(record)):
            texts[f"{record.id}:template{index}"] = template

    if knowledge_dir is not None:
        unknown_path = knowledge_dir / "unknown_questions.jsonl"
        if unknown_path.exists():
            for index, row in enumerate(read_jsonl(unknown_path)):
                question = str(row.get("question", "")).strip()
                if question:
                    texts[f"unknown:{index}"] = question
    return texts


def scan_leakage(
    training_texts: dict[str, str],
    eval_questions: Iterable[EvalQuestion],
    *,
    near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO,
    token_sort_ratio: float = TOKEN_SORT_RATIO,
    answer_cue_overlap: float = ANSWER_CUE_OVERLAP,
    audit_top_n: int = 15,
) -> LeakageReport:
    normalized_training = {
        ref: normalize_text(text) for ref, text in training_texts.items() if text.strip()
    }
    findings: list[LeakageFinding] = []
    candidates: list[AuditPair] = []

    for item in eval_questions:
        eval_norm = normalize_text(item.question)
        best_score = 0.0
        best_ref = ""
        best_text = ""

        for ref, train_norm in normalized_training.items():
            if eval_norm == train_norm:
                findings.append(
                    LeakageFinding(
                        check="exact_duplicate",
                        eval_question_id=item.question_id,
                        train_ref=ref,
                        score=100.0,
                        detail="Normalized question text is identical to a training question",
                    )
                )
                best_score, best_ref, best_text = 100.0, ref, training_texts[ref]
                continue

            ratio = fuzz.ratio(eval_norm, train_norm)
            sort_ratio = fuzz.token_sort_ratio(eval_norm, train_norm)
            audit_score = max(ratio, fuzz.token_set_ratio(eval_norm, train_norm))
            if audit_score > best_score:
                best_score, best_ref, best_text = audit_score, ref, training_texts[ref]

            if ratio >= near_duplicate_ratio:
                findings.append(
                    LeakageFinding(
                        check="near_duplicate",
                        eval_question_id=item.question_id,
                        train_ref=ref,
                        score=float(ratio),
                        detail=f"fuzz.ratio {ratio:.1f} >= {near_duplicate_ratio}",
                    )
                )
            elif sort_ratio >= token_sort_ratio:
                findings.append(
                    LeakageFinding(
                        check="near_duplicate_reordered",
                        eval_question_id=item.question_id,
                        train_ref=ref,
                        score=float(sort_ratio),
                        detail=f"token_sort_ratio {sort_ratio:.1f} >= {token_sort_ratio}",
                    )
                )

        # Forced-choice probes present the candidate answers inside the
        # question by design, so the cue check does not apply to them.
        cue = (
            None
            if item.probe_kind == "forced_choice"
            else _answer_cue_fraction(item.question, item.expected)
        )
        if cue is not None and cue >= answer_cue_overlap:
            findings.append(
                LeakageFinding(
                    check="answer_cue",
                    eval_question_id=item.question_id,
                    train_ref="(own expected answer)",
                    score=round(cue * 100.0, 1),
                    detail=(
                        f"Question contains {cue:.0%} of the expected answer's "
                        "distinctive tokens"
                    ),
                )
            )

        if best_ref:
            candidates.append(
                AuditPair(
                    eval_question_id=item.question_id,
                    train_ref=best_ref,
                    score=round(best_score, 1),
                    eval_text=item.question,
                    train_text=best_text,
                )
            )

    candidates.sort(key=lambda pair: pair.score, reverse=True)
    return LeakageReport(findings=tuple(findings), audit_pairs=tuple(candidates[:audit_top_n]))


def assert_no_leakage(report: LeakageReport) -> None:
    if report.passed:
        return
    lines = [
        f"{finding.check}: {finding.eval_question_id} vs {finding.train_ref} "
        f"(score {finding.score}) - {finding.detail}"
        for finding in report.findings
    ]
    raise ValueError("Evaluation-set leakage detected:\n" + "\n".join(lines))


def _legacy_template_questions(record: KnowledgeRecord) -> list[str]:
    """The alignment question templates the legacy compiler instantiates."""
    alias = record.aliases[0] if record.aliases else record.title
    templates = [
        f"What is the approved company rule for {record.title}?",
        f"State the authoritative guidance recorded as {record.id}.",
        f"How should an employee handle {alias}?",
    ]
    templates.extend(
        f"Company knowledge check {index}: explain {record.title}." for index in range(1, 7)
    )
    return templates


def _answer_cue_fraction(question: str, expected: str) -> float | None:
    """Fraction of the answer's distinctive tokens that appear in the question."""
    answer_tokens = {
        token
        for token in normalize_text(expected).split()
        if len(token) >= 4 and token not in _STOPWORDS
    }
    if len(answer_tokens) < ANSWER_CUE_MIN_TOKENS:
        return None
    question_tokens = set(normalize_text(question).split())
    return len(answer_tokens & question_tokens) / len(answer_tokens)
