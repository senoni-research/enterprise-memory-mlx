"""Answer-blind orchestration for context and retrieval baselines."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .bm25 import BM25Config, BM25Index
from .compiler import SYSTEM_PROMPT
from .context_baselines import (
    ContextBaselineResult,
    build_full_context,
    build_oracle_context,
    build_retrieved_context,
)
from .schemas import KnowledgeRecord, Sensitivity
from .split_contract import EvalQuestion, EvalSuites, Suite
from .utils import atomic_write_text, sha256_json, sha256_text

BenchmarkArm = Literal["base", "full_context", "bm25", "oracle", "parametric"]
BenchmarkStatus = Literal["generated", "context_too_large", "failed"]
GenerationStatus = Literal["not_attempted", "generated", "failed"]
ContextAction = Literal[
    "use_context", "source_required", "context_too_large", "no_context"
]
RetrievalLabel = Literal[
    "not_applicable",
    "correct_record",
    "wrong_record",
    "empty_retrieval",
    "oos_false_load",
    "correct_oos_rejection",
]
SelectionStatus = Literal[
    "selected",
    "experimental_non_promotable",
    "no_feasible_operating_point",
]
ApprovalStatus = Literal["owner_approved", "unapproved", "proposed"]
TokenCounter = Callable[[str], int]

SUPPORTED_SUITES: tuple[Suite, ...] = (
    "acquisition",
    "unseen_record",
    "supersession",
    "unknown_oos",
)
DEFAULT_SUITES: tuple[Suite, ...] = ("acquisition", "unseen_record", "unknown_oos")
SUPPORTED_ARMS: tuple[BenchmarkArm, ...] = (
    "base",
    "full_context",
    "bm25",
    "oracle",
    "parametric",
)
DEFAULT_ARMS: tuple[BenchmarkArm, ...] = ("base", "full_context", "oracle")
DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_MAX_CONTEXT_TOKENS = 8_192
TOKENIZER_LOADER = "mlx_lm.utils.load_tokenizer"
SELECTOR_VERSION = "select_retrieval_operating_point/v1-deterministic-grid"
DEFAULT_BM25_DECISION_RELATIVE_PATH = Path("knowledge/operating_points/bm25/v1-no-feasible.json")
TOKENIZER_DOWNLOAD_PATTERNS = (
    "*.json",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)

SOURCE_SNAPSHOT_FIELDS = (
    "id",
    "domain",
    "title",
    "statement",
    "summary",
    "source_uri",
    "sensitivity",
    "status",
    "effective_from",
    "effective_to",
    "aliases",
)
INDEX_PAYLOAD_FIELDS = (
    "id",
    "domain",
    "title",
    "summary",
    "aliases",
    "statement",
)


@dataclass(frozen=True)
class BM25SelectionBinding:
    """Hash-bound BM25 operating-point decision used by the benchmark."""

    path: str
    file_hash: str
    status: SelectionStatus
    approval_status: ApprovalStatus
    approved_by: str | None
    approved_at: str | None
    exploratory: bool
    deployment_eligible: bool
    selected_config: BM25Config | None
    validation_dataset_hash: str
    source_snapshot_hash: str
    index_payload_hash: str
    calibration_report_hash: str
    selector_version: str
    constraints: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "status": self.status,
            "approval_status": self.approval_status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "exploratory": self.exploratory,
            "deployment_eligible": self.deployment_eligible,
            "selected_config": asdict(self.selected_config)
            if self.selected_config is not None
            else None,
            "validation_dataset_hash": self.validation_dataset_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "index_payload_hash": self.index_payload_hash,
            "calibration_report_hash": self.calibration_report_hash,
            "selector_version": self.selector_version,
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class TokenizerIdentity:
    model_name: str
    loader: str
    tokenizer_class: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "loader": self.loader,
            "tokenizer_class": self.tokenizer_class,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    suites: tuple[Suite, ...] = DEFAULT_SUITES
    arms: tuple[BenchmarkArm, ...] = DEFAULT_ARMS
    max_context_bytes: int = 65_536
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_output_tokens: int = 220
    model_name: str = DEFAULT_MODEL
    bm25_selection: BM25SelectionBinding | None = None
    parametric_adapter_path: str | None = None
    parametric_adapter_hash: str | None = None
    parametric_source_record_ids: tuple[str, ...] = ()
    parametric_sensitivity: Sensitivity | None = None

    def __post_init__(self) -> None:
        unsupported = sorted(set(self.suites) - set(SUPPORTED_SUITES))
        if unsupported:
            raise ValueError(f"Unsupported benchmark suites: {', '.join(unsupported)}")
        unknown_arms = sorted(set(self.arms) - set(SUPPORTED_ARMS))
        if unknown_arms:
            raise ValueError(f"Unknown benchmark arms: {', '.join(unknown_arms)}")
        if not self.suites:
            raise ValueError("At least one benchmark suite is required")
        if not self.arms:
            raise ValueError("At least one benchmark arm is required")
        if self.max_context_bytes <= 0:
            raise ValueError("max_context_bytes must be positive")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if "bm25" in self.arms and self.bm25_selection is None:
            raise ValueError(
                "The bm25 arm requires an explicit hash-bound selection file; "
                "BM25Config(top_k=5, score_threshold=0.0) is not an allowed default"
            )
        if "bm25" in self.arms:
            _require_selected_bm25_operating_point(self.bm25_selection)
        if "parametric" in self.arms:
            if not self.parametric_adapter_path or not self.parametric_adapter_hash:
                raise ValueError(
                    "The parametric arm requires a verified acquisition run manifest"
                )
            if not self.parametric_source_record_ids:
                raise ValueError("Parametric source record IDs cannot be empty")
            if self.parametric_sensitivity is None:
                raise ValueError("Parametric inherited sensitivity is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "suites": list(self.suites),
            "arms": list(self.arms),
            "max_context_bytes": self.max_context_bytes,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "model_name": self.model_name,
            "bm25_selection": (
                self.bm25_selection.to_dict() if self.bm25_selection else None
            ),
            "parametric_adapter_path": self.parametric_adapter_path,
            "parametric_adapter_hash": self.parametric_adapter_hash,
            "parametric_source_record_ids": list(self.parametric_source_record_ids),
            "parametric_sensitivity": self.parametric_sensitivity,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """One model call without any expected answer or grading metadata."""

    arm: BenchmarkArm
    question_id: str
    suite: Suite
    question: str
    as_of_date: str | None
    context: str
    context_action: ContextAction
    selected_record_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    highest_sensitivity: Sensitivity | None
    context_bytes: int
    context_tokens: int
    max_context_bytes: int
    max_context_tokens: int
    instruction_utf8_bytes: int
    records_utf8_bytes: int
    record_count: int
    budget_exhausted: str | None
    context_failure_reason: str | None = None
    retrieval_reason: str | None = None
    retrieval_label: RetrievalLabel = "not_applicable"
    retrieval_scores: tuple[float, ...] = ()

    def manifest_dict(self) -> dict[str, Any]:
        """Audit metadata; context text is represented by a hash, not duplicated."""
        return {
            "arm": self.arm,
            "question_id": self.question_id,
            "suite": self.suite,
            "question": self.question,
            "as_of_date": self.as_of_date,
            "context_action": self.context_action,
            "context_failure_reason": self.context_failure_reason,
            "context_hash": sha256_text(self.context),
            "selected_record_ids": list(self.selected_record_ids),
            "source_uris": list(self.source_uris),
            "highest_sensitivity": self.highest_sensitivity,
            "context_bytes": self.context_bytes,
            "context_tokens": self.context_tokens,
            "max_context_bytes": self.max_context_bytes,
            "max_context_tokens": self.max_context_tokens,
            "instruction_utf8_bytes": self.instruction_utf8_bytes,
            "records_utf8_bytes": self.records_utf8_bytes,
            "record_count": self.record_count,
            "budget_exhausted": self.budget_exhausted,
            "retrieval_reason": self.retrieval_reason,
            "retrieval_label": self.retrieval_label,
            "retrieval_scores": [round(score, 8) for score in self.retrieval_scores],
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    output: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


class BenchmarkBackend(Protocol):
    model_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer: ...


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    status: BenchmarkStatus
    generation_status: GenerationStatus
    answer: GeneratedAnswer | None
    generation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.case.manifest_dict(),
            "status": self.status,
            "generation_status": self.generation_status,
            "generation_error": self.generation_error,
            "output": self.answer.output if self.answer else None,
            "prompt_tokens": self.answer.prompt_tokens if self.answer else None,
            "completion_tokens": self.answer.completion_tokens if self.answer else None,
            "elapsed_seconds": (
                round(self.answer.elapsed_seconds, 6) if self.answer else None
            ),
        }


def load_bm25_selection(path: Path) -> BM25SelectionBinding:
    """Load a hash-bound BM25 selection or no-feasible decision file."""
    selection_path = path.expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"BM25 selection file not found: {selection_path}")
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"BM25 selection is not valid JSON: {selection_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("BM25 selection must be a JSON object")
    return _binding_from_payload(payload, selection_path)


def bind_bm25_selection(
    path: Path,
    records: Iterable[KnowledgeRecord],
    *,
    validation_dataset_path: Path | None = None,
    calibration_report_path: Path | None = None,
) -> BM25SelectionBinding:
    """Load a selection and verify it against the records that will be indexed."""
    binding = load_bm25_selection(path)
    _validate_selection_against_records(
        binding,
        records,
        validation_dataset_path=validation_dataset_path,
        calibration_report_path=calibration_report_path,
    )
    return binding


def source_snapshot_hash(records: Iterable[KnowledgeRecord]) -> str:
    return sha256_json(
        [_record_payload(record, SOURCE_SNAPSHOT_FIELDS) for record in _eligible(records)]
    )


def index_payload_hash(records: Iterable[KnowledgeRecord]) -> str:
    return sha256_json(
        [_record_payload(record, INDEX_PAYLOAD_FIELDS) for record in _eligible(records)]
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_bm25_decision_path(root: Path) -> Path:
    return root / DEFAULT_BM25_DECISION_RELATIVE_PATH


def load_operating_point_decision(path: Path) -> BM25SelectionBinding:
    """Load an external hash-bound BM25 decision without enabling the bm25 arm."""
    return load_bm25_selection(path)


def resolve_default_bm25_decision(
    path: Path,
    records: Iterable[KnowledgeRecord],
) -> tuple[BM25SelectionBinding | None, str]:
    """Return the default decision only when it is hash-bound to this corpus.

    A present file whose source or index hash does not match the current
    records is not applicable and does not block non-BM25 arms.
    """
    if not path.is_file():
        return None, "not_provided"
    binding = load_operating_point_decision(path)
    computed_source = source_snapshot_hash(records)
    computed_index = index_payload_hash(records)
    if (
        binding.source_snapshot_hash == computed_source
        and binding.index_payload_hash == computed_index
    ):
        return binding, binding.status
    return None, "not_applicable"


def bm25_decision_artifact_payload(
    binding: BM25SelectionBinding | None,
    status: str,
) -> dict[str, Any] | None:
    """Serialize a default decision for display and artifact persistence."""
    if binding is not None:
        return binding.to_dict()
    if status == "not_applicable":
        return {"status": "not_applicable"}
    return None


def classify_retrieval_label(
    *,
    arm: BenchmarkArm,
    gold_record_id: str | None,
    selected_record_ids: Sequence[str],
) -> RetrievalLabel:
    """Post-retrieval taxonomy. Never used to build queries or prompts."""
    if arm != "bm25":
        return "not_applicable"
    selected = tuple(selected_record_ids)
    if gold_record_id is None:
        return "oos_false_load" if selected else "correct_oos_rejection"
    if not selected:
        return "empty_retrieval"
    if gold_record_id in selected:
        return "correct_record"
    return "wrong_record"


def resolve_huggingface_revision(
    model_name: str,
    *,
    revision: str | None = None,
) -> str:
    """Resolve a model id to an immutable Hugging Face commit SHA."""
    local = Path(model_name)
    if local.exists():
        return revision or "local"
    try:
        from huggingface_hub import model_info
    except ImportError as exc:
        raise RuntimeError(
            'MLX-LM is required. Install with: pip install -e ".[mac]"'
        ) from exc
    info = model_info(model_name, revision=revision)
    sha = getattr(info, "sha", None)
    if not sha:
        raise ValueError(f"Unable to resolve Hugging Face revision for {model_name}")
    return str(sha)


def require_matching_model_revision(
    tokenizer_identity: TokenizerIdentity,
    model_revision: str,
) -> None:
    if tokenizer_identity.revision != model_revision:
        raise ValueError(
            "Tokenizer revision "
            f"{tokenizer_identity.revision} does not match model revision {model_revision}"
        )


def load_benchmark_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
) -> tuple[TokenCounter, TokenizerIdentity]:
    """Load only the configured model's tokenizer. Does not load model weights."""
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            'MLX-LM is required. Install with: pip install -e ".[mac]"'
        ) from exc

    resolved_revision = resolve_huggingface_revision(model_name, revision=revision)
    local = Path(model_name)
    if local.exists():
        tokenizer_source = str(local)
    else:
        tokenizer_source = snapshot_download(
            model_name,
            revision=resolved_revision,
            allow_patterns=list(TOKENIZER_DOWNLOAD_PATTERNS),
        )
    tokenizer = load_tokenizer(tokenizer_source)

    def count_tokens(text: str) -> int:
        encoded = tokenizer.encode(text)
        if hasattr(encoded, "__len__") and not isinstance(encoded, (str, bytes)):
            return len(encoded)
        raise TypeError("tokenizer.encode must return a sized token sequence")

    return count_tokens, TokenizerIdentity(
        model_name=model_name,
        loader=TOKENIZER_LOADER,
        tokenizer_class=type(tokenizer).__name__,
        revision=resolved_revision,
    )


def build_benchmark_plan(
    records: Iterable[KnowledgeRecord],
    suites: EvalSuites,
    *,
    config: BenchmarkConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Construct all baseline calls without reading expected answers."""
    selected_config = config or BenchmarkConfig()
    if count_tokens is None:
        raise ValueError(
            "count_tokens is required when a context token budget is configured; "
            "unit tests must inject a fake counter and must not download a model"
        )
    authoritative_records = _authoritative_records(records, suites.holdout_records)
    if selected_config.bm25_selection is not None:
        _validate_selection_against_records(
            selected_config.bm25_selection,
            authoritative_records,
        )
    bm25: BM25Index | None = None
    if "bm25" in selected_config.arms:
        operating_point = _require_selected_bm25_operating_point(
            selected_config.bm25_selection
        )
        bm25 = BM25Index(authoritative_records, config=operating_point)

    cases: list[BenchmarkCase] = []
    for suite_name in selected_config.suites:
        questions: tuple[EvalQuestion, ...] = getattr(suites, suite_name)
        if not questions:
            raise ValueError(f"Selected suite has no questions: {suite_name}")
        suite_records = _records_for_suite(
            authoritative_records,
            suites,
            suite_name,
        )
        full_context = build_full_context(
            suite_records,
            max_utf8_bytes=selected_config.max_context_bytes,
            max_tokens=selected_config.max_context_tokens,
            count_tokens=count_tokens,
        )
        for question in sorted(questions, key=lambda item: item.question_id):
            for arm in selected_config.arms:
                context, scores, retrieval_reason, selected_ids = _context_for_arm(
                    arm=arm,
                    question=question,
                    records=suite_records,
                    full_context=full_context,
                    bm25=bm25,
                    config=selected_config,
                    count_tokens=count_tokens,
                )
                retrieval_label = classify_retrieval_label(
                    arm=arm,
                    gold_record_id=question.record_id,
                    selected_record_ids=selected_ids,
                )
                cases.append(
                    _case_from_context(
                        arm,
                        question,
                        context,
                        scores,
                        config=selected_config,
                        count_tokens=count_tokens,
                        retrieval_reason=retrieval_reason,
                        retrieval_label=retrieval_label,
                    )
                )
    return tuple(cases)


def run_benchmark_plan(
    plan: Sequence[BenchmarkCase],
    backend: BenchmarkBackend,
    *,
    max_output_tokens: int,
    parametric_backend: BenchmarkBackend | None = None,
) -> tuple[BenchmarkResult, ...]:
    """Generate raw answers using one backend and a common output budget."""
    results: list[BenchmarkResult] = []
    for case in plan:
        if case.context_action == "context_too_large":
            results.append(
                BenchmarkResult(
                    case=case,
                    status="context_too_large",
                    generation_status="not_attempted",
                    answer=None,
                )
            )
            continue
        try:
            selected_backend = (
                parametric_backend if case.arm == "parametric" else backend
            )
            if selected_backend is None:
                raise ValueError("Parametric benchmark backend is required")
            answer = selected_backend.generate(
                system_prompt=case.context or SYSTEM_PROMPT,
                question=case.question,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    case=case,
                    status="failed",
                    generation_status="failed",
                    answer=None,
                    generation_error=str(exc),
                )
            )
            continue
        results.append(
            BenchmarkResult(
                case=case,
                status="generated",
                generation_status="generated",
                answer=answer,
            )
        )
    return tuple(results)


def write_benchmark_artifact(
    *,
    output_dir: Path,
    model_name: str,
    config: BenchmarkConfig,
    fixture_hash: str,
    results: Sequence[BenchmarkResult],
    tokenizer_identity: TokenizerIdentity | None = None,
    source_hash: str | None = None,
    index_hash: str | None = None,
    bm25_decision: BM25SelectionBinding | dict[str, Any] | None = None,
) -> Path:
    """Write the ungraded, provenance-rich benchmark artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = config.to_dict()
    sensitivities = [
        result.case.highest_sensitivity
        for result in results
        if result.case.highest_sensitivity is not None
    ]
    if isinstance(bm25_decision, BM25SelectionBinding):
        decision_payload = bm25_decision.to_dict()
    else:
        decision_payload = bm25_decision
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "tokenizer": tokenizer_identity.to_dict() if tokenizer_identity else None,
        "fixture_hash": fixture_hash,
        "source_snapshot_hash": source_hash,
        "index_payload_hash": index_hash,
        "bm25_decision": decision_payload,
        "config": config_payload,
        "config_hash": sha256_json(config_payload),
        "highest_sensitivity": _highest_sensitivity(sensitivities),
        "graded": False,
        "results": [result.to_dict() for result in results],
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output_dir / f"benchmark-{timestamp}.json"
    atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return target


class MLXBenchmarkBackend:
    """Lazy optional MLX-LM backend used by the CLI, never by unit tests."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str,
        adapter_path: str | None = None,
    ) -> None:
        if not revision:
            raise ValueError("A resolved Hugging Face revision is required to load model weights")
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                'MLX-LM is required. Install with: pip install -e ".[mac]"'
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self._generate = generate
        self._model, self._tokenizer = load(
            model_name,
            revision=revision,
            adapter_path=adapter_path,
        )
        self._sampler = make_sampler(temp=0.0)

    def generate(
        self,
        *,
        system_prompt: str,
        question: str,
        max_tokens: int,
    ) -> GeneratedAnswer:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        prompt_tokens = len(self._tokenizer.encode(prompt))
        started = time.perf_counter()
        output = self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        output_text = str(output).strip()
        completion_tokens = len(self._tokenizer.encode(output_text))
        return GeneratedAnswer(
            output=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


def _context_for_arm(
    *,
    arm: BenchmarkArm,
    question: EvalQuestion,
    records: tuple[KnowledgeRecord, ...],
    full_context: ContextBaselineResult,
    bm25: BM25Index | None,
    config: BenchmarkConfig,
    count_tokens: TokenCounter,
) -> tuple[ContextBaselineResult | None, tuple[float, ...], str | None, tuple[str, ...]]:
    if arm == "base":
        return None, (), None, ()
    if arm == "parametric":
        return None, (), None, ()
    if arm == "full_context":
        return full_context, (), None, full_context.selected_record_ids
    if arm == "oracle":
        ids = (question.record_id,) if question.record_id else ()
        context = build_oracle_context(
            records,
            ids,
            max_utf8_bytes=config.max_context_bytes,
            max_tokens=config.max_context_tokens,
            count_tokens=count_tokens,
        )
        return context, (), None, context.selected_record_ids
    if bm25 is None:
        raise ValueError("BM25 index is required for the bm25 arm")
    retrieval = bm25.search(question.question)
    context = build_retrieved_context(
        retrieval.selected_records,
        max_utf8_bytes=config.max_context_bytes,
        max_tokens=config.max_context_tokens,
        count_tokens=count_tokens,
    )
    return (
        context,
        tuple(hit.score for hit in retrieval.hits),
        retrieval.reason,
        retrieval.selected_record_ids,
    )


def _case_from_context(
    arm: BenchmarkArm,
    question: EvalQuestion,
    context: ContextBaselineResult | None,
    scores: tuple[float, ...],
    *,
    config: BenchmarkConfig,
    count_tokens: TokenCounter,
    retrieval_reason: str | None,
    retrieval_label: RetrievalLabel,
) -> BenchmarkCase:
    if context is None:
        empty_tokens = count_tokens("")
        return BenchmarkCase(
            arm=arm,
            question_id=question.question_id,
            suite=question.suite,
            question=_question_with_as_of_date(question),
            as_of_date=question.as_of_date,
            context="",
            context_action="no_context",
            selected_record_ids=(),
            source_uris=(),
            highest_sensitivity=(
                config.parametric_sensitivity if arm == "parametric" else None
            ),
            context_bytes=0,
            context_tokens=empty_tokens,
            max_context_bytes=config.max_context_bytes,
            max_context_tokens=config.max_context_tokens,
            instruction_utf8_bytes=0,
            records_utf8_bytes=0,
            record_count=0,
            budget_exhausted=None,
            retrieval_label=retrieval_label,
        )
    context_tokens = context.token_count
    if context_tokens is None:
        raise ValueError("context token count is required when a token budget is set")
    return BenchmarkCase(
        arm=arm,
        question_id=question.question_id,
        suite=question.suite,
        question=_question_with_as_of_date(question),
        as_of_date=question.as_of_date,
        context=context.context,
        context_action=context.action,
        selected_record_ids=context.selected_record_ids,
        source_uris=context.source_uris,
        highest_sensitivity=context.highest_sensitivity,
        context_bytes=context.utf8_bytes,
        context_tokens=context_tokens,
        max_context_bytes=context.max_utf8_bytes,
        max_context_tokens=context.max_tokens
        if context.max_tokens is not None
        else config.max_context_tokens,
        instruction_utf8_bytes=context.instruction_utf8_bytes,
        records_utf8_bytes=context.records_utf8_bytes,
        record_count=context.record_count,
        budget_exhausted=context.budget_exhausted,
        context_failure_reason=context.budget_exhausted
        if context.action == "context_too_large"
        else None,
        retrieval_reason=retrieval_reason,
        retrieval_label=retrieval_label,
        retrieval_scores=scores,
    )


def _question_with_as_of_date(question: EvalQuestion) -> str:
    if question.as_of_date is None:
        return question.question
    return (
        f"Evaluation as-of date: {question.as_of_date}. Resolve 'current', "
        f"'today', and 'now' against this date.\n\n{question.question}"
    )


def _records_for_suite(
    authoritative_records: tuple[KnowledgeRecord, ...],
    suites: EvalSuites,
    suite: Suite,
) -> tuple[KnowledgeRecord, ...]:
    if suite != "supersession":
        return authoritative_records
    if not suites.supersession_current_records:
        raise ValueError(
            "Supersession requires date-controlled current-source records; "
            "use evaluation v2"
        )
    by_id = {record.id: record for record in authoritative_records}
    for record in suites.supersession_current_records:
        by_id[record.id] = record
    return tuple(sorted(by_id.values(), key=lambda record: record.id))


def _authoritative_records(
    records: Iterable[KnowledgeRecord],
    holdout_records: Iterable[KnowledgeRecord],
) -> tuple[KnowledgeRecord, ...]:
    combined = tuple(records) + tuple(holdout_records)
    ids = [record.id for record in combined]
    duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate authoritative record IDs: {', '.join(duplicates)}")
    return tuple(sorted(combined, key=lambda record: record.id))


def _eligible(records: Iterable[KnowledgeRecord]) -> tuple[KnowledgeRecord, ...]:
    return tuple(
        sorted(
            (record for record in records if record.is_trainable()),
            key=lambda record: record.id,
        )
    )


def _record_payload(record: KnowledgeRecord, fields: Sequence[str]) -> dict[str, Any]:
    raw = {
        "id": record.id,
        "domain": record.domain,
        "title": record.title,
        "statement": record.statement,
        "summary": record.summary,
        "source_uri": record.source_uri,
        "sensitivity": record.sensitivity,
        "status": record.status,
        "effective_from": record.effective_from,
        "effective_to": record.effective_to,
        "aliases": list(record.aliases),
    }
    return {field: raw[field] for field in fields}


def _binding_from_payload(
    payload: dict[str, Any],
    selection_path: Path,
) -> BM25SelectionBinding:
    selection = payload.get("selection")
    nested = selection if isinstance(selection, dict) else {}
    index = payload.get("index")
    index_payload = index if isinstance(index, dict) else {}
    implementation = payload.get("implementation")
    implementation_payload = implementation if isinstance(implementation, dict) else {}

    status = _required_status(payload.get("status") or nested.get("status"))
    selected_config = _optional_bm25_config(
        payload.get("selected_config", nested.get("selected_config"))
    )
    exploratory = _required_bool(
        payload.get("exploratory", nested.get("exploratory")),
        "exploratory",
    )
    deployment_eligible = bool(
        payload.get("deployment_eligible", nested.get("deployment_eligible", False))
    )
    constraints = payload.get("constraints", nested.get("constraints"))
    if not isinstance(constraints, dict) or not constraints:
        raise ValueError("BM25 selection is missing constraints")
    selector_version = str(
        payload.get("selector_version")
        or implementation_payload.get("selector_version")
        or ""
    ).strip()
    if not selector_version:
        raise ValueError("BM25 selection is missing selector_version")

    validation_hash = _required_hash(
        payload.get("validation_dataset_hash") or nested.get("validation_dataset_hash"),
        "validation_dataset_hash",
    )
    source_hash = _required_hash(
        payload.get("source_snapshot_hash") or index_payload.get("source_snapshot_hash"),
        "source_snapshot_hash",
    )
    index_hash = _required_hash(
        payload.get("index_payload_hash")
        or index_payload.get("indexed_record_payload_hash"),
        "index_payload_hash",
    )
    report_hash = payload.get("calibration_report_hash")
    if report_hash is None and payload.get("schema_version") is not None and nested:
        report_hash = file_sha256(selection_path)
    calibration_hash = _required_hash(report_hash, "calibration_report_hash")

    approved_by = _optional_text(payload.get("approved_by"))
    approved_at = _optional_text(payload.get("approved_at"))
    approval_status = _approval_status(
        payload.get("approval_status"),
        approved_by=approved_by,
        approved_at=approved_at,
    )
    return BM25SelectionBinding(
        path=str(selection_path),
        file_hash=file_sha256(selection_path),
        status=status,
        approval_status=approval_status,
        approved_by=approved_by,
        approved_at=approved_at,
        exploratory=exploratory,
        deployment_eligible=deployment_eligible,
        selected_config=selected_config,
        validation_dataset_hash=validation_hash,
        source_snapshot_hash=source_hash,
        index_payload_hash=index_hash,
        calibration_report_hash=calibration_hash,
        selector_version=selector_version,
        constraints=dict(constraints),
    )


def _validate_selection_against_records(
    binding: BM25SelectionBinding,
    records: Iterable[KnowledgeRecord],
    *,
    validation_dataset_path: Path | None = None,
    calibration_report_path: Path | None = None,
) -> None:
    computed_source = source_snapshot_hash(records)
    computed_index = index_payload_hash(records)
    if binding.source_snapshot_hash != computed_source:
        raise ValueError(
            "BM25 selection source snapshot hash does not match the records being indexed"
        )
    if binding.index_payload_hash != computed_index:
        raise ValueError(
            "BM25 selection index payload hash does not match the records being indexed"
        )
    if validation_dataset_path is not None:
        if not validation_dataset_path.is_file():
            raise FileNotFoundError(
                f"BM25 selection validation dataset not found: {validation_dataset_path}"
            )
        if binding.validation_dataset_hash != file_sha256(validation_dataset_path):
            raise ValueError(
                "BM25 selection validation dataset hash does not match the approved "
                "validation artifact"
            )
    if calibration_report_path is not None:
        if not calibration_report_path.is_file():
            raise FileNotFoundError(
                f"BM25 selection calibration report not found: {calibration_report_path}"
            )
        if binding.calibration_report_hash != file_sha256(calibration_report_path):
            raise ValueError(
                "BM25 selection calibration report hash does not match the referenced report"
            )


def _require_selected_bm25_operating_point(
    binding: BM25SelectionBinding | None,
) -> BM25Config:
    if binding is None:
        raise ValueError(
            "The bm25 arm requires an explicit hash-bound selection file; "
            "raw top_k/threshold overrides are not accepted"
        )
    if binding.approval_status != "owner_approved":
        raise ValueError(
            "BM25 selection is not owner-approved; the bm25 arm cannot run"
        )
    if not binding.exploratory:
        raise ValueError("BM25 selection must preserve exploratory=true")
    if binding.status == "no_feasible_operating_point":
        raise ValueError(
            "BM25 arm rejected: owner accepted no_feasible_operating_point. "
            "Omit the bm25 arm to run base, full-context, and oracle controls."
        )
    if (
        binding.status == "experimental_non_promotable"
        and binding.deployment_eligible
    ):
        raise ValueError(
            "Experimental BM25 research selection cannot be deployment eligible"
        )
    if binding.selected_config is None:
        raise ValueError("Owner-approved BM25 selection is missing selected_config")
    return binding.selected_config


def _required_status(value: Any) -> SelectionStatus:
    status = str(value or "").strip()
    if status not in {
        "selected",
        "experimental_non_promotable",
        "no_feasible_operating_point",
    }:
        raise ValueError(
            "BM25 selection status must be selected, experimental_non_promotable, "
            "or no_feasible_operating_point"
        )
    return status  # type: ignore[return-value]


def _approval_status(
    value: Any,
    *,
    approved_by: str | None,
    approved_at: str | None,
) -> ApprovalStatus:
    """Approval is never inferred: only an explicit approval_status counts.

    ``approved_by``/``approved_at`` alone are treated as unapproved metadata;
    the owner must write ``approval_status: owner_approved`` deliberately.
    """
    del approved_by, approved_at  # metadata only; never grounds for approval
    explicit = str(value or "").strip()
    if explicit in {"owner_approved", "unapproved", "proposed"}:
        return explicit  # type: ignore[return-value]
    return "unapproved"


def _optional_bm25_config(value: Any) -> BM25Config | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("selected_config must be an object or null")
    try:
        return BM25Config(
            k1=float(value.get("k1", 1.2)),
            b=float(value.get("b", 0.75)),
            top_k=int(value["top_k"]),
            score_threshold=float(value["score_threshold"]),
        )
    except KeyError as exc:
        raise ValueError("selected_config must include top_k and score_threshold") from exc


def _required_hash(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    return digest


def _required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _highest_sensitivity(values: Sequence[Sensitivity]) -> Sensitivity | None:
    if not values:
        return None
    order: dict[Sensitivity, int] = {
        "public": 0,
        "internal_shared": 1,
        "restricted": 2,
        "secret": 3,
    }
    return max(values, key=order.__getitem__)
