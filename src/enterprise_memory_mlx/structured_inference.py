"""Format-only structured inference for the existing Qwen4B review adapter.

This experiment does not train, reopen targets, or replace historical scores.
Default is verification-only. Measured generation requires --execute.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .experiment_profiles import QWEN_4B_MODEL_ID, QWEN_4B_REVISION
from .learning_mechanics import (
    APPROVAL_RECORD_RELATIVE,
    APPROVAL_RECORD_SHA256,
    CombinedExitError,
    WiredLimitGuard,
    expected_prompt_tokens,
    load_approval,
    sha256_file,
    verify_bound_pairs,
    write_json,
)
from .learning_mechanics_eval import (
    REQUIRED_ADAPTER_SHA256,
    SOURCE_RUN_ID,
    ResourceWatchdog,
    mlx_memory_snapshot,
    persist_private_output,
    process_current_rss_bytes,
    process_rss_bytes,
    prospective_memory_policy,
    resolve_source_run,
    sanitized_traceback,
    verify_source_run,
)
from .learning_mechanics_scoring import (
    PERMITTED_DISPOSITIONS,
    PERMITTED_STATUSES,
    REQUIRED_ASSESSMENT_KEYS,
    REQUIRED_EVIDENCE_KEYS,
    REQUIRED_REVIEW_KEYS,
    score_declared_review,
    summarize_corrected_scores,
)
from .utils import sha256_json, sha256_text

PROTOCOL_ID = "senoni-deliverable-review/structured-inference-v1"
IMPLEMENTATION_ID = "qwen4b-structured-inference/v1"
HISTORICAL_EVAL_ID = "eval-20260911T212334Z-from-run-20260910T210406Z-seed42"
REQUIRED_ADAPTER = REQUIRED_ADAPTER_SHA256
PINNED_PACKAGES = {
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.2",
    "llguidance": "1.8.0",
}
MAX_OUTPUT_TOKENS = 4096
CASE_ORDER = ("DEV-001", "DEV-002", "DEV-003", "DEV-004", "DEV-005")
GENERATION_BUDGET = 10
OFFICIAL_AFTER = {
    "complete_valid_outputs": "0/5",
    "requirement_labels": "0/25",
    "dispositions": "0/5",
}
POST_HOC_BODY_ONLY = {
    "label": "post_hoc_body_only_diagnostic",
    "not_original_contract_success": True,
    "complete_valid_objects": "5/5",
    "requirement_labels": "25/25",
    "dispositions": "5/5",
}
FORBIDDEN_TRAINING_NAMES = (
    "run_experiment",
    "load_base_and_adapter",
    "enable_gradient_checkpointing",
    "TrainingAttentionPatch",
    "optimized_training_loss",
    "run_optimizer_step",
    "linear_to_lora_layers",
)
LOCAL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"
    / "snapshots"
    / QWEN_4B_REVISION
)
HISTORICAL_PREFIXES = {
    "DEV-001": "</tool_call>\n\n</tool_call>\n\n",
    "DEV-002": "</tool_call>\n\n</tool_call>\n\n",
    "DEV-003": "</tool_call>\n\n</tool_call>\n\n",
    "DEV-004": "</tool_call>\n\n<tool_call>\n\n",
    "DEV-005": "</tool_call>\n\n</tool_call>\n\n",
}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PINNED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def assert_pinned_environment() -> dict[str, str]:
    versions = package_versions()
    mismatches = {
        name: {"required": required, "installed": versions.get(name)}
        for name, required in PINNED_PACKAGES.items()
        if versions.get(name) != required
    }
    if mismatches:
        raise RuntimeError(
            "Structured inference refuses to migrate the training environment: "
            f"{mismatches}"
        )
    return {name: versions[name] or "" for name in PINNED_PACKAGES}


def assert_module_cannot_train(path: Path | None = None) -> None:
    target = path or Path(__file__)
    tree = ast.parse(target.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    forbidden = (imported | called) & set(FORBIDDEN_TRAINING_NAMES)
    if forbidden:
        raise RuntimeError(
            "structured inference imported or called training symbols: "
            f"{sorted(forbidden)}"
        )


def visible_identifiers(source: Mapping[str, Any]) -> dict[str, Any]:
    case_id = source.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError("Frozen input is missing a visible case_id")
    requirements = source.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise RuntimeError(f"{case_id} frozen input has no visible requirements")
    requirement_ids = []
    for item in requirements:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise RuntimeError(f"{case_id} requirement entry is missing a visible id")
        requirement_ids.append(item["id"])
    if len(set(requirement_ids)) != len(requirement_ids):
        raise RuntimeError(f"{case_id} visible requirement IDs are not unique")
    sources = source.get("source_index")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{case_id} frozen input has no visible source_index")
    source_ids = []
    for item in sources:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_id"), str):
            raise RuntimeError(f"{case_id} source_index entry is missing a visible source_id")
        source_ids.append(item["source_id"])
    return {
        "case_id": case_id,
        "requirement_ids": requirement_ids,
        "source_ids": list(dict.fromkeys(source_ids)),
    }


def declared_review_schema(identifiers: Mapping[str, Any]) -> dict[str, Any]:
    requirement_ids = list(identifiers["requirement_ids"])
    source_ids = list(identifiers["source_ids"])
    count = len(requirement_ids)
    evidence_item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_EVIDENCE_KEYS),
        "properties": {
            "source_id": {"type": "string", "enum": source_ids},
            "location": {"type": "string"},
            "observation": {"type": "string"},
        },
    }
    assessment = {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_ASSESSMENT_KEYS),
        "properties": {
            "requirement_id": {"type": "string", "enum": requirement_ids},
            "status": {"type": "string", "enum": sorted(PERMITTED_STATUSES)},
            "evidence": {"type": "array", "items": evidence_item},
            "gaps": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "declared_review_object",
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_REVIEW_KEYS),
        "properties": {
            "case_id": {"type": "string", "enum": [identifiers["case_id"]]},
            "requirement_assessments": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": assessment,
            },
            "missing_work": {"type": "array", "items": {"type": "string"}},
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
            "authority_conflicts": {"type": "array", "items": {"type": "string"}},
            "unsupported_claims_or_invented_requirements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "overall_disposition": {
                "type": "string",
                "enum": sorted(PERMITTED_DISPOSITIONS),
            },
            "next_actions": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
    }


def schema_contains_case_specific_labels(
    schema: Mapping[str, Any],
    approved_labels: Mapping[str, str],
    approved_disposition: str,
) -> bool:
    blob = json.dumps(schema, ensure_ascii=False)
    for requirement_id, status in approved_labels.items():
        if f'"{requirement_id}": "{status}"' in blob:
            return True
    return f'"correct_disposition": "{approved_disposition}"' in blob


def compile_review_grammar(schema: Mapping[str, Any]) -> str:
    from llguidance import LLMatcher

    grammar = LLMatcher.grammar_from_json_schema(
        schema,
        overrides={"whitespace_flexible": True},
    )
    error = LLMatcher.validate_grammar(grammar)
    if error:
        raise RuntimeError(f"llguidance rejected the review schema: {error}")
    return grammar


_LLG_TOKENIZER_CACHE: dict[int, Any] = {}
_FAST_TOKENIZER = None


def unwrap_hf_tokenizer(tokenizer: Any) -> Any:
    from transformers import PreTrainedTokenizerFast

    if isinstance(tokenizer, PreTrainedTokenizerFast):
        return tokenizer
    inner = getattr(tokenizer, "_tokenizer", None)
    if isinstance(inner, PreTrainedTokenizerFast):
        return inner
    if not LOCAL_SNAPSHOT.exists():
        raise RuntimeError(
            "llguidance requires the pinned fast tokenizer; "
            f"local snapshot missing at {LOCAL_SNAPSHOT}"
        )
    global _FAST_TOKENIZER
    if _FAST_TOKENIZER is None:
        from transformers import AutoTokenizer

        _FAST_TOKENIZER = AutoTokenizer.from_pretrained(
            str(LOCAL_SNAPSHOT),
            local_files_only=True,
            trust_remote_code=False,
        )
    return _FAST_TOKENIZER


def llguidance_tokenizer_from_hf(hf_tokenizer: Any):
    import llguidance.hf
    from transformers import PreTrainedTokenizerFast

    resolved = unwrap_hf_tokenizer(hf_tokenizer)
    if not isinstance(resolved, PreTrainedTokenizerFast):
        raise RuntimeError("llguidance.hf.from_tokenizer requires PreTrainedTokenizerFast")
    cache_key = id(resolved)
    cached = _LLG_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    built = llguidance.hf.from_tokenizer(resolved)
    _LLG_TOKENIZER_CACHE[cache_key] = built
    return built


class LLGuidanceJSONLogitsProcessor:
    """Existing llguidance mask applied through mlx-lm's logits_processor hook."""

    def __init__(self, matcher: Any, ll_tokenizer: Any) -> None:
        import llguidance.numpy

        self.matcher = matcher
        self.ll_tokenizer = ll_tokenizer
        self.started = False
        self.bitmask = llguidance.numpy.allocate_token_bitmask(1, ll_tokenizer.vocab_size)

    def __call__(self, tokens: Any, logits: Any) -> Any:
        import llguidance.mlx
        import llguidance.numpy
        import mlx.core as mx

        if self.started:
            last = tokens[-1]
            token_id = int(last.item()) if hasattr(last, "item") else int(last)
            if not self.matcher.consume_token(token_id):
                raise RuntimeError(
                    "llguidance could not consume the sampled token: "
                    f"{self.matcher.get_error() or token_id}"
                )
        else:
            self.started = True
        llguidance.numpy.fill_next_token_bitmask(self.matcher, self.bitmask, 0)
        original_ndim = int(logits.ndim)
        work = logits if original_ndim == 2 else mx.expand_dims(logits, 0)
        vocab = int(work.shape[-1])
        tok_vocab = int(self.ll_tokenizer.vocab_size)
        clipped = work[:, :tok_vocab] if vocab > tok_vocab else work
        masked = llguidance.mlx.apply_token_bitmask(clipped, self.bitmask)
        if vocab > tok_vocab:
            tail = mx.full((work.shape[0], vocab - tok_vocab), -float("inf"), dtype=work.dtype)
            masked = mx.concatenate([masked, tail], axis=1)
        return masked if original_ndim == 2 else masked[0]


def first_allowed_token_ids(
    matcher: Any, ll_tokenizer: Any, candidates: Mapping[str, int]
) -> dict[str, bool]:
    import llguidance.numpy

    bitmask = llguidance.numpy.allocate_token_bitmask(1, ll_tokenizer.vocab_size)
    llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, 0)
    words = bitmask[0]
    allowed = {}
    for name, token_id in candidates.items():
        word = token_id // 32
        bit = token_id % 32
        allowed[name] = bool(int(words[word]) & (1 << bit))
    return allowed


def dummy_valid_review(identifiers: Mapping[str, Any]) -> dict[str, Any]:
    assessments = []
    first_source = identifiers["source_ids"][0]
    for requirement_id in identifiers["requirement_ids"]:
        assessments.append(
            {
                "requirement_id": requirement_id,
                "status": "met",
                "evidence": [
                    {
                        "source_id": first_source,
                        "location": "probe:1",
                        "observation": "schema probe",
                    }
                ],
                "gaps": [],
            }
        )
    return {
        "case_id": identifiers["case_id"],
        "requirement_assessments": assessments,
        "missing_work": [],
        "missing_evidence": [],
        "authority_conflicts": [],
        "unsupported_claims_or_invented_requirements": [],
        "overall_disposition": "proceed",
        "next_actions": [],
        "rationale": "schema probe",
    }


def matcher_accepts_text(grammar: str, ll_tokenizer: Any, hf_tokenizer: Any, text: str) -> bool:
    from llguidance import LLMatcher

    matcher = LLMatcher(ll_tokenizer, grammar)
    token_ids = hf_tokenizer.encode(text.rstrip("\n"), add_special_tokens=False)
    if not matcher.consume_tokens(list(token_ids)):
        return False
    return bool(matcher.is_accepting()) and not matcher.is_error()


def historical_eval_dir(root: Path) -> Path:
    return root / "artifacts" / "qwen4b-learning-mechanics-v1" / HISTORICAL_EVAL_ID


def extract_first_object(text: str) -> tuple[dict[str, Any], int, int] | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, consumed = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value, start, start + consumed


def prepare_content_review(root: Path, verified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    packet = root / "artifacts" / "qwen4b-structured-inference-v1" / "content-review"
    packet.mkdir(parents=True, exist_ok=True)
    eval_dir = historical_eval_dir(root)
    cases = {}
    for item in verified:
        case_id = item["case_id"]
        case_dir = packet / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        after_path = eval_dir / "after" / f"{case_id}.json"
        payload = json.loads(after_path.read_text(encoding="utf-8"))
        raw = payload["raw_output"]
        parsed = payload["output"]
        extracted = extract_first_object(parsed)
        if extracted is None:
            raise RuntimeError(f"{case_id} historical after-output has no review object")
        body, start, end = extracted
        prefix = parsed[:start]
        if prefix != HISTORICAL_PREFIXES[case_id]:
            raise RuntimeError(f"{case_id} historical prefix changed")
        source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
        evidence_index = [
            {
                "source_id": row["source_id"],
                "path": row.get("path"),
                "sha256": row.get("sha256"),
                "line_ranges": row.get("line_ranges"),
                "kind": row.get("kind"),
            }
            for row in source["source_index"]
        ]
        body_path = case_dir / "review-body.json"
        prefix_path = case_dir / "excluded-prefix.json"
        evidence_path = case_dir / "local-evidence-index.json"
        checklist_path = case_dir / "reviewer-checklist.json"
        if not prefix_path.exists():
            write_json(
                prefix_path,
                {
                    "case_id": case_id,
                    "excluded_prefix": prefix,
                    "original_raw_sha256": sha256_text(raw),
                    "original_output_sha256": sha256_text(parsed),
                    "original_after_path": str(after_path.relative_to(root)),
                    "answers_not_edited": True,
                },
            )
        if not body_path.exists():
            write_json(body_path, body)
        if not evidence_path.exists():
            write_json(
                evidence_path,
                {
                    "case_id": case_id,
                    "input_sha256": item["input_sha256"],
                    "target_sha256": item["target_sha256"],
                    "local_sources_only": True,
                    "external_service_used": False,
                    "sources": evidence_index,
                },
            )
        if not checklist_path.exists():
            write_json(
                checklist_path,
                {
                    "case_id": case_id,
                    "assess": [
                        "evidence_support",
                        "invented_requirements",
                        "correct_treatment_of_missing_evidence_and_authority",
                        "harmful_or_contradictory_next_actions",
                    ],
                    "do_not": [
                        "edit_answers",
                        "reopen_targets",
                        "fabricate_human_labels",
                        "send_private_evidence_to_an_external_service",
                    ],
                    "status": "pending_authorized_reviewer",
                },
            )
        cases[case_id] = {
            "excluded_prefix": prefix,
            "review_body_sha256": sha256_file(body_path),
            "original_raw_sha256": sha256_text(raw),
            "original_output_sha256": sha256_text(parsed),
            "evidence_source_count": len(evidence_index),
        }
    status = {
        "protocol_id": PROTOCOL_ID,
        "semantic_assessment": "pending",
        "blocks_full_task_success": True,
        "does_not_block_format_only_engineering_comparison": True,
        "human_labels_fabricated": False,
        "external_service_used": False,
        "original_answers_edited": False,
        "official_result_unchanged": OFFICIAL_AFTER,
        "post_hoc_body_only_unchanged": POST_HOC_BODY_ONLY,
        "cases": cases,
    }
    write_json(packet / "status.json", status)
    readme = packet / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Local content review\n\n"
            "Authorized local reviewer only. Private review bodies and evidence "
            "paths stay on this machine.\n\n"
            "Assess evidence support, invented requirements, missing-evidence and "
            "authority treatment, and harmful or contradictory next actions.\n\n"
            "Do not edit the original raw outputs. The excluded prefixes are "
            "retained beside each extracted body.\n\n"
            "Status is pending until an authorized reviewer records an assessment. "
            "Pending does not block the format-only engineering comparison.\n",
            encoding="utf-8",
        )
    return status


def experiment_root(root: Path) -> Path:
    return root / "artifacts" / "qwen4b-structured-inference-v1"


def freeze_dir(root: Path) -> Path:
    return experiment_root(root) / "freeze"


def build_case_contracts(root: Path, verified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases = {}
    for item in verified:
        source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
        identifiers = visible_identifiers(source)
        schema = declared_review_schema(identifiers)
        if schema_contains_case_specific_labels(schema, item["labels"], item["disposition"]):
            raise RuntimeError(f"{item['case_id']} schema leaked approved labels")
        cases[item["case_id"]] = {
            "identifiers": identifiers,
            "schema": schema,
            "schema_sha256": sha256_json(schema),
            "input_sha256": item["input_sha256"],
            "target_sha256": item["target_sha256"],
            "approved_labels_used_in_schema": False,
        }
    return cases


def write_freeze(root: Path, verified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = freeze_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    versions = assert_pinned_environment()
    cases = build_case_contracts(root, verified)
    contract = {
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "model_id": QWEN_4B_MODEL_ID,
        "model_revision": QWEN_4B_REVISION,
        "adapter_sha256": REQUIRED_ADAPTER,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "generation_budget": GENERATION_BUDGET,
        "sampler": "greedy_among_grammar_permitted_continuations",
        "temperature": 0.0,
        "tools_enabled": False,
        "second_chat_template_forbidden": True,
        "target_bearing_prefill_forbidden": True,
        "case_specific_correct_labels_forbidden_in_grammar": True,
        "strict_post_generation_validation": True,
        "grammar_compliance_is_not_factual_correctness": True,
        "truncated_output_fails": True,
        "official_result_preserved": OFFICIAL_AFTER,
        "post_hoc_body_only_preserved": POST_HOC_BODY_ONLY,
        "cases": {
            case_id: {
                "schema_sha256": row["schema_sha256"],
                "identifiers": row["identifiers"],
                "input_sha256": row["input_sha256"],
                "target_sha256": row["target_sha256"],
            }
            for case_id, row in cases.items()
        },
    }
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "files": {
            rel: file_digest(root / rel)
            for rel in (
                "src/enterprise_memory_mlx/structured_inference.py",
                "src/enterprise_memory_mlx/learning_mechanics_scoring.py",
                "src/enterprise_memory_mlx/learning_mechanics_eval.py",
                "src/enterprise_memory_mlx/benchmark.py",
            )
        },
        "package_versions": versions,
    }
    write_json(target / "measurement-contract.json", contract)
    write_json(target / "implementation-identity.json", identity)
    write_json(target / "dependency-pin.json", {"required": PINNED_PACKAGES, "installed": versions})
    write_json(
        target / "schemas.json",
        {case_id: row["schema"] for case_id, row in cases.items()},
    )
    freeze = {
        "measurement_contract_sha256": sha256_file(target / "measurement-contract.json"),
        "implementation_identity_sha256": sha256_file(target / "implementation-identity.json"),
        "dependency_pin_sha256": sha256_file(target / "dependency-pin.json"),
        "schemas_sha256": sha256_file(target / "schemas.json"),
        "adapter_sha256": REQUIRED_ADAPTER,
        "optimizer_updates": 0,
    }
    write_json(target / "freeze-manifest.json", freeze)
    return {"cases": cases, "freeze": freeze, "versions": versions}


def load_freeze(root: Path) -> dict[str, Any]:
    target = freeze_dir(root)
    if not (target / "freeze-manifest.json").exists():
        raise RuntimeError("Structured inference freeze is missing")
    return {
        "manifest": json.loads((target / "freeze-manifest.json").read_text(encoding="utf-8")),
        "contract": json.loads((target / "measurement-contract.json").read_text(encoding="utf-8")),
        "schemas": json.loads((target / "schemas.json").read_text(encoding="utf-8")),
        "identity": json.loads(
            (target / "implementation-identity.json").read_text(encoding="utf-8")
        ),
    }


def render_verified_prompt(
    tokenizer: Any,
    *,
    case_id: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    expected = expected_prompt_tokens()[case_id]
    if len(token_ids) != expected:
        raise RuntimeError(f"{case_id} rendered prompt tokens {len(token_ids)} != bound {expected}")
    return {
        "prompt_text": prompt,
        "prompt_token_ids": token_ids,
        "prompt_tokens": len(token_ids),
        "prompt_sha256": sha256_text(prompt),
        "template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", ""))),
    }


def generate_constrained(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    schema: Mapping[str, Any],
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    from llguidance import LLMatcher
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    grammar = compile_review_grammar(schema)
    hf_tokenizer = unwrap_hf_tokenizer(tokenizer)
    ll_tokenizer = llguidance_tokenizer_from_hf(hf_tokenizer)
    matcher = LLMatcher(ll_tokenizer, grammar)
    processor = LLGuidanceJSONLogitsProcessor(matcher, ll_tokenizer)
    prompt_cache = make_prompt_cache(model)
    started = time.perf_counter()
    generated_ids: list[int] = []
    final = None
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=list(prompt_token_ids),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            logits_processors=[processor],
            prompt_cache=prompt_cache,
        ):
            generated_ids.append(int(response.token))
            final = response
    finally:
        prompt_cache = None
    elapsed = time.perf_counter() - started
    if final is None:
        raise RuntimeError("Constrained generation returned no response")
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    content_ids = [token for token in generated_ids if token not in eos_ids]
    raw_output = tokenizer.decode(content_ids)
    return {
        "raw_output": raw_output,
        "generated_token_ids": generated_ids,
        "completion_tokens": int(final.generation_tokens),
        "prompt_tokens": int(final.prompt_tokens),
        "finish_reason": final.finish_reason,
        "truncated": final.finish_reason == "length",
        "elapsed_seconds": elapsed,
        "peak_memory_gb": final.peak_memory,
        "grammar_accepting": bool(matcher.is_accepting()),
        "grammar_error": matcher.get_error() or None,
        "matcher_stop_reason": matcher.stop_reason(),
    }


def score_constrained_output(
    *,
    case_id: str,
    item: Mapping[str, Any],
    raw_output: str,
    truncated: bool,
) -> dict[str, Any]:
    return score_declared_review(
        case_id=case_id,
        approved_labels=item["labels"],
        approved_disposition=item["disposition"],
        raw_text=raw_output,
        truncated=truncated,
        parse_status="plain",
    )


def not_attempted_row(item: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "case_id": item["case_id"],
        "status": "not_attempted",
        "reason": reason,
        "complete_valid": False,
        "label_matches": 0,
        "label_count": len(item["labels"]),
        "disposition_agrees": False,
        "approved_labels": dict(item["labels"]),
        "model_labels": {},
        "approved_disposition": item["disposition"],
        "model_disposition": "",
        "evidence_support": "pending",
        "action_safety": "pending",
        "full_task_success": False,
    }


def run_arm(
    *,
    root: Path,
    run_dir: Path,
    arm_id: str,
    adapter_path: str | None,
    verified: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, Any],
    watchdog: ResourceWatchdog,
    mx: Any,
) -> list[dict[str, Any]]:
    from .benchmark import MLXBenchmarkBackend

    arm_dir = run_dir / arm_id
    (arm_dir / "outputs").mkdir(parents=True, exist_ok=True)
    backend = MLXBenchmarkBackend(
        QWEN_4B_MODEL_ID,
        revision=QWEN_4B_REVISION,
        adapter_path=adapter_path,
        max_context_tokens=262144,
    )
    if hasattr(backend._model, "eval"):
        backend._model.eval()
    public_rows: list[dict[str, Any]] = []
    try:
        for item in verified:
            case_id = item["case_id"]
            if watchdog.breach or watchdog.monitor_failed:
                row = not_attempted_row(item, watchdog.breach or watchdog.monitor_failed)
                persist_private_output(arm_dir / "outputs" / f"{case_id}.score.json", row)
                public_rows.append(row)
                continue
            source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
            try:
                rendered = render_verified_prompt(
                    backend._tokenizer,
                    case_id=case_id,
                    system_prompt=source["system_prompt"],
                    user_prompt=source["user_prompt"],
                )
                generated = generate_constrained(
                    model=backend._model,
                    tokenizer=backend._tokenizer,
                    prompt_token_ids=rendered["prompt_token_ids"],
                    schema=schemas[case_id],
                )
                mx.synchronize()
                scored = score_constrained_output(
                    case_id=case_id,
                    item=item,
                    raw_output=generated["raw_output"],
                    truncated=bool(generated["truncated"]),
                )
                output_payload = {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "input_sha256": item["input_sha256"],
                    "adapter_path": adapter_path,
                    "prompt_sha256": rendered["prompt_sha256"],
                    "prompt_tokens": generated["prompt_tokens"],
                    "template_sha256": rendered["template_sha256"],
                    "raw_output": generated["raw_output"],
                    "generated_token_ids": generated["generated_token_ids"],
                    "completion_tokens": generated["completion_tokens"],
                    "finish_reason": generated["finish_reason"],
                    "truncated": generated["truncated"],
                    "elapsed_seconds": generated["elapsed_seconds"],
                    "peak_memory_gb": generated["peak_memory_gb"],
                    "grammar_accepting": generated["grammar_accepting"],
                    "grammar_error": generated["grammar_error"],
                    "matcher_stop_reason": generated["matcher_stop_reason"],
                    "mlx_memory": mlx_memory_snapshot(mx),
                    "process_rss_bytes": process_current_rss_bytes(),
                    "ru_maxrss_bytes": process_rss_bytes(),
                }
                persist_private_output(arm_dir / "outputs" / f"{case_id}.json", output_payload)
                public = {
                    "case_id": case_id,
                    "status": "completed_output" if not generated["truncated"] else "failed_output",
                    "input_sha256": item["input_sha256"],
                    "prompt_sha256": rendered["prompt_sha256"],
                    "prompt_tokens": generated["prompt_tokens"],
                    "completion_tokens": generated["completion_tokens"],
                    "elapsed_seconds": generated["elapsed_seconds"],
                    "finish_reason": generated["finish_reason"],
                    "truncated": generated["truncated"],
                    "generated_token_count": len(generated["generated_token_ids"]),
                    "grammar_accepting": generated["grammar_accepting"],
                    **scored,
                    "evidence_support": "pending",
                    "action_safety": "pending",
                    "full_task_success": False,
                }
                persist_private_output(arm_dir / "outputs" / f"{case_id}.score.json", public)
                public_rows.append(public)
                print(
                    f"STRUCTURED {arm_id} {case_id} valid={public['complete_valid']} "
                    f"labels={public['label_matches']}/{public['label_count']}",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "case_id": case_id,
                    "status": "failed",
                    "error": sanitized_traceback(exc),
                }
                persist_private_output(arm_dir / "outputs" / f"{case_id}.failure.json", failure)
                row = not_attempted_row(item, f"{type(exc).__name__}: {exc}")
                row["status"] = "failed"
                persist_private_output(arm_dir / "outputs" / f"{case_id}.score.json", row)
                public_rows.append(row)
                print(f"STRUCTURED {arm_id} {case_id} failed: {type(exc).__name__}", flush=True)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    summary = summarize_corrected_scores(public_rows)
    summary["evidence_support"] = "pending"
    summary["action_safety"] = "pending"
    summary["full_task_success_cases"] = 0
    write_json(arm_dir / "summary.json", summary)
    return public_rows


def compare_arms(
    arm_a: Mapping[str, Any],
    arm_b: Mapping[str, Any],
    review_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "historical_official_unconstrained_after": OFFICIAL_AFTER,
        "historical_official_not_replaced": True,
        "post_hoc_body_only": POST_HOC_BODY_ONLY,
        "post_hoc_body_only_not_replaced": True,
        "arm_a_unadapted": arm_a,
        "arm_b_existing_adapter": arm_b,
        "constraint_alone_is_not_adapter_credit": True,
        "adapter_improvement_requires_arm_b_minus_arm_a": True,
        "semantic_assessment": review_status.get("semantic_assessment"),
        "full_task_success_claimed": False,
        "engineering_and_exact_example_fitting_only": True,
    }


def run_verification(root: Path, source_run: str) -> dict[str, Any]:
    assert_module_cannot_train()
    versions = assert_pinned_environment()
    run_dir = resolve_source_run(root, source_run)
    bound = verify_source_run(root, run_dir)
    approval = load_approval(root)
    verified = verify_bound_pairs(root, approval)
    review = prepare_content_review(root, verified)
    cases = build_case_contracts(root, verified)
    return {
        "status": "verified_only",
        "protocol_id": PROTOCOL_ID,
        "package_versions": versions,
        "adapter_sha256": bound["adapter_sha256"],
        "adapter_digest_unchanged": bound["adapter_sha256"] == REQUIRED_ADAPTER,
        "optimizer_updates": 0,
        "model_loaded": False,
        "generation_calls": 0,
        "schema_count": len(cases),
        "content_review": {
            "semantic_assessment": review["semantic_assessment"],
            "blocks_full_task_success": True,
        },
    }


def run_structured_inference(root: Path, source_run: str) -> dict[str, Any]:
    import mlx.core as mx

    assert_module_cannot_train()
    versions = assert_pinned_environment()
    run_dir_src = resolve_source_run(root, source_run)
    bound = verify_source_run(root, run_dir_src)
    load_approval(root)
    if sha256_file(root / APPROVAL_RECORD_RELATIVE) != APPROVAL_RECORD_SHA256:
        raise RuntimeError("Approval record hash does not match")
    verified = bound["verified"]
    review = prepare_content_review(root, verified)
    frozen = write_freeze(root, verified)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = experiment_root(root) / f"run-{stamp}"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        out_dir / "source-link.json",
        {
            "source_run_id": SOURCE_RUN_ID,
            "historical_eval_id": HISTORICAL_EVAL_ID,
            "adapter_sha256": REQUIRED_ADAPTER,
            "verified_adapter_sha256": bound["adapter_sha256"],
            "freeze_manifest_sha256": frozen["freeze"]["measurement_contract_sha256"],
        },
    )
    policy = prospective_memory_policy(mx)
    write_json(out_dir / "resource-policy.json", policy)
    watchdog = ResourceWatchdog(stop_rss_bytes=policy["stop_rss_bytes"])
    schemas = {case_id: row["schema"] for case_id, row in frozen["cases"].items()}
    arm_a_rows: list[dict[str, Any]] = []
    arm_b_rows: list[dict[str, Any]] = []
    status = "completed"
    stop_reason = None
    guard = WiredLimitGuard(mx, policy["selected_limit_bytes"])
    try:
        guard.install()
        with contextlib.suppress(Exception):
            mx.set_cache_limit(min(8 * 2**30, policy["selected_limit_bytes"] // 4))
        watchdog.start()
        if watchdog.monitor_failed:
            raise RuntimeError(f"monitoring_failed:{watchdog.monitor_failed}")
        print("STRUCTURED_ARM_A_START", flush=True)
        arm_a_rows = run_arm(
            root=root,
            run_dir=out_dir,
            arm_id="arm-a-unadapted",
            adapter_path=None,
            verified=verified,
            schemas=schemas,
            watchdog=watchdog,
            mx=mx,
        )
        if sha256_file(Path(bound["adapter_dir"]) / "adapters.safetensors") != REQUIRED_ADAPTER:
            raise RuntimeError("Adapter digest changed before arm B")
        print("STRUCTURED_ARM_B_START", flush=True)
        arm_b_rows = run_arm(
            root=root,
            run_dir=out_dir,
            arm_id="arm-b-adapter",
            adapter_path=bound["adapter_dir"],
            verified=verified,
            schemas=schemas,
            watchdog=watchdog,
            mx=mx,
        )
        if sha256_file(Path(bound["adapter_dir"]) / "adapters.safetensors") != REQUIRED_ADAPTER:
            raise RuntimeError("Adapter digest changed after arm B")
    except Exception as exc:
        status = "failed"
        stop_reason = f"{type(exc).__name__}: {exc}"
        write_json(out_dir / "run-failure.json", sanitized_traceback(exc))
        raise
    finally:
        watchdog.stop()
        with contextlib.suppress(CombinedExitError):
            guard.uninstall()
    if len(arm_a_rows) != 5 or len(arm_b_rows) != 5:
        status = "partial"
    arm_a_summary = summarize_corrected_scores(arm_a_rows) if arm_a_rows else {}
    arm_b_summary = summarize_corrected_scores(arm_b_rows) if arm_b_rows else {}
    for summary in (arm_a_summary, arm_b_summary):
        if summary:
            summary["evidence_support"] = "pending"
            summary["action_safety"] = "pending"
            summary["full_task_success_cases"] = 0
    comparison = compare_arms(arm_a_summary, arm_b_summary, review)
    write_json(out_dir / "comparison.json", comparison)
    generation_calls = sum(
        1
        for row in [*arm_a_rows, *arm_b_rows]
        if row.get("status") in {"completed_output", "failed_output", "failed"}
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": status,
        "stop_reason": stop_reason,
        "source_run_id": SOURCE_RUN_ID,
        "historical_eval_id": HISTORICAL_EVAL_ID,
        "adapter_sha256": REQUIRED_ADAPTER,
        "adapter_digest_unchanged": True,
        "optimizer_updates": 0,
        "generation_calls": generation_calls,
        "generation_budget": GENERATION_BUDGET,
        "package_versions": versions,
        "freeze_manifest_sha256": frozen["freeze"]["measurement_contract_sha256"],
        "comparison_sha256": sha256_file(out_dir / "comparison.json"),
        "content_review_semantic_assessment": review["semantic_assessment"],
        "official_result_unchanged": OFFICIAL_AFTER,
        "post_hoc_body_only_unchanged": POST_HOC_BODY_ONLY,
        "full_task_success_claimed": False,
        "deployment_authorized": False,
        "new_training_authorized": False,
        "generalization_claimed": False,
        "watchdog_samples": len(watchdog.samples),
        "watchdog_breach": watchdog.breach,
    }
    write_json(out_dir / "manifest.json", manifest)
    return {
        "status": status,
        "run_dir": str(out_dir),
        "manifest": manifest,
        "comparison": comparison,
        "optimizer_updates": 0,
        "adapter_sha256": REQUIRED_ADAPTER,
    }
