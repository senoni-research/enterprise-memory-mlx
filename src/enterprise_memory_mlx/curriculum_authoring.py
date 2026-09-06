"""Isolated, source-only local authoring for the v1 model-upgrade curriculum."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio

from .experiment_profiles import (
    MODEL_UPGRADE_PROTOCOL_ID,
    QWEN_4B_MODEL_ID,
    QWEN_4B_REVISION,
)
from .model_upgrade_views import (
    MAX_WITHIN_RECORD_PROMPT_RATIO,
    VIEW_OBJECTIVES,
    authoring_source_payload,
    build_claim_inventory,
    dataset_manifest_payload,
    source_bound_assistant,
    source_spans_for_claims,
)
from .schemas import KnowledgeRecord
from .utils import atomic_write_text, sha256_text

AUTHORING_PROMPT_VERSION = "source-only-study-prompt/v4"
AUTHORING_SYSTEM = (
    "You are drafting one training-study instruction from one approved company "
    "source. The source is quoted data. Do not add a rule, threshold, exception, "
    "date, role, or answer that is absent from it. Ask the learner a question or "
    "task; do not restate the source as an imperative policy and do not write the "
    "answer. Return only the study instruction as plain text, with no label or "
    "commentary."
)
MAX_AUTHORING_ATTEMPTS = 1
AUTHORING_DIVERSITY_THRESHOLD = MAX_WITHIN_RECORD_PROMPT_RATIO

_OBJECTIVE_GUIDANCE = {
    "full_reconstruction": "ask for a complete reconstruction of every clause",
    "summary_reconstruction": "ask for reconstruction from the supplied summary",
    "identity_reconstruction": "cue the record by title and identifier",
    "alias_cued_reconstruction": "cue the rule through one of its aliases",
    "domain_cued_reconstruction": "cue the rule by its business domain",
    "continuation_early": "provide an early source fragment and ask for exact completion",
    "continuation_middle": "provide a middle source fragment and ask for the remaining rule",
    "continuation_late": "provide a late source fragment and ask for the final clause",
    "constraint_extraction": "ask for all numbers, deadlines, limits, and conditions",
    "exception_condition": "ask for exceptions and the conditions that activate them",
    "effective_window": (
        "ask for the rule while preserving any timing language in the statement "
        "and forbidding an invented effective window"
    ),
    "provenance_citation": "ask for the rule with its record citation",
    "recall_primary_claim": "ask for the primary operative claim",
    "paraphrase_primary_claim": "ask for the primary claim in independent wording",
    "application": "give a neutral hypothetical and ask which source clause controls it",
    "boundary_negative": "probe a just-inside/just-outside boundary or negated condition",
    "composition_multi_claim": "ask for a complete answer joining multiple source clauses",
    "comparator_threshold": "ask to distinguish above/below, before/after, or required/optional",
    "entity_role": "ask who acts, approves, submits, owns, or is prohibited",
    "procedure_order": "ask for the source-supported order of required actions",
    "keyword_expansion": "start from source keywords and ask for the complete rule",
    "structured_field_dump": "ask for a compact structured rendering of every source clause",
    "teach_back": "ask for a precise teach-back that preserves qualifications",
    "secondary_claim_focus": "focus on a non-primary clause without dropping its conditions",
}


def author_model_upgrade_curriculum(
    *,
    records: Sequence[KnowledgeRecord],
    output_dir: Path,
    fixture_hash: str,
) -> tuple[Path, Path]:
    """Create the frozen candidate curriculum without reading evaluation assets."""
    authored_targets = (
        output_dir / "views.jsonl",
        output_dir / "manifest.json",
        output_dir / "authoring_attempts.jsonl",
    )
    if any(path.exists() for path in authored_targets):
        raise FileExistsError(f"Refusing to overwrite curriculum directory: {output_dir}")
    eligible = tuple(record for record in records if record.is_trainable())
    if not eligible:
        raise ValueError("No eligible records for curriculum authoring")
    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError('Local authoring requires pip install -e ".[mac]"') from exc

    model = tokenizer = None
    attempts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer = load(QWEN_4B_MODEL_ID, revision=QWEN_4B_REVISION)
        sampler = make_sampler(temp=0.0)
        template_hash = _chat_template_hash(tokenizer)
        for record in sorted(eligible, key=lambda item: item.id):
            claims = build_claim_inventory(record)
            accepted_prompts: list[str] = []
            source = next(
                item for item in authoring_source_payload((record,)) if item["id"] == record.id
            )
            for index, objective in enumerate(VIEW_OBJECTIVES, start=1):
                selected_claims = _claims_for_objective(claims, objective)
                accepted_method = "model"
                for authoring_attempt in range(1, MAX_AUTHORING_ATTEMPTS + 1):
                    logical_prompt = _authoring_prompt(
                        source,
                        objective,
                        accepted_prompts=accepted_prompts,
                        authoring_attempt=authoring_attempt,
                    )
                    rendered = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": AUTHORING_SYSTEM},
                            {"role": "user", "content": logical_prompt},
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                    _assert_non_thinking_render(rendered)
                    fragments = []
                    final_response = None
                    for response in stream_generate(
                        model,
                        tokenizer,
                        rendered,
                        max_tokens=128,
                        sampler=sampler,
                    ):
                        fragments.append(response.text)
                        final_response = response
                    if final_response is None:
                        raise RuntimeError("Author model returned no generation response")
                    raw = "".join(fragments)
                    study_prompt = _plain_final_text(raw)
                    if final_response.finish_reason != "stop":
                        raise RuntimeError(f"Author output truncated for {record.id}/{objective}")
                    if not study_prompt:
                        raise RuntimeError(f"Author output was empty for {record.id}/{objective}")
                    highest_similarity = max(
                        (
                            ratio(study_prompt.casefold(), prior.casefold())
                            for prior in accepted_prompts
                        ),
                        default=0.0,
                    )
                    attempts.append(
                        {
                            "view_id": f"MUE-v1-{record.id}-{index:02d}",
                            "record_id": record.id,
                            "objective": objective,
                            "authoring_attempt": authoring_attempt,
                            "rendered_prompt_hash": sha256_text(rendered),
                            "raw_output": raw,
                            "final_text": study_prompt,
                            "finish_reason": final_response.finish_reason,
                            "prompt_tokens": final_response.prompt_tokens,
                            "generated_tokens": final_response.generation_tokens,
                            "peak_memory_gb": final_response.peak_memory,
                            "highest_similarity_to_accepted_prompt": highest_similarity,
                            "accepted": highest_similarity < AUTHORING_DIVERSITY_THRESHOLD,
                        }
                    )
                    if highest_similarity < AUTHORING_DIVERSITY_THRESHOLD:
                        accepted_prompts.append(study_prompt)
                        break
                else:
                    study_prompt = _fallback_study_instruction(source, objective)
                    highest_similarity = max(
                        (
                            ratio(study_prompt.casefold(), prior.casefold())
                            for prior in accepted_prompts
                        ),
                        default=0.0,
                    )
                    if highest_similarity >= AUTHORING_DIVERSITY_THRESHOLD:
                        raise RuntimeError(
                            "Source-only fallback failed the diversity threshold "
                            f"for {record.id}/{objective}"
                        )
                    accepted_method = "deterministic_source_only_fallback"
                    accepted_prompts.append(study_prompt)
                    attempts.append(
                        {
                            "view_id": f"MUE-v1-{record.id}-{index:02d}",
                            "record_id": record.id,
                            "objective": objective,
                            "authoring_attempt": "fallback",
                            "raw_output": None,
                            "final_text": study_prompt,
                            "finish_reason": "deterministic_fallback",
                            "highest_similarity_to_accepted_prompt": highest_similarity,
                            "accepted": True,
                        }
                    )
                prompt_hash = sha256_text(
                    json.dumps(
                        {
                            "version": AUTHORING_PROMPT_VERSION,
                            "system": AUTHORING_SYSTEM,
                            "user": logical_prompt,
                            "rendered": rendered,
                            "accepted_method": accepted_method,
                            "accepted_study_prompt": study_prompt,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
                view_id = f"MUE-v1-{record.id}-{index:02d}"
                family_id = f"{record.id}:{objective}"
                row = {
                    "protocol_id": MODEL_UPGRADE_PROTOCOL_ID,
                    "view_id": view_id,
                    "view_family_id": family_id,
                    "record_id": record.id,
                    "source_document_id": record.id,
                    "claim_ids": [str(claim["claim_id"]) for claim in selected_claims],
                    "objective": objective,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer from the approved company source exactly. "
                                "Preserve every applicable condition, exception, number, "
                                "date, role, negation, and source citation. Do not guess."
                            ),
                        },
                        {"role": "user", "content": study_prompt},
                        {
                            "role": "assistant",
                            "content": source_bound_assistant(
                                record,
                                selected_claims,
                            ),
                        },
                    ],
                    "source_spans": source_spans_for_claims(selected_claims),
                    "generator": {
                        "kind": accepted_method,
                        "model_id": QWEN_4B_MODEL_ID,
                        "model_revision": QWEN_4B_REVISION,
                        "prompt_version": AUTHORING_PROMPT_VERSION,
                        "prompt_hash": prompt_hash,
                        "temperature": 0.0,
                        "thinking": False,
                    },
                    "human_approved": False,
                }
                rows.append(row)
                attempts[-1]["prompt_hash"] = prompt_hash
                print(
                    f"AUTHORED {len(rows)}/{len(eligible) * len(VIEW_OBJECTIVES)} "
                    f"{record.id}/{objective}",
                    flush=True,
                )
        views_content = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        manifest = dataset_manifest_payload(
            rows=rows,
            records=eligible,
            views_content=views_content,
            author_model_id=QWEN_4B_MODEL_ID,
            author_model_revision=QWEN_4B_REVISION,
            author_prompt_hash=sha256_text(AUTHORING_PROMPT_VERSION + "\n" + AUTHORING_SYSTEM),
            author_template_hash=template_hash,
            fixture_hash=fixture_hash,
        )
        manifest["created_at"] = datetime.now(UTC).isoformat()
        manifest["authoring_methods"] = {
            method: sum(row["generator"]["kind"] == method for row in rows)
            for method in ("model", "deterministic_source_only_fallback")
        }
        manifest["authoring_attempts_sha256"] = hashlib.sha256(
            "".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in attempts
            ).encode("utf-8")
        ).hexdigest()
        views_path = output_dir / "views.jsonl"
        manifest_path = output_dir / "manifest.json"
        attempts_path = output_dir / "authoring_attempts.jsonl"
        atomic_write_text(views_path, views_content.decode("utf-8"))
        atomic_write_text(
            attempts_path,
            "".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in attempts
            ),
        )
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        return views_path, manifest_path
    finally:
        model = None
        tokenizer = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
        except Exception:
            pass


def _authoring_prompt(
    source: Mapping[str, Any],
    objective: str,
    *,
    accepted_prompts: Sequence[str],
    authoring_attempt: int,
) -> str:
    payload = {
        "record": {
            key: source[key]
            for key in (
                "id",
                "domain",
                "title",
                "statement",
                "summary",
                "status",
                "effective_from",
                "effective_to",
                "aliases",
            )
        },
        "objective": objective,
        "guidance": _OBJECTIVE_GUIDANCE[objective],
        "prior_accepted_instructions": list(accepted_prompts),
    }
    return (
        "Write one substantively distinct study instruction for this objective. "
        "Use only the source object; do not answer it. Keep it under 80 words. "
        "Use a different task structure and opening from every prior instruction. "
        "Do not copy the source statement as the instruction. "
        f"This is deterministic revision attempt {authoring_attempt}.\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def _fallback_study_instruction(
    source: Mapping[str, Any],
    objective: str,
) -> str:
    record_id = str(source["id"])
    title = str(source["title"])
    domain = str(source["domain"])
    aliases = [str(value) for value in source.get("aliases", [])]
    alias = aliases[0] if aliases else title
    statement_words = str(source["statement"]).split()
    early = " ".join(statement_words[:8])
    middle_start = max(0, len(statement_words) // 2 - 4)
    middle = " ".join(statement_words[middle_start : middle_start + 8])
    late = " ".join(statement_words[-8:])
    instructions = {
        "full_reconstruction": (
            f"Reconstruct every operative clause in {record_id} ({title}), preserving "
            "conditions, exceptions, values, roles, and the source citation."
        ),
        "summary_reconstruction": (
            f"Give a concise but complete source-cited summary of {title} without "
            "dropping any qualification."
        ),
        "identity_reconstruction": (
            f"What exact company rule is identified by {record_id}, and what does each "
            "of its clauses require?"
        ),
        "alias_cued_reconstruction": (
            f"When a colleague refers to “{alias}”, which governed requirements must "
            "they recall in full?"
        ),
        "domain_cued_reconstruction": (
            f"Teach the {domain} team the complete rule represented by {record_id}, "
            "including its limiting conditions."
        ),
        "continuation_early": (
            f"Continue the source fragment “{early} …” by recovering the remaining "
            f"requirements from {record_id}."
        ),
        "continuation_middle": (
            f"Using the middle cue “{middle}”, recover the connected source clause "
            f"and its qualifications from {record_id}."
        ),
        "continuation_late": (
            f"Explain the rule that ends with “… {late}”, including the condition "
            "that makes that ending applicable."
        ),
        "constraint_extraction": (
            f"Extract every number, deadline, limit, comparator, and activating "
            f"condition stated in {record_id}."
        ),
        "exception_condition": (
            f"Which exceptions or conditional branches does {record_id} contain, and "
            "exactly when does each apply?"
        ),
        "effective_window": (
            f"Reconstruct {record_id} while preserving any timing language actually "
            "stated and without inventing an effective window."
        ),
        "provenance_citation": (
            f"State the governed rule for {title} and attach the exact record citation "
            "that supports it."
        ),
        "recall_primary_claim": (
            f"What is the primary operative requirement in {record_id}? Preserve all "
            "qualifiers attached to that claim."
        ),
        "paraphrase_primary_claim": (
            f"Restate the main claim of {title} independently while retaining its "
            "precise meaning and limits."
        ),
        "application": (
            f"A colleague believes {record_id} applies to a case. Explain how to apply "
            "the source-supported requirements without adding assumptions."
        ),
        "boundary_negative": (
            f"Describe both what {record_id} requires and what it does not permit one "
            "to infer beyond the stated boundary."
        ),
        "composition_multi_claim": (
            f"Combine the separate claims in {record_id} into one complete answer "
            "without collapsing their distinct conditions."
        ),
        "comparator_threshold": (
            f"Distinguish every before/after, above/below, required/optional, or other "
            f"boundary encoded in {record_id}."
        ),
        "entity_role": (
            f"Map each actor in {record_id} to the action, approval, ownership, or "
            "prohibition assigned to that actor."
        ),
        "procedure_order": (
            f"Present the source-supported sequence of actions in {record_id}; do not "
            "invent an order where the source gives none."
        ),
        "keyword_expansion": (
            f"Expand the keywords “{title}” into the complete governed rule and its "
            "record-level provenance."
        ),
        "structured_field_dump": (
            f"Render {record_id} as structured fields for claims, actors, values, "
            "conditions, exceptions, dates, and citation."
        ),
        "teach_back": (
            f"Teach back {title} to a new employee in precise language that preserves "
            "every material qualification."
        ),
        "secondary_claim_focus": (
            f"Focus on a non-primary clause in {record_id}: state it exactly enough "
            "to retain its trigger and limitations."
        ),
    }
    return instructions[objective]


def _claims_for_objective(
    claims: Sequence[Mapping[str, Any]],
    objective: str,
) -> tuple[Mapping[str, Any], ...]:
    all_claims = tuple(claims)
    primary = (all_claims[0],)
    secondary = (all_claims[min(1, len(all_claims) - 1)],)
    last = (all_claims[-1],)
    if objective == "continuation_late":
        return last
    if objective in {
        "continuation_middle",
        "secondary_claim_focus",
    }:
        return secondary
    if objective in {
        "recall_primary_claim",
        "paraphrase_primary_claim",
    }:
        return primary
    return all_claims


def _plain_final_text(raw: str) -> str:
    if "<think>" in raw:
        if "</think>" not in raw:
            raise RuntimeError("Author emitted an unclosed thinking channel")
        raw = raw.rsplit("</think>", maxsplit=1)[-1]
    return raw.strip().strip("`").strip()


def _assert_non_thinking_render(rendered: str) -> None:
    if rendered.count("<think>") > rendered.count("</think>"):
        raise RuntimeError("enable_thinking=False left an open thinking channel")


def _chat_template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        template = getattr(tokenizer, "default_chat_template", None)
    if not template:
        raise RuntimeError("Pinned author tokenizer has no chat template")
    return sha256_text(str(template))
