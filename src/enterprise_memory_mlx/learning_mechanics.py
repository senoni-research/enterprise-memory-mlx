"""Dedicated Qwen4B learning-mechanics path for senoni-deliverable-review.

This is not the smoke or model-upgrade-exploratory/v1 runtime. Frozen
acquisition profiles are not modified.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition_runtime import (
    NonThinkingTokenizer,
    _memory_snapshot,
    _swap_snapshot,
    _verify_template_policy,
    build_target_coverage,
)
from .acquisition_training import materialize_model_snapshot
from .benchmark import MLXBenchmarkBackend
from .experiment_profiles import (
    QWEN_4B_MODEL_ID,
    QWEN_4B_PROFILE,
    QWEN_4B_REVISION,
    model_profile,
)
from .utils import atomic_write_text, sha256_text

PROTOCOL_ID = "senoni-deliverable-review/qwen4b-learning-mechanics-v1"
APPROVAL_RECORD_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/"
    "supervision-candidate-v1/owner-approval-development-reference-v1.json"
)
APPROVAL_RECORD_SHA256 = "af6a26c0cf18f050751410cabd29d355722717ee4cd237f8c40ade9dbe610371"
AUTHORIZATION_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/"
    "qwen4b-learning-mechanics-v1/execution-authorization.json"
)
RETRY_AUTHORIZATION_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/"
    "qwen4b-learning-mechanics-v1/retry-authorization-v1.json"
)
RETRY_AUTHORIZATION_SHA256 = "7a0cd768559da9f789db6bb6e0a4807f38a4866982ffc1ab565f281f5f5e3ddc"
MEMORY_OPTIMIZATION_AUTHORIZATION_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/"
    "qwen4b-learning-mechanics-v1/memory-optimization-authorization-v1.json"
)
MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256 = (
    "ae1a7b983307546b5c9240c514c1241c20518d1d65fed6070cc1aa0ba67acf75"
)
PACK_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/"
    "supervision-candidate-v1"
)
REHEARSAL_V2_RELATIVE = (
    "knowledge/private/company-task-specialization/senoni-deliverable-review/rehearsal-v2"
)
REHEARSAL_V2_MANIFEST_SHA256 = (
    "8aebaa3fb5e1752b9942c479db1d201184cd0acb95204e8223689c97fc13d5b1"
)
APPLICATION_MEMORY_LIMIT_GIB = 96
PROPOSED_MAX_SEQ_LENGTH = 35406
PREFLIGHT_CASE_ORDER = ("DEV-004", "DEV-002", "DEV-004")
TRAIN_CASE_ORDER = ("DEV-001", "DEV-002", "DEV-003", "DEV-004", "DEV-005")
EXPECTED_COMPLETE_TOKENS = {
    "DEV-001": 13980,
    "DEV-002": 8183,
    "DEV-003": 21201,
    "DEV-004": 35406,
    "DEV-005": 27252,
}
PAD_TO = 32
HARMFUL_ACTION_MARKERS = (
    "execute the experiment",
    "run the models",
    "run the model",
    "construct a supervision candidate",
    "proceed despite unresolved authority",
    "unknown_oos execution",
    "production enforcement",
    "load or stress testing",
)

FROZEN_SETTINGS = {
    "protocol_id": PROTOCOL_ID,
    "model_id": QWEN_4B_MODEL_ID,
    "model_revision": QWEN_4B_REVISION,
    "rank": 16,
    "scale": 2.0,
    "dropout": 0.05,
    "num_layers": 36,
    "target_modules": list(QWEN_4B_PROFILE.target_modules),
    "expected_target_count": QWEN_4B_PROFILE.expected_target_count,
    "gradient_checkpointing": True,
    "optimizer": "adamw",
    "learning_rate": 5e-5,
    "learning_rate_schedule": "constant",
    "weight_decay": 0.01,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "examples_per_epoch": 5,
    "epochs": 8,
    "measured_optimizer_updates": 40,
    "max_seq_length": PROPOSED_MAX_SEQ_LENGTH,
    "thinking": False,
    "mask_prompt": True,
    "do_not_inherit_smoke_warmup_cosine": True,
}

_GRAD_CHECKPOINTED_LAYER_TYPES: set[type[Any]] = set()


class MechanicsTokenizer(NonThinkingTokenizer):
    """Non-thinking template, with tokenize=True implemented as text+encode.

    This keeps loader-produced IDs aligned with the preparation-script counts
    that matched sealed rehearsal-v2 prompt tokens.
    """

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        requested = kwargs.pop("enable_thinking", False)
        if requested is not False:
            raise ValueError("Learning-mechanics training requires enable_thinking=False")
        tokenize = kwargs.pop("tokenize", True)
        kwargs.pop("return_dict", False)
        text = self._tokenizer.apply_chat_template(
            *args,
            enable_thinking=False,
            tokenize=False,
            return_dict=False,
            **kwargs,
        )
        if not tokenize:
            return text
        return self._tokenizer.encode(text, add_special_tokens=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_approval(root: Path) -> dict[str, Any]:
    path = root / APPROVAL_RECORD_RELATIVE
    if sha256_file(path) != APPROVAL_RECORD_SHA256:
        raise RuntimeError("Approval record hash does not match the authorized binding")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bound_pairs(root: Path, approval: Mapping[str, Any]) -> list[dict[str, Any]]:
    pack = root / PACK_RELATIVE
    verified = []
    for row in approval["approved_targets"]:
        input_path = pack / row["source_input_path"]
        target_path = pack / row["path"]
        input_sha = sha256_file(input_path)
        target_sha = sha256_file(target_path)
        if input_sha != row["source_input_sha256"]:
            raise RuntimeError(f"{row['case_id']} input hash mismatch")
        if target_sha != row["sha256"]:
            raise RuntimeError(f"{row['case_id']} target hash mismatch")
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        if "proposed_target_review" not in payload:
            raise RuntimeError(f"{row['case_id']} is missing proposed_target_review")
        verified.append(
            {
                "case_id": row["case_id"],
                "input_path": str(input_path),
                "target_path": str(target_path),
                "input_sha256": input_sha,
                "target_sha256": target_sha,
                "labels": dict(row["labels"]),
                "disposition": row["disposition"],
            }
        )
    if [row["case_id"] for row in verified] != list(TRAIN_CASE_ORDER):
        raise RuntimeError("Approved target order does not match the five-case contract")
    return verified


def serialize_training_rows(
    root: Path,
    verified: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in verified:
        source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
        target = json.loads(Path(item["target_path"]).read_text(encoding="utf-8"))
        review = target["proposed_target_review"]
        if not isinstance(review, dict):
            raise RuntimeError(f"{item['case_id']} proposed_target_review is not an object")
        leaked = {
            "citation_appendix",
            "owner_approved",
            "training_eligible",
            "owner_reference_sha256",
            "candidate_id",
        } & set(review)
        if leaked:
            raise RuntimeError(
                f"{item['case_id']} assistant target leaked audit metadata keys: {sorted(leaked)}"
            )
        assistant = json.dumps(review, indent=2, ensure_ascii=False) + "\n"
        rows.append(
            {
                "case_id": item["case_id"],
                "messages": [
                    {"role": "system", "content": source["system_prompt"]},
                    {"role": "user", "content": source["user_prompt"]},
                    {"role": "assistant", "content": assistant},
                ],
                "input_sha256": item["input_sha256"],
                "target_sha256": item["target_sha256"],
            }
        )
    return rows


def preparation_token_counts(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, int]:
    messages = list(row["messages"])
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    return {
        "prompt_tokens": len(prompt_ids),
        "complete_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - len(prompt_ids),
    }


def validate_loader_row(
    *,
    tokenizer: MechanicsTokenizer,
    row: Mapping[str, Any],
    expected_prompt_tokens: int,
    max_seq_length: int,
    max_position_embeddings: int,
) -> dict[str, Any]:
    from mlx_lm.tuner.datasets import ChatDataset

    dataset = ChatDataset([{"messages": row["messages"]}], tokenizer, mask_prompt=True)
    tokens, offset = dataset.process(dataset[0])
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"{row['case_id']} loader did not produce a token list")
    prepared = preparation_token_counts(tokenizer, row)
    if len(tokens) != prepared["complete_tokens"] or offset != prepared["prompt_tokens"]:
        raise RuntimeError(
            f"{row['case_id']} loader tokenization disagrees with preparation counts: "
            f"loader_full={len(tokens)} loader_offset={offset} "
            f"prepared_full={prepared['complete_tokens']} "
            f"prepared_prompt={prepared['prompt_tokens']}"
        )
    expected_complete = EXPECTED_COMPLETE_TOKENS[row["case_id"]]
    if offset != expected_prompt_tokens:
        raise RuntimeError(
            f"{row['case_id']} prompt offset {offset} != sealed rehearsal {expected_prompt_tokens}"
        )
    if len(tokens) != expected_complete:
        raise RuntimeError(
            f"{row['case_id']} loader full length {len(tokens)} != proposed "
            f"{expected_complete}. Serialization discrepancy; refusing to crop."
        )
    if offset <= 0 or offset >= len(tokens):
        raise RuntimeError(f"{row['case_id']} assistant loss boundary is invalid")
    eos_id = tokenizer.eos_token_id
    terminator_tail = list(tokens[-2:])
    if eos_id not in terminator_tail:
        raise RuntimeError(
            f"{row['case_id']} full row does not end with EOS/im_end; tail={terminator_tail}"
        )
    if len(tokens) > max_seq_length:
        raise RuntimeError(
            f"{row['case_id']} has {len(tokens)} tokens and would be truncated "
            f"by the proposed cap {max_seq_length}"
        )
    if len(tokens) > max_position_embeddings:
        raise RuntimeError(
            f"{row['case_id']} length {len(tokens)} exceeds model max_position_embeddings "
            f"{max_position_embeddings}"
        )
    return {
        "case_id": row["case_id"],
        "loader_full_tokens": len(tokens),
        "loader_prompt_offset": offset,
        "supervised_tokens": len(tokens) - offset,
        "ends_with_eos": True,
        "terminator_tail": terminator_tail,
        "truncated": False,
        "within_positional_config": True,
        "tokens": tokens,
        "offset": offset,
    }


def build_batch(
    tokens: Sequence[int],
    offset: int,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    if len(tokens) > max_seq_length:
        raise RuntimeError("Refusing to truncate a training row")
    padded_length = 1 + PAD_TO * ((len(tokens) + PAD_TO - 1) // PAD_TO)
    if padded_length > max_seq_length:
        padded_length = len(tokens)
    padded = list(tokens) + [0] * (padded_length - len(tokens))
    return {
        "batch": [padded],
        # default_loss numbers target positions from one. The final real target
        # is therefore len(tokens) - 1; using len(tokens) would supervise the
        # first padding token whenever this row is padded.
        "lengths": [[offset, len(tokens) - 1]],
        "unpadded_length": len(tokens),
        "padded_shape": [1, padded_length],
        "padding_positions": padded_length - len(tokens),
        "prompt_positions_excluded_from_loss": offset,
        "padding_excluded_from_loss": True,
    }


def expected_prompt_tokens() -> dict[str, int]:
    return {
        "DEV-001": 12682,
        "DEV-002": 7620,
        "DEV-003": 20390,
        "DEV-004": 33494,
        "DEV-005": 25917,
    }


def recommended_working_set_bytes(mx: Any) -> int | None:
    for owner in (mx.metal, mx):
        candidate = getattr(owner, "get_recommended_max_working_set_size", None)
        if callable(candidate):
            return int(candidate())
    return None


def application_memory_limit_bytes(mx: Any) -> tuple[int, dict[str, Any]]:
    ceiling = APPLICATION_MEMORY_LIMIT_GIB * 2**30
    info = dict(mx.device_info())
    device_bytes = int(info.get("memory_size") or ceiling)
    recommended = recommended_working_set_bytes(mx)
    info_recommended = info.get("max_recommended_working_set_size")
    if recommended is None and info_recommended:
        recommended = int(info_recommended)
    limit = min(ceiling, device_bytes)
    if recommended:
        limit = min(limit, recommended)
    return limit, {
        "application_limit_gib": limit / 2**30,
        "configured_ceiling_gib": APPLICATION_MEMORY_LIMIT_GIB,
        "device_memory_gib": device_bytes / 2**30,
        "recommended_working_set_gib": None if recommended is None else recommended / 2**30,
        "device_info": info,
    }


class CombinedExitError(Exception):
    """Primary body exception plus a cleanup failure; both remain observable."""

    def __init__(self, primary: BaseException, cleanup: BaseException) -> None:
        self.primary = primary
        self.cleanup = cleanup
        super().__init__(
            f"primary {type(primary).__name__}: {primary}; "
            f"cleanup {type(cleanup).__name__}: {cleanup}"
        )


def _require_wired_limit_int(requested: Any, *, label: str) -> int:
    if requested is None:
        raise TypeError(f"{label} rejected unexpected None")
    if type(requested) is not int:
        raise TypeError(f"{label} requires int, got {type(requested).__name__}")
    return requested


def install_memory_ceiling(mx: Any, limit: int) -> Any:
    """Install a wired-limit cap that preserves the native integer return.

    This is a wired-memory setting, not a total-process allocation cap.
    The original int(None) training-run crash is a strong matching
    hypothesis for a discarded setter return, not a proven historical
    call site: that traceback was not captured.
    """
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal device is unavailable")
    limit = _require_wired_limit_int(limit, label="wired-limit ceiling")
    native = mx.set_wired_limit
    pre_entry = native(limit)
    _require_wired_limit_int(pre_entry, label="native set_wired_limit return")

    def capped(requested: Any) -> int:
        value = _require_wired_limit_int(requested, label="set_wired_limit")
        return native(min(value, limit))

    capped._native = native
    capped._pre_entry = pre_entry
    capped._protected_limit = limit
    mx.set_wired_limit = capped
    return native


def restore_memory_ceiling(mx: Any, native: Any, pre_entry: int) -> None:
    """Restore the native setter and the pre-entry numerical wired limit."""
    _require_wired_limit_int(pre_entry, label="pre-entry wired limit")
    mx.set_wired_limit = native
    native(pre_entry)


class WiredLimitGuard:
    """Scoped wired-limit wrapper that restores callable and numeric setting."""

    def __init__(self, mx: Any, limit: int) -> None:
        self.mx = mx
        self.limit = _require_wired_limit_int(limit, label="wired-limit ceiling")
        self._native = mx.set_wired_limit
        self._pre_entry: int | None = None
        self._installed = False

    def install(self) -> WiredLimitGuard:
        if not self.mx.metal.is_available():
            raise RuntimeError("MLX Metal device is unavailable")
        pre_entry = self._native(self.limit)
        self._pre_entry = _require_wired_limit_int(
            pre_entry, label="native set_wired_limit return"
        )

        def capped(requested: Any) -> int:
            value = _require_wired_limit_int(requested, label="set_wired_limit")
            return self._native(min(value, self.limit))

        self.mx.set_wired_limit = capped
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        native = self._native
        pre_entry = self._pre_entry
        self._installed = False
        restore_memory_ceiling(self.mx, native, int(pre_entry))

    def __enter__(self) -> WiredLimitGuard:
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            self.uninstall()
        except BaseException as cleanup:
            if exc is not None:
                raise CombinedExitError(exc, cleanup) from cleanup
            raise
        return False


def seed_all(seed: int, mx: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def epoch_order(seed: int) -> list[int]:
    import numpy as np

    rng = np.random.RandomState(seed)
    order = list(range(len(TRAIN_CASE_ORDER)))
    rng.shuffle(order)
    return order


def tensor_digest(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.array(value)).tobytes()).hexdigest()


def enable_gradient_checkpointing(model: Any, grad_checkpoint: Any) -> list[str]:
    """Enable MLX-LM's class-level checkpoint wrapper exactly once per type."""
    from .learning_mechanics_exact import CHECKPOINT_MARKER

    active: set[str] = set()
    for layer in model.layers:
        layer_type = type(layer)
        name = f"{layer_type.__module__}.{layer_type.__qualname__}"
        current = layer_type.__call__
        already_marked = getattr(current, CHECKPOINT_MARKER, False)
        if already_marked or layer_type in _GRAD_CHECKPOINTED_LAYER_TYPES:
            active.add(name)
            continue
        if getattr(current, "__name__", "") == "checkpointed_fn":
            raise RuntimeError(
                f"Duplicate checkpoint installation on {name}: the layer class "
                "is already wrapped and a second wrap would nest checkpointing"
            )
        grad_checkpoint(layer)
        wrapped = layer_type.__call__
        if wrapped is not current:
            with contextlib.suppress(AttributeError, TypeError):
                setattr(wrapped, CHECKPOINT_MARKER, True)
        _GRAD_CHECKPOINTED_LAYER_TYPES.add(layer_type)
        active.add(name)
    return sorted(active)


def extract_review(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def review_labels(review: Mapping[str, Any] | None) -> dict[str, str]:
    if not review:
        return {}
    labels = {}
    for item in review.get("requirement_assessments", []):
        if isinstance(item, Mapping) and item.get("requirement_id"):
            labels[str(item["requirement_id"])] = str(item.get("status", ""))
    return labels


def harmful_next_actions(review: Mapping[str, Any] | None) -> list[str]:
    if not review:
        return ["missing_review"]
    found = []
    invented = review.get("unsupported_claims_or_invented_requirements") or []
    if invented:
        found.extend(str(item) for item in invented)
    for action in review.get("next_actions") or []:
        lowered = str(action).casefold()
        if (
            "do not execute" in lowered
            or "not permission to run" in lowered
            or "not authorization to run" in lowered
        ):
            continue
        if any(marker in lowered for marker in HARMFUL_ACTION_MARKERS):
            found.append(str(action))
    return found


def score_review(
    *,
    case_id: str,
    approved_labels: Mapping[str, str],
    approved_disposition: str,
    review: Mapping[str, Any] | None,
    schema_valid: bool,
) -> dict[str, Any]:
    labels = review_labels(review)
    disposition = str(review.get("overall_disposition", "")) if review else ""
    label_matches = sum(
        labels.get(requirement_id) == status
        for requirement_id, status in approved_labels.items()
    )
    harmful = harmful_next_actions(review)
    return {
        "case_id": case_id,
        "schema_valid": schema_valid,
        "model_labels": labels,
        "approved_labels": dict(approved_labels),
        "label_matches": label_matches,
        "label_count": len(approved_labels),
        "model_disposition": disposition,
        "approved_disposition": approved_disposition,
        "disposition_agrees": disposition == approved_disposition,
        "invented_or_harmful_actions": harmful,
        "full_task_success": (
            schema_valid
            and label_matches == len(approved_labels)
            and disposition == approved_disposition
            and not harmful
        ),
    }


def load_rehearsal_before(
    root: Path, verified: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rehearsal = root / REHEARSAL_V2_RELATIVE
    manifest = rehearsal / "result-manifest.json"
    if sha256_file(manifest) != REHEARSAL_V2_MANIFEST_SHA256:
        raise RuntimeError("Sealed rehearsal-v2 manifest hash changed")
    rows = []
    for item in verified:
        path = rehearsal / "outputs" / "qwen4b" / f"{item['case_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        review = extract_review(payload.get("output"))
        rows.append(
            {
                "source": "rehearsal-v2-qwen4b",
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_memory_gb": payload.get("peak_memory_gb"),
                **score_review(
                    case_id=item["case_id"],
                    approved_labels=item["labels"],
                    approved_disposition=item["disposition"],
                    review=review,
                    schema_valid=bool(payload.get("validation", {}).get("schema_valid")),
                ),
            }
        )
    return rows


def summarize_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_matches = sum(int(row["label_matches"]) for row in rows)
    label_total = sum(int(row["label_count"]) for row in rows)
    dispositions = sum(1 for row in rows if row["disposition_agrees"])
    valid = sum(1 for row in rows if row["schema_valid"])
    insufficient = []
    not_met = []
    non_proceed = []
    referrals = []
    met_preserved = 0
    proceed_preserved = 0
    for row in rows:
        approved = row["approved_labels"]
        model = row["model_labels"]
        for requirement_id, status in approved.items():
            if status == "met" and model.get(requirement_id) == "met":
                met_preserved += 1
            if status == "insufficient_evidence":
                insufficient.append(
                    {
                        "case_id": row["case_id"],
                        "requirement_id": requirement_id,
                        "recovered": model.get(requirement_id) == "insufficient_evidence",
                    }
                )
            if status == "not_met":
                not_met.append(
                    {
                        "case_id": row["case_id"],
                        "requirement_id": requirement_id,
                        "recovered": model.get(requirement_id) == "not_met",
                    }
                )
        if row["approved_disposition"] != "proceed":
            non_proceed.append(
                {
                    "case_id": row["case_id"],
                    "approved": row["approved_disposition"],
                    "model": row["model_disposition"],
                    "recovered": row["disposition_agrees"],
                }
            )
        else:
            proceed_preserved += int(row["disposition_agrees"])
        if row["approved_disposition"] == "refer_to_source":
            referrals.append(
                {
                    "case_id": row["case_id"],
                    "recovered": row["model_disposition"] == "refer_to_source",
                }
            )
    return {
        "complete_valid_outputs": f"{valid}/5",
        "requirement_labels": f"{label_matches}/{label_total}",
        "dispositions": f"{dispositions}/5",
        "insufficient_evidence": insufficient,
        "not_met": not_met,
        "non_proceed": non_proceed,
        "authority_referral": referrals,
        "met_labels_preserved": met_preserved,
        "proceed_dispositions_preserved": proceed_preserved,
        "full_task_success_cases": sum(1 for row in rows if row["full_task_success"]),
        "training_set_fitting_only": True,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def prepare_run_directory(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        root
        / "artifacts"
        / "qwen4b-learning-mechanics-v1"
        / f"run-{stamp}-seed42"
    )
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _release_model(model: Any, mx: Any) -> None:
    del model
    gc.collect()
    try:
        mx.synchronize()
        mx.clear_cache()
    except Exception:
        pass


def run_optimizer_step(
    *,
    model: Any,
    optimizer: Any,
    mx: Any,
    nn: Any,
    default_loss: Any,
    tokens: Sequence[int],
    offset: int,
    max_seq_length: int,
    case_id: str,
    update_index: int,
) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    from .learning_mechanics_exact import (
        EXPECTED_DEV004_SUPERVISED,
        optimized_training_loss,
        supervised_hidden_span,
    )

    batch_info = build_batch(tokens, offset, max_seq_length=max_seq_length)
    batch = mx.array(batch_info["batch"])
    lengths = mx.array(batch_info["lengths"])
    span = supervised_hidden_span(offset, batch_info["unpadded_length"])
    if case_id == "DEV-004" and span["supervised_token_count"] != EXPECTED_DEV004_SUPERVISED:
        raise RuntimeError(
            "DEV-004 supervised span is "
            f"{span['supervised_token_count']}, expected {EXPECTED_DEV004_SUPERVISED}"
        )
    model.train()
    loss_and_grad = nn.value_and_grad(model, default_loss or optimized_training_loss)
    (loss, ntoks), gradients = loss_and_grad(model, batch, lengths)
    mx.eval(loss, ntoks, gradients)
    loss_value = float(loss.item())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"Non-finite loss on {case_id} update {update_index}")
    gradient_values = [value for _name, value in tree_flatten(gradients)]
    if not gradient_values:
        raise RuntimeError("No gradients produced")
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in gradient_values):
        raise RuntimeError(f"Non-finite gradients on {case_id} update {update_index}")
    if not any(bool(mx.any(value != 0).item()) for value in gradient_values):
        raise RuntimeError(f"All-zero gradients on {case_id} update {update_index}")
    before = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    after = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    changed = sum(1 for name in before if before[name] != after[name])
    if changed == 0:
        raise RuntimeError(f"Optimizer update changed no adapter tensors on {case_id}")
    return {
        "case_id": case_id,
        "update_index": update_index,
        "loss": loss_value,
        "supervised_tokens": int(ntoks.item()),
        "learning_rate": 5e-5,
        "unpadded_length": batch_info["unpadded_length"],
        "padded_shape": batch_info["padded_shape"],
        "padding_positions": batch_info["padding_positions"],
        "changed_adapter_tensors": changed,
        "supervised_hidden_span": span,
        "first_supervised_hidden_index": span["first_supervised_hidden_index"],
        "runtime_loss": "supervised_position_projection",
    }


def save_adapter(model: Any, mx: Any, path: Path) -> dict[str, str]:
    from mlx.utils import tree_flatten

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite adapter: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)
    return {name: tensor_digest(value) for name, value in weights.items()}


def reload_adapter_check(model: Any, mx: Any, path: Path, expected: Mapping[str, str]) -> None:
    from mlx.utils import tree_flatten

    for _name, value in tree_flatten(model.trainable_parameters()):
        value[:] = mx.zeros_like(value)
    model.load_weights(str(path), strict=False)
    mx.eval(model.trainable_parameters())
    reloaded = {
        name: tensor_digest(value)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    if dict(reloaded) != dict(expected):
        raise RuntimeError("Adapter save/reload integrity check failed")


def load_base_and_adapter(
    *,
    snapshot_path: str,
    seed: int,
    mx: Any,
    profile: Any,
    snapshot_config: Mapping[str, Any],
) -> tuple[Any, MechanicsTokenizer, dict[str, Any]]:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    seed_all(seed, mx)
    model, tokenizer = load(
        snapshot_path,
        tokenizer_config={"trust_remote_code": False},
    )
    wrapped = MechanicsTokenizer(tokenizer)
    template = _verify_template_policy(wrapped, profile)
    model.freeze()
    linear_to_lora_layers(
        model,
        profile.num_layers,
        {
            "rank": FROZEN_SETTINGS["rank"],
            "scale": FROZEN_SETTINGS["scale"],
            "dropout": FROZEN_SETTINGS["dropout"],
            "keys": list(FROZEN_SETTINGS["target_modules"]),
        },
    )
    coverage = build_target_coverage(model, profile, snapshot_config)
    checkpointed_layer_types = []
    if FROZEN_SETTINGS["gradient_checkpointing"]:
        checkpointed_layer_types = enable_gradient_checkpointing(model, grad_checkpoint)
    coverage["gradient_checkpointing"] = {
        "enabled": FROZEN_SETTINGS["gradient_checkpointing"],
        "class_level_wrapper_applied_once_per_layer_type": True,
        "layer_types": checkpointed_layer_types,
    }
    return model, wrapped, {
        "chat_template_hash": sha256_text(str(template)),
        "coverage": coverage,
        "checkpointed_layer_types": checkpointed_layer_types,
        "gradient_checkpointing_enabled": FROZEN_SETTINGS["gradient_checkpointing"],
    }


def generate_after_reviews(
    *,
    root: Path,
    run_dir: Path,
    verified: Sequence[Mapping[str, Any]],
    adapter_dir: Path,
) -> list[dict[str, Any]]:
    backend = MLXBenchmarkBackend(
        QWEN_4B_MODEL_ID,
        revision=QWEN_4B_REVISION,
        adapter_path=str(adapter_dir),
        max_context_tokens=262144,
    )
    results = []
    try:
        for item in verified:
            source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
            answer = backend.generate(
                system_prompt=source["system_prompt"],
                question=source["user_prompt"],
                max_tokens=4096,
            )
            parsed = extract_review(answer.output)
            schema_valid = bool(parsed and parsed.get("requirement_assessments"))
            scored = score_review(
                case_id=item["case_id"],
                approved_labels=item["labels"],
                approved_disposition=item["disposition"],
                review=parsed,
                schema_valid=schema_valid,
            )
            record = {
                "case_id": item["case_id"],
                "parse_status": answer.parse_status,
                "finish_reason": answer.finish_reason,
                "truncated": answer.truncated,
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "elapsed_seconds": answer.elapsed_seconds,
                "peak_memory_gb": answer.peak_memory_gb,
                "raw_output": answer.raw_output,
                "output": answer.output,
                "review": parsed,
                **scored,
            }
            write_json(run_dir / "after" / f"{item['case_id']}.json", record)
            results.append(record)
    finally:
        backend.close()
    return results


def run_experiment(root: Path) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from .learning_mechanics_exact import (
        IMPLEMENTATION_ID,
        TrainingAttentionPatch,
        diagnose_runtime,
        optimized_training_loss,
        run_model_equivalence,
    )

    approval = load_approval(root)
    verified = verify_bound_pairs(root, approval)
    authorization_path = root / AUTHORIZATION_RELATIVE
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization["source_approval_record_sha256"] != APPROVAL_RECORD_SHA256:
        raise RuntimeError("Execution authorization is not bound to the approved hashes")
    retry_authorization_path = root / RETRY_AUTHORIZATION_RELATIVE
    if sha256_file(retry_authorization_path) != RETRY_AUTHORIZATION_SHA256:
        raise RuntimeError("Retry authorization hash does not match the bounded retry")
    retry_authorization = json.loads(retry_authorization_path.read_text(encoding="utf-8"))
    if (
        retry_authorization.get("protocol_id") != PROTOCOL_ID
        or retry_authorization.get("source_approval_record_sha256") != APPROVAL_RECORD_SHA256
        or retry_authorization.get("automatic_further_retry_authorized") is not False
    ):
        raise RuntimeError("Retry authorization does not match the experiment boundary")
    memory_authorization_path = root / MEMORY_OPTIMIZATION_AUTHORIZATION_RELATIVE
    if sha256_file(memory_authorization_path) != MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256:
        raise RuntimeError("Memory-optimization authorization hash does not match")
    memory_authorization = json.loads(memory_authorization_path.read_text(encoding="utf-8"))
    if (
        memory_authorization.get("protocol_id") != PROTOCOL_ID
        or memory_authorization.get("data_changed") is not False
        or memory_authorization.get("objective_changed") is not False
    ):
        raise RuntimeError("Memory-optimization authorization is not an exact-runtime revision")
    rows = serialize_training_rows(root, verified)
    run_dir = prepare_run_directory(root)
    print(f"LEARNING_MECHANICS_RUN_DIR {run_dir}", flush=True)
    write_json(run_dir / "frozen-settings.json", FROZEN_SETTINGS)
    write_json(
        run_dir / "bound-pairs.json",
        {
            "approval_sha256": APPROVAL_RECORD_SHA256,
            "authorization_sha256": sha256_file(authorization_path),
            "retry_authorization_sha256": RETRY_AUTHORIZATION_SHA256,
            "memory_optimization_authorization_sha256": MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256,
            "pairs": [
                {key: item[key] for key in ("case_id", "input_sha256", "target_sha256")}
                for item in verified
            ],
        },
    )
    export_path = run_dir / "train.jsonl"
    atomic_write_text(
        export_path,
        "".join(
            json.dumps(
                {"case_id": row["case_id"], "messages": row["messages"]},
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
    )

    snapshot_path = materialize_model_snapshot(QWEN_4B_MODEL_ID, QWEN_4B_REVISION)
    profile = model_profile(QWEN_4B_MODEL_ID, QWEN_4B_REVISION)
    snapshot_config = json.loads((Path(snapshot_path) / "config.json").read_text(encoding="utf-8"))
    max_position = int(snapshot_config["max_position_embeddings"])
    if max_position < PROPOSED_MAX_SEQ_LENGTH:
        raise RuntimeError("Proposed cap exceeds model positional configuration")

    memory_limit, memory_policy = application_memory_limit_bytes(mx)
    original_limit = install_memory_ceiling(mx, memory_limit)
    started = time.perf_counter()
    preflight_report: dict[str, Any] = {}
    model = tokenizer = optimizer = None
    processed: dict[str, Any] = {}
    attention_patch = TrainingAttentionPatch()
    try:
        diagnosis = diagnose_runtime(mx)
        write_json(run_dir / "runtime-diagnosis.json", diagnosis)
        print(
            f"RUNTIME {diagnosis['versions']} "
            f"sdpa={diagnosis['sdpa']['qualname']} "
            f"attn_dtype={diagnosis['short_sequence_attention']['activation_dtype']}",
            flush=True,
        )
        model, tokenizer, loaded = load_base_and_adapter(
            snapshot_path=snapshot_path,
            seed=7,
            mx=mx,
            profile=profile,
            snapshot_config=snapshot_config,
        )
        equivalence = run_model_equivalence(
            model=model,
            mx=mx,
            nn=nn,
            tokens=list(range(80)),
            offset=40,
            max_seq_length=PROPOSED_MAX_SEQ_LENGTH,
            build_batch=build_batch,
        )
        write_json(run_dir / "equivalence-report.json", equivalence)
        print(f"EQUIVALENCE passed={equivalence['passed']}", flush=True)
        _release_model(model, mx)
        model = tokenizer = None
        if not equivalence["passed"]:
            write_json(
                run_dir / "run-manifest.json",
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "status": "failed_equivalence",
                    "implementation_id": IMPLEMENTATION_ID,
                    "training_eligible": False,
                    "source_approval_sha256": APPROVAL_RECORD_SHA256,
                    "execution_authorization_sha256": sha256_file(authorization_path),
                    "retry_authorization_sha256": RETRY_AUTHORIZATION_SHA256,
                    "memory_optimization_authorization_sha256": (
                        MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256
                    ),
                    "equivalence": equivalence,
                    "runtime_diagnosis": diagnosis,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            return {
                "status": "failed_equivalence",
                "run_dir": str(run_dir),
                "equivalence": equivalence,
            }
        attention_patch.install()
        try:
            model, tokenizer, loaded = load_base_and_adapter(
                snapshot_path=snapshot_path,
                seed=42,
                mx=mx,
                profile=profile,
                snapshot_config=snapshot_config,
            )
            prompt_expected = expected_prompt_tokens()
            processed = {
                row["case_id"]: validate_loader_row(
                    tokenizer=tokenizer,
                    row=row,
                    expected_prompt_tokens=prompt_expected[row["case_id"]],
                    max_seq_length=PROPOSED_MAX_SEQ_LENGTH,
                    max_position_embeddings=max_position,
                )
                for row in rows
            }
            write_json(
                run_dir / "loader-validation.json",
                {
                    "proposed_max_seq_length": PROPOSED_MAX_SEQ_LENGTH,
                    "max_position_embeddings": max_position,
                    "expected_complete_tokens": EXPECTED_COMPLETE_TOKENS,
                    "rows": [
                        {key: value[key] for key in value if key not in {"tokens", "offset"}}
                        for value in processed.values()
                    ],
                },
            )
            write_json(run_dir / "target-coverage-preflight.json", loaded["coverage"])
            optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
            preflight_steps = []
            preflight_started = time.perf_counter()
            started_memory = _memory_snapshot(mx)
            started_swap = _swap_snapshot()
            mx.reset_peak_memory()
            for index, case_id in enumerate(PREFLIGHT_CASE_ORDER, start=1):
                row = processed[case_id]
                step = run_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    mx=mx,
                    nn=nn,
                    default_loss=optimized_training_loss,
                    tokens=row["tokens"],
                    offset=row["offset"],
                    max_seq_length=PROPOSED_MAX_SEQ_LENGTH,
                    case_id=case_id,
                    update_index=index,
                )
                step["peak_memory_gb"] = mx.get_peak_memory() / 1e9
                preflight_steps.append(step)
                print(
                    f"PREFLIGHT {index}/3 {case_id} loss={step['loss']:.6f} "
                    f"shape={step['padded_shape']} peak_gb={step['peak_memory_gb']:.2f}",
                    flush=True,
                )
            disposable = run_dir / "preflight" / "preflight-adapters.safetensors"
            hashes = save_adapter(model, mx, disposable)
            reload_adapter_check(model, mx, disposable, hashes)
            preflight_report = {
                "schema_version": 1,
                "status": "passed",
                "disposable": True,
                "configured_path_feasible": True,
                "not_a_statement_about_qwen4b_or_supervised_learning_in_general": True,
                "steps": preflight_steps,
                "save_reload_parity": True,
                "elapsed_seconds": time.perf_counter() - preflight_started,
                "memory": {
                    **memory_policy,
                    "before": started_memory,
                    "after": _memory_snapshot(mx),
                    "swap_before": started_swap,
                    "swap_after": _swap_snapshot(),
                },
            }
            write_json(run_dir / "preflight" / "preflight-report.json", preflight_report)
        except Exception as exc:
            preflight_report = {
                "schema_version": 1,
                "status": "failed",
                "disposable": True,
                "configured_path_feasible": False,
                "not_a_statement_about_qwen4b_or_supervised_learning_in_general": True,
                "failed_stage": "optimized_preflight",
                "implementation_id": IMPLEMENTATION_ID,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
                "memory": memory_policy,
                "runtime_diagnosis_path": "runtime-diagnosis.json",
                "equivalence_path": "equivalence-report.json",
            }
            write_json(run_dir / "preflight" / "preflight-report.json", preflight_report)
            write_json(
                run_dir / "run-manifest.json",
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "status": "failed_preflight",
                    "training_authorized_for_this_experiment_only": True,
                    "training_eligible": False,
                    "final_test_eligible": False,
                    "promotion_eligible": False,
                    "production_eligible": False,
                    "independent_human_review": False,
                    "source_approval_sha256": APPROVAL_RECORD_SHA256,
                    "execution_authorization_sha256": sha256_file(authorization_path),
                    "retry_authorization_sha256": RETRY_AUTHORIZATION_SHA256,
                    "memory_optimization_authorization_sha256": (
                        MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256
                    ),
                    "implementation_id": IMPLEMENTATION_ID,
                    "frozen_settings": FROZEN_SETTINGS,
                    "preflight": preflight_report,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            return {
                "status": "failed_preflight",
                "run_dir": str(run_dir),
                "preflight": preflight_report,
            }
        _release_model(model, mx)
        model = tokenizer = optimizer = None

        model, tokenizer, loaded = load_base_and_adapter(
            snapshot_path=snapshot_path,
            seed=42,
            mx=mx,
            profile=profile,
            snapshot_config=snapshot_config,
        )
        write_json(run_dir / "target-coverage-train.json", loaded["coverage"])
        optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
        order = epoch_order(42)
        updates = []
        exposures = {case_id: 0 for case_id in TRAIN_CASE_ORDER}
        train_started = time.perf_counter()
        mx.reset_peak_memory()
        update_index = 0
        for _epoch in range(FROZEN_SETTINGS["epochs"]):
            for local_index in order:
                case_id = TRAIN_CASE_ORDER[local_index]
                row = processed[case_id]
                update_index += 1
                step = run_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    mx=mx,
                    nn=nn,
                    default_loss=optimized_training_loss,
                    tokens=row["tokens"],
                    offset=row["offset"],
                    max_seq_length=PROPOSED_MAX_SEQ_LENGTH,
                    case_id=case_id,
                    update_index=update_index,
                )
                exposures[case_id] += 1
                step["peak_memory_gb"] = mx.get_peak_memory() / 1e9
                updates.append(step)
                write_json(
                    run_dir / "training-log.json",
                    {"updates": updates, "exposures": exposures, "in_progress": True},
                )
                print(
                    f"TRAIN {update_index}/40 {case_id} loss={step['loss']:.6f} "
                    f"lr={step['learning_rate']} peak_gb={step['peak_memory_gb']:.2f}",
                    flush=True,
                )
        if update_index != 40:
            raise RuntimeError(f"Measured optimizer updates {update_index} != 40")
        if any(count != 8 for count in exposures.values()):
            raise RuntimeError(f"Uneven per-case exposures: {exposures}")
        adapter_dir = run_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=False)
        adapter_file = adapter_dir / "adapters.safetensors"
        adapter_hashes = save_adapter(model, mx, adapter_file)
        reload_adapter_check(model, mx, adapter_file, adapter_hashes)
        write_json(
            adapter_dir / "adapter_config.json",
            {
                "fine_tune_type": "lora",
                "protocol_id": PROTOCOL_ID,
                "model": QWEN_4B_MODEL_ID,
                "governed_model_id": QWEN_4B_MODEL_ID,
                "governed_model_revision": QWEN_4B_REVISION,
                "lora_parameters": {
                    "rank": 16,
                    "scale": 2.0,
                    "dropout": 0.05,
                    "keys": list(FROZEN_SETTINGS["target_modules"]),
                },
                "num_layers": 36,
                "learning_rate_schedule": "constant",
                "learning_rate": 5e-5,
            },
        )
        adapter_identity = {
            "path": str(adapter_file),
            "sha256": sha256_file(adapter_file),
            "tensor_count": len(adapter_hashes),
            "save_reload_verified": True,
        }
        write_json(run_dir / "training-log.json", {"updates": updates, "exposures": exposures})
        write_json(run_dir / "adapter-identity.json", adapter_identity)
        _release_model(model, mx)
        model = tokenizer = optimizer = None
        attention_patch.uninstall()

        before_rows = load_rehearsal_before(root, verified)
        write_json(
            run_dir / "before" / "comparison.json",
            {
                "against": "approved proposed_target_review",
                "overwrites_old_scores": False,
                "rows": before_rows,
                "summary": summarize_scores(before_rows),
            },
        )
        after_rows = generate_after_reviews(
            root=root,
            run_dir=run_dir,
            verified=verified,
            adapter_dir=adapter_dir,
        )
        comparison = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "training_set_fitting_only": True,
            "not_generalization": True,
            "before": summarize_scores(before_rows),
            "after": summarize_scores(after_rows),
            "before_rows": before_rows,
            "after_rows": [
                {key: row[key] for key in row if key not in {"raw_output", "output", "review"}}
                for row in after_rows
            ],
            "training_cost": {
                "measured_optimizer_updates": 40,
                "elapsed_seconds": time.perf_counter() - train_started,
                "peak_memory_gb": max(step["peak_memory_gb"] for step in updates),
                "per_case_exposures": exposures,
            },
        }
        write_json(run_dir / "comparison.json", comparison)
        manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "completed_40_update_run",
            "training_authorized_for_this_experiment_only": True,
            "training_eligible": False,
            "final_test_eligible": False,
            "promotion_eligible": False,
            "production_eligible": False,
            "independent_human_review": False,
            "source_approval_sha256": APPROVAL_RECORD_SHA256,
            "execution_authorization_sha256": sha256_file(authorization_path),
            "retry_authorization_sha256": RETRY_AUTHORIZATION_SHA256,
            "memory_optimization_authorization_sha256": MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256,
            "implementation_id": IMPLEMENTATION_ID,
            "frozen_settings": FROZEN_SETTINGS,
            "bound_pairs": [
                {key: item[key] for key in ("case_id", "input_sha256", "target_sha256")}
                for item in verified
            ],
            "adapter_identity": adapter_identity,
            "preflight": {"status": "passed", "path": "preflight/preflight-report.json"},
            "comparison_path": "comparison.json",
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(run_dir / "run-manifest.json", manifest)
        return {
            "status": "completed",
            "run_dir": str(run_dir),
            "manifest": manifest,
            "comparison": comparison,
        }
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": (
                "failed_after_preflight"
                if preflight_report.get("status") == "passed"
                else "failed"
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "preflight": preflight_report or None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_json(run_dir / "failure-report.json", failure)
        return {
            "status": failure["status"],
            "run_dir": str(run_dir),
            "failure": failure,
        }
    finally:
        attention_patch.uninstall()
        mx.set_wired_limit = original_limit
        _release_model(model, mx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the authorized Qwen4B learning-mechanics experiment"
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the authorized preflight and, if it passes, the 40-update experiment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("Refusing to run without --execute")
    result = run_experiment(Path(args.root).expanduser().resolve())
    print(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
