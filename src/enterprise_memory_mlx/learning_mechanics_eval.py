"""Inference-only completion for the existing Qwen4B learning-mechanics run.

This module must not train, preflight, install gradient checkpoints, or
patch training attention. Default is verification-only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import resource
import subprocess
import threading
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import MLXBenchmarkBackend
from .experiment_profiles import QWEN_4B_MODEL_ID, QWEN_4B_REVISION
from .learning_mechanics import (
    APPLICATION_MEMORY_LIMIT_GIB,
    APPROVAL_RECORD_SHA256,
    AUTHORIZATION_RELATIVE,
    MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256,
    PROTOCOL_ID,
    REHEARSAL_V2_MANIFEST_SHA256,
    REHEARSAL_V2_RELATIVE,
    RETRY_AUTHORIZATION_SHA256,
    TRAIN_CASE_ORDER,
    CombinedExitError,
    WiredLimitGuard,
    application_memory_limit_bytes,
    expected_prompt_tokens,
    load_approval,
    sha256_file,
    verify_bound_pairs,
    write_json,
)
from .learning_mechanics_scoring import (
    SCORER_ID,
    score_declared_review,
    summarize_corrected_scores,
)
from .utils import atomic_write_text, sha256_text

EVAL_IMPLEMENTATION_ID = "qwen4b-learning-mechanics-eval-only/v1"
SOURCE_RUN_ID = "run-20260910T210406Z-seed42"
REQUIRED_ADAPTER_SHA256 = "7a5cd28ed8f469a98e82f285bce34ed83801d1ee71f01b69c9a9b20f984a67ba"
EVAL_CASE_ORDER = ("DEV-001", "DEV-002", "DEV-003", "DEV-004", "DEV-005")
MAX_OUTPUT_TOKENS = 4096
HEADROOM_GIB = 8
SAMPLER_INTERVAL_S = 0.75
HISTORICAL_BEFORE_LABELS = "22/25"
HISTORICAL_BEFORE_DISPOSITIONS = "3/5"
FORBIDDEN_TRAINING_NAMES = (
    "run_experiment",
    "load_base_and_adapter",
    "enable_gradient_checkpointing",
    "TrainingAttentionPatch",
    "optimized_training_loss",
    "run_optimizer_step",
    "linear_to_lora_layers",
)
OUTSTANDING_LIMITATIONS = (
    "Old Q/K/V checker omitted explicit differentiation arguments.",
    "Full-model equivalence did not cross the production chunk sizes.",
    "Gradient/update tolerances were insufficiently sensitive.",
    "Historical executed-code identity is incomplete.",
    "Historical application-memory compliance is not established.",
    "Historical training-only elapsed time is unavailable.",
    (
        "The original int(None) traceback was not captured; discarded "
        "wired-limit return is a strong matching hypothesis, not a proven "
        "historical call site."
    ),
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_source_run(root: Path, source_run: str) -> Path:
    run_dir = Path(source_run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if run_dir.name != SOURCE_RUN_ID:
        raise RuntimeError(f"Source run identity mismatch: {run_dir.name}")
    expected = root / "artifacts" / "qwen4b-learning-mechanics-v1" / SOURCE_RUN_ID
    if run_dir != expected.resolve():
        raise RuntimeError("Source run is not the bound local training directory")
    return run_dir


def verify_source_run(root: Path, run_dir: Path) -> dict[str, Any]:
    if (run_dir / "after").exists() or (run_dir / "comparison.json").exists():
        raise RuntimeError("After-output artifacts already exist on the training run")
    identity = json.loads((run_dir / "adapter-identity.json").read_text(encoding="utf-8"))
    adapter_file = run_dir / "adapter" / "adapters.safetensors"
    digest = sha256_file(adapter_file)
    if digest != REQUIRED_ADAPTER_SHA256 or digest != identity["sha256"]:
        raise RuntimeError("Final adapter digest does not match the authorized binding")
    config = json.loads((run_dir / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((run_dir / "frozen-settings.json").read_text(encoding="utf-8"))
    if (
        frozen["model_id"] != QWEN_4B_MODEL_ID
        or frozen["model_revision"] != QWEN_4B_REVISION
        or config.get("governed_model_revision") != QWEN_4B_REVISION
        or config.get("num_layers") != 36
        or config.get("lora_parameters", {}).get("rank") != 16
    ):
        raise RuntimeError("Adapter or base identity does not match the frozen 4B settings")
    log = json.loads((run_dir / "training-log.json").read_text(encoding="utf-8"))
    updates = log["updates"]
    exposures = log["exposures"]
    if [row["update_index"] for row in updates] != list(range(1, 41)):
        raise RuntimeError("Training log is not exactly 40 measured updates")
    if any(exposures.get(case_id) != 8 for case_id in TRAIN_CASE_ORDER):
        raise RuntimeError("Per-case exposures are not eight each")
    approval = load_approval(root)
    verified = verify_bound_pairs(root, approval)
    bound = json.loads((run_dir / "bound-pairs.json").read_text(encoding="utf-8"))
    if bound["approval_sha256"] != APPROVAL_RECORD_SHA256:
        raise RuntimeError("Source-run approval binding changed")
    if [row["case_id"] for row in verified] != list(EVAL_CASE_ORDER):
        raise RuntimeError("Verified case order does not match the evaluation contract")
    before = json.loads((run_dir / "before" / "comparison.json").read_text(encoding="utf-8"))
    rehearsal = root / REHEARSAL_V2_RELATIVE
    if sha256_file(rehearsal / "result-manifest.json") != REHEARSAL_V2_MANIFEST_SHA256:
        raise RuntimeError("Sealed rehearsal-v2 manifest hash changed")
    before_bindings = []
    for row, item in zip(before["rows"], verified, strict=True):
        path = rehearsal / "outputs" / "qwen4b" / f"{item['case_id']}.json"
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{item['case_id']} sealed before-output hash mismatch")
        if row["case_id"] != item["case_id"]:
            raise RuntimeError("Before-comparison case order mismatch")
        before_bindings.append({"case_id": item["case_id"], "sha256": row["sha256"]})
    if before["summary"].get("requirement_labels") != HISTORICAL_BEFORE_LABELS:
        raise RuntimeError("Historical before label summary changed")
    if before["summary"].get("dispositions") != HISTORICAL_BEFORE_DISPOSITIONS:
        raise RuntimeError("Historical before disposition summary changed")
    return {
        "run_dir": str(run_dir),
        "adapter_sha256": digest,
        "adapter_tensor_count": identity["tensor_count"],
        "save_reload_verified": identity["save_reload_verified"],
        "model_id": QWEN_4B_MODEL_ID,
        "model_revision": QWEN_4B_REVISION,
        "updates": 40,
        "exposures": exposures,
        "verified_pairs": [
            {key: item[key] for key in ("case_id", "input_sha256", "target_sha256")}
            for item in verified
        ],
        "before_output_bindings": before_bindings,
        "historical_before_labels": HISTORICAL_BEFORE_LABELS,
        "historical_before_dispositions": HISTORICAL_BEFORE_DISPOSITIONS,
        "verified": verified,
        "adapter_dir": str(run_dir / "adapter"),
    }


def implementation_identity(root: Path) -> dict[str, Any]:
    files = [
        "src/enterprise_memory_mlx/learning_mechanics_eval.py",
        "src/enterprise_memory_mlx/learning_mechanics_scoring.py",
        "src/enterprise_memory_mlx/learning_mechanics.py",
        "src/enterprise_memory_mlx/benchmark.py",
    ]
    hashes = {path: file_digest(root / path) for path in files}
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *files],
            cwd=root,
            text=True,
        )
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        status = f"unavailable:{exc}"
        head = "unavailable"
    versions = {}
    try:
        from importlib import metadata

        for name in ("mlx", "mlx-lm", "mlx-metal"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    except Exception:
        versions = {"error": "importlib.metadata unavailable"}
    return {
        "implementation_id": EVAL_IMPLEMENTATION_ID,
        "does_not_backdate_training_code_identity": True,
        "git_head": head,
        "git_status_porcelain": status,
        "file_sha256": hashes,
        "package_versions": versions,
    }


def prospective_memory_policy(mx: Any) -> dict[str, Any]:
    limit, policy = application_memory_limit_bytes(mx)
    headroom = HEADROOM_GIB * 2**30
    stop_bytes = max(limit - headroom, 0)
    return {
        "metric": "process_rss_bytes_as_available_os_process_measurement",
        "process_footprint_distinct_from_rss": True,
        "wired_limit_is_not_total_process_cap": True,
        "allocator_guideline_is_not_hard_cap": True,
        "configured_ceiling_gib": APPLICATION_MEMORY_LIMIT_GIB,
        "selected_limit_bytes": limit,
        "headroom_bytes": headroom,
        "stop_rss_bytes": stop_bytes,
        "sampler_interval_s": SAMPLER_INTERVAL_S,
        "polling_does_not_prove_no_transient_exceedance": True,
        "policy": policy,
    }


def process_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage)


def process_current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        )
        return int(output.strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def system_swap_snapshot() -> dict[str, Any] | None:
    try:
        raw = subprocess.check_output(("sysctl", "-n", "vm.swapusage"), text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    values = {
        key.casefold(): float(value)
        for key, value in re.findall(r"(total|used|free)\s*=\s*([0-9.]+)M", raw)
    }
    return {"raw": raw.strip(), "parsed_M": values, "not_per_process": True}


def memory_pressure_snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(("memory_pressure"), text=True, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    lowered = raw.casefold()
    status = "unknown"
    if "critical" in lowered:
        status = "critical"
    elif "warn" in lowered:
        status = "warn"
    elif "normal" in lowered:
        status = "normal"
    return {"available": True, "status": status, "raw": raw.strip()[:2000]}


def mlx_memory_snapshot(mx: Any | None) -> dict[str, Any] | None:
    if mx is None:
        return None
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "unit": "mlx_allocator_bytes",
        "not_process_footprint": True,
    }


class ResourceWatchdog:
    def __init__(self, *, stop_rss_bytes: int) -> None:
        self.stop_rss_bytes = stop_rss_bytes
        self.samples: list[dict[str, Any]] = []
        self.breach: str | None = None
        self.monitor_failed: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="eval-watchdog", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.wait(SAMPLER_INTERVAL_S):
                sample = {
                    "ts": datetime.now(UTC).isoformat(),
                    "ru_maxrss_bytes": process_rss_bytes(),
                    "ps_rss_bytes": process_current_rss_bytes(),
                    "swap": system_swap_snapshot(),
                    "pressure": memory_pressure_snapshot(),
                }
                self.samples.append(sample)
                rss = sample["ps_rss_bytes"] or sample["ru_maxrss_bytes"]
                if rss >= self.stop_rss_bytes:
                    self.breach = f"process_rss_bytes {rss} >= stop {self.stop_rss_bytes}"
                    break
                pressure = sample["pressure"]
                if pressure.get("status") in {"warn", "critical"}:
                    self.breach = f"memory_pressure:{pressure.get('status')}"
                    break
        except Exception as exc:
            self.monitor_failed = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.monitor_failed = self.monitor_failed or "watchdog_join_timeout"


def sanitized_traceback(exc: BaseException) -> dict[str, Any]:
    frames = traceback.extract_tb(exc.__traceback__)[-8:]
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "frames": [
            {"file": frame.filename, "line": frame.lineno, "name": frame.name}
            for frame in frames
        ],
        "locals_omitted": True,
        "private_prompts_omitted": True,
    }


def prepare_eval_directory(root: Path, run_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    eval_dir = (
        root
        / "artifacts"
        / "qwen4b-learning-mechanics-v1"
        / f"eval-{stamp}-from-{SOURCE_RUN_ID}"
    )
    if eval_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=False)
    (eval_dir / "after").mkdir()
    (eval_dir / "attempts").mkdir()
    write_json(
        eval_dir / "source-run-link.json",
        {
            "source_run_id": SOURCE_RUN_ID,
            "source_run_dir": str(run_dir.relative_to(root)),
            "immutable_training_artifacts": True,
        },
    )
    return eval_dir


def score_sealed_before(
    root: Path, verified: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rehearsal = root / REHEARSAL_V2_RELATIVE
    rows = []
    for item in verified:
        path = rehearsal / "outputs" / "qwen4b" / f"{item['case_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        scored = score_declared_review(
            case_id=item["case_id"],
            approved_labels=item["labels"],
            approved_disposition=item["disposition"],
            raw_text=payload.get("output"),
            truncated=bool(payload.get("truncated")),
            parse_status=payload.get("parse_status"),
        )
        rows.append(
            {
                "source": "rehearsal-v2-qwen4b",
                "output_sha256": sha256_file(path),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                **scored,
            }
        )
    return rows


def verify_rendered_prompt(
    backend: MLXBenchmarkBackend,
    *,
    case_id: str,
    system_prompt: str,
    user_prompt: str,
    expected_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = backend._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    token_ids = backend._tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) != expected_tokens:
        raise RuntimeError(
            f"{case_id} rendered prompt tokens {len(token_ids)} != bound {expected_tokens}"
        )
    if expected_tokens + MAX_OUTPUT_TOKENS > 262144:
        raise RuntimeError(f"{case_id} prompt plus reserve exceeds model context")
    return {
        "prompt_tokens": len(token_ids),
        "prompt_sha256": sha256_text(prompt),
        "template_sha256": sha256_text(str(getattr(backend._tokenizer, "chat_template", ""))),
        "thinking": False,
    }


def persist_private_output(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def generate_one_case(
    *,
    backend: MLXBenchmarkBackend,
    mx: Any,
    item: Mapping[str, Any],
    eval_dir: Path,
    adapter_sha256: str,
    watchdog: ResourceWatchdog,
) -> dict[str, Any]:
    case_id = item["case_id"]
    source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
    start = {
        "case_id": case_id,
        "input_sha256": item["input_sha256"],
        "adapter_sha256": adapter_sha256,
        "started_at": datetime.now(UTC).isoformat(),
        "stage": "attempt_start",
    }
    persist_private_output(eval_dir / "attempts" / f"{case_id}.start.json", start)
    print(f"EVAL_ATTEMPT_START {case_id}", flush=True)
    record: dict[str, Any] = {
        **start,
        "status": "failed",
        "stage": "render",
    }
    raw_path = eval_dir / "after" / f"{case_id}.json"
    try:
        rendered = verify_rendered_prompt(
            backend,
            case_id=case_id,
            system_prompt=source["system_prompt"],
            user_prompt=source["user_prompt"],
            expected_tokens=expected_prompt_tokens()[case_id],
        )
        record.update(rendered)
        record["stage"] = "generation"
        mx.random.seed(42)
        answer = backend.generate(
            system_prompt=source["system_prompt"],
            question=source["user_prompt"],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        record["stage"] = "synchronization"
        mx.synchronize()
        output_payload = {
            "case_id": case_id,
            "input_sha256": item["input_sha256"],
            "adapter_sha256": adapter_sha256,
            "prompt_tokens": answer.prompt_tokens,
            "prompt_sha256": answer.prompt_hash or rendered["prompt_sha256"],
            "template_sha256": answer.rendered_template_hash or rendered["template_sha256"],
            "raw_output": answer.raw_output,
            "output": answer.output,
            "parse_status": answer.parse_status,
            "finish_reason": answer.finish_reason,
            "truncated": answer.truncated,
            "completion_tokens": answer.completion_tokens,
            "elapsed_seconds": answer.elapsed_seconds,
            "peak_memory_gb": answer.peak_memory_gb,
            "mlx_memory": mlx_memory_snapshot(mx),
            "process_rss_bytes": process_current_rss_bytes(),
            "ru_maxrss_bytes": process_rss_bytes(),
        }
        persist_private_output(raw_path, output_payload)
        record["stage"] = "scoring"
        failed_output = bool(
            answer.truncated
            or answer.output is None
            or answer.parse_status not in {"plain"}
        )
        scored = score_declared_review(
            case_id=case_id,
            approved_labels=item["labels"],
            approved_disposition=item["disposition"],
            raw_text=answer.output if not failed_output else answer.raw_output,
            truncated=bool(answer.truncated) or failed_output,
            parse_status=answer.parse_status,
        )
        public = {
            "case_id": case_id,
            "status": "completed_output" if not failed_output else "failed_output",
            "input_sha256": item["input_sha256"],
            "adapter_sha256": adapter_sha256,
            "prompt_tokens": answer.prompt_tokens,
            "prompt_sha256": output_payload["prompt_sha256"],
            "completion_tokens": answer.completion_tokens,
            "elapsed_seconds": answer.elapsed_seconds,
            "finish_reason": answer.finish_reason,
            "truncated": answer.truncated,
            "parse_status": answer.parse_status,
            "private_output_path": str(raw_path.name),
            **scored,
        }
        persist_private_output(eval_dir / "after" / f"{case_id}.score.json", public)
        print(f"EVAL_ATTEMPT_DONE {case_id} status={public['status']}", flush=True)
        return public
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = sanitized_traceback(exc)
        persist_private_output(eval_dir / "attempts" / f"{case_id}.failure.json", record)
        raise
    finally:
        if watchdog.breach or watchdog.monitor_failed:
            record.setdefault("watchdog", watchdog.breach or watchdog.monitor_failed)


def run_verification(root: Path, source_run: str) -> dict[str, Any]:
    run_dir = resolve_source_run(root, source_run)
    bound = verify_source_run(root, run_dir)
    identity = implementation_identity(root)
    return {
        "status": "verified_only",
        "source_run": bound,
        "implementation": identity,
        "optimizer_updates": 0,
        "model_loaded": False,
    }


def run_evaluation(root: Path, source_run: str) -> dict[str, Any]:
    import mlx.core as mx

    run_dir = resolve_source_run(root, source_run)
    bound = verify_source_run(root, run_dir)
    identity = implementation_identity(root)
    eval_dir = prepare_eval_directory(root, run_dir)
    write_json(
        eval_dir / "execution-authorization.json",
        {
            "authorization_kind": "inference_only_completion",
            "protocol_id": PROTOCOL_ID,
            "source_run_id": SOURCE_RUN_ID,
            "source_approval_record_sha256": APPROVAL_RECORD_SHA256,
            "execution_authorization_sha256": sha256_file(root / AUTHORIZATION_RELATIVE),
            "retry_authorization_sha256": RETRY_AUTHORIZATION_SHA256,
            "memory_optimization_authorization_sha256": MEMORY_OPTIMIZATION_AUTHORIZATION_SHA256,
            "required_adapter_sha256": REQUIRED_ADAPTER_SHA256,
            "verified_adapter_sha256": bound["adapter_sha256"],
            "training_authorized": False,
            "optimizer_updates_authorized": 0,
            "generation_calls_authorized": 5,
            "outstanding_limitations": list(OUTSTANDING_LIMITATIONS),
            "technical_review_file_in_repo": False,
            "technical_review_used": (
                "session review 2026-09-11 evaluation-pending technical review"
            ),
        },
    )
    write_json(eval_dir / "implementation-identity.json", identity)
    policy = prospective_memory_policy(mx)
    write_json(eval_dir / "resource-policy.json", policy)
    watchdog = ResourceWatchdog(stop_rss_bytes=policy["stop_rss_bytes"])
    public_rows: list[dict[str, Any]] = []
    status = "evaluation_completed"
    stop_reason = None
    backend = None
    guard = WiredLimitGuard(mx, policy["selected_limit_bytes"])
    try:
        guard.install()
        with contextlib.suppress(Exception):
            mx.set_cache_limit(min(8 * 2**30, policy["selected_limit_bytes"] // 4))
        watchdog.start()
        if watchdog.monitor_failed:
            raise RuntimeError(f"monitoring_failed:{watchdog.monitor_failed}")
        backend = MLXBenchmarkBackend(
            QWEN_4B_MODEL_ID,
            revision=QWEN_4B_REVISION,
            adapter_path=bound["adapter_dir"],
            max_context_tokens=262144,
        )
        if hasattr(backend._model, "eval"):
            backend._model.eval()
        adapter_file = Path(bound["adapter_dir"]) / "adapters.safetensors"
        if sha256_file(adapter_file) != REQUIRED_ADAPTER_SHA256:
            raise RuntimeError("Adapter digest changed after load")
        expected = expected_prompt_tokens()
        for case_id in EVAL_CASE_ORDER:
            if expected[case_id] != {
                "DEV-001": 12682,
                "DEV-002": 7620,
                "DEV-003": 20390,
                "DEV-004": 33494,
                "DEV-005": 25917,
            }[case_id]:
                raise RuntimeError("Bound prompt counts changed")
        for item in bound["verified"]:
            if watchdog.breach or watchdog.monitor_failed:
                stop_reason = watchdog.breach or watchdog.monitor_failed
                status = "evaluation_partially_failed"
                persist_private_output(
                    eval_dir / "after" / f"{item['case_id']}.score.json",
                    {
                        "case_id": item["case_id"],
                        "status": "not_attempted",
                        "reason": stop_reason,
                    },
                )
                public_rows.append(
                    {
                        "case_id": item["case_id"],
                        "status": "not_attempted",
                        "complete_valid": False,
                        "label_matches": 0,
                        "label_count": len(item["labels"]),
                        "disposition_agrees": False,
                        "approved_labels": dict(item["labels"]),
                        "model_labels": {},
                        "approved_disposition": item["disposition"],
                        "model_disposition": "",
                        "evidence_support": "not_assessed",
                        "action_safety": "not_assessed",
                        "full_task_success": False,
                    }
                )
                continue
            try:
                public_rows.append(
                    generate_one_case(
                        backend=backend,
                        mx=mx,
                        item=item,
                        eval_dir=eval_dir,
                        adapter_sha256=bound["adapter_sha256"],
                        watchdog=watchdog,
                    )
                )
            except Exception as exc:
                stop_reason = f"{type(exc).__name__}: {exc}"
                status = "evaluation_partially_failed"
                remaining = [
                    row
                    for row in bound["verified"]
                    if row["case_id"] not in {item["case_id"], *[r["case_id"] for r in public_rows]}
                ]
                failed_row = {
                    "case_id": item["case_id"],
                    "status": "failed",
                    "complete_valid": False,
                    "label_matches": 0,
                    "label_count": len(item["labels"]),
                    "disposition_agrees": False,
                    "approved_labels": dict(item["labels"]),
                    "model_labels": {},
                    "approved_disposition": item["disposition"],
                    "model_disposition": "",
                    "evidence_support": "not_assessed",
                    "action_safety": "not_assessed",
                    "full_task_success": False,
                    "error": sanitized_traceback(exc),
                }
                score_path = eval_dir / "after" / f"{item['case_id']}.score.json"
                if not score_path.exists():
                    persist_private_output(score_path, failed_row)
                public_rows.append(failed_row)
                for later in remaining:
                    persist_private_output(
                        eval_dir / "after" / f"{later['case_id']}.score.json",
                        {
                            "case_id": later["case_id"],
                            "status": "not_attempted",
                            "reason": "batch_stopped_after_failure",
                        },
                    )
                    public_rows.append(
                        {
                            "case_id": later["case_id"],
                            "status": "not_attempted",
                            "complete_valid": False,
                            "label_matches": 0,
                            "label_count": len(later["labels"]),
                            "disposition_agrees": False,
                            "approved_labels": dict(later["labels"]),
                            "model_labels": {},
                            "approved_disposition": later["disposition"],
                            "model_disposition": "",
                            "evidence_support": "not_assessed",
                            "action_safety": "not_assessed",
                            "full_task_success": False,
                        }
                    )
                break
    except CombinedExitError as exc:
        status = "evaluation_partially_failed"
        stop_reason = str(exc)
        write_json(
            eval_dir / "failure-report.json",
            {
                "primary": sanitized_traceback(exc.primary),
                "cleanup": sanitized_traceback(exc.cleanup),
            },
        )
    except Exception as exc:
        status = "evaluation_partially_failed"
        stop_reason = f"{type(exc).__name__}: {exc}"
        write_json(eval_dir / "failure-report.json", sanitized_traceback(exc))
    finally:
        watchdog.stop()
        if backend is not None:
            try:
                backend.close()
            except Exception as cleanup:
                write_json(
                    eval_dir / "cleanup-failure.json",
                    sanitized_traceback(cleanup),
                )
        try:
            guard.uninstall()
        except Exception as cleanup:
            write_json(
                eval_dir / "wired-limit-restore-failure.json",
                sanitized_traceback(cleanup),
            )

    if len(public_rows) != 5 and status == "evaluation_completed":
        status = "evaluation_partially_failed"
    while len(public_rows) < 5:
        missing_id = EVAL_CASE_ORDER[len(public_rows)]
        public_rows.append(
            {
                "case_id": missing_id,
                "status": "not_attempted",
                "complete_valid": False,
                "label_matches": 0,
                "label_count": next(
                    len(item["labels"])
                    for item in bound["verified"]
                    if item["case_id"] == missing_id
                ),
                "disposition_agrees": False,
                "approved_labels": dict(
                    next(
                        item["labels"]
                        for item in bound["verified"]
                        if item["case_id"] == missing_id
                    )
                ),
                "model_labels": {},
                "approved_disposition": next(
                    item["disposition"]
                    for item in bound["verified"]
                    if item["case_id"] == missing_id
                ),
                "model_disposition": "",
                "evidence_support": "not_assessed",
                "action_safety": "not_assessed",
                "full_task_success": False,
            }
        )
        status = "evaluation_partially_failed"

    before_rows = score_sealed_before(root, bound["verified"])
    after_summary = summarize_corrected_scores(public_rows)
    before_summary = summarize_corrected_scores(before_rows)
    comparison = {
        "schema_version": 1,
        "scorer_id": SCORER_ID,
        "separately_versioned": True,
        "does_not_overwrite_historical_before_scores": True,
        "historical_unrevised_before": {
            "requirement_labels": HISTORICAL_BEFORE_LABELS,
            "dispositions": HISTORICAL_BEFORE_DISPOSITIONS,
            "note": (
                "Preserved from the training-run before/comparison.json. "
                "Stricter validation may differ."
            ),
        },
        "corrected_before": before_summary,
        "corrected_after": after_summary,
        "before_rows": before_rows,
        "after_rows": [
            {key: row[key] for key in row if key not in {"raw_output", "output", "review"}}
            for row in public_rows
        ],
        "stricter_validation_difference": {
            "historical_before_labels": HISTORICAL_BEFORE_LABELS,
            "corrected_before_labels": before_summary["requirement_labels"],
            "historical_before_dispositions": HISTORICAL_BEFORE_DISPOSITIONS,
            "corrected_before_dispositions": before_summary["dispositions"],
            "explanation": (
                "The historical 22/25 and 3/5 used the old nonempty-assessments rule. "
                "This report reapplies the declared-schema scorer to the same sealed outputs."
            ),
        },
        "training_set_fitting_only": True,
        "semantic_support_and_safety": "pending",
    }
    write_json(eval_dir / "comparison.json", comparison)
    write_json(
        eval_dir / "watchdog-samples.json",
        {
            "breach": watchdog.breach,
            "monitor_failed": watchdog.monitor_failed,
            "samples": watchdog.samples,
            "machine_wide_swap_not_attributed_to_this_process": True,
        },
    )
    if sha256_file(Path(bound["adapter_dir"]) / "adapters.safetensors") != REQUIRED_ADAPTER_SHA256:
        raise RuntimeError("Adapter digest changed during evaluation")
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "implementation_id": EVAL_IMPLEMENTATION_ID,
        "not_the_missing_original_final_run_manifest": True,
        "source_run_id": SOURCE_RUN_ID,
        "training_recorded_as_completed": True,
        "evaluation_status": status,
        "semantic_support_safety": "pending",
        "unresolved_historical_limitations": list(OUTSTANDING_LIMITATIONS),
        "adapter_sha256": REQUIRED_ADAPTER_SHA256,
        "adapter_digest_unchanged": True,
        "optimizer_updates": 0,
        "generation_calls_authorized": 5,
        "stop_reason": stop_reason,
        "eval_dir": str(eval_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "training_eligible": False,
        "promotion_eligible": False,
        "production_eligible": False,
    }
    write_json(eval_dir / "evaluation-completion-manifest.json", manifest)
    return {
        "status": status,
        "eval_dir": str(eval_dir),
        "manifest": manifest,
        "comparison": comparison,
        "stop_reason": stop_reason,
    }


def assert_eval_module_cannot_train() -> None:
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        if name in FORBIDDEN_TRAINING_NAMES:
            raise RuntimeError(f"Evaluation module calls forbidden training name {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-only completion for the existing Qwen4B learning-mechanics run"
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument(
        "--source-run",
        required=True,
        help="Bound training run directory (must be run-20260910T210406Z-seed42)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the five authorized inference attempts after verification",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    assert_eval_module_cannot_train()
    if args.execute:
        result = run_evaluation(root, args.source_run)
    else:
        result = run_verification(root, args.source_run)
    print(json.dumps({"status": result["status"], "eval_dir": result.get("eval_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
