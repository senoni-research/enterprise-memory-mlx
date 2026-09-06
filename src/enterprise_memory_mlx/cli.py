from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from .acquisition_compiler import compile_acquisition_dataset
from .acquisition_diagnostics import run_general_diagnostic
from .acquisition_training import (
    DEFAULT_ACQUISITION_MODEL,
    AcquisitionConfig,
    build_mlx_acquisition_config,
    execute_training_command,
    load_verified_acquisition_adapter,
    materialize_model_snapshot,
    run_acquisition,
)
from .benchmark import (
    DEFAULT_ARMS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SUITES,
    SUPPORTED_ARMS,
    SUPPORTED_SUITES,
    BenchmarkConfig,
    MLXBenchmarkBackend,
    bind_bm25_selection,
    bm25_decision_artifact_payload,
    build_benchmark_plan,
    default_bm25_decision_path,
    index_payload_hash,
    load_benchmark_tokenizer,
    require_matching_model_revision,
    resolve_default_bm25_decision,
    resolve_huggingface_revision,
    run_benchmark_plan,
    source_snapshot_hash,
    write_benchmark_artifact,
)
from .benchmark_review import (
    prepare_benchmark_review,
    write_benchmark_review_report,
    write_model_regrading_bridge,
)
from .compiler import compile_knowledge, load_records
from .curriculum_authoring import author_model_upgrade_curriculum
from .evaluation import evaluate_models
from .experiment_profiles import (
    MODEL_UPGRADE_CONFIG,
    MODEL_UPGRADE_PROFILE,
    QWEN_4B_MODEL_ID,
    QWEN_27B_MODEL_ID,
    model_profile,
)
from .gemma_advisory import run_gemma_advisory, run_gemma_preflight
from .grading import grade_benchmark_artifact, write_grading_report
from .hardware import PRESETS, resolve_preset
from .inference import interactive_chat
from .legacy_guard import LEGACY_COMMANDS, block_legacy_command
from .model_upgrade_report import write_model_upgrade_comparison
from .registry import find_adapter
from .review_ui import (
    DEFAULT_MAPPING,
    DEFAULT_PACKET,
    serve_review_ui,
)
from .router import route_query
from .semantic_neighbors import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_MODEL_REVISION,
    MLXEmbeddingBackend,
)
from .specialization_audit import (
    prepare_specialization_audit,
    validate_specialization_audit_overlay,
)
from .specialization_evaluator import verify_evaluator_v2_contract
from .specialization_fact_state_experiment import (
    load_fact_state_assets,
    prepare_fact_state_review,
    run_fact_state_comparison,
    run_fact_state_regression,
    score_fact_state_review,
)
from .specialization_qualification import (
    load_qualification_assets,
    prepare_qualification_review,
    run_qualification_comparison,
    score_qualification_advisory,
)
from .specialization_real_work import validate_real_work_seed
from .specialization_remedy_experiment import (
    load_remedy_experiment_assets,
    prepare_remedy_review,
    run_remedy_experiment,
    score_remedy_review,
)
from .split_contract import load_eval_suites, verify_frozen_assets
from .task_specialization import (
    load_specialization_assets,
    run_specialization_pilot,
)
from .training import doctor_report, train_pipeline
from .utils import atomic_write_text, read_jsonl, slugify

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emmlx",
        description=(
            "Local enterprise-memory research harness. Legacy training commands "
            "are disabled until the revised acquisition contract is implemented."
        ),
    )
    parser.add_argument("--root", default=".", help="Repository root. Default: current directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Inspect the Mac and recommend a training preset")

    compile_parser = subparsers.add_parser("compile", help="DISABLED legacy dataset compiler")
    compile_parser.add_argument("--knowledge-dir", default="knowledge")
    compile_parser.add_argument("--output-dir", default="artifacts/datasets")
    compile_parser.add_argument("--seed", type=int, default=42)
    compile_parser.add_argument("--include-restricted", action="store_true")
    compile_parser.add_argument("--acknowledge-weight-acl-risk", action="store_true")
    compile_parser.add_argument("--no-per-domain", action="store_true")

    train_parser = subparsers.add_parser("train", help="DISABLED legacy adapter trainer")
    train_parser.add_argument(
        "--stage",
        choices=["vanilla", "inject", "align", "recover", "all"],
        default="all",
    )
    train_parser.add_argument("--preset", choices=["auto", *PRESETS.keys()], default="auto")
    train_parser.add_argument("--model", default=None, help="Override the preset model")
    train_parser.add_argument("--domain", default="global")
    train_parser.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="DISABLED legacy lexical evaluator")
    eval_parser.add_argument(
        "--suite", choices=["domain", "retention", "general"], default="domain"
    )
    eval_parser.add_argument("--adapter", default=None)
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--domain", default="global")
    eval_parser.add_argument("--base-only", action="store_true")
    eval_parser.add_argument("--skip-base", action="store_true")
    eval_parser.add_argument("--max-tokens", type=int, default=220)

    route_parser = subparsers.add_parser("route", help="DISABLED legacy lexical adapter router")
    route_parser.add_argument("query")
    route_parser.add_argument("--threshold", type=float, default=48.0)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run answer-blind base, full-context, and oracle controls",
    )
    benchmark_parser.add_argument(
        "--suite",
        action="append",
        choices=SUPPORTED_SUITES,
        help="Suite to run; repeat to select several. Default: all supported suites.",
    )
    benchmark_parser.add_argument(
        "--eval-version",
        choices=["v1", "v2"],
        default="v1",
        help="v1 provides non-temporal suites; v2 is the date-controlled supersession suite.",
    )
    benchmark_parser.add_argument(
        "--arm",
        action="append",
        choices=SUPPORTED_ARMS,
        help=(
            "Arm to run; repeat to select several. Default: base, full_context, and "
            "oracle. The bm25 arm requires an owner-approved selected operating point."
        ),
    )
    benchmark_parser.add_argument(
        "--bm25-selection",
        default=None,
        help=(
            "Hash-bound BM25 selection file. Required for --arm bm25. "
            "A no_feasible_operating_point decision rejects the bm25 arm."
        ),
    )
    benchmark_parser.add_argument(
        "--acquisition-run",
        default=None,
        help="Verified trained acquisition run manifest; required for --arm parametric.",
    )
    benchmark_parser.add_argument("--max-context-bytes", type=int, default=65_536)
    benchmark_parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    benchmark_parser.add_argument("--max-tokens", type=int, default=220)
    benchmark_parser.add_argument(
        "--experiment-id",
        default=None,
        help="Immutable experiment/run identity used in the benchmark artifact name.",
    )
    benchmark_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    benchmark_parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Optional Hugging Face revision. When omitted, the resolved commit SHA "
            "of --model is recorded and reused for model weight loading."
        ),
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and summarize the answer-blind plan using tokenizer-only loading; "
            "do not load model weights."
        ),
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Launch the localhost-only blinded human-review UI and status dashboard",
    )
    review_parser.add_argument(
        "--packet",
        default=str(DEFAULT_PACKET),
        help="Blinded review packet ZIP, relative to --root by default.",
    )
    review_parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING),
        help="Private review-ID mapping, relative to --root by default.",
    )
    review_parser.add_argument(
        "--reviewer",
        required=True,
        help=(
            "Human reviewer identity stored in the per-reviewer overlay. "
            "Required; the identity is asserted, not authenticated, and the "
            "invoking OS user is recorded alongside every decision."
        ),
    )
    review_parser.add_argument(
        "--state-root",
        default=None,
        help=(
            "Review state/output directory, relative to --root by default. "
            "Use a benchmark-specific path to isolate it from judge calibration."
        ),
    )
    review_parser.add_argument("--port", type=int, default=8765)
    review_parser.add_argument("--no-browser", action="store_true")

    benchmark_review_parser = subparsers.add_parser(
        "benchmark-review",
        help="Prepare or report a blinded human review of benchmark outputs",
    )
    benchmark_review_actions = benchmark_review_parser.add_subparsers(
        dest="benchmark_review_action",
        required=True,
    )
    prepare_review_parser = benchmark_review_actions.add_parser(
        "prepare",
        help="Create a blinded packet plus private arm mapping",
    )
    prepare_review_parser.add_argument("--benchmark", required=True)
    prepare_review_parser.add_argument("--grading", required=True)
    prepare_review_parser.add_argument(
        "--eval-version",
        choices=["v1", "v2"],
        default="v1",
    )
    prepare_review_parser.add_argument(
        "--output-root",
        default="artifacts/review-packets",
    )
    report_review_parser = benchmark_review_actions.add_parser(
        "report",
        help="Unblind a completed human overlay and apply the stopping rule",
    )
    report_review_parser.add_argument("--benchmark", required=True)
    report_review_parser.add_argument("--grading", required=True)
    report_review_parser.add_argument("--packet", required=True)
    report_review_parser.add_argument("--mapping", required=True)
    report_review_parser.add_argument("--overlay", required=True)
    report_review_parser.add_argument(
        "--overlay-manifest",
        default=None,
        help="Defaults to the overlay path with .manifest.json suffix.",
    )
    report_review_parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark-review",
    )

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="Compile and optionally run the revised non-promotable MLX acquisition smoke",
    )
    acquire_parser.add_argument(
        "--profile",
        choices=["smoke_non_promotable", MODEL_UPGRADE_PROFILE],
        default="smoke_non_promotable",
    )
    acquire_parser.add_argument("--rank", type=int, choices=[16], default=16)
    acquire_parser.add_argument("--target-exposures", type=int, default=None)
    acquire_parser.add_argument("--batch-size", type=int, default=1)
    acquire_parser.add_argument("--grad-accumulation", type=int, default=8)
    acquire_parser.add_argument("--seed", type=int, default=42)
    acquire_parser.add_argument("--model", default=None)
    acquire_parser.add_argument("--model-revision", default=None)
    acquire_parser.add_argument(
        "--study-views-root",
        default="knowledge/model_upgrade_exploratory/v1",
    )
    acquire_parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    acquire_parser.add_argument("--semantic-model-revision", default=None)
    acquire_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag only compile and emit a dry-run config.",
    )

    model_upgrade_parser = subparsers.add_parser(
        "model-upgrade",
        help="Run governed stages of model-upgrade-exploratory/v1",
    )
    model_upgrade_actions = model_upgrade_parser.add_subparsers(
        dest="model_upgrade_action",
        required=True,
    )
    author_parser = model_upgrade_actions.add_parser(
        "author-curriculum",
        help="Author the 24-view source-only curriculum with local Qwen",
    )
    author_parser.add_argument(
        "--output-dir",
        default="knowledge/model_upgrade_exploratory/v1",
    )
    preflight_parser = model_upgrade_actions.add_parser(
        "preflight",
        help="Run disposable target-coverage, backward, and reload checks",
    )
    preflight_parser.add_argument(
        "--model",
        choices=[QWEN_4B_MODEL_ID, QWEN_27B_MODEL_ID],
        required=True,
    )
    preflight_parser.add_argument(
        "--study-views-root",
        default="knowledge/model_upgrade_exploratory/v1",
    )
    preflight_parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    preflight_parser.add_argument("--semantic-model-revision", default=None)
    preflight_parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Immutable preflight attempt number; increase only after a disclosed defect.",
    )
    upgrade_train_parser = model_upgrade_actions.add_parser(
        "train",
        help="Run the measured seed-42 training trajectory after preflight",
    )
    upgrade_train_parser.add_argument(
        "--model",
        choices=[QWEN_4B_MODEL_ID, QWEN_27B_MODEL_ID],
        required=True,
    )
    upgrade_benchmark_parser = model_upgrade_actions.add_parser(
        "benchmark",
        help="Run frozen acquisition/OOS arms for one trained model",
    )
    upgrade_benchmark_parser.add_argument(
        "--model",
        choices=[QWEN_4B_MODEL_ID, QWEN_27B_MODEL_ID],
        required=True,
    )
    upgrade_benchmark_parser.add_argument("--acquisition-run", required=True)
    upgrade_benchmark_parser.add_argument(
        "--bm25-selection",
        default="knowledge/operating_points/bm25/v1-research-tradeoff.json",
        help="Frozen experimental BM25 selection used only for the 27B arm.",
    )
    upgrade_benchmark_parser.add_argument("--dry-run", action="store_true")
    general_parser = model_upgrade_actions.add_parser(
        "general-diagnostic",
        help="Compare base and adapter on the frozen small general-behaviour fixture",
    )
    general_parser.add_argument("--acquisition-run", required=True)
    judge_preflight_parser = model_upgrade_actions.add_parser(
        "judge-preflight",
        help="Run one developer-only Gemma structured-output check",
    )
    judge_preflight_parser.add_argument(
        "--output",
        default="artifacts/model-upgrade/preflight/gemma4-structured-output.json",
    )
    judge_parser = model_upgrade_actions.add_parser(
        "judge",
        help="Grade one raw benchmark with frozen local Gemma",
    )
    judge_parser.add_argument("--benchmark", required=True)
    judge_parser.add_argument(
        "--eval-version",
        choices=["v1", "v2"],
        default="v1",
    )
    judge_parser.add_argument(
        "--output-dir",
        default="artifacts/model-upgrade/gemma-advisories",
    )
    compare_parser = model_upgrade_actions.add_parser(
        "compare",
        help="Build the bounded 4B/27B model-upgrade comparison report",
    )
    compare_parser.add_argument("--advisory-4b", required=True)
    compare_parser.add_argument("--advisory-27b", required=True)
    compare_parser.add_argument("--general-4b", required=True)
    compare_parser.add_argument("--general-27b", required=True)
    compare_parser.add_argument(
        "--output-dir",
        default="artifacts/model-upgrade/report",
    )
    bridge_parser = model_upgrade_actions.add_parser(
        "bridge",
        help="Compare published Cursor-model labels with Gemma on the same outputs",
    )
    bridge_parser.add_argument("--original-advisory", required=True)
    bridge_parser.add_argument("--gemma-advisory", required=True)
    bridge_parser.add_argument(
        "--output-dir",
        default="artifacts/model-upgrade/historical-bridge",
    )

    specialization_parser = subparsers.add_parser(
        "specialization",
        help="Run governed stages of company-task-specialization/v1",
    )
    specialization_actions = specialization_parser.add_subparsers(
        dest="specialization_action",
        required=True,
    )
    specialization_actions.add_parser(
        "validate-contract",
        help="Verify the frozen task, split, source, and development-case hashes",
    )
    specialization_actions.add_parser(
        "validate-evaluator-v2",
        help="Verify the draft obligation-review evaluator contract and seed schema",
    )
    specialization_actions.add_parser(
        "validate-qualification-v2",
        help="Verify the frozen supplier-workflow qualification contract",
    )
    qualification_run_parser = specialization_actions.add_parser(
        "run-qualification-v2",
        help="Compare three bounded Qwen27 prompt strategies on fresh cases",
    )
    qualification_run_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v2",
    )
    qualification_review_parser = specialization_actions.add_parser(
        "prepare-qualification-review",
        help="Blind a qualification-v2 comparison for GPT advisory review",
    )
    qualification_review_parser.add_argument("--comparison", required=True)
    qualification_review_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v2/reviews",
    )
    qualification_score_parser = specialization_actions.add_parser(
        "score-qualification-review",
        help="Unblind GPT qualification labels and apply the frozen gate",
    )
    qualification_score_parser.add_argument("--comparison", required=True)
    qualification_score_parser.add_argument("--packet", required=True)
    qualification_score_parser.add_argument("--mapping", required=True)
    qualification_score_parser.add_argument("--advisory", required=True)
    qualification_score_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v2/decisions",
    )
    specialization_actions.add_parser(
        "validate-remedy-v3",
        help="Verify the frozen remedy-logic experiment and fresh cases",
    )
    remedy_run_parser = specialization_actions.add_parser(
        "run-remedy-v3",
        help="Run the two-arm single-call Qwen27 remedy-logic experiment",
    )
    remedy_run_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v3-remedy-logic",
    )
    remedy_review_parser = specialization_actions.add_parser(
        "prepare-remedy-review",
        help="Blind remedy-logic outputs for arm-label-blinded GPT review",
    )
    remedy_review_parser.add_argument("--comparison", required=True)
    remedy_review_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v3-remedy-logic/reviews",
    )
    remedy_score_parser = specialization_actions.add_parser(
        "score-remedy-review",
        help="Apply the frozen remedy-logic gate to completed GPT labels",
    )
    remedy_score_parser.add_argument("--comparison", required=True)
    remedy_score_parser.add_argument("--packet", required=True)
    remedy_score_parser.add_argument("--mapping", required=True)
    remedy_score_parser.add_argument("--advisory", required=True)
    remedy_score_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v3-remedy-logic/decisions",
    )
    specialization_actions.add_parser(
        "validate-fact-state-v4",
        help="Verify the frozen fact-state contrast experiment",
    )
    fact_state_run_parser = specialization_actions.add_parser(
        "run-fact-state-v4",
        help="Run the two-arm single-call Qwen27 fact-state comparison",
    )
    fact_state_run_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v4-fact-state",
    )
    fact_state_review_parser = specialization_actions.add_parser(
        "prepare-fact-state-review",
        help="Blind fact-state outputs for arm-label-blinded GPT review",
    )
    fact_state_review_parser.add_argument("--comparison", required=True)
    fact_state_review_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v4-fact-state/reviews",
    )
    fact_state_score_parser = specialization_actions.add_parser(
        "score-fact-state-review",
        help="Apply frozen learning and qualification rules to GPT labels",
    )
    fact_state_score_parser.add_argument("--comparison", required=True)
    fact_state_score_parser.add_argument("--packet", required=True)
    fact_state_score_parser.add_argument("--mapping", required=True)
    fact_state_score_parser.add_argument("--advisory", required=True)
    fact_state_score_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v4-fact-state/decisions",
    )
    fact_state_regression_parser = specialization_actions.add_parser(
        "run-fact-state-regression-v4",
        help="Run the sole v3 regression pass after a successful fresh decision",
    )
    fact_state_regression_parser.add_argument("--fresh-decision", required=True)
    fact_state_regression_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v4-fact-state/regression",
    )
    specialization_pilot_parser = specialization_actions.add_parser(
        "pilot",
        help="Run the bounded 4B baseline, Qwen27 repair, and Gemma advisory batch",
    )
    specialization_pilot_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/v1",
    )
    specialization_audit_parser = specialization_actions.add_parser(
        "prepare-audit",
        help="Blind the stopped pilot's teacher and repair outputs for human audit",
    )
    specialization_audit_parser.add_argument("--pilot", required=True)
    specialization_audit_parser.add_argument(
        "--output-root",
        default="artifacts/company-task-specialization/human-audit",
    )
    specialization_validate_audit = specialization_actions.add_parser(
        "validate-audit",
        help="Validate a completed human specialization audit overlay",
    )
    specialization_validate_audit.add_argument("--packet", required=True)
    specialization_validate_audit.add_argument("--mapping", required=True)
    specialization_validate_audit.add_argument("--overlay", required=True)
    specialization_validate_audit.add_argument("--output", required=True)
    specialization_real_seed = specialization_actions.add_parser(
        "validate-real-seed",
        help="Validate a private human-reviewed one-workflow real-work seed",
    )
    specialization_real_seed.add_argument("--input", required=True)
    specialization_real_seed.add_argument("--output", required=True)

    grade_parser = subparsers.add_parser(
        "grade",
        help="Apply deterministic strict/provenance grading to a raw benchmark artifact",
    )
    grade_parser.add_argument("--benchmark", required=True)
    grade_parser.add_argument(
        "--eval-version",
        choices=["v1", "v2"],
        default="v1",
    )
    grade_parser.add_argument(
        "--parametric-source-record",
        action="append",
        default=[],
        help="Record ID authorized for provenance from a parametric adapter.",
    )
    grade_parser.add_argument(
        "--output-dir",
        default="artifacts/grading",
    )

    chat_parser = subparsers.add_parser("chat", help="DISABLED legacy adapter chat")
    chat_parser.add_argument("--adapter", default=None)
    chat_parser.add_argument("--model", default=None)
    chat_parser.add_argument("--domain", default="global")
    chat_parser.add_argument("--max-tokens", type=int, default=320)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    try:
        if args.command in LEGACY_COMMANDS:
            block_legacy_command(args.command)
        elif args.command == "doctor":
            _doctor()
        elif args.command == "benchmark":
            _benchmark(root, args)
        elif args.command == "review":
            _review(root, args)
        elif args.command == "benchmark-review":
            _benchmark_review(root, args)
        elif args.command == "grade":
            _grade(root, args)
        elif args.command == "acquire":
            _acquire(root, args)
        elif args.command == "model-upgrade":
            _model_upgrade(root, args)
        elif args.command == "specialization":
            _specialization(root, args)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0


def _doctor() -> None:
    report = doctor_report()
    hardware = report["hardware"]
    table = Table(title="MLX hardware report")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("System", f"{hardware['system']} {hardware['machine']}")
    table.add_row("Chip", str(hardware["chip"]))
    table.add_row("Unified/physical memory", f"{hardware['memory_gib']} GiB")
    table.add_row("GPU cores", str(hardware["gpu_cores"] or "not detected"))
    table.add_row("OS version", str(hardware["os_version"]))
    table.add_row("Apple Silicon", str(hardware["is_apple_silicon"]))
    table.add_row("Python", report["python"])
    table.add_row("MLX-LM installed", str(report["mlx_lm_installed"]))
    table.add_row("MLX-LM version", str(report["mlx_lm_version"] or "not installed"))
    table.add_row("Training status", "BLOCKED — legacy pipeline is scientifically invalid")
    console.print(table)


def _compile(root: Path, args: argparse.Namespace) -> None:
    if args.include_restricted and not args.acknowledge_weight_acl_risk:
        raise ValueError(
            "Restricted knowledge is excluded because model weights cannot enforce "
            "source-level ACLs. "
            "Repeat with --acknowledge-weight-acl-risk only after an explicit governance decision."
        )
    result = compile_knowledge(
        root / args.knowledge_dir,
        root / args.output_dir,
        seed=args.seed,
        include_restricted=args.include_restricted,
        per_domain=not args.no_per_domain,
    )
    console.print(
        f"[green]Compiled[/green] {result.records_included} records "
        f"({result.records_excluded} excluded) across {', '.join(result.domains)}."
    )
    console.print(f"Manifest: {result.manifest_path}")


def _train(root: Path, args: argparse.Namespace) -> None:
    commands = train_pipeline(
        root,
        stage=args.stage,
        preset_name=args.preset,
        model_override=args.model,
        domain=args.domain,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        console.print("[bold]Commands that would run:[/bold]")
        for command in commands:
            console.print("  " + shlex.join(command))
    else:
        console.print(f"[green]Completed[/green] {len(commands)} training stage(s).")


def _evaluate(root: Path, args: argparse.Namespace) -> None:
    registry_path = root / "artifacts" / "registry" / "adapters.json"
    registered = find_adapter(registry_path, domain=args.domain, stage="recover")
    adapter_path: Path | None
    if args.base_only:
        adapter_path = None
    elif args.adapter:
        adapter_path = Path(args.adapter).expanduser().resolve()
    elif registered:
        adapter_path = Path(str(registered["adapter_path"]))
    else:
        candidate = root / "artifacts" / "adapters" / "recover"
        adapter_path = candidate if candidate.exists() else None

    if args.model:
        model_name = args.model
    elif registered:
        model_name = str(registered["base_model"])
    else:
        model_name = resolve_preset("auto").model

    suite_root = root / "artifacts" / "datasets"
    if args.domain != "global":
        suite_root = suite_root / "domains" / slugify(args.domain)
    suite_path = suite_root / "eval" / f"{args.suite}.jsonl"
    if args.suite == "general" and args.domain != "global":
        suite_path = root / "artifacts" / "datasets" / "eval" / "general.jsonl"

    if adapter_path is None and args.skip_base:
        raise ValueError("No adapter was found and --skip-base would leave nothing to evaluate")

    report_path = evaluate_models(
        model_name=model_name,
        suite_path=suite_path,
        output_dir=root / "artifacts" / "eval" / "results",
        adapter_path=adapter_path,
        include_base=not args.skip_base,
        max_tokens=args.max_tokens,
    )
    console.print(f"[green]Evaluation written:[/green] {report_path}")


def _route(root: Path, args: argparse.Namespace) -> None:
    decision = route_query(
        query=args.query,
        knowledge_dir=root / "knowledge",
        registry_path=root / "artifacts" / "registry" / "adapters.json",
        threshold=args.threshold,
    )
    console.print_json(json.dumps(decision.to_dict()))


def _benchmark(root: Path, args: argparse.Namespace) -> None:
    eval_dir = root / "knowledge" / "eval_frozen"
    if args.eval_version == "v2":
        eval_dir = eval_dir / "v2"
    freeze_problems = verify_frozen_assets(eval_dir)
    if freeze_problems:
        raise ValueError(
            "Frozen evaluation assets failed verification:\n" + "\n".join(freeze_problems)
        )

    freeze_manifest = json.loads((eval_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    records = load_records(root / "knowledge")
    suites = load_eval_suites(eval_dir)
    selected_arms = tuple(args.arm or DEFAULT_ARMS)
    acquisition_adapter = None
    if args.acquisition_run:
        run_manifest = Path(args.acquisition_run).expanduser()
        if not run_manifest.is_absolute():
            run_manifest = root / run_manifest
        acquisition_adapter = load_verified_acquisition_adapter(run_manifest)
        if acquisition_adapter.model_id != args.model:
            raise ValueError("Acquisition adapter base model does not match --model")
    authoritative_by_id = {record.id: record for record in (*records, *suites.holdout_records)}
    for record in suites.supersession_current_records:
        authoritative_by_id[record.id] = record
    authoritative = tuple(sorted(authoritative_by_id.values(), key=lambda record: record.id))
    selection = None
    if args.bm25_selection:
        selection = bind_bm25_selection(
            Path(args.bm25_selection),
            authoritative,
            validation_dataset_path=(
                root / "knowledge" / "retrieval_validation" / "v1" / "queries.jsonl"
            ),
            calibration_report_path=(
                root / "artifacts" / "retrieval-calibration" / "v1" / "report.json"
            ),
        )
    bm25_decision, decision_status = resolve_default_bm25_decision(
        default_bm25_decision_path(root),
        authoritative,
    )
    decision_payload = bm25_decision_artifact_payload(bm25_decision, decision_status)
    config = BenchmarkConfig(
        suites=tuple(
            args.suite or (("supersession",) if args.eval_version == "v2" else DEFAULT_SUITES)
        ),
        arms=selected_arms,
        max_context_bytes=args.max_context_bytes,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_tokens,
        model_name=args.model,
        bm25_selection=selection,
        parametric_adapter_path=(
            str(acquisition_adapter.adapter_path) if acquisition_adapter else None
        ),
        parametric_adapter_hash=(acquisition_adapter.adapter_hash if acquisition_adapter else None),
        parametric_source_record_ids=(
            acquisition_adapter.source_record_ids if acquisition_adapter else ()
        ),
        parametric_sensitivity=(
            acquisition_adapter.inherited_classification if acquisition_adapter else None
        ),
        thinking_enabled=False,
        temperature=0.0,
        experiment_id=args.experiment_id,
    )
    count_tokens, tokenizer_identity = load_benchmark_tokenizer(
        args.model,
        revision=(
            acquisition_adapter.model_revision if acquisition_adapter else args.model_revision
        ),
    )
    plan = build_benchmark_plan(
        records,
        suites,
        config=config,
        count_tokens=count_tokens,
    )

    if args.dry_run:
        table = Table(title="Answer-blind benchmark plan")
        table.add_column("Arm")
        table.add_column("Cases", justify="right")
        table.add_column("Source required", justify="right")
        table.add_column("Context too large", justify="right")
        for arm in config.arms:
            cases = [case for case in plan if case.arm == arm]
            table.add_row(
                arm,
                str(len(cases)),
                str(sum(case.context_action == "source_required" for case in cases)),
                str(sum(case.context_action == "context_too_large" for case in cases)),
            )
        console.print(table)
        console.print(
            f"Fixture hash: {freeze_manifest['combined_hash']}\n"
            f"Source snapshot hash: {source_snapshot_hash(authoritative)}\n"
            f"Tokenizer: {tokenizer_identity.loader} "
            f"({tokenizer_identity.tokenizer_class}) revision={tokenizer_identity.revision}\n"
            f"BM25 default/production decision: {decision_status}\n"
            f"BM25 active research selection: "
            f"{selection.status if selection is not None and 'bm25' in selected_arms else 'none'}\n"
            "Tokenizer-only loading was used; no model weights were loaded and no "
            "answers were generated."
        )
        return

    ordinary_plan = tuple(case for case in plan if case.arm != "parametric")
    parametric_plan = tuple(case for case in plan if case.arm == "parametric")
    ordinary_results = ()
    if ordinary_plan:
        backend = MLXBenchmarkBackend(
            args.model,
            revision=tokenizer_identity.revision,
        )
        try:
            require_matching_model_revision(tokenizer_identity, backend.revision)
            ordinary_results = run_benchmark_plan(
                ordinary_plan,
                backend,
                max_output_tokens=config.max_output_tokens,
            )
        finally:
            backend.close()
    parametric_results = ()
    if parametric_plan:
        if acquisition_adapter is None:
            raise ValueError("Parametric arm requires --acquisition-run")
        adapter_backend = MLXBenchmarkBackend(
            args.model,
            revision=tokenizer_identity.revision,
            adapter_path=str(acquisition_adapter.adapter_path),
        )
        try:
            require_matching_model_revision(
                tokenizer_identity,
                adapter_backend.revision,
            )
            parametric_results = run_benchmark_plan(
                parametric_plan,
                adapter_backend,
                max_output_tokens=config.max_output_tokens,
                parametric_backend=adapter_backend,
            )
        finally:
            adapter_backend.close()
    results_by_key = {
        (result.case.question_id, result.case.arm): result
        for result in (*ordinary_results, *parametric_results)
    }
    results = tuple(results_by_key[(case.question_id, case.arm)] for case in plan)
    artifact = write_benchmark_artifact(
        output_dir=root / "artifacts" / "benchmark",
        model_name=config.model_name,
        config=config,
        fixture_hash=str(freeze_manifest["combined_hash"]),
        results=results,
        tokenizer_identity=tokenizer_identity,
        source_hash=source_snapshot_hash(authoritative),
        index_hash=index_payload_hash(authoritative),
        bm25_decision=decision_payload,
        run_id=args.experiment_id,
    )
    console.print(f"[green]Ungraded benchmark written:[/green] {artifact}")


def _review(root: Path, args: argparse.Namespace) -> None:
    packet = Path(args.packet).expanduser()
    mapping = Path(args.mapping).expanduser()
    state_root = Path(args.state_root).expanduser() if args.state_root is not None else None
    if not packet.is_absolute():
        packet = root / packet
    if not mapping.is_absolute():
        mapping = root / mapping
    if state_root is not None and not state_root.is_absolute():
        state_root = root / state_root
    serve_review_ui(
        root=root,
        packet_path=packet,
        mapping_path=mapping,
        reviewer_id=args.reviewer,
        port=args.port,
        open_browser=not args.no_browser,
        state_root=state_root,
    )


def _benchmark_review(root: Path, args: argparse.Namespace) -> None:
    benchmark_path = _rooted_path(root, args.benchmark)
    grading_path = _rooted_path(root, args.grading)
    if args.benchmark_review_action == "prepare":
        output_root = _rooted_path(root, args.output_root)
        eval_dir = root / "knowledge" / "eval_frozen"
        if args.eval_version == "v2":
            eval_dir = eval_dir / "v2"
        prepared = prepare_benchmark_review(
            benchmark_path=benchmark_path,
            deterministic_grading_path=grading_path,
            eval_dir=eval_dir,
            output_root=output_root,
        )
        state_root = root / "artifacts" / "human-reviews" / benchmark_path.stem
        console.print(f"[green]Blinded packet written:[/green] {prepared.packet_path}")
        console.print(f"Private mapping: {prepared.mapping_path}")
        console.print(f"Cases: {prepared.case_count}")
        console.print(
            "Next: emmlx review "
            f"--packet {prepared.packet_path} "
            f"--mapping {prepared.mapping_path} "
            f"--state-root {state_root} "
            '--reviewer "Your Name"'
        )
        return
    if args.benchmark_review_action == "report":
        packet_path = _rooted_path(root, args.packet)
        mapping_path = _rooted_path(root, args.mapping)
        overlay_path = _rooted_path(root, args.overlay)
        overlay_manifest_path = (
            _rooted_path(root, args.overlay_manifest)
            if args.overlay_manifest is not None
            else overlay_path.with_suffix(".manifest.json")
        )
        output_dir = _rooted_path(root, args.output_dir)
        paths = write_benchmark_review_report(
            benchmark_path=benchmark_path,
            deterministic_grading_path=grading_path,
            packet_path=packet_path,
            mapping_path=mapping_path,
            overlay_path=overlay_path,
            overlay_manifest_path=overlay_manifest_path,
            output_dir=output_dir,
        )
        console.print(f"[green]Completed smoke decision:[/green] {paths.decision}")
        console.print(f"JSON report: {paths.json_path}")
        console.print(f"Human-readable report: {paths.markdown_path}")
        console.print(
            "[bold yellow]Single-human diagnostic only; promotion and "
            "headline claims remain blocked.[/bold yellow]"
        )
        return
    raise ValueError(f"Unsupported benchmark-review action: {args.benchmark_review_action}")


def _grade(root: Path, args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    eval_dir = root / "knowledge" / "eval_frozen"
    if args.eval_version == "v2":
        eval_dir = eval_dir / "v2"
    suites = load_eval_suites(eval_dir)
    report = grade_benchmark_artifact(
        benchmark_path,
        eval_dir,
        suites,
        allowed_parametric_record_ids=args.parametric_source_record,
    )
    json_path, markdown_path = write_grading_report(report, output_dir)
    console.print(f"[green]Deterministic grading written:[/green] {json_path}")
    console.print(f"Human-readable report: {markdown_path}")
    console.print(
        "[bold yellow]Semantic review unavailable; promotion remains blocked.[/bold yellow]"
    )


def _acquire(root: Path, args: argparse.Namespace) -> None:
    is_upgrade = args.profile == MODEL_UPGRADE_PROFILE
    model_id = args.model or (QWEN_27B_MODEL_ID if is_upgrade else DEFAULT_ACQUISITION_MODEL)
    target_exposures = args.target_exposures
    if target_exposures is None:
        target_exposures = MODEL_UPGRADE_CONFIG.target_exposures_per_record if is_upgrade else 24
    semantic_backend = MLXEmbeddingBackend(
        args.semantic_model,
        revision=(
            args.semantic_model_revision
            or (
                DEFAULT_SEMANTIC_MODEL_REVISION
                if is_upgrade and args.semantic_model == DEFAULT_SEMANTIC_MODEL
                else None
            )
        ),
    )
    effective_batch = args.batch_size * args.grad_accumulation
    compilation = compile_acquisition_dataset(
        knowledge_dir=root / "knowledge",
        eval_dir=root / "knowledge" / "eval_frozen",
        output_root=root / "artifacts" / "acquisition",
        semantic_backend=semantic_backend,
        profile=args.profile,
        target_exposures_per_fact=target_exposures,
        effective_batch_size=effective_batch,
        micro_batch_size=args.batch_size,
        seed=args.seed,
        study_views_root=(_rooted_path(root, args.study_views_root) if is_upgrade else None),
    )
    model_revision = resolve_huggingface_revision(
        model_id,
        revision=args.model_revision,
    )
    architecture = model_profile(model_id, model_revision)
    preflight_path = preflight_hash = None
    if is_upgrade and args.execute:
        preflight_path, preflight_hash = _require_model_upgrade_preflight(
            root,
            architecture,
            compilation,
        )
    config = AcquisitionConfig(
        model_id=model_id,
        model_revision=model_revision,
        rank=args.rank,
        scale=MODEL_UPGRADE_CONFIG.scale if is_upgrade else 2.0,
        learning_rate=(MODEL_UPGRADE_CONFIG.learning_rate if is_upgrade else 2e-4),
        dropout=MODEL_UPGRADE_CONFIG.dropout if is_upgrade else 0.05,
        num_layers=architecture.num_layers,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation,
        max_seq_length=MODEL_UPGRADE_CONFIG.max_sequence_length,
        seed=args.seed,
        profile=args.profile,
        weight_decay=MODEL_UPGRADE_CONFIG.weight_decay,
        preflight_report_path=preflight_path,
        preflight_report_hash=preflight_hash,
    )
    run = run_acquisition(
        root=root,
        compilation=compilation,
        config=config,
        execute=args.execute,
    )
    status = "trained" if run.executed else "dry-run only"
    console.print(f"[green]Revised acquisition {status}:[/green] {run.run_manifest_path}")
    console.print(f"Config: {run.config_path}")
    if run.executed:
        console.print(f"Adapter: {run.adapter_path}")
    else:
        console.print(
            "[yellow]No training ran. Repeat with --execute only after the "
            "required disposable preflight has passed.[/yellow]"
        )


def _model_upgrade(root: Path, args: argparse.Namespace) -> None:
    if args.model_upgrade_action in {
        "author-curriculum",
        "preflight",
        "train",
        "benchmark",
        "general-diagnostic",
        "judge-preflight",
        "judge",
    }:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.model_upgrade_action == "author-curriculum":
        output_dir = _rooted_path(root, args.output_dir)
        freeze_manifest = json.loads(
            (root / "knowledge" / "eval_frozen" / "freeze_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        views, manifest = author_model_upgrade_curriculum(
            records=load_records(root / "knowledge"),
            output_dir=output_dir,
            fixture_hash=str(freeze_manifest["combined_hash"]),
        )
        console.print(f"[green]Curriculum authored:[/green] {views}")
        console.print(f"Manifest: {manifest}")
        return
    if args.model_upgrade_action == "preflight":
        _model_upgrade_preflight(root, args)
        return
    if args.model_upgrade_action == "train":
        _model_upgrade_train(root, args)
        return
    if args.model_upgrade_action == "benchmark":
        _model_upgrade_benchmark(root, args)
        return
    if args.model_upgrade_action == "general-diagnostic":
        adapter = load_verified_acquisition_adapter(_rooted_path(root, args.acquisition_run))
        report = run_general_diagnostic(
            rows=read_jsonl(root / "knowledge" / "general_eval.jsonl"),
            adapter=adapter,
            output_dir=root / "artifacts" / "model-upgrade" / "general-diagnostics",
        )
        console.print(f"[green]General diagnostic written:[/green] {report}")
        return
    if args.model_upgrade_action == "judge-preflight":
        report = run_gemma_preflight(output_path=_rooted_path(root, args.output))
        console.print(f"[green]Gemma preflight passed:[/green] {report}")
        return
    if args.model_upgrade_action == "judge":
        eval_dir = root / "knowledge" / "eval_frozen"
        if args.eval_version == "v2":
            eval_dir = eval_dir / "v2"
        json_path, markdown_path = run_gemma_advisory(
            benchmark_path=_rooted_path(root, args.benchmark),
            eval_dir=eval_dir,
            knowledge_dir=root / "knowledge",
            output_dir=_rooted_path(root, args.output_dir),
        )
        console.print(f"[green]Gemma advisory written:[/green] {json_path}")
        console.print(f"Human-readable report: {markdown_path}")
        console.print(
            "[bold yellow]Model-only advisory; not human-approved and not "
            "promotion eligible.[/bold yellow]"
        )
        return
    if args.model_upgrade_action == "compare":
        json_path, markdown_path = write_model_upgrade_comparison(
            advisory_4b_path=_rooted_path(root, args.advisory_4b),
            advisory_27b_path=_rooted_path(root, args.advisory_27b),
            general_4b_path=_rooted_path(root, args.general_4b),
            general_27b_path=_rooted_path(root, args.general_27b),
            output_dir=_rooted_path(root, args.output_dir),
        )
        console.print(f"[green]Comparison written:[/green] {json_path}")
        console.print(f"Human-readable report: {markdown_path}")
        return
    if args.model_upgrade_action == "bridge":
        paths = write_model_regrading_bridge(
            original_model_advisory_path=_rooted_path(root, args.original_advisory),
            gemma_advisory_path=_rooted_path(root, args.gemma_advisory),
            output_dir=_rooted_path(root, args.output_dir),
        )
        console.print(f"[green]Historical bridge written:[/green] {paths.json_path}")
        console.print(f"Human-readable report: {paths.markdown_path}")
        return
    raise ValueError(f"Unsupported model-upgrade action: {args.model_upgrade_action}")


def _specialization(root: Path, args: argparse.Namespace) -> None:
    if args.specialization_action == "validate-contract":
        assets = load_specialization_assets(root)
        console.print(
            f"[green]Specialization contract verified:[/green] "
            f"{len(assets.cases)} synthetic development cases"
        )
        console.print(f"Protocol SHA-256: {assets.protocol_hash}")
        console.print(f"Cases SHA-256: {assets.cases_hash}")
        console.print(
            "[bold yellow]No human labels; development assets are not training eligible."
            "[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-evaluator-v2":
        manifest = verify_evaluator_v2_contract(root)
        console.print(f"[green]Evaluator-v2 contract verified:[/green] {manifest['status']}")
        console.print(
            "[bold yellow]Awaiting human validation; teacher requalification "
            "and student training remain blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-qualification-v2":
        assets = load_qualification_assets(root)
        console.print(
            "[green]Qualification-v2 contract verified:[/green] "
            f"{len(assets.cases)} fresh cases; {assets.protocol['workflow']}"
        )
        console.print(
            "[bold yellow]Model-advisory-only qualification; "
            "training and stronger-teacher work remain blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "run-qualification-v2":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifacts = run_qualification_comparison(
            root=root,
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Qualification-v2 comparison:[/green] {artifacts.report_path}")
        console.print(
            "[bold yellow]Awaiting blinded GPT advisory review; "
            "not training or production evidence.[/bold yellow]"
        )
        return
    if args.specialization_action == "prepare-qualification-review":
        artifacts = prepare_qualification_review(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Qualification review packet:[/green] {artifacts.packet_path}")
        console.print(f"Private mapping: {artifacts.mapping_path}")
        console.print(f"Model-advisory template: {artifacts.template_path}")
        console.print(
            "[bold yellow]Share only the packet. Prompt arms and references "
            "remain private.[/bold yellow]"
        )
        return
    if args.specialization_action == "score-qualification-review":
        artifacts = score_qualification_advisory(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            packet_path=_rooted_path(root, args.packet),
            mapping_path=_rooted_path(root, args.mapping),
            advisory_path=_rooted_path(root, args.advisory),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Qualification decision:[/green] {artifacts.markdown_path}")
        console.print(
            "[bold yellow]Model-advisory-only decision; "
            "training and stronger-teacher work remain blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-remedy-v3":
        assets = load_remedy_experiment_assets(root)
        console.print(
            f"[green]Remedy-logic v3 contract verified:[/green] {len(assets.cases)} fresh cases"
        )
        console.print(
            "[bold yellow]Synthetic model-advisory experiment; "
            "training remains blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "run-remedy-v3":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifacts = run_remedy_experiment(
            root=root,
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Remedy-logic comparison:[/green] {artifacts.report_path}")
        console.print(
            "[bold yellow]Awaiting arm-label-blinded GPT review; "
            "training remains blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "prepare-remedy-review":
        artifacts = prepare_remedy_review(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Remedy review packet:[/green] {artifacts.packet_path}")
        console.print(f"Private mapping: {artifacts.mapping_path}")
        console.print(f"Model-advisory template: {artifacts.template_path}")
        console.print(
            "[bold yellow]Share only the packet. The schemas may reveal treatment, "
            "so only arm-label blindness is claimed.[/bold yellow]"
        )
        return
    if args.specialization_action == "score-remedy-review":
        artifacts = score_remedy_review(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            packet_path=_rooted_path(root, args.packet),
            mapping_path=_rooted_path(root, args.mapping),
            advisory_path=_rooted_path(root, args.advisory),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Remedy-logic decision:[/green] {artifacts.markdown_path}")
        console.print(
            "[bold yellow]Passing may designate a bounded advisory configuration, "
            "but does not authorize training.[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-fact-state-v4":
        assets = load_fact_state_assets(root)
        console.print(
            "[green]Fact-state v4 contract verified:[/green] "
            f"{len(assets.cases)} cases in 8 contrast pairs"
        )
        console.print(
            "[bold yellow]One fresh comparison is allowed; training remains blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "run-fact-state-v4":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifacts = run_fact_state_comparison(
            root=root,
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Fact-state comparison:[/green] {artifacts.report_path}")
        console.print(
            "[bold yellow]Awaiting arm-label-blinded GPT review; no regression "
            "or training is yet authorized.[/bold yellow]"
        )
        return
    if args.specialization_action == "prepare-fact-state-review":
        artifacts = prepare_fact_state_review(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Fact-state review packet:[/green] {artifacts.packet_path}")
        console.print(f"Private mapping: {artifacts.mapping_path}")
        console.print(f"Model-advisory template: {artifacts.template_path}")
        console.print("[bold yellow]Share only the blinded packet.[/bold yellow]")
        return
    if args.specialization_action == "score-fact-state-review":
        artifacts = score_fact_state_review(
            root=root,
            comparison_path=_rooted_path(root, args.comparison),
            packet_path=_rooted_path(root, args.packet),
            mapping_path=_rooted_path(root, args.mapping),
            advisory_path=_rooted_path(root, args.advisory),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Fact-state decision:[/green] {artifacts.markdown_path}")
        console.print(
            "[bold yellow]Only a fresh success may authorize one v3 regression "
            "pass; training remains blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "run-fact-state-regression-v4":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifacts = run_fact_state_regression(
            root=root,
            fresh_decision_path=_rooted_path(root, args.fresh_decision),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Fact-state v3 regression:[/green] {artifacts.report_path}")
        console.print(
            "[bold yellow]This is the sole conditional regression pass; "
            "training remains blocked.[/bold yellow]"
        )
        return
    if args.specialization_action == "pilot":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        artifacts = run_specialization_pilot(
            root=root,
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Specialization pilot written:[/green] {artifacts.report_path}")
        console.print(f"Candidate repair batch: {artifacts.candidate_repairs_path}")
        console.print(f"Human-readable report: {artifacts.markdown_path}")
        console.print(
            "[bold yellow]Synthetic, model-reviewed development evidence only; "
            "not training or deployment eligible.[/bold yellow]"
        )
        return
    if args.specialization_action == "prepare-audit":
        artifacts = prepare_specialization_audit(
            root=root,
            pilot_path=_rooted_path(root, args.pilot),
            output_root=_rooted_path(root, args.output_root),
        )
        console.print(f"[green]Blinded audit packet:[/green] {artifacts.packet_path}")
        console.print(f"Private mapping: {artifacts.mapping_path}")
        console.print(f"Review template: {artifacts.template_path}")
        console.print(
            "[bold yellow]Share the packet, not the private mapping. "
            "Human review does not replace the stopped pilot's scores.[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-audit":
        report = validate_specialization_audit_overlay(
            packet_path=_rooted_path(root, args.packet),
            mapping_path=_rooted_path(root, args.mapping),
            overlay_path=_rooted_path(root, args.overlay),
            output_path=_rooted_path(root, args.output),
        )
        console.print(f"[green]Human audit validated:[/green] {report}")
        console.print(
            "[bold yellow]Single-review development evidence; "
            "second review and adjudication remain required.[/bold yellow]"
        )
        return
    if args.specialization_action == "validate-real-seed":
        report = validate_real_work_seed(
            root=root,
            seed_path=_rooted_path(root, args.input),
            output_path=_rooted_path(root, args.output),
        )
        console.print(f"[green]Private real-work seed manifest:[/green] {report}")
        console.print("Case content remains under the ignored knowledge/private boundary.")
        return
    raise ValueError(f"Unsupported specialization action: {args.specialization_action}")


def _model_upgrade_preflight(root: Path, args: argparse.Namespace) -> None:
    semantic_backend = MLXEmbeddingBackend(
        args.semantic_model,
        revision=(
            args.semantic_model_revision
            or (
                DEFAULT_SEMANTIC_MODEL_REVISION
                if args.semantic_model == DEFAULT_SEMANTIC_MODEL
                else None
            )
        ),
    )
    compilation = compile_acquisition_dataset(
        knowledge_dir=root / "knowledge",
        eval_dir=root / "knowledge" / "eval_frozen",
        output_root=root / "artifacts" / "acquisition",
        semantic_backend=semantic_backend,
        profile=MODEL_UPGRADE_PROFILE,
        target_exposures_per_fact=MODEL_UPGRADE_CONFIG.target_exposures_per_record,
        effective_batch_size=(
            MODEL_UPGRADE_CONFIG.batch_size * MODEL_UPGRADE_CONFIG.gradient_accumulation_steps
        ),
        micro_batch_size=MODEL_UPGRADE_CONFIG.batch_size,
        seed=MODEL_UPGRADE_CONFIG.seed,
        study_views_root=_rooted_path(root, args.study_views_root),
    )
    architecture = model_profile(
        args.model,
        (model_profile(args.model, model_profile_revision(args.model)).revision),
    )
    model_snapshot = materialize_model_snapshot(
        architecture.model_id,
        architecture.revision,
    )
    config = AcquisitionConfig(
        model_id=architecture.model_id,
        model_revision=architecture.revision,
        rank=MODEL_UPGRADE_CONFIG.rank,
        scale=MODEL_UPGRADE_CONFIG.scale,
        learning_rate=MODEL_UPGRADE_CONFIG.learning_rate,
        dropout=MODEL_UPGRADE_CONFIG.dropout,
        num_layers=architecture.num_layers,
        batch_size=MODEL_UPGRADE_CONFIG.batch_size,
        grad_accumulation_steps=MODEL_UPGRADE_CONFIG.gradient_accumulation_steps,
        max_seq_length=MODEL_UPGRADE_CONFIG.max_sequence_length,
        seed=MODEL_UPGRADE_CONFIG.seed,
        profile=MODEL_UPGRADE_PROFILE,
        weight_decay=MODEL_UPGRADE_CONFIG.weight_decay,
    )
    model_slug = architecture.model_id.rsplit("/", maxsplit=1)[-1].lower()
    identity = f"{model_slug}--{architecture.revision[:10]}--{compilation.schedule.total_rows}rows"
    if args.attempt <= 0:
        raise ValueError("--attempt must be positive")
    if args.attempt > 1:
        identity += f"--attempt-{args.attempt}"
    preflight_dir = root / "artifacts" / "model-upgrade" / "preflight" / identity
    if preflight_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed/disposable preflight: {preflight_dir}"
        )
    adapter_path = preflight_dir / "unused-adapter-output"
    mlx_config = build_mlx_acquisition_config(
        compilation=compilation,
        config=config,
        model_path=model_snapshot,
        adapter_path=adapter_path,
    )
    config_path = preflight_dir / "preflight-config.yaml"
    coverage_path = preflight_dir / "target_coverage.json"
    report_path = preflight_dir / "preflight-report.json"
    log_path = preflight_dir / "preflight.log"
    atomic_write_text(config_path, yaml.safe_dump(mlx_config, sort_keys=False))
    command = (
        sys.executable,
        "-m",
        "enterprise_memory_mlx.acquisition_runtime",
        "preflight",
        "--config",
        str(config_path),
        "--coverage",
        str(coverage_path),
        "--report",
        str(report_path),
    )
    execute_training_command(command, cwd=root, log_path=log_path)
    console.print(f"[green]Preflight passed:[/green] {report_path}")
    console.print(f"Target coverage: {coverage_path}")


def _model_upgrade_train(root: Path, args: argparse.Namespace) -> None:
    train_args = argparse.Namespace(
        profile=MODEL_UPGRADE_PROFILE,
        rank=MODEL_UPGRADE_CONFIG.rank,
        target_exposures=MODEL_UPGRADE_CONFIG.target_exposures_per_record,
        batch_size=MODEL_UPGRADE_CONFIG.batch_size,
        grad_accumulation=MODEL_UPGRADE_CONFIG.gradient_accumulation_steps,
        seed=MODEL_UPGRADE_CONFIG.seed,
        model=args.model,
        model_revision=model_profile_revision(args.model),
        study_views_root="knowledge/model_upgrade_exploratory/v1",
        semantic_model=DEFAULT_SEMANTIC_MODEL,
        semantic_model_revision=DEFAULT_SEMANTIC_MODEL_REVISION,
        execute=True,
    )
    _acquire(root, train_args)


def _model_upgrade_benchmark(root: Path, args: argparse.Namespace) -> None:
    model_slug = "27b" if args.model == QWEN_27B_MODEL_ID else "4b"
    adapter = load_verified_acquisition_adapter(_rooted_path(root, args.acquisition_run))
    if adapter.profile != MODEL_UPGRADE_PROFILE or adapter.model_id != args.model:
        raise ValueError("Model-upgrade benchmark requires its matching v1 adapter")
    run_suffix = adapter.run_identity.rsplit("--", maxsplit=1)[-1]
    arms = ["base", "parametric", "full_context", "oracle"]
    bm25_selection = None
    if args.model == QWEN_27B_MODEL_ID:
        arms.append("bm25")
        bm25_selection = args.bm25_selection
    benchmark_args = argparse.Namespace(
        eval_version="v1",
        suite=["acquisition", "unknown_oos"],
        arm=arms,
        bm25_selection=bm25_selection,
        acquisition_run=args.acquisition_run,
        max_context_bytes=65_536,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        max_tokens=MODEL_UPGRADE_CONFIG.generator_max_output_tokens,
        experiment_id=f"model-upgrade-v1-{model_slug}-seed42-{run_suffix}",
        model=args.model,
        model_revision=model_profile_revision(args.model),
        dry_run=args.dry_run,
    )
    _benchmark(root, benchmark_args)


def model_profile_revision(model_id: str) -> str:
    if model_id == QWEN_4B_MODEL_ID:
        from .experiment_profiles import QWEN_4B_REVISION

        return QWEN_4B_REVISION
    if model_id == QWEN_27B_MODEL_ID:
        from .experiment_profiles import QWEN_27B_REVISION

        return QWEN_27B_REVISION
    raise ValueError(f"Unsupported model-upgrade model: {model_id}")


def _require_model_upgrade_preflight(
    root: Path,
    architecture: Any,
    compilation: Any,
) -> tuple[str, str]:
    model_slug = architecture.model_id.rsplit("/", maxsplit=1)[-1].lower()
    identity = f"{model_slug}--{architecture.revision[:10]}--{compilation.schedule.total_rows}rows"
    report_paths = [
        root
        / "artifacts"
        / "model-upgrade"
        / "preflight"
        / (identity if attempt == 1 else f"{identity}--attempt-{attempt}")
        / "preflight-report.json"
        for attempt in range(10, 0, -1)
    ]
    report_path = next((path for path in report_paths if path.is_file()), None)
    if report_path is None:
        raise FileNotFoundError(
            f"Measured training requires a completed disposable preflight for {identity}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("disposable") is not True
        or report.get("model_id") != architecture.model_id
        or report.get("model_revision") != architecture.revision
        or report.get("model_profile") != architecture.profile_id
        or report.get("execution_revision") != MODEL_UPGRADE_CONFIG.execution_revision
        or report.get("seeding", {}).get("seed") != MODEL_UPGRADE_CONFIG.seed
        or report.get("seeding", {}).get("applied_before_lora_initialization_and_dataset_iteration")
        is not True
        or report.get("training_data", {}).get("row_count") != compilation.schedule.total_rows
        or report.get("training_data", {}).get("truncated_rows") != 0
        or report.get("backward", {}).get("exercised_sequence_length")
        != report.get("training_data", {}).get("longest_rendered_tokens")
        or report.get("backward", {}).get("configured_max_sequence_length")
        != MODEL_UPGRADE_CONFIG.max_sequence_length
    ):
        raise ValueError("Model-upgrade preflight does not match the measured training contract")
    expected_dataset_hash = hashlib.sha256(
        (compilation.dataset_dir / "train.jsonl").read_bytes()
    ).hexdigest()
    if report.get("dataset", {}).get("train_sha256") != expected_dataset_hash:
        raise ValueError("Model-upgrade preflight used different training data")
    return (
        str(report_path.resolve()),
        hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )


def _rooted_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _chat(root: Path, args: argparse.Namespace) -> None:
    registry_path = root / "artifacts" / "registry" / "adapters.json"
    registered = find_adapter(registry_path, domain=args.domain, stage="recover")
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    if adapter_path is None and registered:
        adapter_path = Path(str(registered["adapter_path"]))
    if adapter_path is None:
        candidate = root / "artifacts" / "adapters" / "recover"
        if candidate.exists():
            adapter_path = candidate
    if adapter_path is None:
        raise FileNotFoundError("No recover adapter found. Run: emmlx train --stage all")

    model_name = args.model or (
        str(registered["base_model"]) if registered else resolve_preset("auto").model
    )
    interactive_chat(model_name, adapter_path, max_tokens=args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
