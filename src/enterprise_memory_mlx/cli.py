from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .acquisition_compiler import compile_acquisition_dataset
from .acquisition_training import (
    DEFAULT_ACQUISITION_MODEL,
    AcquisitionConfig,
    load_verified_acquisition_adapter,
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
)
from .compiler import compile_knowledge, load_records
from .evaluation import evaluate_models
from .grading import grade_benchmark_artifact, write_grading_report
from .hardware import PRESETS, resolve_preset
from .inference import interactive_chat
from .legacy_guard import LEGACY_COMMANDS, block_legacy_command
from .registry import find_adapter
from .review_ui import (
    DEFAULT_MAPPING,
    DEFAULT_PACKET,
    serve_review_ui,
)
from .router import route_query
from .semantic_neighbors import (
    DEFAULT_SEMANTIC_MODEL,
    MLXEmbeddingBackend,
)
from .split_contract import load_eval_suites, verify_frozen_assets
from .training import doctor_report, train_pipeline
from .utils import slugify

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

    compile_parser = subparsers.add_parser(
        "compile", help="DISABLED legacy dataset compiler"
    )
    compile_parser.add_argument("--knowledge-dir", default="knowledge")
    compile_parser.add_argument("--output-dir", default="artifacts/datasets")
    compile_parser.add_argument("--seed", type=int, default=42)
    compile_parser.add_argument("--include-restricted", action="store_true")
    compile_parser.add_argument("--acknowledge-weight-acl-risk", action="store_true")
    compile_parser.add_argument("--no-per-domain", action="store_true")

    train_parser = subparsers.add_parser(
        "train", help="DISABLED legacy adapter trainer"
    )
    train_parser.add_argument(
        "--stage",
        choices=["vanilla", "inject", "align", "recover", "all"],
        default="all",
    )
    train_parser.add_argument(
        "--preset", choices=["auto", *PRESETS.keys()], default="auto"
    )
    train_parser.add_argument("--model", default=None, help="Override the preset model")
    train_parser.add_argument("--domain", default="global")
    train_parser.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser(
        "evaluate", help="DISABLED legacy lexical evaluator"
    )
    eval_parser.add_argument(
        "--suite", choices=["domain", "retention", "general"], default="domain"
    )
    eval_parser.add_argument("--adapter", default=None)
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--domain", default="global")
    eval_parser.add_argument("--base-only", action="store_true")
    eval_parser.add_argument("--skip-base", action="store_true")
    eval_parser.add_argument("--max-tokens", type=int, default=220)

    route_parser = subparsers.add_parser(
        "route", help="DISABLED legacy lexical adapter router"
    )
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
        choices=["smoke_non_promotable"],
        default="smoke_non_promotable",
    )
    acquire_parser.add_argument("--rank", type=int, choices=[16], default=16)
    acquire_parser.add_argument("--target-exposures", type=int, default=24)
    acquire_parser.add_argument("--batch-size", type=int, default=1)
    acquire_parser.add_argument("--grad-accumulation", type=int, default=8)
    acquire_parser.add_argument("--seed", type=int, default=42)
    acquire_parser.add_argument("--model", default=DEFAULT_ACQUISITION_MODEL)
    acquire_parser.add_argument("--model-revision", default=None)
    acquire_parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    acquire_parser.add_argument("--semantic-model-revision", default=None)
    acquire_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag only compile and emit a dry-run config.",
    )

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

    chat_parser = subparsers.add_parser(
        "chat", help="DISABLED legacy adapter chat"
    )
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
            "Frozen evaluation assets failed verification:\n"
            + "\n".join(freeze_problems)
        )

    freeze_manifest = json.loads(
        (eval_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
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
            raise ValueError(
                "Acquisition adapter base model does not match --model"
            )
    authoritative_by_id = {
        record.id: record for record in (*records, *suites.holdout_records)
    }
    for record in suites.supersession_current_records:
        authoritative_by_id[record.id] = record
    authoritative = tuple(
        sorted(authoritative_by_id.values(), key=lambda record: record.id)
    )
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
            args.suite
            or (("supersession",) if args.eval_version == "v2" else DEFAULT_SUITES)
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
        parametric_adapter_hash=(
            acquisition_adapter.adapter_hash if acquisition_adapter else None
        ),
        parametric_source_record_ids=(
            acquisition_adapter.source_record_ids if acquisition_adapter else ()
        ),
        parametric_sensitivity=(
            acquisition_adapter.inherited_classification
            if acquisition_adapter
            else None
        ),
    )
    count_tokens, tokenizer_identity = load_benchmark_tokenizer(
        args.model,
        revision=(
            acquisition_adapter.model_revision
            if acquisition_adapter
            else args.model_revision
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
            f"BM25 decision: {decision_status}\n"
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
        require_matching_model_revision(tokenizer_identity, backend.revision)
        ordinary_results = run_benchmark_plan(
            ordinary_plan,
            backend,
            max_output_tokens=config.max_output_tokens,
        )
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
        adapter_backend.close()
    results_by_key = {
        (result.case.question_id, result.case.arm): result
        for result in (*ordinary_results, *parametric_results)
    }
    results = tuple(
        results_by_key[(case.question_id, case.arm)] for case in plan
    )
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
    )
    console.print(f"[green]Ungraded benchmark written:[/green] {artifact}")


def _review(root: Path, args: argparse.Namespace) -> None:
    packet = Path(args.packet).expanduser()
    mapping = Path(args.mapping).expanduser()
    state_root = (
        Path(args.state_root).expanduser() if args.state_root is not None else None
    )
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
        state_root = (
            root / "artifacts" / "human-reviews" / benchmark_path.stem
        )
        console.print(
            f"[green]Blinded packet written:[/green] {prepared.packet_path}"
        )
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
        console.print(
            f"[green]Completed smoke decision:[/green] {paths.decision}"
        )
        console.print(f"JSON report: {paths.json_path}")
        console.print(f"Human-readable report: {paths.markdown_path}")
        console.print(
            "[bold yellow]Single-human diagnostic only; promotion and "
            "headline claims remain blocked.[/bold yellow]"
        )
        return
    raise ValueError(
        f"Unsupported benchmark-review action: {args.benchmark_review_action}"
    )


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
    semantic_backend = MLXEmbeddingBackend(
        args.semantic_model,
        revision=args.semantic_model_revision,
    )
    effective_batch = args.batch_size * args.grad_accumulation
    compilation = compile_acquisition_dataset(
        knowledge_dir=root / "knowledge",
        eval_dir=root / "knowledge" / "eval_frozen",
        output_root=root / "artifacts" / "acquisition",
        semantic_backend=semantic_backend,
        profile=args.profile,
        target_exposures_per_fact=args.target_exposures,
        effective_batch_size=effective_batch,
        micro_batch_size=args.batch_size,
        seed=args.seed,
    )
    model_revision = resolve_huggingface_revision(
        args.model,
        revision=args.model_revision,
    )
    config = AcquisitionConfig(
        model_id=args.model,
        model_revision=model_revision,
        rank=args.rank,
        scale=2.0,
        learning_rate=2e-4,
        dropout=0.05,
        num_layers=36,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation,
        max_seq_length=2048,
        seed=args.seed,
        profile=args.profile,
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
            "[yellow]No training ran. Repeat with --execute only for the "
            "non-promotable synthetic smoke.[/yellow]"
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
