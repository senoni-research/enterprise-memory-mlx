# Validation status

## Scope

This document validates the scientific foundation and safety boundary. It is
not evidence that parametric knowledge acquisition works, and it does not
authorize training.

Current status: **NOT READY FOR CONFIRMATORY TRAINING OR PROMOTION.**

## Trusted checks

Run from the repository root:

```bash
source .venv/bin/activate
pytest
ruff check .
python -m compileall -q src tests
bash -n scripts/*.sh
```

Test totals are intentionally not copied into this document because they
become stale. The command result is authoritative.

## Supported runtime checks

```bash
emmlx doctor
emmlx benchmark --dry-run
emmlx review --reviewer "Philippe Dagher"
emmlx acquire
# Explicit synthetic machinery smoke only:
emmlx acquire --execute
emmlx grade --benchmark artifacts/benchmark/<raw-result>.json
python scripts/package_source.py
```

The benchmark dry run uses tokenizer-only loading. It must report the resolved
tokenizer revision and must not load model weights or generate answers.

## Required fail-closed checks

Each command below must exit non-zero before invoking legacy code:

```bash
emmlx compile
emmlx train --dry-run
emmlx evaluate
emmlx route "test"
emmlx chat
```

The Makefile legacy targets and `scripts/run_demo.sh` /
`scripts/run_ablation.sh` must fail the same way.

## Frozen and versioned assets

Validation must confirm:

- the frozen evaluation manifest matches every recorded byte hash;
- judge-calibration v1, v2, and v3 candidate hashes remain unchanged;
- retrieval-validation and BM25 decision hashes remain intact;
- review packets contain no proposed labels or private ID map;
- human review exports remain separate from model reviews;
- model reviews never populate `human_*` approval fields.

## Accepted component boundary

Accepted as libraries:

- split/freeze/leakage contracts;
- confidence-bound gates;
- full-context/oracle and token budgets;
- answer-blind benchmark planning;
- BM25 selector and accepted no-feasible decision;
- strict typed grading;
- provenance/OOS grading;
- fake-backend semantic-judge calibration;
- blinded human-review state/export.
- date-controlled supersession v2;
- deterministic-only benchmark grading;
- governed smoke compiler with pinned MLX semantic-neighbour scanning;
- experimental non-promotable BM25 research control;
- all-linear 36-layer rank-16 smoke acquisition.

Not integrated end to end:

- real local judge backends and certification;
- human-complete semantic grading;
- confirmatory 24-view-per-fact rank/exposure acquisition matrix;
- powered general-capability/retention evaluation;
- promotion registry.

## Known-invalid internals

`compiler.py`, `training.py`, `hardware.py`, `evaluation.py`, `router.py`, and
legacy adapter inference remain in place for historical reference. Supported
commands cannot reach them.

Their direct library APIs now fail closed: `compile_knowledge`,
`build_stage_config`, `train_pipeline`, `evaluate_models`, `score_answer`,
`route_query`, and `interactive_chat` raise `LegacyPipelineDisabledError`
unless the caller passes `allow_scientifically_invalid=True`. Only
historical-reference tests may pass that flag. The new `emmlx acquire`
command imports neither legacy compiler nor legacy trainer.

## Latest smoke boundary

The corrected smoke run is:

```text
artifacts/acquisition/runs/
  smoke_non_promotable-v2-micro-iterations-r16-e24-s42.json
```

It records 240 micro-iterations, 30 optimizer updates, all 36 layers, all seven
attention/MLP projections, rank 16, and MLX scale 2.0. Its adapter and every
downstream benchmark/grading artifact are explicitly non-promotable.

The earlier 30-micro-iteration run is marked
`invalidated_micro_iteration_accounting` and must not be compared.

## Human-review boundary

The local review UI is blinded from:

- original case IDs;
- proposed/model labels;
- error categories;
- strict/provenance outcomes;
- certification strata.

It requires a human attestation and saves each decision atomically. A completed
primary overlay remains `human_approved: false` until second review and
adjudication.

Reviewer identity is asserted, not authenticated: `emmlx review` has no
default reviewer, `--reviewer` is mandatory, and every decision and export
records the invoking OS account plus
`identity_verification: asserted_only_not_authenticated`.

## Smoke-decision boundary

The benchmark human-review packet contains all 160 smoke outputs. Every case is
arm-blinded and excludes question IDs, suite, deterministic outcomes, retrieval
metadata, and source records. Source records are deliberately empty because
their count distinguishes base/parametric, oracle, and full-context arms.

Preparation binds the packet and private mapping to:

- the raw benchmark byte hash;
- the deterministic grading byte hash and grader configuration;
- the frozen fixture hash;
- the pre-registered continuation rule.

The report command fails closed unless the human overlay is complete and every
artifact hash matches. It emits direct human scores for audit and separate
governed scores. A deterministic hard failure remains `0.0` regardless of the
human semantic score.

Passing the continuation rule authorizes only one redesigned
`non-promotable` diagnostic. It requires all three:

- parametric governed mean score at least `0.10` above base;
- at least four additional fully-correct parametric cases;
- more paired parametric wins than losses.

Failing stops parametric research in favor of context/RAG. Passing does not
authorize confirmatory training, promotion, judge certification, or headline
accuracy claims. The single-review overlay remains `human_approved: false`.

## Clean archive boundary

The source archive must exclude:

- `.venv`, build caches, bytecode, and egg metadata;
- `artifacts` and `dev`;
- `.specstory` and macOS resource files;
- private knowledge, downloaded models, and model weights;
- credentials and local configuration.

The archive must include its own checksum manifest and pass an automated
contents test.
