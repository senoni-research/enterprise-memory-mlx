# Validation status

## Scope

This document validates the scientific foundation and safety boundary. It is
not certified evidence that parametric knowledge acquisition works, and it
does not authorize confirmatory or additional exploratory training.

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
- architecture-profiled Qwen3.8 hybrid QLoRA with fail-closed target coverage;
- frozen local single-Gemma source-aware advisory verification;
- immutable matched 4B/27B exploratory orchestration and comparison reporting.

Not integrated end to end:

- certified dual-family local judging;
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

## Model-upgrade exploratory boundary

`model-upgrade-exploratory/v1` completed one matched, explicitly seeded seed-42
cycle using the same 192-row, 24-view curriculum and 96-exposure budget for
Qwen3-4B and Qwen3.8-27B. The first runtime attempt is retained but invalidated
because the low-level training path had not explicitly seeded MLX and NumPy.
Execution revision `v2-explicit-seeding-and-output-integrity` seeds Python,
NumPy, and MLX before LoRA initialization and dataset iteration. The 27B
preflight verified all 496 expected targets across 48 linear-attention, 16
full-attention, and 64 MLP blocks. The corrected measured runs each recorded
768 micro-iterations and 96 optimizer updates.

The local Gemma advisory found a 27B adapter mean uplift of `0.171875` and five
additional fully-correct acquisition answers over the same-model base, but
unknown/OOS failures worsened from 0 to 14 and the adapter remained `0.578125`
below full context. The frozen decision is
`end_model_upgrade_exploratory_v1`; seed 43 is not authorized.

The 27B backward preflight covered the actual longest curriculum row (192
tokens) with no truncation. Artificially padding that row to the configured
2,048-token cap exceeded MLX Metal's graph resource-count limit. This
limitation is preserved in failed preflight attempts; no layer, target, rank,
or measured exposure reduction was used to bypass it.

All Gemma labels remain `single_local_judge_advisory`, `human_approved: false`,
and `promotion_eligible: false`. The September 1 candidate, artifacts,
stopping rule, and `human_review_pending` status remain unchanged.

## Company-task specialization boundary

`company-task-specialization/v1` verifies a separately frozen task, split,
source snapshot, protected historical evaluation hash, model identities,
teacher gate, and bounded repair-selection rule. Its CLI loads the 4B
generator, releases it, loads Qwen27 for teacher and repair generation,
releases it, and only then loads Gemma for advisory scoring.

The initial synthetic development run is retained but invalidated because its
decision and field semantics were ambiguous. The corrected execution revision
produced complete, non-truncated generations with zero invalid Gemma outputs,
but the Qwen27 teacher candidate passed only 7/14 deterministic cases and
failed the frozen `0.90` teacher threshold. Consequently:

- the repair batch is auditable but unresolved;
- no repair is advisory-accepted;
- every candidate records `human_review_required: true`;
- every candidate records `training_eligible: false`;
- no model training or autonomous corpus admission is authorized.

The v1 pass count is not an expert-judged material-error count.
`structured-policy-assessment/v1` used free-text substring omission as a hard
failure and skipped Gemma on those rows. Correct paraphrases could therefore
fail, while a polarity-reversed sentence containing the required phrases could
reach semantic review. Historical scores and admission decisions remain
preserved.

Draft evaluator `structured-policy-assessment/v2-obligation-review` makes only
parse/schema and permitted-evidence membership failures authoritative for the
legacy-shaped answers. Free-text obligation completeness, polarity, equivalent
deadlines, and decision equivalence become `semantic_review_required`. It
cannot produce a passing semantic grade until calibrated against the blinded
human audit and private real-work seed.

Audit packet v2 separates measurement from diagnosis. Blinded Phase A reviewers
record obligations, satisfaction, unsafe claims, next-step usefulness, and
ambiguity without seeing or diagnosing historical results. Only after those
labels are frozen does the maintainer use the private mapping for a Phase B
disagreement report. Neither phase replaces the historical pilot scores.

The synthetic Northstar cases validate pipeline mechanics only. A future
specialization claim requires approved real work and a separately frozen,
untouched test. A future closed-book acquisition claim must use unseen
applications of trained facts without policy evidence and must be reported
separately.

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
