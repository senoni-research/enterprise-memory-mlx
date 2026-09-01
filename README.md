# Enterprise Memory MLX

A local MLX research and engineering platform for testing whether governed
company knowledge can be acquired in model adaptations without losing safety,
general capability, provenance, or updateability.

## Current status

**NOT READY FOR CONFIRMATORY TRAINING OR PROMOTION.**

The original compiler, trainer, lexical evaluator, adapter chat, and router are
scientifically invalid and disabled through every supported CLI entry point.
There is no override flag.

Why they are disabled:

- train/test identity and template leakage;
- q/v-only LoRA over only part of the model;
- fixed iteration counts instead of fact-exposure budgets;
- mis-calibrated MLX LoRA scale;
- substring, token-F1, and keyword answer scoring;
- unvalidated continual writes, recovery, routing, and deletion.

The code remains in the repository only as historical research material. See
[`docs/legacy-research-prototype.md`](docs/legacy-research-prototype.md).

## Safe commands

On Apple Silicon:

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate

emmlx doctor
emmlx benchmark --dry-run
emmlx review --reviewer "Your Name"   # identity is asserted, not authenticated
emmlx acquire                         # governed compile + dry-run config
# Explicitly non-promotable synthetic smoke only:
emmlx acquire --execute
emmlx grade --benchmark artifacts/benchmark/<raw-result>.json
pytest
ruff check .
python scripts/package_source.py
```

`emmlx benchmark --dry-run` downloads/loads only the tokenizer for the pinned
Qwen revision; it does not load model weights or generate answers.

`emmlx acquire` is a new governed path and is not an alias for the disabled
legacy trainer. Without `--execute` it compiles, runs lexical/semantic leakage
checks, and writes a dry-run configuration. `--execute` is limited to the
explicitly non-promotable rank-16 synthetic smoke.

These commands are deliberately blocked:

```text
emmlx compile
emmlx train
emmlx evaluate
emmlx route
emmlx chat
```

## Blinded human-review UI

Launch:

```bash
emmlx review --reviewer "Philippe Dagher"
```

The localhost-only UI provides:

- a blinded one-case-at-a-time review workflow;
- required `1.0` / `0.5` / `0.0` score, reason, confidence, and human
  attestation;
- atomic autosave and resume per reviewer;
- no proposed/model labels, original case IDs, error categories, or
  certification strata;
- a status dashboard showing scientific blockers;
- export only after all cases are reviewed.

The primary-review export remains
`single_human_review_complete_not_adjudicated` and `human_approved: false`.
A second human and adjudication are still required by the calibration
contract.

Default inputs:

```text
artifacts/review-packets/judge-calibration-v3-model-review.zip
artifacts/review-packets/judge-calibration-v3-private/review_id_map.json
```

Review state and exports remain under `artifacts/human-reviews/` and are not
committed.

## Blinded smoke decision

Training is paused while all 160 outputs from the corrected smoke benchmark
receive a direct human review. Prepare the hash-bound packet:

```bash
emmlx benchmark-review prepare \
  --benchmark artifacts/benchmark/benchmark-20260901T101652Z.json \
  --grading artifacts/grading/deterministic-grading-20260901T101703Z.json
```

Review it with state isolated from judge calibration:

```bash
emmlx review \
  --packet artifacts/review-packets/benchmark-20260901T101652Z-human-review.zip \
  --mapping artifacts/review-packets/benchmark-20260901T101652Z-human-review-private/review_id_map.json \
  --state-root artifacts/human-reviews/benchmark-20260901T101652Z \
  --reviewer "Your Name"
```

The benchmark packet omits arm, question ID, grader outcomes, retrieval
metadata, and source records. Source records are omitted because their count
would fingerprint the benchmark arm. Use the UI's export button only after all
160 cases are complete, then write the diagnostic report:

```bash
emmlx benchmark-review report \
  --benchmark artifacts/benchmark/benchmark-20260901T101652Z.json \
  --grading artifacts/grading/deterministic-grading-20260901T101703Z.json \
  --packet artifacts/review-packets/benchmark-20260901T101652Z-human-review.zip \
  --mapping artifacts/review-packets/benchmark-20260901T101652Z-human-review-private/review_id_map.json \
  --overlay artifacts/human-reviews/benchmark-20260901T101652Z/overlays/<reviewer-slug>.jsonl
```

The report unblinds only after completion. It reports direct human scores
separately from governed final scores, where deterministic hard failures
remain `0.0`. The pre-registered rule authorizes at most one redesigned
non-promotable diagnostic only when parametric beats base by at least `0.10`
mean score, gains at least four fully-correct cases, and has more paired wins
than losses. Regardless of that result, promotion, judge certification, and
headline accuracy remain blocked.

A completed blinded **model-review advisory** (not human labels) for this
benchmark is summarized with full hash bindings in
[`docs/results/2026-09-01-smoke-advisory.md`](docs/results/2026-09-01-smoke-advisory.md).
Its advisory outcome was `stop_parametric_research`; the operative status
remains `human_review_pending`.

## Current scientific foundation

Implemented:

- governed knowledge records and default restricted-data exclusion;
- frozen acquisition, unseen-record, supersession, and unknown/OOS suites;
- split semantics and lexical leakage checks;
- exact confidence-bound and independent-unit gate accounting;
- deterministic full-context and oracle controls;
- token/byte context budgeting and provenance;
- answer-blind benchmark generation;
- BM25 retrieval and validation-only operating-point selection;
- typed factual checks;
- citation/provenance/OOS checks;
- a fake-backend semantic-judge calibration harness;
- blinded local human-review tooling.

The BM25 validation experiment produced
`no_feasible_operating_point`: no lexical threshold met both acquisition and
OOS constraints. BM25 is disabled by default. Base, full-context, and oracle
controls remain available.

Not yet complete:

- certified local semantic judges;
- human-complete semantic grading and final accuracy;
- confirmatory 24-view-per-fact datasets and rank 8/16/32 multi-seed matrix;
- full public general-capability benchmark suite;
- sequential-write, deletion, recovery, routing, and extraction experiments;
- promotion and governed adapter registry.

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Evaluation principles

The repository separates:

- **Acquisition:** trained fact, unseen question family.
- **Unseen-record generalisation:** record never trained.
- **Supersession:** controlled old/new versions.
- **Unknown/OOS:** no authoritative record; refuse or consult the live source.

Model generation is answer-blind. Expected answers and grading metadata enter
only after generation. Deterministic critical-slot and provenance failures
cannot be overridden by an LLM judge. Repeated prompts about one fact do not
count as independent facts for promotion confidence.

The authoritative source remains external. Current facts, record-level ACLs,
revocable information, and audit evidence must not rely on weights.

## Hardware target

- One M4 Max Mac.
- 40-core integrated Apple GPU.
- 128 GB unified memory.
- MLX / MLX-LM, no CUDA or multi-GPU assumptions.
- Qwen3-4B as the principal experimental model.
- Full BF16 4B training only after a representative feasibility preflight.

Legacy and confirmatory training remain blocked. One explicitly
`smoke_non_promotable` acquisition profile is available for machinery checks.

## Latest acquisition smoke

The current rank-16 smoke uses Qwen3-4B, all 36 layers, all q/k/v/o and
gate/up/down projections, MLX scale 2.0, and an exposure-derived
three-epoch/30-optimizer-update schedule.

Artifacts:

```text
artifacts/acquisition/runs/
  smoke_non_promotable-v2-micro-iterations-r16-e24-s42.json
artifacts/benchmark/benchmark-20260901T101652Z.json
artifacts/grading/deterministic-grading-20260901T101703Z.json
artifacts/acquisition/diagnostics/general-diagnostic.json
```

The first attempted smoke was explicitly invalidated after runtime evidence
showed that MLX `iters` counts microbatches rather than optimizer updates. The
corrected run processed 240 micro-iterations / 30 optimizer updates.

Deterministic hard failures on the acquisition suite were:

- base: 28/32;
- experimental BM25: 21/32;
- full context: 7/32;
- oracle context: 4/32;
- parametric adapter: 23/32.

These are certain typed/provenance failures, not semantic accuracy. Every
non-hard-failure row still requires certified semantic review, so the smoke
does not establish that weights outperform context or retrieval.

## Data handling

- Restricted and secret records are excluded by default.
- Generated adapters/checkpoints inherit the highest source classification.
- Calibration and review artifacts are never model-training data.
- Model reviews do not count as human labels.
- Confidential judging must remain local.
- Anything under `knowledge/private/`, downloaded models, weights, local
  credentials, and generated artifacts is excluded from source archives.

## Repository map

```text
knowledge/                  governed synthetic records and versioned contracts
src/enterprise_memory_mlx/ current libraries plus disabled legacy internals
tests/                      contract and component tests
docs/                       ADR, security, roadmap, and historical notes
artifacts/                  generated local outputs; ignored
dev/                        local plans and dispatch briefs; ignored
scripts/                    bootstrap and clean source packaging
```

## Documentation

- [`docs/adr/0001-purpose-hardware-and-first-boundary.md`](docs/adr/0001-purpose-hardware-and-first-boundary.md)
- [`docs/ROADMAP.md`](docs/ROADMAP.md)
- [`docs/security.md`](docs/security.md)
- [`docs/data-contract.md`](docs/data-contract.md)
- [`docs/research-notes.md`](docs/research-notes.md)
- [`VALIDATION.md`](VALIDATION.md)

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

MLX inference must run on Apple Silicon. All ordinary contract, grading,
review-state, and packaging tests remain model-free.
