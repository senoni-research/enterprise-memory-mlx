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
# Separately versioned, non-promotable model-upgrade experiment:
emmlx model-upgrade --help
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
Its serialized rule result was `stop_parametric_research`; the operative
interpretation is to stop that candidate, not to treat model-only labels as a
general disproof of parametric acquisition. Its status remains
`human_review_pending`.

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
- a frozen local single-Gemma advisory backend that cannot certify or promote;
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
- Qwen3.8-27B 4-bit as the new exploratory candidate, with a matched
  Qwen3-4B 4-bit control.
- Gemma 4 31B 8-bit as a sequentially loaded, frozen advisory verifier.

Legacy and confirmatory training remain blocked. The historical
`smoke_non_promotable` profile and the separately versioned
`model-upgrade-exploratory/v1` profile are non-promotable.

## Model-upgrade exploratory v1

The frozen protocol is
[`knowledge/model_upgrade_exploratory/v1/protocol.json`](knowledge/model_upgrade_exploratory/v1/protocol.json).
It compares each 4B/27B adapter with its own base and with direct evidence,
using the same 24-view-per-record curriculum and 96-exposure budget. Qwen
generation and Gemma verification run in separate processes with thinking
disabled, greedy decoding, fresh per-case state, and a 1,024-token cap.

Run stages explicitly:

```bash
emmlx model-upgrade preflight --model mlx-community/Qwen3-4B-Instruct-2507-4bit
emmlx model-upgrade preflight --model mlx-community/Qwen3.8-27B-4bit
emmlx model-upgrade judge-preflight
emmlx model-upgrade train --model mlx-community/Qwen3-4B-Instruct-2507-4bit
emmlx model-upgrade train --model mlx-community/Qwen3.8-27B-4bit
```

The generated curriculum records model authorship and any deterministic
source-only fallback, but remains `human_approved: false`. Local result
artifacts are ignored by Git. The experiment is complete only after the
measured 4B/27B benchmark, local Gemma regrading, historical bridge, general
diagnostic, and bounded comparison report exist.

The seed-42 experiment is now complete. Its single-Gemma advisory found:

- no generator-quality gain from 27B over 4B on this diagnostic;
- 27B closed-book adapter uplift of `+0.171875` over its own base, with five
  additional fully-correct answers and 9 paired wins versus 3 losses;
- 27B adapter mean `0.265625`, versus `0.84375` with full context, `0.875`
  with oracle context, and `0.34375` with experimental BM25;
- unknown/OOS failures worsening from 0 for the 27B base to 14 for its
  adapter.

The frozen continuation gate therefore ends this experiment without a seed-43
repeat. These are model-review labels, not human-approved evidence. The local
comparison and historical judge bridge are under
`artifacts/model-upgrade/report-runtime-v2/` and
`artifacts/model-upgrade/historical-bridge-runtime-v2/`; generated artifacts
remain uncommitted. The first runtime attempt is retained but invalidated
because the custom low-level trainer had not explicitly seeded MLX and NumPy.
Execution revision `v2-explicit-seeding-and-output-integrity` reran both
models from the untouched bases and produced the figures above. The 27B
preflight covered all 496 expected adapter targets and the actual fixed-dataset
maximum of 192 tokens. Its separate artificial
2,048-token padded stress attempt hit MLX Metal's graph resource-count limit,
which is retained as a disclosed runtime limitation.

## Company task specialization v1

`company-task-specialization/v1` is a separate source-grounded task experiment,
not a reinterpretation of the closed-book acquisition result. Its frozen
contract and synthetic development cases are under
`knowledge/company_task_specialization/v1/`.

```bash
emmlx specialization validate-contract
emmlx specialization validate-evaluator-v2
emmlx specialization pilot
emmlx specialization prepare-audit \
  --pilot artifacts/company-task-specialization/v1/pilot-0276f15bf32ef4a4/specialization-pilot.json
```

The pilot runs the untrained 4B baseline, the Qwen27 teacher/repairer candidate,
and the Gemma advisory evaluator sequentially. It never trains a model. The
first development pass was retained and invalidated after exposing ambiguous
decision and field semantics. The corrected
`v2-explicit-decision-and-field-semantics` pass found:

- untrained 4B with evidence: 6/14 deterministic passes, mean `0.429`;
- Qwen27 teacher candidate: 7/14 deterministic passes, mean `0.500`;
- Qwen27 repairs: 5/8 deterministic passes, mean `0.562`;
- zero advisory-accepted or training-eligible repairs.

The Qwen27 candidate therefore failed the frozen implementation's teacher
qualification. Review subsequently found that free-text substring omissions
were treated as hard failures, so the result is not an expert-judged teacher
error rate. No SFT, GRPO, gisting, autonomous admission, or deployment is
authorized. A blinded audit, approved real work, and a human-calibrated
obligation-level evaluator are required before training can be considered.
The shareable, hash-bound summary is
[`docs/results/2026-09-06-company-task-specialization-v1-advisory.md`](docs/results/2026-09-06-company-task-specialization-v1-advisory.md).
The draft evaluator-v2 and private real-work schema are under
`knowledge/company_task_specialization/evaluator_v2/`. Real case content must
remain under the ignored `knowledge/private/` boundary.

The audit command writes a shareable ZIP, a private unblinding map, and a
review template under `artifacts/company-task-specialization/human-audit/`.
Send only the ZIP to a reviewer. After they return a completed template:

```bash
emmlx specialization validate-audit \
  --packet <specialization-output-audit.zip> \
  --mapping <private-review-id-map.json> \
  --overlay <completed-review.jsonl> \
  --output <audit-report.json>

emmlx specialization validate-real-seed \
  --input knowledge/private/company-task-specialization/real-work.jsonl \
  --output artifacts/company-task-specialization/real-work-seed-manifest.json
```

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
- [`docs/adr/0002-model-upgrade-exploratory-v1.md`](docs/adr/0002-model-upgrade-exploratory-v1.md)
- [`docs/adr/0003-company-task-specialization-v1.md`](docs/adr/0003-company-task-specialization-v1.md)
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
