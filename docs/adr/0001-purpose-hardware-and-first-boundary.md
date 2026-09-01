# ADR 0001: Purpose, hardware target, and the first implementation boundary

- Status: accepted
- Date: 2026-08-31
- Applies to: the whole repository

## Mission

This repository is a **local MLX research and engineering platform that
compiles governed knowledge into model adaptations, measures whether the
knowledge was acquired, and refuses promotion unless the adaptation beats
appropriate context/retrieval baselines and passes retention, supersession,
privacy and capability gates.**

Knowledge acquisition in weights is the objective, not a curiosity. Context
and retrieval are mandatory controls, not replacements for that objective.
The scientific null hypothesis stays neutral:

> Parametric adaptation does not outperform the best non-parametric baseline
> under the workload's actual constraints.

### Users and outputs

| User | Uses the repository to |
|---|---|
| ML engineer / researcher | Train a local model on governed knowledge |
| Architect | Decide whether a resulting adapter is worth deploying |
| Governance reviewer | Decide whether an artifact is safe to promote |

Outputs are a trained adapter or checkpoint **plus** an evaluation report, a
source/configuration manifest, a classification record, and a promotion or
rejection decision. The target CLI shape is `emmlx benchmark` (context vs RAG
vs weights), `emmlx acquire`, `emmlx validate`, `emmlx promote`.

## Hardware contract

- Target: **one M4 Max Mac, 40-core integrated GPU, 128 GB unified memory.**
- No CUDA assumptions; no multi-GPU batch sizes copied from cluster papers;
  no cloud or rented-GPU requirement.
- MLX / MLX-LM is the primary implementation.
- **Qwen3-4B is the principal experimental model.**
- Full-parameter BF16 4B fine-tuning is a **feasibility-gated challenger**:
  before any full run, a preflight of 50–100 optimizer updates at the real
  sequence length, optimizer and checkpointing settings must record peak
  active memory, peak cache memory, memory trend, tokens/second, checkpoint
  save/load success, and a general benchmark before/after. Sequence length
  must not be reduced just to pass the preflight unless the same reduced
  length is used in the real experiment. If full 4B training is infeasible,
  fall back to full fine-tuning of the corresponding 1.7B model rather than
  dropping the full-parameter control.
- Larger local models (for example ~27B/24B 4-bit instruct models) may be
  loaded **sequentially as judges only**, never as training targets.

## Frozen legacy declaration

The pipeline that existed before this ADR (`compiler.py` datasets,
`training.py` presets, lexical evaluation and router) is **frozen and its
ablation results are invalid**. Known defects, kept for the record:

1. Train/test leakage: `_split_rows` reuses identical rows across splits for
   1–2 item collections, and templated paraphrases with identical answers
   cross splits.
2. LoRA scale mis-calibration: MLX-LM's `scale` is the direct multiplier on
   the LoRA update (PEFT equivalent `alpha / r`). The legacy presets used
   scale 20.0–32.0, an effective PEFT-alpha of 160–512. Correct equivalent of
   alpha=32, r=16 is `scale: 2.0`.
3. Substrate: q/v-only attention LoRA over 8–24 layers is rejected as the
   primary storage substrate; the revised default is rank 16 over all seven
   attention and MLP projections on all 36 layers, with r8 and r32 arms.
4. Scoring: substring/token-F1 passing is replaced by typed critical-slot
   checks, counterfactuals, provenance checks and blinded semantic grading.

No training may be modified or run until the split contract in this PR has
been reviewed.

## Split contract (implemented in `split_contract.py`, `leakage.py`)

Four frozen suites with distinct validity rules:

| Suite | Fact trained? | Question form seen? | Measures |
|---|---|---|---|
| `acquisition` | yes | no | Access to stored facts through new phrasings |
| `unseen_record` | no | no | Recipe generalisation to untrained records |
| `supersession` | old/new controlled | no | Adopting new values without stale answers |
| `unknown_oos` | no record exists | no | Refusal instead of invention |

Rules:

- Sharing the **fact** between training and acquisition evaluation is
  intentional; sharing a question family, template, wording or generator is a
  violation. The unseen-record suite alone is record-disjoint; making every
  suite record-disjoint would stop measuring internalization.
- Test assets must be authored by a generator unavailable to the training
  compiler (different model family or independent human), with recorded
  generator identity and prompt hash.
- Assets are frozen and hashed (`freeze_manifest.json`) **before** training
  data is compiled; CI fails when a frozen file changes.
- Compilation must fail on exact duplicates, normalized near-duplicates and
  answer-cue overlap, and must emit a nearest-pair audit list for human
  review. Forced-choice probes are exempt from the answer-cue check because
  they present candidate answers by design.
- A record with fewer than three independent question families is train-only.
- A semantic (embedding-based) nearest-neighbour scan is a planned addition on
  the Mac side; the shipped scanner is lexical and the audit list exists so a
  human reviews the closest pairs regardless.

## Gate contract (implemented in `gates.py`)

- Every rate gate declares a denominator unit, a one-sided Clopper-Pearson
  bound at 95% confidence, a cluster key and a critical-failure policy.
- Zero observed failures support a ≤1% claim only with ≥299 independent
  units, and ≤0.5% only with ≥598. Observed failures use the exact bound,
  not the rule-of-three approximation.
- Repeated probes of the same fact cluster into **one** independent unit; a
  unit fails if any probe in it fails.
- Pilot-scale corpora (the shipped 9-record corpus) cannot reach these
  denominators. Pilot results are diagnostic and **never promotable**.

## Precision contract (for later training PRs)

| Artifact | Precision |
|---|---|
| Canonical base and sequentially merged state | BF16 |
| Dense adapter-delta math, TIES/DARE/scaling | FP32 or BF16 |
| SVD refactorisation | FP32 |
| Serving artifact | optionally 4-bit, quantised once at the end |

Sequential-write protocols must be compared at identical base precision. The
final quantisation tax (accuracy before/after one quantisation, per-fact
regressions) is reported separately. Adapter deltas may be reconstructed
directly from LoRA A/B matrices in FP32 when all adapters share a base; a
BF16 base is required for merge-per-write sequences, fused-checkpoint
reconstruction, checkpoint-subtraction deltas, and pre/post-quantisation
comparison.

## Evaluation confidentiality

- Synthetic/public corpora: external API judges permitted, with logging and
  reproducibility.
- Confidential/restricted corpora: nothing (source, expected, generated
  answer, judge prompt or output) leaves the approved environment. Order of
  authority: deterministic typed critical-slot checks first (a failed
  critical slot cannot be overridden by any judge), then two sequential local
  judges from model families different from the trained model, then human
  adjudication. Judges require a frozen human-labelled calibration set,
  temperature 0, fixed version and prompt hash, blind to arm, structured
  JSON output, and false-pass analysis before their scores can support a
  promotion decision.

## Deferred, not dropped

- **Residual-error post-training (Wnuan-style GRPO)**: deferred. The
  residual-error pool schema (error, category, producing checkpoint, expert
  correction, reward inputs, random/full-pool comparison sets) is to be kept
  from the first training PR. Before any local GRPO backend, run the cheaper
  three-arm residual-SFT experiment (residual vs size-matched random vs
  full-pool). Community MLX GRPO implementations are experimental
  dependencies, not trusted core.
- **Domain adapters and learned routing**: no domain adapters until a routing
  benchmark passes and the oracle-adapter gap justifies specialisation
  (oracle must beat no-adapter end-to-end accuracy by ≥10 points before any
  learned router is built).

## First implementation boundary (this PR)

Contains only: this ADR, the split-contract schemas, leakage scans, frozen
synthetic evaluation fixtures with their freeze manifest, and the gate schema
with confidence-bound tests. Training code is untouched and was not run.
