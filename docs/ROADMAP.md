# Revised programme roadmap

## Programme rule

No model training may run until the evidence path is capable of rejecting a
bad model. The legacy path is disabled rather than treated as a convenient
smoke test.

## Current accepted foundation

- Governed records with default restricted/secret exclusion.
- Frozen acquisition, unseen-record, supersession, and unknown/OOS suites.
- Split semantics, lexical leakage checks, and freeze verification.
- Exact confidence-bound and independent-unit gate accounting.
- Full-context/oracle construction with byte/token budgets.
- Answer-blind benchmark generation with pinned model/tokenizer revision.
- BM25 retrieval study resulting in `no_feasible_operating_point`.
- Deterministic typed and provenance/OOS graders.
- Fake-backend semantic-judge calibration harness.
- Blinded local human-review workflow.
- Supported legacy commands fail closed.
- Date-controlled supersession v2 with preserved v1.
- Deterministic benchmark grading and gate reporting.
- Local MLX semantic-neighbour audit in the governed smoke compiler.
- Hash-bound experimental BM25 research control.
- Corrected all-linear rank-16 acquisition smoke and adapter benchmark arm.

## Next scientific milestone: trustworthy grading

1. Complete primary and secondary human labels for judge-calibration v3.
2. Independently adjudicate disagreements and freeze the label overlay.
3. Implement and calibrate two local judge families sequentially.
4. Connect raw benchmark output to:
   - strict critical-slot grading;
   - provenance/OOS grading;
   - certified semantic judging;
   - aggregate metrics and confidence bounds;
   - release gates.
5. Add `emmlx grade`/`validate` without invoking legacy `evaluation.py`.

## Temporal evaluation correction

Supersession v1 is frozen but contains relative words such as “current” and
“today” without a controlled date. It must not be used for temporal claims.

Supersession v2 now:

- adds `as_of_date` to every temporal scenario/question;
- resolves all relative language against that date;
- includes the date in model prompts and grading inputs;
- includes current-source record overrides;
- has a separate freeze manifest without overwriting v1.

Independent expert review remains required before v2 supports confirmatory
temporal claims.

## Governed compiler replacement

Compilation must execute in this order:

```text
verify frozen hashes
  -> load suites and governed source snapshot
  -> validate split contract
  -> build proposed training views
  -> exact/normalized/lexical/semantic leakage scans
  -> persist nearest-pair audit
  -> write datasets and manifest
```

Any failure prevents dataset and manifest issuance. The legacy `_split_rows`
implementation is not repaired or reused.

This path is implemented for `smoke_non_promotable`, including a pinned local
MLX MiniLM semantic scan. Confirmatory compilation remains blocked until every
fact has at least 24 independently authored study-view families and the
semantic nearest-pair audit is independently approved.

## Retrieval research decision

The approved BM25 study found no production-feasible lexical operating point.
BM25 therefore remains unavailable through normal benchmark configuration.

A separate owner-approved `experimental_non_promotable` trade-off now exists
to quantify end-to-end retrieval failure. It maximizes balanced validation
utility and preserves its poor OOS behavior in the artifact. It does not
relax or rewrite the accepted no-feasible production result.

Hybrid or embedding retrieval is a separate challenger and requires its own
validation set, operating-point selection, and provenance.

## Revised acquisition milestone

Confirmatory training remains blocked until semantic grading and study-view
milestones pass. One synthetic smoke profile is implemented.

The first smoke used:

- Qwen3-4B as principal model;
- all 36 transformer blocks;
- all attention and MLP projections;
- rank 16;
- a target of 24 semantic exposures per fact using only 10 available view
  families, explicitly recorded as non-promotable;
- MLX scale equivalent to `alpha / rank`;
- full-parameter BF16 control only after the documented memory preflight;
- base, full-context, oracle, and approved retrieval controls;
- general-capability and retention measurements.

The confirmatory experiment still requires ranks 8/16/32, exposure checkpoints
24/96/192, three seeds, independently scheduled confirmation runs, certified
semantic grading, and full general-capability benchmarks.

## Bounded model-upgrade exploratory run

`model-upgrade-exploratory/v1` is a new non-promotable experiment, not a
revision of the September 1 candidate. It uses pinned Qwen3.8-27B and
Qwen3-4B 4-bit checkpoints on the same 24-view curriculum and 96-exposure
trajectory, followed by one pinned Gemma 4 31B 8-bit advisory verifier.

Its report must keep three outcomes separate:

1. stronger 27B generation versus 4B generation;
2. adapter uplift versus the same model's base;
3. competitiveness with full context and the frozen experimental BM25 arm.

Passing the frozen acquisition screen authorizes only one seed-43 repeat.
Neither a pass nor a single-Gemma label authorizes promotion or changes the
historical candidate's `human_review_pending` status.

The corrected, explicitly seeded seed-42 cycle is complete. The first runtime
attempt is retained but invalidated because MLX and NumPy were not explicitly
seeded before LoRA initialization and dataset iteration. Under execution
revision `v2-explicit-seeding-and-output-integrity`, the 27B adapter passed the
three acquisition uplift thresholds (`+0.171875` mean, five additional
fully-correct answers, 9 paired wins versus 3 losses), but failed the required
unknown/OOS condition:
failures increased from 0 for the base to 14 for the adapter. It also remained
`0.578125` below full context. The decision is therefore
`end_model_upgrade_exploratory_v1`; no seed-43 repeat or tuning campaign is
authorized. The result remains a single-local-Gemma advisory rather than
human-approved evidence.

## Company task specialization

`company-task-specialization/v1` tests a distinct claim: source-grounded task
specialization through qualified examples and repaired failures. It keeps
evidence-assisted task competence, replacement value, and closed-book
parametric access as separate endpoints.

The task/split/evaluation contract was frozen before development generation.
The initial pass exposed an evaluator-contract defect and was preserved as
invalid. After explicitly defining decision and field semantics, the corrected
synthetic pass still failed teacher qualification: Qwen27 achieved 7/14
deterministic passes and a `0.500` advisory governed mean. Five of eight repairs
passed deterministic checks, but no repair was admitted because the teacher
qualification gate failed.

Review then identified a measurement limitation: free-text substring omissions
were authoritative hard failures. The pass rate therefore measures the frozen
implementation, not an expert-adjudicated material-error rate. The stopped
pilot and all admission decisions remain unchanged.

The next milestone is not training. First, audit all 14 teacher and eight repair
answers blindly. Reviewers judge obligations, satisfaction, safety, next-step
usefulness, and ambiguity; evaluator/reference diagnosis occurs only after
their labels are frozen and privately compared with historical results. Then
collect approximately 25 approved, anonymized cases from one real workflow,
with accepted outcomes and human-defined obligations.
Evaluator v2 routes free-text completeness and polarity to calibrated review
while retaining syntax and provenance hard checks. Only after that validation
may fresh teacher cases and bounded prompt/orchestration comparisons be frozen.
A stronger teacher, SFT, GRPO, and gisting remain deferred.

The project owner subsequently elected to use a blinded GPT review instead of
collecting human labels for this exploratory cycle. That review rated 16 of 22
outputs acceptable, three minor revision, and three unacceptable. Private
comparison found eight possible evaluator false failures, two possible false
acceptances, and one supported model failure. This permits a model-advisory-only
task-contract correction and fresh synthetic Qwen comparison; it does not
create human evidence or authorize production claims. The next bounded step is
to correct multipart decisions, alternatives, unknown-versus-unmet states, and
unsupported consequences before retesting the existing Qwen system. Training
and stronger-teacher work remain blocked.

### Supplier-workflow model-advisory qualification

`company-task-specialization/v2-model-advisory` freezes 14 fresh synthetic
cases from one supplier-onboarding and first-invoice workflow. It compares the
same pinned Qwen27 with a baseline single pass, an explicit
obligation-by-obligation prompt, and one completeness revision pass. The
revision arm reuses the obligation output as its draft and reports the combined
latency and token cost.

The 42 generations completed without failure or truncation and all parsed under
the multipart schema. The blinded GPT review rated all three arms identically:
13/14 acceptable, one minor revision, and zero unacceptable. Every arm repeated
the same overexclusive exception-remedy wording on the same targeted case and
therefore failed the frozen zero-unsafe and targeted-case gates.

The result is `no_arm_qualified`. The longer obligation prompt and two-pass
revision produced no quality gain, so neither advances. A future bounded
development change should represent alternative remedies explicitly rather
than add more generic prompting or a stronger model. This experiment cannot
authorize student training, a stronger teacher, or production use.

### Remedy-logic representation

`company-task-specialization/v3-remedy-logic` preserves v2 and isolates that
single defect. The frozen design compares the unchanged one-call Qwen27
baseline with a one-call structured-remedy challenger on 24 fresh synthetic
cases. It adds explicit `any_of`, `all_of`, and nested remedy groups while
separating known current blockers from future actions.

Typed model-free tests run before inference. Structural and source-membership
checks are machine-authoritative; policy interpretation remains subject to an
arm-label-blinded GPT review. The old 14 cases are retained as regression-only
evidence and are not counted toward the new qualification result.

The prospective gate requires a 90% acceptable rate, no unacceptable or unsafe
output, every targeted logic and multipart case acceptable, and no generation,
truncation, or structure failure. Challenger advancement additionally requires
paired alternative-preservation improvement without a mandatory-condition
regression. Equal quality selects the simpler control.

Even a passing arm is limited to bounded advisory example generation and needs
a separately approved admission protocol. Student training, deployment, and
human-validation claims remain blocked.

The completed review found a real paired benefit but no passing arm. Control
scored 16/24 acceptable; the structured challenger scored 22/24 acceptable,
with five alternative-preservation wins and no losses. Both arms nevertheless
invented an attachment or verification requirement when completion was already
established. The challenger also downgraded one known-complete control to
unknown. The preserved decision is `no_arm_passes_substantive_gate`.

The next specialization milestone is therefore a bounded investigation of
policy interpretation and fact-state adherence, not student training or a new
model-selection campaign. Any repair run and any advisory-example admission
protocol require separate approval.

### Fact-state stop boundary

`company-task-specialization/v4-fact-state` retained structured remedies and
ran the one authorized fresh fact-state comparison. The 16 scenarios were
organized as 8 contrast pairs with generator-visible authority metadata.

The amendment improved four matched scenarios and regressed one. It fixed the
recorded-completion/absent-attachment contrast and improved state-authority
handling, but failed to eliminate unknown-versus-unmet and collective-completion
errors. It also introduced one remedy-logic and two unsafe-waiver regressions.
The candidate scored 10/16 acceptable versus control's 8/16; neither passed the
substantive gate.

The decision is `no_success_stop_synthetic_prompt_iteration`. The conditional
v3 regression was not authorized. The next specialization milestone is to
inspect the evidence contract against an approved real-work seed or validated
workflow state. Another synthetic prompt tournament, stronger-model campaign,
student training, and automatic example admission remain blocked.

## Later milestones

### Recovery

- replay as a training baseline;
- scalar adapter-delta shrink;
- DARE delta recovery;
- TIES across independent seed deltas;
- FP32/BF16 delta math and one-time deployment quantization.

### Continual writes and deletion

- resumed-adapter writes;
- BF16 fresh-adapter merge-per-write;
- rebuild from the original base;
- retention and probability-lift curves;
- supersession and stale-answer gates;
- context-rescue measurement.

### Registry and security enforcement

- inherited classification;
- adapter/checkpoint/source hashes;
- source record and manifest IDs;
- approval state and expiry;
- evaluator/judge versions;
- extraction-test status;
- external-judging prohibition.

### Routing

No domain adapters or learned router until:

- a labelled routing benchmark exists;
- oracle adapter selection materially improves answer quality;
- wrong-load, false-load, false-fallback, and cross-domain metrics pass;
- prompt-level learned routing is compared with transparent baselines.

## Promotion

The pilot remains non-promotable regardless of point estimates because it does
not contain enough independent facts/scenarios for the configured confidence
bounds. `pass`, `fail`, and `insufficient_data` remain distinct outcomes.
