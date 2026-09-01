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
