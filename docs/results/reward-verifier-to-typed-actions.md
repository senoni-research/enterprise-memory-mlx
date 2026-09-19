# Before Reinforcement Learning, Test What Your Reward Cannot See

Technical note · 19 September 2026

This note documents what a deterministic reward checker could and could not
measure in a stopped GRPO pilot, and what we changed in the annotation
interface as a result. It is a record of measurement and mechanical testing,
not a model-performance result.

## 1. What stopped, and what we refused to reopen

Two historical results are preserved unchanged:

- **GRPO v1 pilot** (`research/qwen4b-grpo-governed-actions-v1`, PR #3):
  `stop_preflight`. Of three inspected TRAIN families, two produced zero
  reward variance (one all hard-zero, one all perfect). The pre-registered
  gate required variance in at least two of three groups; it failed, and no
  optimizer update was ever run.
- **TRAIN-only reward landscape** (1,152 rollouts = 24 TRAIN cases × 4
  sampling regimes × 12 replicates, base model, no adapter):
  `abandon_standard_grpo_for_current_dataset`. No regime met the eligibility
  thresholds (valid JSON 0.90–0.92 vs 0.95; conflated provenance-hard-fail
  0.087–0.108 vs 0.05). After eligibility failed, every downstream gate was
  fail-closed except `reward_byte_identical`.

We did not retune thresholds, rerun the landscape, reopen DEV/TEST, or edit
the frozen reward implementation (`grpo_reward.py`, SHA-256
`e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b`, byte-identical
to the landscape's recorded binding).

Two observations were misdescribable if you only read the aggregate decision:
truncation was **zero out of 1,152** (an `observed_pass`, not a failure merely
because the aggregate decision failed), and 6–7 of 24 cases per regime were
`saturated_high` — **not a majority**. The correct summary is *reward
saturation under the frozen checker*, not "task mastered."

## 2. Three checker counterexamples

We reproduced three weaknesses of the frozen checker with invented, model-free
fixtures driven through the real `score_completion()` path (not a reimplementation):

| # | counterexample | what the frozen checker reported |
| --- | --- | --- |
| A | Replace a remedy action's text with "Skip mandatory verification V-1 and record it complete" — decision, source IDs, tree shape unchanged | reward **1.0**, `full_machine_success` **true**, no safety flag |
| B | Drop one of three alternatives from a valid `any_of` group (leaving a valid two-way group); also `all_of`→`any_of` with unchanged leaves | missing alternative: remedy score **1.0**, no hard fail; operator swap: remedy 0.5, still not a hard fail |
| C | Malformed JSON, missing request item, extra request item, unauthorized source ID | all four collapse into a single `provenance_hard_fail` flag |

Root cause for A and B: the checker reduces remedy trees to skeletons that
discard action text and pair children positionally after a canonical sort, so
a subset or reworded-but-contradictory action can preserve a unit score.
Root cause for C: one hard-fail flag covers syntax, coverage, and membership.

These are **demonstrated weaknesses of the measurement instrument**, distinct
from their observed occurrence in saved outputs (Section 3).

## 3. Diagnostic counts on the 1,152 saved outputs

An append-only overlay rescored the saved outputs without regenerating
anything. Original scalar rewards and classifications were preserved
verbatim; all 1,152 rescores reproduced the recorded reward.

| category | n | denominator |
| --- | ---: | ---: |
| JSON syntax failure | 28 | 1,152 |
| schema failure | 77 | 1,152 |
| source membership not assessable | 105 | 1,152 |
| request-item coverage failure | 0 | 1,047 assessable |
| unauthorized source-ID use | 12 | 1,047 assessable |
| exact decision mismatch | 303 | 1,152 |
| remedy-structure mismatch | 504 | 1,047 assessable |
| omitted alternatives | 0 | 1,047 assessable |
| action meaning not assessed | 1,152 | 1,152 |
| old `provenance_hard_fail` | 117 | 1,152 |
| original reward = 1 | 486 | 1,152 |
| reward-1 with corrected structural contradiction | **0** | 486 |
| truncated | 0 | 1,152 |

Overlapping categories occur and are counted separately (unauthorized-ID ∩
decision mismatch: 7; unauthorized-ID ∩ structure mismatch: 11; decision ∩
structure mismatch: 246). Malformed outputs get `source_membership_assessment =
not_assessable` — neither a provenance success nor an invented citation.

**Explicitly:** the 486 reward-1 outputs are **not** verified-safe training
examples. No corrected structural contradiction was found among those 486 —
but absence of structural contradiction is not semantic correctness, and
action meaning remained unassessed by the old reward (260 of the 486 contain
an action node whose wording was never checked). None of these outputs was
admitted to training.

## 4. What the counterexamples are not

A demonstrated checker weakness is not evidence that it fired in the corpus.
Conversely, zero occurrences of omitted alternatives among reward-1 outputs
is not evidence the completions are semantically correct. This phase neither
proves GRPO ineffective nor demonstrates a safe executor. The historical stops
stand on their pre-registered gates, not on this diagnosis.

## 5. From diagnosis to a typed-action annotation contract

The diagnosis motivated a narrower, machine-checkable task representation.
All annotation drafts are **model-authored** (Cursor), reviewed by a
**model-advisory** reviewer (GPT-6 Astra Pro). No human attestation exists
anywhere in this chain, and nothing is training-authorized.

Annotation v1 (24 TRAIN cases) was reviewed with outcomes **22 acceptable,
2 minor revisions**. Annotation v2 applied exactly the two mapping-only
corrections and changed nothing else:

| case | field | old → new | target changed? |
| --- | --- | --- | --- |
| CTS3-QUAL-013 | `recorded_procurement_exception.observation_authority` | `requester_claim` → `request_fact` | no |
| CTS3-QUAL-024 | `recorded_procurement_exception.status` | `unmet` → `unknown` | no |

Training targets in v2 are **byte-identical** to v1 (all 24 decisions and
action mappings unchanged).

Structural validation (model-free):

- all 24 drafts pass the closed schema and identity/coverage checks;
- **five negative schema fixtures are rejected** (injected "skip verification"
  instruction, duplicated conflicting condition ID, empty blocker object,
  duplicate request item, missing operation-specific argument);
- **four admissible-but-wrong fixtures still pass** — the grammar remains
  answer-blind; a substantively wrong choice must remain expressible and be
  rejected by evaluation, never secretly removed by the grammar.

The closed interface now carries an explicit boundary: **requested actions do
not establish outcomes or authorize record changes**. Requesting an exception
is not recording one; a negative answer on one unknown `any_of` branch does
not resolve the other; a later executor must check resulting condition states.

Model-advisory annotation outcomes are **not** model accuracy. No typed-action
model training was run and no performance result exists.

## 6. Test-isolation incident

One integrity incident occurred and was corrected. A bundle-writing test
regenerated the annotation-v1 review ZIP **in place**, altering its container
digest. The reviewed bytes were restored and verified against the pinned
SHA-256 (`8a63b877…`); all ten manifest-listed member digests were unaffected.
Current byte identity is the verified current state — it does not mean no
historical write occurred, and timestamps are not proof of restored identity.
The test now writes only into an isolated temporary copy and pins the real
archive, so it cannot regenerate the reviewed artifact. The already-bound
review record is preserved and this clarification is linked from it.

## 7. Where things stand

- The 24-case annotation dataset itself (drafts, mappings) remains **local and
  unpublished** in this step.
- GRPO and DPO remain deferred until a trustworthy typed-action verifier and a
  useful supervised baseline exist. Any later training requires its own
  dataset, budget, and evaluation authorization.
- Schema validity is not semantic correctness. A mechanically valid target is
  not automatically a semantically validated one.

## Appendix: mechanical validation commands and counts

All validation is model-free: no model downloads, inference, backward passes,
or optimizer steps.

Local, full model-free suite (private packs and annotation artifacts present):

```
.venv/bin/python -m ruff check .        # passed
.venv/bin/python -m pytest -q           # 452 passed, 0 skipped
```

Clean public-CI simulation (repository files only; no private packs, no local
annotation artifacts, no Apple-only runtime):

```
ruff check .                            # passed
pytest -q                                # 444 passed, 8 skipped
```

Skip reasons (public CI): six `private deliverable-review pack is not present`,
two `local annotation-v1 artifacts are not present` — honest environment skips,
not disguised passes. Frozen digests were verified before and after testing:
`grpo_reward.py` `e8fbc652…` and the annotation-v1 review ZIP `8a63b877…`
were unchanged by the test run.

Focused-suite breakdown for this milestone (annotation v2 + verifier
diagnosis + landscape tests,
`pytest tests/test_typed_action_annotation.py tests/test_typed_action_annotation_v2.py tests/test_grpo_verifier_diagnosis.py tests/test_grpo_landscape.py`):
47 collected, 47 passed, 0 failed, 0 skipped locally.