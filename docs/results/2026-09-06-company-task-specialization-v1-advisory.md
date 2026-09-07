# Company task specialization v1 — advisory development result

**Date:** 6 September 2026
**Status:** bounded synthetic development result
**Evidence status:** single local Gemma advisory; not human-approved
**Training status:** not authorized

## Research question

Can the existing local pipeline qualify Qwen3.8-27B as a teacher and repairer
for a small source-grounded operational-policy task before any Qwen3-4B
student training is attempted?

This experiment is distinct from closed-book parametric knowledge acquisition.
Every ordinary task case supplies its permitted policy evidence. Success would
show task competence with evidence, not storage of mutable company facts in
model weights.

## Frozen task

Given an operational request and the permitted policy evidence, return a
structured assessment containing:

- a decision;
- required actions;
- missing information;
- applicable exceptions;
- supporting record IDs and claims.

The corrected development set contains 14 synthetic cases covering known
policy families, compositional requests, ambiguous requests, absent sources,
live facts, and restricted material. Existing frozen acquisition questions are
not repair-eligible.

## Models and controls

- Student baseline: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- Teacher/repairer candidate: `mlx-community/Qwen3.8-27B-4bit`
- Advisory verifier: `mlx-community/gemma-4-31b-it-8bit`

All use pinned revisions. Generation is greedy, thinking is disabled, outputs
are capped at 1,024 tokens, and each case receives fresh state. Models are
loaded sequentially.

## Corrective freeze

The first development pass exposed a task-contract problem rather than a model
result: decision labels were underspecified, known unmet controls were treated
as missing information, and explicitly absent alternatives were treated as
applicable exceptions.

That run is preserved and marked
`invalidated_task_contract_ambiguity`. No final or held-out data was inspected.
Execution revision `v2-explicit-decision-and-field-semantics` then froze
explicit meanings and corrected the synthetic development references before a
new run.

## Corrected results

- Untrained 4B with evidence: 6 of 14 deterministic passes, governed advisory
  mean `0.429`, six fully correct, zero generation failures.
- Qwen27 teacher with evidence: 7 of 14 deterministic passes, governed advisory
  mean `0.500`, seven fully correct, zero generation failures.
- Qwen27 repair of eight selected 4B failures: 5 of 8 deterministic passes,
  governed advisory mean `0.562`, four fully correct, zero generation failures.

The Qwen27 teacher failed the frozen implementation's requirements of at least
`0.90` deterministic pass rate and at least `0.90` Gemma advisory mean,
including complete referral behaviour.

Consequently:

- teacher qualification failed;
- no repaired example was admitted;
- every candidate remains human-review-required;
- zero examples are training-eligible;
- no SFT, GRPO, gisting, or autonomous admission is authorized.

## Measurement limitation discovered in review

The v1 deterministic evaluator treated a missing required substring in a
free-text field as an authoritative hard failure. Those answers received zero
and were not sent to Gemma. Phrase presence is not a dependable semantic test:
an equivalent paraphrase can miss the substring, while a polarity-reversed
instruction can contain every required phrase.

The result therefore establishes that Qwen27 failed the frozen
implementation's qualification gate. It does **not** establish that an expert
would judge seven of fourteen answers materially wrong. The existing answers
require a blinded obligation-level human audit to separate genuine task errors,
evaluator false failures, reference ambiguity, and output-contract problems.

Zero accepted repairs also reflects the global teacher gate. It does not mean
that every repair failed its own checks: five of eight passed the deterministic
stage and four received a fully-correct governed score. They remain rejected
under this experiment. The repair mean must not be compared directly with the
teacher mean because repairs cover a selected subset of 4B failures rather than
the same case distribution.

## Interpretation

This result does not show that task distillation is ineffective. It shows that
the current Qwen27 configuration was not qualified by the frozen measurement.
Training the 4B student on these repairs would convert unresolved evaluator and
teacher behaviour into supervision.

The next justified input is a blinded audit of the existing answers plus
approved, anonymized real work and human task labels. Those labels should
validate a separately versioned obligation-level evaluator and clarify one
coherent workflow before Qwen is reassessed. Prompt and orchestration
improvements should be tested before a stronger teacher. The old gate and
scores must not be changed retrospectively.

## Measurement follow-up

The repository now includes the draft, separately versioned
`company-task-specialization/evaluator-v2` contract. For legacy-shaped answers
it retains hard failures for parse/schema defects and unauthorized evidence
IDs, while routing free-text obligation completeness, polarity, equivalent
deadlines, decision equivalence, and multipart conclusions to
`semantic_review_required`.

The follow-up also adds:

- a blinded packet builder for all 14 teacher and eight repair answers;
- a structured Phase A human overlay requiring obligation-by-obligation
  satisfaction, unsafe claims, next-step usefulness, and observed ambiguity;
- a validator that freezes those labels before privately comparing them with
  historical results in a separate Phase B report;
- a private, hash-manifested real-work seed schema targeting approximately 25
  approved cases from one workflow.

Evaluator v2 remains `draft_awaiting_human_validation`. Teacher
requalification and student training remain blocked.

## Integrity bindings

- Corrected protocol SHA-256:
  `79d8e2eaa125f79f22811b31d3ec9df9a9d9e226adb9e9e9f6e0e2c796989bc9`
- Corrected development cases SHA-256:
  `f8c1a31eec50d4ede346b49254ddde6aa4d3bbe0b249ea9452febc40708e5dfe`
- Corrected pilot JSON SHA-256:
  `0e4f4690ca4a164dd19182e4ed996d7c7751cf75b2115f28ae43046a2706580c`
- Candidate repair batch SHA-256:
  `ba1e6115e5ecba7f74251c6e4005baa94d231345d7424a04d0c9af8c1ce643cd`
- Draft evaluator-v2 protocol SHA-256:
  `c6408ed7790e3e63c16d2e664ce9e91866a96b23b758ad4530573f9243a0cc47`
- Private real-work seed schema SHA-256:
  `10ec5a7625a70e3d118473f7c3a009cbc87ee5454c1830c74b8e2a44e445d67b`

Raw local generation and verifier artifacts remain excluded from Git because
they contain detailed model outputs and runtime metadata. The frozen contract,
implementation, tests, decision record, and this summary are included for
review.
