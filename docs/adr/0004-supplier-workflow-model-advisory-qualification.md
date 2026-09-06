# ADR 0004: Supplier-workflow model-advisory qualification

Date: 6 September 2026

Status: complete; no arm qualified

## Context

The blinded GPT audit of the v1 pilot found both likely lexical false failures
and likely false acceptances. It also identified real candidate defects:
cumulative treatment of alternative remedies and unsupported operational
consequences. The project owner elected to continue with GPT as the sole
advisory reviewer rather than collect human labels for this exploratory cycle.

## Decision

Open `company-task-specialization/v2-model-advisory` as a distinct bounded
experiment. It makes no human-validation or production claim.

The experiment:

- focuses on one synthetic supplier-onboarding and first-invoice workflow;
- freezes 14 fresh cases before generation;
- represents multipart requests with one decision per request item;
- distinguishes known unmet conditions from unknown states;
- tests alternative remedies and unsupported-consequence behavior;
- compares the same pinned Qwen27 under three prompt strategies;
- uses identical policy evidence and a 1,024-token output cap for every arm;
- reports the two-pass revision arm's total latency and token cost;
- sends every output to a blinded external GPT advisory review.

## Frozen continuation gate

An arm requires at least a `0.90` acceptable rate, no unacceptable output, no
unsafe-claim case, complete success on multipart/unknown/unmet/alternative
cases, and no generation or structured-output failure. Prompt improvement must
not reduce acceptable count. Passing this model-advisory gate does not
authorize training.

## Execution

All 42 expected generations completed:

- baseline: 14/14 structured, zero generation failures;
- obligation prompt: 14/14 structured, zero generation failures;
- obligation plus revision: 14/14 structured, zero generation failures.

There were no truncations. Five outputs repeated the same authorized source ID
for distinct claims and retain the frozen duplicate-citation machine warning.

The blinded GPT review rated each arm 13/14 acceptable, one minor revision, and
zero unacceptable. All 42 answers supported the next step. The same targeted
invoice case failed the zero-unsafe and all-targeted-cases-acceptable checks in
every arm because its action made a recorded Procurement exception sound
compulsory while omitting the approved-purchase-order alternative. Therefore,
the frozen decision is `no_arm_qualified`.

## Consequences

The review packet hides prompt arm, case ID, reference answer, machine result,
and latency. The private mapping is retained locally. The obligation prompt and
revision pass produced no advisory improvement over baseline while increasing
cost. Student training, stronger-teacher comparison, GRPO, gisting, autonomous
admission, and deployment remain blocked.
