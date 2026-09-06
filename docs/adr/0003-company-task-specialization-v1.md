# ADR 0003: Company task specialization v1

## Status

Accepted as a bounded, non-training development experiment on 6 September
2026. It does not revise or rescue `model-upgrade-exploratory/v1`.

## Context

The completed model-upgrade experiment found measurable 27B closed-book
acquisition uplift but failed its unknown/OOS continuation gate and remained
far behind direct evidence. A separate question remains: can a smaller model
become better at a stable, source-grounded operational task through validated
examples and repaired failures?

This is task specialization, not evidence that mutable company facts can
replace a versioned factual store. The repository currently contains synthetic
Northstar policies rather than approved historical company work, so the first
batch can test only pipeline mechanics and candidate-example quality.

## Decision

Open `company-task-specialization/v1` with the task:

> Given an operational request and the permitted policy evidence, identify the
> decision, required actions, missing information, applicable exceptions, and
> supporting records.

Use the existing pinned models sequentially:

- untrained Qwen3-4B as the student baseline;
- Qwen3.8-27B as a teacher and repairer candidate;
- Gemma 4 31B as a separate advisory semantic evaluator.

Freeze the task, development cases, endpoint-specific split contract, teacher
qualification, repair selection, and future comparison gates before
generation. Preserve the old acquisition suites outside the repair pool.

No repair hint, Gemma rationale, held-out test material, or unavailable
deployment evidence may enter a candidate example. Deterministic failures
remain authoritative. Gemma does not admit examples by itself.

## Split and claim boundary

Specialization supplies applicable evidence and holds out requests, scenario
combinations, and transfer cases. Closed-book acquisition may train on target
facts but must hold out their evaluation questions and applications. Policy
updating requires explicit previous and updated snapshots plus independent
current, historical, and retention probes.

An untouched final test must be authored and frozen separately. Development
cases may be diagnosed and repaired; final-test cases may not.

## Corrective development pass

The first development pass is preserved at
`artifacts/company-task-specialization/v1/pilot-1f30bc7a492e1844/`.
It exposed ambiguous decision labels and field semantics: known unmet controls
had been treated as missing information, and explicitly absent alternatives
had been treated as applicable exceptions. That pass is
`invalidated_task_contract_ambiguity`.

Execution revision `v2-explicit-decision-and-field-semantics` froze explicit
decision meanings and corrected development references before rerunning.
The corrected report is at
`artifacts/company-task-specialization/v1/pilot-0276f15bf32ef4a4/`.

The corrected advisory result was:

- untrained 4B with evidence: 6/14 deterministic passes, governed mean `0.429`;
- Qwen27 teacher candidate: 7/14 deterministic passes, governed mean `0.500`;
- Qwen27 repairs: 5/8 deterministic passes, governed mean `0.562`;
- teacher qualification: failed;
- advisory-accepted repairs: zero;
- training-eligible repairs: zero.

The teacher omitted material actions on several cases and did not satisfy the
pre-registered referral requirements. It is therefore not qualified to create
training data under this contract.

## Consequences

No student training, GRPO, gisting, continuous admission, or deployment is
authorized. The next useful input is approved real work and human task labels,
not another automated repair pass over these synthetic cases.

Research continuation and deployment eligibility remain separate decisions.
Even a later successful evidence-assisted specialist would not establish
closed-book parametric acquisition.
