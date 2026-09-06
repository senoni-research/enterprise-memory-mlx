# ADR 0006: Bounded fact-state amendment

Date: 6 September 2026

Status: complete; stop synthetic prompt iteration

## Context

V3 established two separate findings:

- explicit structured remedies improved alternative preservation, with five
  paired wins and no losses;
- the resulting configuration still failed qualification after reopening facts
  already settled by authenticated workflow state.

The remaining defect concerns policy-to-state application, not remedy
representation or model size. Missing documentation was treated as evidence
that established completion needed verification, and a collective
all-other-controls-complete statement was not preserved.

## Decision

Retain the v3 structured-remedy contract and pinned Qwen27 single-call
configuration. Open `company-task-specialization/v4-fact-state` with exactly
one prompt amendment:

- preserve established authenticated workflow facts;
- treat empty workflow status as unresolved;
- distinguish unverified requester claims from authenticated state;
- treat document observations as document availability only;
- require attachments or verification only when policy says so;
- preserve scoped collective completion statements;
- surface unresolved authorized-record conflicts and honor explicit
  supersession;
- tie every blocker to a policy requirement and unmet state;
- keep no-blocker conclusions scoped to supplied policy and evidence.

The model receives the state-authority metadata that a deployment interface
would need. Expected decisions and evaluator-derived state labels remain
private.

Freeze eight two-scenario contrast pairs before inference. Run all 16 fresh
scenarios through:

1. the unchanged v3 structured prompt;
2. the same prompt plus the compact fact-state amendment.

Both configurations use one call, the same policy and operational state,
Qwen27 revision, deterministic decoding, and 1,024-token cap.

## Measurement

Report targeted learning separately from qualification.

Targeted learning requires more paired fact-state wins than losses, no
remedy-logic loss, and no unsafe-waiver loss. Qualification retains the strict
90% aggregate threshold, zero unacceptable and unsafe outputs, every contrast
case acceptable, and zero generation, truncation, or structural failures.

Review is arm-label blinded with the same output schema. Machine checks remain
limited to structure and source membership; policy-to-state application
requires semantic review.

## Stop boundary

Only one fresh comparison is authorized. If the amendment both improves
fact-state handling and passes the substantive gate, one amended-prompt pass
over the existing 24 v3 cases is authorized as a regression check.

If the defect persists, stop synthetic prompt iteration and inspect the
real-work seed and evidence contract. If fresh and regression gates pass, the
next milestone is a separately approved bounded example-admission protocol,
not another prompt tournament.

No outcome authorizes student training, automatic admission, a stronger-model
campaign, deployment, or a human-validation claim.

## Result

All 32 generations completed with valid structures and no truncation or
generation failure. A model-free runtime correction added the supplied
operational-state IDs to the provenance allowlist; it changed no output, prompt,
case, token count, or timing and added no model call.

The blinded GPT advisory review found:

- control: 8/16 acceptable, three minor, five unacceptable;
- amendment: 10/16 acceptable, four minor, two unacceptable;
- paired fact-state outcomes: four wins and one loss;
- remedy-logic losses: one;
- unsafe-waiver losses: two.

The amendment fixed the primary recorded-completion/absent-attachment contrast
and improved authority handling, but it still converted unknown state into
mandatory action, reopened collectively completed controls, and introduced an
unsafe instruction to record unresolved verification as complete.

The candidate failed the substantive gate and the preregistered no-regression
learning rule. The decision is
`no_success_stop_synthetic_prompt_iteration`. The conditional v3 regression is
not authorized and was not run.

The structured-remedy representation remains retained. The next investigation
must use the approved real-work seed or validated structured workflow state
rather than another synthetic prompt iteration.
