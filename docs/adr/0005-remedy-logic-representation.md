# ADR 0005: Explicit remedy-logic representation

Date: 6 September 2026

Status: contract frozen; generation pending

## Context

`company-task-specialization/v2-model-advisory` ended with
`no_arm_qualified`. Every arm correctly blocked the affected invoice but made
one recorded Procurement exception sound compulsory, erasing the approved
purchase-order alternative. Longer prompts and a revision call gave no measured
quality benefit.

This points to a representation defect worth isolating before model selection
or training. The flat action array has no explicit way to distinguish
alternative routes from cumulative requirements.

## Decision

Open `company-task-specialization/v3-remedy-logic` without modifying v2.

The challenger uses recursive business-logic remedy nodes:

- `any_of`: at least one listed route must be completed;
- `all_of`: every listed requirement must be completed;
- nested groups: cumulative controls may contain alternative routes.

Current blockers remain separate from future remedies. A proposed remedy is not
evidence that it is complete, and recommending one route must not erase another
valid route.

First validate the logic model without inference, including operator swaps,
dropped routes, unsupported routes, verbal-versus-recorded exceptions, and a
separate security requirement. Machine authority stops at operator vocabulary,
node shape, request-item coverage, and source-ID membership. Policy support,
operator correctness, route completeness, and free-text action mapping require
semantic review.

Then compare exactly two arms on 24 frozen fresh cases:

1. the existing Qwen27 single-pass baseline and flat-action contract;
2. the same Qwen27, evidence, decoding settings, and one-call budget with
   explicit remedy groups.

The prior 14 cases remain bound as regression-only evidence and are not counted
as fresh qualification evidence. Review is arm-label blinded, not claimed to be
perfectly treatment blinded because the schemas differ.

## Frozen decision rule

An arm requires at least 90% acceptable outputs, no unacceptable or
unsafe/unsupported output, every targeted remedy-logic and multipart case
acceptable, and zero generation, truncation, or structural failures.

The challenger must also have more paired alternative-preservation wins than
losses and no mandatory-condition regression. If both pass without a challenger
quality advantage, choose the simpler control unless a separate operational
benefit is demonstrated.

Report complete single-call token and end-to-end runtime costs, including local
rendering and validation. Do not add an explanation model call.

## Consequences

A passing and improved challenger may receive a bounded advisory teacher
designation only after scoring under this frozen rule. It does not authorize
student training, autonomous admission, deployment, or a human-validation
claim. A 4B training batch and random-example versus failure-repair comparison
remain a separately authorized milestone.
