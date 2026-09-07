# Real-work evidence readiness review

Date: 7 September 2026

Status: inputs missing; no reviewed training seed exists

## Research phase closure

The synthetic prompt-iteration phase is closed.

- Synthetic prompt iteration: stopped.
- V3 remedy-representation finding: retained.
- V4 fact-state stop decision: retained.
- Teacher qualification: not achieved.
- Student training: not authorized.
- Conditional V3 regression: not authorized and not run.
- Next research input: approved real work with validated state semantics.

The V4 provenance correction remains a separate model-free correction. It added
generator-visible workflow-state IDs to the citation allowlist. It did not
change any of the 32 generated answers, prompts, cases, labels, timings, or
token counts.

## Available real-work evidence

No approved real-work seed is available to this repository or local validation
path:

- `knowledge/private/company-task-specialization/real-work.jsonl` is absent;
- the frozen evaluator-v2 manifest records `real_work_seed_present: false`;
- no workflow owner, selected real workflow, approved policy version, or
  human-reviewed case count is recorded;
- no hash-bound real-work seed manifest has been produced.

The repository contains a schema and validator for approximately 25 approved,
redacted cases from one workflow. It does not contain those cases. Synthetic
supplier-onboarding examples cannot be substituted and called real work.

Result: a reviewed training seed is **not established**. The number of approved
cases that may exist outside the project is unknown, not zero.

## Current schema assessment

`company-task-specialization-real-work-seed/v1` is a useful private collection
shell. It requires redaction, research authorization, policy evidence, an
accepted resolution, and at least one approving human review.

It does not yet establish the evidence semantics exposed by V4:

- which policy version applied at the decision time and why;
- which system or role may establish each operational fact;
- whether a record is authoritative, an attributed claim, or only a document
  observation;
- the decision-time state snapshot, conflicts, and supersession;
- whether document presence establishes completion;
- explicit `any_of`/`all_of` remedy logic;
- actions or state changes the system must never infer or perform;
- whether a workflow owner approved the authority model;
- whether an independent reviewer confirmed the accepted resolution.

Do not treat rows conforming only to the current schema as automatically
training-ready.

## Proposed one-workflow task contract

The workflow owner must first select and approve one actual workflow. The
synthetic supplier workflow is not automatically that workflow.

The workflow-level record should establish:

1. **Scope**
   - workflow identifier and owner;
   - decision types the system may assist with;
   - explicit out-of-scope decisions;
   - applicable jurisdictions, business units, and time range.

2. **Policy authority**
   - approved policy identifiers and versions;
   - effective dates and supersession;
   - owner-confirmed mapping from each decision type to applicable rules;
   - mandatory controls, permitted exceptions, and escalation points.

3. **State authority**
   - source systems or roles allowed to establish each fact;
   - authority of workflow status, uploaded documents, approvals, and
     attributed claims;
   - conflict and supersession rules;
   - facts that require independent verification.

4. **Permitted assistance**
   - interpret the request;
   - locate approved policy and state evidence;
   - apply validated logic where available;
   - expose unresolved policy or state questions;
   - draft a scoped explanation.

5. **Prohibited behavior**
   - invent a missing policy requirement;
   - convert unknown state into an unmet condition;
   - record or recommend recording completion without evidence;
   - waive a mandatory control through an unrelated exception;
   - alter an operational record;
   - present “no blocker under supplied policy” as unconditional approval.

## Proposed per-case evidence contract

Each approved, redacted case should preserve the decision-time view rather than
a hindsight reconstruction.

### Request

- stable private case ID;
- workflow and decision timestamp;
- original redacted request;
- separately identified request items.

### Applicable rules

- policy ID, version, effective period, and approved source;
- workflow-owner explanation of applicability;
- mandatory controls and permitted exceptions;
- explicit remedy logic when it is sufficiently formalized.

### Operational state

For every material fact:

- state record ID and statement;
- source system or attributed person/role;
- authority classification;
- observation and effective timestamps;
- document availability as a separate property;
- verification status where required;
- conflict or supersession links;
- redaction and research-use approval.

An LLM-generated state table is not authoritative evidence.

### Accepted resolution

- one decision per request item;
- established, unresolved, and known-unmet conditions kept separate;
- correct next action;
- complete remedy alternatives and cumulative requirements;
- applicable exceptions;
- supporting policy and state record IDs;
- required escalations;
- forbidden inferences and forbidden state changes.

### Human review

- workflow-owner attestation that policy applicability and source-authority
  rules are correct;
- independent review of the decision and next action;
- recorded disagreements left unresolved until adjudicated;
- no model-generated label substituted for either human review.

## Division of responsibility to validate with the owner

- Approved policy owner: defines mandatory controls and permitted exceptions.
- Authorized source system or reviewed evidence: establishes operational state
  and provenance.
- Deterministic logic: applies conditions only where policy and state semantics
  are sufficiently formalized and validated.
- LLM assistance: interprets requests, locates evidence, drafts explanations,
  and exposes unresolved questions.
- Existing workflow permissions: authorize operational-record changes.

This is a proposed boundary for owner review, not an implemented rules engine,
knowledge graph, or system architecture.

## Training-seed assessment

Current decision: **no reviewed training seed exists**.

A future seed may contain human-authored or independently corrected examples;
the teacher does not need to be flawless. However, no case should be admitted
merely because a model marked it acceptable.

Before any training proposal, the project needs:

- one owner-approved workflow and authority model;
- approximately 25 approved, redacted real cases using the existing target;
- independently reviewed accepted resolutions;
- explicit unresolved-case handling;
- a private validated seed manifest;
- a separate decision on example admission and audit sampling.

No inference run, architecture implementation, admission protocol, or training
is authorized by this review.
