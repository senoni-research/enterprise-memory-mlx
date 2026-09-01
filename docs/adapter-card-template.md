# Adapter card: `<name>`

## Identity

- Adapter version:
- Base model and immutable revision:
- MLX-LM version:
- Knowledge snapshot hash:
- Training configuration hash:
- Training date:
- Owner:
- Reviewer:

## Intended use

Describe the bounded tasks and authorised user population. State whether the
adapter may answer closed-book or must consult the current source system.

## Explicit non-use

List decisions, domains, users and sensitivity classes for which the adapter
must not be used. Include current facts, legal conclusions, safety-critical
actions and any record requiring source-level access control when relevant.

## Knowledge scope

- Included source records:
- Effective-date range:
- Excluded records and reasons:
- Known conflicts:
- Authoritative source location:

## Training method

- Baseline or staged path:
- Inject objectives:
- Alignment examples:
- General replay corpus:
- Iterations and learning rates:
- Adapted layers, rank and scale:

## Evaluation

| Suite | Base | Vanilla | Staged | Acceptance threshold |
|---|---:|---:|---:|---:|
| Held-out domain | | | | |
| Full retention | | | | |
| General capability | | | | |
| Unknown/refusal | | | | |
| Expert blind review | | | | |

Attach the machine-readable and Markdown evaluation reports. Record every
manual override of the automated score.

## Known failures

Document counterexamples, unsupported paraphrases, routing failures, source
conflicts, leakage tests and any behaviour that worsened from the base model.

## Security and governance

- Sensitivity classification of the adapter:
- Storage and distribution controls:
- Extraction/red-team results:
- Deletion/revocation plan:
- Rollback adapter:
- Re-evaluation trigger:

## Approval

- Technical approval:
- Domain-owner approval:
- Security/privacy approval:
- Expiry or mandatory review date:
