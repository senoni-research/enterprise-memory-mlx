# Remedy-logic v3: final model-advisory result

Date: 6 September 2026

Status: complete; no arm passed the frozen gate

## Claim boundary

This is one synthetic, model-advisory comparison. It is not human validation,
production evidence, or authorization to train a student.

`company-task-specialization/v2-model-advisory` remains unchanged with
`no_arm_qualified`. V3 froze a new protocol, 24 fresh cases, both prompts,
business-logic schema, review rubric, and decision rule before inference in
commit `57651fb`.

## Design

The comparison used the pinned
`mlx-community/Qwen3.8-27B-4bit` revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, deterministic decoding, and a
1,024-token output cap.

- Control: the existing single-pass baseline and flat action list.
- Challenger: one call to the same model with recursive `any_of`, `all_of`, and
  nested remedy groups.
- Budget: 24 fresh cases per arm, 48 outputs total.
- Regression boundary: the old 14 cases remained bound but were not regenerated
  or counted as fresh qualification evidence.

Before inference, model-free tests covered neither/either/both invoice routes,
a verbally claimed but unrecorded exception, nested security controls, both
operator reversals, a dropped valid route, and an unsupported route.

Machine checks were limited to recursive node validity, operator vocabulary,
request-item coverage, and supplied source-ID membership. Policy meaning
remained subject to semantic review.

## Generation integrity

Both arms completed all 24 calls:

- zero generation failures;
- zero truncations;
- 24/24 structurally valid outputs per arm;
- zero machine hard failures.

The review packet hid arm labels, case IDs, references, machine results, and
cost. Because the output schemas differ visibly, the review is described as
arm-label blinded rather than perfectly treatment blinded. GPT-5.6 Sol supplied
all 48 advisory labels with `human_attested: false`.

## Frozen-gate results

Control:

- 16/24 acceptable (`66.7%`);
- 0 minor revision;
- 8 unacceptable;
- 6 unsafe/unsupported-claim cases;
- failed acceptable-rate, zero-unacceptable, zero-unsafe, and targeted-case
  checks.

Structured challenger:

- 22/24 acceptable (`91.7%`);
- 1 minor revision;
- 1 unacceptable;
- 1 unsafe/unsupported-claim case;
- passed the aggregate acceptable-rate and execution-integrity checks;
- failed zero-unacceptable, zero-unsafe, and all-targeted-cases-acceptable
  checks.

The frozen result is `no_arm_passes_substantive_gate`. Neither arm receives a
bounded advisory teacher designation.

## Did explicit representation help?

Yes, on the defect it was designed to isolate:

- 5 paired alternative-preservation wins and 0 losses;
- 7 paired overall-outcome wins and 0 losses;
- all 13 challenger cases with applicable alternatives preserved those
  alternatives;
- the challenger had no mandatory-condition regression relative to control;
- the challenger correctly represented the nested
  `all_of(security, any_of(purchase_order, recorded_exception))` case.

The control repeated the flat-action failure on five fresh cases by omitting a
valid invoice route or making one route sound exclusive. The structured
challenger did not repeat that defect. The representation therefore has a
demonstrated paired quality advantage; this is not merely a cleaner schema.

## Why the challenger still failed

Case `CTS3-QUAL-006` stated that the workflow recorded the security
questionnaire as complete while its PDF was not attached. The policy required
completion but did not require an attachment.

- Control demanded that the PDF be attached.
- Challenger changed the decision to `needs_information` and demanded
  verification of the already established completion.

The reviewer judged both outputs unacceptable and unsupported. This shared
fact-state error is outside the alternative-remedy representation defect but is
material under the prospective zero-unsafe gate.

On `CTS3-QUAL-023`, the challenger correctly kept the combined operation
blocked on the incomplete security questionnaire, but unnecessarily
reclassified signed data-processing terms as unknown even though the facts
said every other onboarding control was complete. That output was a minor
revision and caused the all-targeted-cases check to fail independently.

## Cost

Control:

- end-to-end time: `206.091` seconds;
- prompt tokens: `11,043`;
- completion tokens: `4,175`;
- total tokens: `15,218`;
- peak memory: `16.478 GB`.

Structured challenger:

- end-to-end time: `391.317` seconds;
- prompt tokens: `17,307`;
- completion tokens: `6,975`;
- total tokens: `24,282`;
- peak memory: `16.798 GB`.

The challenger cost `1.90x` the control's end-to-end time and `1.60x` its total
tokens. These totals include generation, parsing, deterministic rendering, and
validation. No explanation or revision model call was added.

## Decision and next milestone

The remedy representation worked, but the candidate configuration did not pass
the frozen safety gate. Preserve this failed qualification result.

The next investigation should target policy interpretation and fact-state
adherence: established completion must not be downgraded merely because an
attachment is absent, and known-complete controls must not be reintroduced as
unknown. Do not train these mistakes into a student.

No stronger-model campaign is indicated. Student training, autonomous example
admission, deployment, and human-validation claims remain blocked. Any repair
run or admission protocol requires separate approval. The approved real-work
seed may still be collected independently.

## Artifact bindings

- protocol SHA-256:
  `65353b7c15f3ae0b1e5707d395da834fc054d9fef53cc4ae4b8df31e2c2c98ee`;
- fresh cases SHA-256:
  `85db86bcfb4d0221f6db807acc3aa6b4c08157422895477ecbabc6c15ca6ba06`;
- comparison SHA-256:
  `3ba154f91e5ad5dd466ef176ef41ecccdaa69b461e719c41344436a73de15ff5`;
- blinded packet SHA-256:
  `10928f4aeefbfb015b319dd7e37ae2484c4e6fe82a3615f168160091c3d8efc3`;
- completed advisory SHA-256:
  `17fd61d1795b360c4db14fc22cb375309a1c142b93f1d11f71901b8988055788`.
