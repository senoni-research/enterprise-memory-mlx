# Fact-state v4: final model-advisory result

Date: 6 September 2026

Status: complete; stop synthetic prompt iteration

## Claim boundary

This is a synthetic, model-advisory development comparison. It is not human
validation, production evidence, a teacher designation, or authorization to
train.

V3 remains unchanged:

- the structured-remedy representation produced five paired
  alternative-preservation wins with no loss;
- the overall v3 decision remains `no_arm_passes_substantive_gate`.

V4 retained that representation and tested only a compact fact-state amendment.
The protocol, policy snapshot, 8 two-scenario contrast pairs, prompts, rubric,
decision rules, and stop boundary were frozen before inference in commit
`3c4c24a`.

## Design

Both arms used the pinned
`mlx-community/Qwen3.8-27B-4bit` revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, deterministic decoding, one
generation call, the same state and policy evidence, and a 1,024-token cap.

- Control: the unchanged v3 structured-remedy prompt.
- Candidate: the same prompt plus the fact-state amendment.
- Fresh budget: 16 scenarios across 8 contrast pairs and 2 arms, for 32
  generations.

Generator-visible state metadata distinguished authenticated workflow records,
empty workflow status, unverified requester claims, document observations, and
explicit record supersession. Expected decisions and evaluator state labels
were not supplied to the model.

The contrasts covered recorded versus unknown versus incomplete completion,
document presence and absence, explicit attachment and verification
requirements, collective versus partial completion, conflicting and superseded
records, and requester claims versus authenticated state.

## Model-free provenance correction

The first generated artifact incorrectly treated operational-state IDs as
unauthorized citations because the runtime allowlist included policy IDs but
omitted the state IDs that were visibly supplied to the generator.

This implementation defect was corrected before semantic review:

- the original artifact was preserved;
- all 32 outputs, prompts, cases, tokens, and timings remained unchanged;
- no generation call was added;
- the corrected allowlist contains both supplied policy and operational-state
  IDs;
- all 32 corrected machine grades are non-hard-fail.

The corrected comparison is bound to the original comparison SHA-256
`b816abac0a41a6018487ee6efba5d05d0b0a0d9a41eebae24a6e301247f37c2d`.

## Generation integrity

Both arms completed all 16 calls:

- zero generation failures;
- zero truncations;
- 16/16 valid structured outputs;
- zero corrected machine hard failures.

The semantic review hid arm, case, contrast-pair, reference, machine result,
timing, and token information. Both arms used the same output schema. GPT-5.6
Sol supplied 32 model-advisory labels with `human_attested: false`.

## Qualification results

Control:

- 8/16 acceptable (`50.0%`);
- 3 minor revision;
- 5 unacceptable;
- 7 unsafe/unsupported-claim cases;
- failed the acceptable-rate, all-contrast, zero-unacceptable, and zero-unsafe
  checks.

Fact-state amendment:

- 10/16 acceptable (`62.5%`);
- 4 minor revision;
- 2 unacceptable;
- 5 unsafe/unsupported-claim cases;
- failed the acceptable-rate, all-contrast, zero-unacceptable, and zero-unsafe
  checks.

Neither arm met the substantive gate.

## Targeted learning

The amendment produced directional improvement:

- 4 paired fact-state wins and 1 loss;
- 4 paired overall-outcome wins and 1 loss;
- 2 established-fact-preservation wins and 0 losses;
- 2 documentation-scope wins and 0 losses;
- 5 state-authority wins and 0 losses;
- 2 invented-requirement wins and 0 losses.

It fixed both scenarios in the first contrast pair:

- recorded completion plus an absent, non-required attachment no longer reopened
  completion;
- unknown completion plus an absent, non-required attachment was correctly
  treated as unresolved.

These gains remain useful model-advisory evidence. They are not erased by the
failed gate.

The amendment nevertheless failed the preregistered targeted-learning rule:

- 1 remedy-logic loss;
- 2 unsafe-waiver losses;
- 1 paired fact-state loss.

It therefore did not establish the required improvement without regression.

## Remaining and introduced defects

Unknown state still became a demanded action:

- an empty questionnaire status was sometimes converted into “complete the
  questionnaire” rather than “establish whether it is complete”;
- an unresolved verification status was converted into an instruction to record
  verification as complete without establishing that verification occurred.

Collective completion was still reopened:

- data-processing terms were reintroduced after an authenticated statement
  established all other applicable controls as complete.

Partial state produced unsafe remedy logic:

- unknown sanctions, conflict-of-interest, and data-processing-term states were
  treated as unmet;
- the candidate then omitted sanctions and conflict-of-interest from a purported
  complete `all_of` remedy, implying that fewer controls could resolve the case.

One completion-only case also introduced an unsupported questionnaire-approval
requirement.

These are policy-to-state and workflow-state defects, not failures of the
retained `any_of`/`all_of` representation.

## Cost

Control:

- end-to-end time: `298.445` seconds;
- prompt tokens: `17,097`;
- completion tokens: `5,691`;
- total tokens: `22,788`;
- peak memory: `17.284 GB`.

Fact-state amendment:

- end-to-end time: `325.161` seconds;
- prompt tokens: `20,409`;
- completion tokens: `5,126`;
- total tokens: `25,535`;
- peak memory: `17.566 GB`.

The amendment used `1.09x` the control's end-to-end time and `1.12x` its total
tokens. It added no model call.

## Decision

The frozen decision is `no_success_stop_synthetic_prompt_iteration`.

The amendment showed measurable learning but failed both qualification and the
no-regression requirement. The conditional 24-case v3 regression pass is not
authorized and was not run.

Per the preregistered boundary:

- keep the structured-remedy representation and its v3 finding;
- stop synthetic prompt iteration;
- inspect the evidence contract against the approved real-work seed, or use
  validated structured workflow state;
- do not launch another prompt tournament or stronger-model campaign;
- do not train or automatically admit examples.

No approved private real-work seed is created by this experiment. Any real-work
inspection or bounded example-admission protocol remains a separate
authorization.

## Artifact bindings

- protocol SHA-256:
  `2a5cb788452a5e433cfe977deba30175575b23018b2c0d1ec3fe69e21cd813eb`;
- fresh cases SHA-256:
  `7afe689491c5a6039110a2331f9785f82c52d1a7286006d92b4ca87da753bfe5`;
- policy snapshot SHA-256:
  `518442c84fab4001ddead5d6f4344fa3b79b47ef3dc40e82fa7ff4411bed20e8`;
- corrected comparison SHA-256:
  `764a6f4fc70e541b073a47cbdb8a0732e7ad1fde3c4faaaed3dbbfd568a43f5c`;
- blinded packet SHA-256:
  `c5761fcd78ad06742d43c8394c10b5b608e792dc9b63b6f0a122292a23b3d93b`;
- completed advisory SHA-256:
  `3668aa074045bf38bac76968bf5b13d1928d733437fb046bce96d7e3352a9647`.
