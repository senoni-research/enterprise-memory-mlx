# Company task specialization — model-advisory audit v2

Date: 6 September 2026

Status: model-advisory development evidence only

## Scope

GPT-6 Astra Pro reviewed all 22 blinded outputs from the stopped
`company-task-specialization/v1` pilot: 14 teacher outputs and eight selected
repair outputs. It saw each operational request, candidate assessment, and
authorized fictional policy evidence. It did not see model identity,
teacher-versus-repair role, case IDs, historical deterministic results, Gemma
labels, governed scores, or reference answers.

The resulting labels are not human evidence, independent agreement, production
validation, or a replacement for the historical pilot.

## Advisory outcomes

Across all 22 outputs:

- acceptable: 16;
- minor revision: 3;
- unacceptable: 3;
- cannot assess: 0;
- supports the next step: 17 yes, two unclear, three no;
- material obligations: 97 satisfied, eight unsatisfied, one unclear.

For the 14 teacher outputs:

- acceptable: 9;
- minor revision: 3;
- unacceptable: 2.

For the eight selected repairs:

- acceptable: 7;
- unacceptable: 1.

The repair distribution remains selected and cannot be compared directly with
the teacher distribution.

## Private Phase B comparison

After the advisory outcomes were frozen, the maintainer compared them with the
preserved historical deterministic results:

- eight acceptable outputs had historically hard-failed: possible evaluator
  false failures;
- two unacceptable outputs had historically passed: possible evaluator false
  acceptances;
- one unacceptable output had historically hard-failed: supported model
  failure;
- two minor-revision outputs had historically passed;
- one minor-revision output had historically hard-failed;
- eight acceptable outputs agreed with historical passes.

These are investigation categories derived from one model review. They are not
adjudicated correctness labels.

## What changed in the interpretation

The frozen implementation's `7/14` teacher pass count substantially mixed
lexical measurement failures with task failures. Five of the seven teacher
hard failures were rated acceptable by the blinded advisory reviewer, one
needed a minor revision, and one was rated unacceptable.

The audit also found the opposite error. The historical evaluator and Gemma
accepted unsupported forfeiture instructions in the leave cases even though
the supplied policy stated a carry-over cap without defining the ultimate
disposition of excess days.

The current evidence therefore does not support either extreme claim that
Qwen failed half the task or that Qwen is already qualified. It supports
correcting the task representation and evaluator before reassessing the same
Qwen system.

## Contract corrections indicated

The next task contract should:

- assess obligations semantically rather than by required substrings;
- represent multipart requests with one decision per request item;
- represent alternative remedies as alternatives rather than independent
  mandatory actions;
- distinguish a condition known to be unmet from evidence that has merely not
  been supplied;
- reject unsupported operational consequences even when the quoted policy
  excerpt is correct;
- treat equivalent relative and absolute deadlines consistently;
- preserve hard failures for malformed output and unauthorized provenance.

## Next bounded experiment

Open a new model-advisory-only development revision. Focus on one coherent
workflow and freeze fresh synthetic qualification cases before generation.
Compare the existing Qwen27 configuration against:

1. an explicit obligation-by-obligation prompt;
2. one bounded completeness-check-and-revision pass.

Use identical deployment-available evidence and report end-to-end latency.
Do not train the 4B student, add a stronger teacher, run GRPO, or introduce
gisting during this comparison.

Any claim about actual company work or production readiness remains out of
scope until approved real-work evidence exists.

## Integrity bindings

- Audit packet SHA-256:
  `b366e6893d5576874b3c5ebb7d19425b3032710b1ef25f6d51bd7844e26c8ce2`
- Model-advisory JSONL SHA-256:
  `a18b20a7d791be709353084101480cc06a0dfba3389f763032b9792e4b03d319`
- Advisory summary SHA-256:
  `bdd1da11fd98666a517370c0b16545eba9d75b6ad33a187e98177c6ba12c0965`
- Private mapping SHA-256:
  `447d52bd23d5cfbd9107d22f2dfd30961e3f9f65340b60c572002ce2fbc0cf56`
- Phase B report SHA-256:
  `428b61f9bd5dd924c23edb2d1cc28e4bc08c87b0d656653cb0d049f02ed8c2d8`
- Preserved pilot SHA-256:
  `0e4f4690ca4a164dd19182e4ed996d7c7751cf75b2115f28ae43046a2706580c`
