# Supplier-workflow qualification v2 — final model-advisory decision

Date: 6 September 2026

Decision: `no_arm_qualified`

Status: complete model-advisory evidence; not human validation

## Result

The blinded GPT review completed all 42 Qwen27 outputs. Each prompt arm received
the same outcome distribution:

- 13 of 14 acceptable (`0.929`);
- one minor revision;
- zero unacceptable;
- all 14 support the correct next operational step;
- one case with an unsafe or unsupported wording concern.

All three minor revisions map to `CTS2-QUAL-014`, one from each arm. Each answer
correctly blocked the current invoice because neither an approved purchase
order nor a recorded Procurement exception existed. Each then worded the
recorded exception as if it were compulsory, without clearly preserving the
alternative remedy of correcting the invoice to quote an approved purchase
order.

## Frozen gate

Every arm passed:

- minimum case count;
- zero generation failures;
- zero invalid structured outputs;
- minimum `0.90` acceptable rate;
- zero unacceptable outputs;
- prompt non-inferiority;
- revision-latency reporting.

Every arm failed:

- zero unsafe-claim cases;
- all targeted multipart, unknown/unmet, and alternative-remedy cases
  acceptable.

The gate is not relaxed retrospectively. No arm is qualified.

## Prompt comparison

The obligation-focused prompt did not improve a single advisory outcome over
the baseline. It used 2,357 completion tokens and 121.875 seconds, versus 2,240
tokens and 105.096 seconds for the baseline.

The revision pass also produced no advisory improvement. Including its draft,
it used 4,661 completion tokens and 254.227 seconds—more than twice the
baseline cost.

Because quality outcomes were identical, the baseline is the descriptive
operational choice among these three configurations. That is not a
qualification decision.

## Interpretation

The corrected evaluator and multipart contract materially changed the picture
from the v1 pilot: the same Qwen27 system handled 13 of 14 fresh supplier cases
acceptably, with no unacceptable output. The remaining defect is narrow and
repeated across all three prompts. More general instruction text and a generic
revision pass did not solve it.

The next intervention should target representation of alternative remedies,
not model size. A future development contract could encode remedy groups
explicitly as `any_of` or `all_of` so that alternatives are machine-visible.
That requires a new freeze and fresh cases; it must not rescue this result.

## Governance

- Historical v1 scores remain unchanged.
- These labels are from one external GPT review, not human evidence or
  independent agreement.
- No student training, stronger-teacher run, GRPO, gisting, autonomous example
  admission, or deployment is authorized.
- No claim is made about actual company work or production readiness.

## Integrity bindings

- Protocol SHA-256:
  `4cd777d1d16c1369418a40d8d8a003c4b8fa4de684593037c7016ff362553e37`
- Qualification cases SHA-256:
  `bc14d0cbba19f65c4cb808f521dd0ecfe06532d8af5ee8155cc21a9f5bd25574`
- Comparison JSON SHA-256:
  `7973f288cc1327193369f70daa43119e9c054489a77371c16611672791b055a7`
- Blinded packet SHA-256:
  `5f1f0d3a7a25c8431c3b3f4b97d2b09ccc35325fb517813fa7314199f74ebfdf`
- Completed advisory JSONL SHA-256:
  `40ff24cf3259a782e9e86194abef8d7e074205d3b03d13c4245531276a2a2134`
- Advisory summary SHA-256:
  `74d39ff91a491c321ee3bcbbfd2139fcccd225e8a991844366360cd51105c8b6`
- Private decision JSON SHA-256:
  `da6ff3738ba6ef6070d3be9a3a1be50494aedf060091e77c08e54b2c9e314971`
