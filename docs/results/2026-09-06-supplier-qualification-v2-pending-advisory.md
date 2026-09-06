# Supplier-workflow qualification v2 — generation report

Date: 6 September 2026

Status: blinded GPT advisory pending

Classification: synthetic, model-advisory development evidence only

## Design

Fourteen fresh cases were frozen for the synthetic
supplier-onboarding-and-first-invoice workflow. The same pinned
`mlx-community/Qwen3.8-27B-4bit` revision generated all three arms:

- baseline single pass;
- obligation-focused single pass;
- obligation-focused draft followed by one bounded completeness revision.

Thinking was disabled, temperature was zero, the output cap was 1,024 tokens,
and conversation state was reset for each attempt. The revision arm reports
draft-plus-revision cost.

## Generation results

Baseline single pass:

- 14 cases, 14 structured outputs;
- zero generation failures or truncations;
- 2,240 completion tokens;
- 105.096 seconds;
- 16.477 GB peak memory.

Obligation-focused single pass:

- 14 cases, 14 structured outputs;
- zero generation failures or truncations;
- 2,357 completion tokens;
- 121.875 seconds;
- 16.700 GB peak memory.

Obligation plus revision:

- 14 cases, 14 structured outputs;
- zero generation failures or truncations;
- 4,661 end-to-end completion tokens;
- 254.227 end-to-end seconds;
- 16.791 GB peak memory.

Five attempts repeated an authorized source ID while assigning it multiple
distinct claims. The frozen machine checker records those as duplicate-citation
warnings. They are not omitted from advisory review and do not establish a
semantic failure.

## Pending decision

All 42 outputs are included in a randomized blinded packet. The reviewer sees
the request, request-item IDs, candidate answer, and fictional policy evidence,
but not the prompt arm, case ID, reference, machine warning, model identity, or
latency.

No arm is qualified before those labels return. The frozen gate requires at
least a `0.90` acceptable rate, zero unacceptable outputs, zero unsafe-claim
cases, and complete success on the targeted multipart and state-distinction
cases. Passing remains model-advisory only and does not authorize training.

## Integrity bindings

- Protocol SHA-256:
  `4cd777d1d16c1369418a40d8d8a003c4b8fa4de684593037c7016ff362553e37`
- Qualification cases SHA-256:
  `bc14d0cbba19f65c4cb808f521dd0ecfe06532d8af5ee8155cc21a9f5bd25574`
- Comparison JSON SHA-256:
  `7973f288cc1327193369f70daa43119e9c054489a77371c16611672791b055a7`
- Blinded review packet SHA-256:
  `5f1f0d3a7a25c8431c3b3f4b97d2b09ccc35325fb517813fa7314199f74ebfdf`

Raw model outputs, the private unblinding map, and the blank advisory overlay
remain excluded from Git.
