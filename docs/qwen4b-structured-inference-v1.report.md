# Qwen4B structured exact-example fitting

Experiment: `senoni-deliverable-review/structured-inference-v1`  
Implementation: `qwen4b-structured-inference/v1`  
Measured run: `run-20260912T112818Z`

This is a format-only engineering and exact-example-fitting comparison. It
does not authorize deployment, new training, new cases, or a generalization
claim. Semantic support and action safety were not assessed here.

## Finding

Under identical structured decoding, the existing adapter achieved
25/25 labels and 5/5 dispositions on five exact training examples;
the base achieved 22/25 and 3/5. Both produced 5/5 valid outputs.
Semantic support and action safety remain pending.

## Preserved historical results

Official unconstrained after scores remain **0/5 complete valid, 0/25
labels, 0/5 dispositions**.

Evaluation comparison SHA-256 is unchanged:
`89b1a872abcb4ee82a6d1cebe14272951d50d66247e4f35d1ca26fcd0d70693e`

The separately labeled post-hoc body-only diagnostic remains **25/25
labels and 5/5 dispositions**, marked `not_original_contract_success`.
It is not official contract success and does not replace the 0/5.

## Bindings

| Item | Value |
|---|---|
| Base | `mlx-community/Qwen3-4B-Instruct-2507-4bit` @ `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` |
| Adapter SHA-256 | `7a5cd28ed8f469a98e82f285bce34ed83801d1ee71f01b69c9a9b20f984a67ba` |
| Adapter digest after run | unchanged |
| Optimizer updates | 0 |
| Environment | mlx 0.32.2, mlx-lm 0.31.3, mlx-metal 0.32.2, llguidance 1.8.0 |
| Structured backend | installed `llguidance` 1.8.0, used as an mlx-lm logits processor |
| Sampler | greedy among grammar-permitted continuations |
| Max generated tokens | 4096 |
| Measured generation calls | 10 (one per case per arm) |
| Tools | disabled |

Recorded `freeze_manifest_sha256` field:
`d043dc33bf3576a43e634717f8a92fd3e3c31a81a5b5020d1494ef706ca53058`

That field stores the measurement-contract digest, not the hash of
`freeze-manifest.json`. See
[`qwen4b-structured-inference-v1.provenance-clarification.json`](qwen4b-structured-inference-v1.provenance-clarification.json).

This comparison SHA-256:
`68baed93bbb12e3bba85403105a107a8f15627cdacd2f48d7f42e735b72e6e9d`

A prior execute, `run-20260912T112732Z`, failed before decoding because
the mlx tokenizer wrapper was unwrapped past `PreTrainedTokenizerFast`.
Those ten records are setup failures, not measured decoder calls. They
were preserved and not overwritten.

## Constrained arms

Both arms used the same frozen inputs, the same already-rendered prompts,
the same declared-review schema, and the same resource controls.
Case-specific correct labels were not placed in the grammar, prompt, or
logits processor. Identifiers were restricted only from fields already
visible in each frozen input. Strict post-generation validation remained
in force.

| Arm | Complete valid | Labels | Dispositions | Rare labels recovered |
|---|---|---|---|---|
| A. Unadapted pinned Qwen4B | 5/5 | 22/25 | 3/5 | no |
| B. Same base + existing adapter | 5/5 | 25/25 | 5/5 | yes |

Arm A missed three requirement labels and both non-proceed dispositions.
Arm B recovered those three labels and both non-proceed dispositions.

Constraints alone made the official whole-text JSON contract pass. Arm A
shows that format recovery is not adapter credit: 5/5 valid outputs still
miss the three rare labels and both non-proceed dispositions.

The adapter increment is Arm B minus Arm A: +3/25 labels and +2/5
dispositions. That increment is exact-example fitting under a structured
boundary. It is not full task success, not evidence-support certification,
and not a replacement for the official unconstrained 0/5.

## Known limitations

- Semantic support and action safety remain pending. A blinded ten-output
  review packet exists locally and has not been judged.
- The field named `freeze_manifest_sha256` on both execute records is the
  measurement-contract digest, not a hash of `freeze-manifest.json`.
- No per-run freeze snapshot was copied; later `write_freeze` overwrites
  the freeze directory.
- Exact-memory training-path limitations remain open. See
  [`exact-memory-long-context-qlora.qualification-2026-09-11.md`](exact-memory-long-context-qlora.qualification-2026-09-11.md).
- This result does not authorize training, promotion, deployment, or new
  cases.

## Not published here

Raw outputs, prompts, targets, token IDs, weights, owner/approval
metadata, private mappings, and the blinded review ZIP stay local.
