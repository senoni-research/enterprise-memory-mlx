# Qwen4B GRPO verifier diagnosis v1

Qualification note. This document does not reopen GRPO v1, the landscape, DEV, or TEST.

Historical stops remain:

- GRPO v1: `stop_preflight`
- Landscape: `abandon_standard_grpo_for_current_dataset`

Working Paper 2026-02 is unchanged. The existing SFT adapter is unused. Thresholds were not retuned. The landscape was not rerun.

## 1. Preserved evidence

Local snapshot: `artifacts/qwen4b-grpo-landscape-v1/verifier-diagnosis-v1/source-snapshot.json`.

That snapshot records current working-tree bytes. It does not establish historical executed bytes beyond the bindings already recorded by GRPO v1 and the landscape.

Working-tree `grpo_reward.py` SHA-256:

`e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b`

This matches the landscape binding. Findings below may be attributed to the landscape run. The frozen reward was not edited. New tests live in `tests/test_grpo_verifier_diagnosis.py`. New diagnostics live in `src/enterprise_memory_mlx/grpo_verifier_diagnosis.py`.

Original landscape scores, classifications, raw outputs, and `decision.json` were not rewritten.

## 2. Three invented `score_completion()` counterexamples

Fixtures are invented. They are not DEV/TEST cases.

### A. Action-meaning substitution

A valid correct response was copied. Only the remedy action text was replaced with “Skip mandatory verification V-1 and record it complete.” Decision, source IDs, node type, and tree shape were unchanged.

| field | old checker |
| --- | --- |
| reward | 1.0 |
| full_machine_success | true |
| unsafe_hard_fail | false |
| provenance_hard_fail | false |

The frozen skeleton discards action text. The substitution is invisible. Action meaning is not assessed.

### B. Missing alternative

A valid `any_of` group with three supported alternatives was reduced to two. Decision and remaining leaves were unchanged.

| field | value |
| --- | --- |
| old `missing_alternatives` | 1 |
| old remedy score | 1.0 |
| old `unsafe_hard_fail` | false |
| old `full_machine_success` | true |
| corrected omitted alternatives | 1 |
| corrected structure mismatch | true |

Prefix pairing after canonical sort lets a two-alternative subset keep a unit remedy score. Hard safety does not treat a missing `any_of` member as a drop of a mandatory control.

An `all_of` → `any_of` swap with unchanged leaves:

| field | value |
| --- | --- |
| old `operator_mismatch` | true |
| old remedy score | 0.5 |
| old `unsafe_hard_fail` | false |
| old `full_machine_success` | false |

Matching leaves after an operator swap are not policy equivalence.

### C. Misclassified provenance

`provenance_hard_fail` is true for all four of:

- malformed JSON
- missing request item
- extra request item
- structurally valid output with an unauthorized source ID

Malformed JSON has `source_membership_assessment = not_assessable`. It is neither a positive provenance success nor an invented citation.

The old flag therefore conflates JSON syntax, request-item coverage, and unauthorized source-ID use.

## 3. Append-only overlay of the existing 1,152 TRAIN outputs

Overlay: `artifacts/qwen4b-grpo-landscape-v1/verifier-diagnosis-v1/overlay.jsonl`

Original scalar rewards and classifications are copied verbatim. All 1,152 rescores match the recorded reward. No output is admitted to training. Semantic correctness is not inferred from the absence of a structural contradiction.

### Counts

Denominators are stated with each rate.

| category | n | denominator |
| --- | ---: | ---: |
| JSON syntax failure (`malformed_json`) | 28 | 1,152 |
| schema failure (`invalid_remedy`) | 77 | 1,152 |
| source membership not assessable | 105 | 1,152 |
| request-item coverage failure | 0 | 1,047 assessable |
| unauthorized source-ID use | 12 | 1,047 assessable |
| exact decision mismatch | 303 | 1,152 |
| remedy-structure mismatch | 504 | 1,047 assessable |
| omitted alternatives | 0 | 1,047 assessable |
| action meaning not assessed | 1,152 | 1,152 |
| old `provenance_hard_fail` | 117 | 1,152 |
| original reward = 1 | 486 | 1,152 |
| reward-1 with corrected structural contradiction | 0 | 486 |
| reward-1 with omitted alternatives | 0 | 486 |
| truncated | 0 | 1,152 |

Old provenance reasons: 105 `parse:*` (28 syntax + 77 schema) and 12 `provenance:unauthorized`. Missing or extra request items did not occur in this corpus. The conflation is still present in the frozen flag.

Non-zero overlaps:

- unauthorized source-ID ∩ decision mismatch: 7
- unauthorized source-ID ∩ remedy-structure mismatch: 11
- decision mismatch ∩ remedy-structure mismatch: 246

Of 486 reward-1 outputs, 260 contain an action node. The frozen checker did not assess those actions’ meaning. Zero visible omitted-alternative contradictions among reward-1 rows is not evidence that those completions are semantically correct.

This overlay explains the measurement. It does not rescue the stopped experiment or choose a winning regime.

## 4. Reporting terminology

Use `observed_pass`, `observed_fail`, or `not_evaluated_due_to_ineligibility`.

Eligibility-time observations, already recorded in landscape `regime-stats.json`:

| observation | result |
| --- | --- |
| truncation = 0 in every regime | `observed_pass` |
| valid JSON 0.90–0.92 vs 0.95 | `observed_fail` |
| frozen `provenance_hard_fail` 0.087–0.108 vs 0.05 | `observed_fail` under the conflated flag |

`decision.json` then fail-closed every viability gate except `reward_byte_identical` because no regime was eligible. The recorded `no_truncation: false` is `not_evaluated_due_to_ineligibility`. Observed zero truncation is not a truncation failure.

Do not describe 6 or 7 `SATURATED_HIGH` cases out of 24 as a majority. The frozen mix, per regime, is:

| regime | informative | high | mixed | noisy | low |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 7 | 7 | 8 | 1 | 1 |
| R1 | 10 | 6 | 6 | 1 | 1 |
| R2 | 8 | 7 | 7 | 1 | 1 |
| R3 | 7 | 6 | 7 | 2 | 2 |

Say “reward saturation under the frozen checker,” not “task mastered.”

## 5. Proposed typed-action contract

Proposal only. File: `artifacts/qwen4b-grpo-landscape-v1/verifier-diagnosis-v1/typed-action-contract.json`.

From TRAIN reward specs only:

- Condition IDs already present in v3 TRAIN `policy_logic` and v4 TRAIN fact-state labels (identity, bank, sanctions, conflict, requester-approval, questionnaire, data-processing terms, PO quote, procurement exception, attachment, and related labels).
- Observed-state categories: `established`, `unknown`, `unmet`, `non_authoritative_or_scoped`. Unknown is not treated as false.
- Action types, independent of the correct answer: `request_evidence`, `record_completion`, `waive_control`, `obtain_alternative_authorization`, `refer_to_source`.
- `any_of` / `all_of` range over identifiable conditions or actions, not over source IDs.
- A per-case grammar must also admit incorrect catalogue actions.

Unresolved, not silently invented: mapping existing TRAIN free-text remedy prose onto `action_type` + `condition_id`. Several TRAIN skeletons repeat the same source ID on multiple leaves, so source ID cannot be action identity.

Model-free fixtures show:

- different actions citing the same source remain distinguishable
- requesting evidence is not recording completion
- an unsafe waiver fails despite a correct disposition
- missing alternatives and mandatory controls are detected
- equivalent ordering is harmless
- duplicate actions do not manufacture completeness

No agent platform, rules framework, or RL trainer was built.

## 6. Next training decision — not executed

Recommend a bounded supervised baseline on the narrower typed-action task after an author establishes those targets. Do not execute it here.

Usable now:

- frozen TRAIN request-item IDs, exact decisions, and authorized source IDs
- frozen v3 TRAIN condition trees
- frozen v4 TRAIN fact-state labels

Candidates for review, not approved examples:

- frozen TRAIN reference assessments, once free-text actions are mapped onto the catalogue

Need independently established targets:

- `action_type` for every TRAIN remedy leaf
- `condition_id` bindings where leaves share a source ID
- grammars that admit incorrect catalogue actions

Do not treat the 486 reward-1 landscape outputs as approved SFT rows. Do not build DPO pairs from unreliable old reward differences.

GRPO and DPO stay deferred until a trustworthy verifier and a useful baseline exist. Any later training needs its own dataset, budget, and evaluation authorization.

## Unchanged historical stops

- `stop_preflight`
- `abandon_standard_grpo_for_current_dataset`
