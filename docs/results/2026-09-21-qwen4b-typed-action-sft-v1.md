# From Correct Labels to Supported Actions: A Local Qwen4B Fine-Tuning Experiment

Result note · 21–27 September 2026 · Experiment
`company-task-specialization/qwen4b-typed-action-sft-v1`

This note reports one supervised fine-tune on the 24 reviewed typed-action
TRAIN targets and its matched before/after evaluation. It links to Working
Paper 2026-02, which it does not modify, and adds the later experiment's
outcome. Historical results are preserved unchanged: GRPO v1
`stop_preflight`, the TRAIN-only landscape
`abandon_standard_grpo_for_current_dataset`, the frozen reward
(`grpo_reward.py` `e8fbc652becf93b849ebf905dbbcf9f7f0ce8fc1ae3769488c5feffa90231b0b` —
a historical artifact whose `remedy_skeleton()` discards action text; it was
**not** the checker used for this experiment), the five-example
learning-mechanics adapter, and the working-paper tag
`working-paper-2026-02-v1.0`.

## Main result

We trained a Qwen4B QLoRA adapter (rank 16, scale 2.0, all 36 layers,
q/k/v/o + gate/up/down; AdamW, constant LR 5e-5, weight decay 0.01; exactly
192 measured optimizer updates — 8 epochs × 24 cases, seed 42; ~14 minutes on
an M4 Max) on 24 model-authored, model-advisory-reviewed typed-action
examples using MLX. On eight previously investigated TEST cases excluded from
that training trajectory, the adapted model produced **eight valid
responses, compared with one from the identically constrained base**. Six
adapted responses matched the complete frozen references. A subsequent
arm-label-blinded model-advisory review, joined locally after completion,
judged **seven adapted responses acceptable and one in need of an
alternative-coverage correction; the base produced one acceptable and seven
unacceptable responses**. The improvement combines output-completion
reliability with better evidence-relative decisions under the tested
interface. It does not establish broad generalization, autonomous-action
safety or superiority to reinforcement learning.

| TEST measurement | Base Qwen4B | Qwen4B + SFT adapter |
| --- | ---: | ---: |
| Complete, valid responses | 1/8 | 8/8 |
| Complete agreement with frozen references | 0/8 | 6/8 |
| Current-decision agreement | 1/8 | 8/8 |
| Acceptable in the model-advisory review | 1/8 | 7/8 |
| Minor revision | 0/8 | 1/8 |
| Unacceptable | 7/8 | 0/8 |

The 6/8 and 7/8 figures answer different questions and both are reported:
6/8 is exact agreement with the frozen typed reference; 7/8 is substantive
acceptability under the supplied evidence and task per the blinded advisory.
The divergence is informative — one response (`CTS4-FRESH-08B`) failed exact
reference agreement on a strict supporting-source-set comparison while the
review judged its shorter source list substantively acceptable; the frozen
score stands, and the semantic finding is recorded separately.

### DEV result (published beside TEST, not buried)

On the eight DEV cases (secondary diagnostic): SFT **2/8 complete reference
agreement** (base 0/8) with **8/8 decision agreement** (base 1/8). The TEST
advisory does not resolve the DEV disagreements, and the TEST–DEV gap remains
an open observation: the model is more consistent at selecting the current
decision than at composing the complete typed answer (blockers, remedy
trees, exception citations, source associations).

## Sequence and lesson

This chapter follows a coherent arc. First, the attempted GRPO route exposed
weaknesses in the reward checker (action text ignored; a dropped `any_of`
alternative could keep a perfect score; a conflated provenance flag). No
reinforcement-learning updates occurred, so we established nothing about
GRPO's effectiveness — only about the measurability of the target. Second,
those weaknesses motivated a narrower typed-action representation:
requesting evidence, performing a control and recording completion became
distinct operations with machine-checkable identities. Third, supervised
training on the reviewed typed targets improved the complete
model-plus-decoder system on cases excluded from those updates, and a
source-based review supported the substance of most adapted answers.

The strategic lesson: **before increasing optimisation pressure, make the
target behaviour explicit enough to teach and evaluate; then test a simple
training baseline.** The experiment is consistent with that lesson. It does
not isolate the separate contribution of every representation change, and
it does not compare trained SFT against trained GRPO.

## The two TEST disagreements, kept visible

- **`CTS3-QUAL-020` (SFT, minor revision):** supported decisions and a
  legitimate invoice remedy (`correct_nonconformance` — quote an approved
  PO), but the `any_of` alternative route (a recorded Procurement exception)
  was omitted. The semantic review agreed the remedy coverage is incomplete
  under the alternative-preservation requirement. This is a real remaining
  defect, left uncorrected; there is no retraining to convert it into a
  cleaner headline.
- **`CTS4-FRESH-08B` (SFT, frozen mismatch / advisory acceptable):** the
  answer cited only `PROC-SEC-VER-004`, the policy directly supporting the
  special condition; the review contract did not require an exhaustive source
  list. Useful evidence about the measurement, not permission to revise the
  original score.

The base's seven truncated responses spent the whole 2,048-token budget on
runaway recursive remedy-group nesting (70–76 nested groups, zero action
leaves). The blinded review additionally found **visible decision, state or
source errors before the truncation** in several of them (treating a
recorded-complete questionnaire as blocked on a document observation;
inventing a PDF requirement; reopening controls a collective statement
covers; misapplying the invoice PO-quote rule to PO issuance). So "it only
learned formatting" is an inadequate description of the observed answers —
though we have no clean decomposition of the gain into formatting versus
policy reasoning, and no experiment isolating the interface factor.

## Material qualifications

1. **The evaluation is small and previously investigated.** Eight TEST cases
   span five families; they were excluded from this weight-update trajectory
   but not untouched by the wider research programme. No significance claim
   is made from reused cases.
2. **The base comparison is interface-dependent.** Seven base responses
   exhausted the shared budget through recursive nesting. A differently
   configured base might perform differently; the visible substantive
   mistakes strengthen the result but do not measure that counterfactual.
3. **DEV remains weaker** (2/8 complete reference agreement), published
   beside TEST above.
4. **The review is model-advisory, not human certification.** The advisory
   is a source-based finding, not fully independent corroboration: the
   reviewer participated in the design discussion and saw aggregate
   descriptions before reviewing. It supports a limited finding about these
   outputs — not a safe executor, real-company performance, or reliable
   behaviour across later policy updates.

These limits constrain the claim; they do not erase the positive result.

## Provenance and integrity

- Sealed adapter: `814fee23a53df4a43f735c41900335eba33025e8ac42a3b28170279be624a197`
  (reload-verified; the eval copy `adapters.safetensors` is byte-identical).
- Frozen digests verified before and after training and evaluation:
  `grpo_reward.py` `e8fbc652…`; annotation-v1 review ZIP `8a63b877…`;
  annotation-v2 TRAIN targets `5cdf5629…`.
- Evaluation references: model-authored, `human_attested=false`,
  `independent_review=pending` — agreement with them is a reference-relative
  measurement, not validated truth.
- Advisory packets: parent `e4c56120…`; self-contained derivative
  `b1b47fcb…` (task contract, truncation status, seeded order, empty
  template); completed blinded advisory `0ba965c4…`; the arm-label join is a
  separate append-only local record (`advisory-join-20260923/`).
- All raw outputs, weights, references and private mappings remain local.
  Aggregate sanitized results are exported under
  `docs/results/public/qwen4b-typed-action-sft-v1/`.
- Model-advisory outcomes are not model accuracy; `human_attested=false` and
  `training_authorized=false` throughout. No GRPO/DPO was restarted. This
  phase neither proves RL ineffective nor demonstrates a safe executor.

## Reproduction pointers

Implementation: `src/enterprise_memory_mlx/typed_action_sft_v1.py`,
`typed_action_sft_v1_runtime.py`; stage driver:
`scripts/typed_action_sft_v1_driver.py`; tests:
`tests/test_typed_action_sft_v1.py` (model-free; Metal/snapshot-dependent
tests skip honestly when unavailable). Run directory (local, gitignored):
`artifacts/qwen4b-typed-action-sft-v1/run-20260921T093006Z`.