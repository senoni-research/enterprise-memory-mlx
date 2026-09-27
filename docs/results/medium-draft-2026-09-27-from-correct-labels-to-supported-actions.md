# From Correct Labels to Supported Actions: A Local Qwen4B Fine-Tuning Experiment

*Draft for publication. The reviewer was an AI assistant participating in the
research programme; all evaluations were reference-relative and
model-advisory, with no human attestation. Everything ran on one M4 Max.*

## The lesson our failed experiment taught us

We set out to fine-tune a small model with GRPO on governed business
decisions — supplier onboarding, invoice processing — and we stopped before a
single optimizer update. Not because the model was bad, but because our
reward function was measuring the wrong thing.

When we stress-tested the frozen reward checker with invented fixtures, three
cracks appeared. Swap a remedy action's text to *"skip the mandatory
verification and record it complete"* — same source citation, same tree
shape — and the reward stayed a perfect 1.0. Drop one alternative from a
valid three-way "any of these routes" group, and it still scored full marks.
And one overloaded "provenance" flag couldn't tell malformed JSON from a
missing request item from an invented source citation.

No reinforcement-learning updates occurred. We hadn't shown GRPO was
ineffective — we'd shown our target wasn't measurable. There is a world of
difference between those two findings.

That failure pointed at a lesson we now believe generalizes:

**Before increasing optimisation pressure, make the target behaviour
explicit enough to teach and evaluate. Then test a simple training baseline.**

## Making actions typed

The old task let the model write free-text remedies. "Complete the security
questionnaire" and "Skip the security questionnaire" were, to the checker,
the same action — as long as they cited the same policy source. So we
re-built the task around typed operations with distinct meanings:

- `request_evidence` — establish an unknown condition status
- `perform_unmet_control` — carry out a known-unmet required control
- `correct_nonconformance` — fix a document, e.g. quote the missing PO
- `record_completion` — record completion *after* evidence and permission exist
- `request_alternative_authorization` — ask Procurement to record an exception
- `resolve_record_conflict` — surface conflicting authorized records

Action identity became operation + condition + scope, never the source ID
alone. Unknown stopped collapsing into "unmet". A document's presence stopped
counting as a completed control. And one boundary got written into the
contract itself: **a requested action does not establish its outcome**.
Requesting an exception is not recording one.

We authored typed targets for all 24 training cases, ran them through a
model-advisory review (22 acceptable, 2 minor mapping corrections), fixed a
real schema bug the review caught (a union that validated nothing), and
froze a closed, answer-blind output schema: wrong decisions remain legal to
express, and evaluation — not the grammar — rejects them.

## The experiment

One supervised fine-tune. Qwen3-4B (4-bit), fresh rank-16 QLoRA over all 36
layers, AdamW at a constant 5e-5, batch size 1, seed 42. Exactly 192 measured
optimizer updates — eight epochs over the 24 typed examples. Fourteen minutes
on an M4 Max, peak process memory around 3.3 GB.

Then a matched comparison on eight TEST cases that were excluded from those
weight updates (though previously investigated in the broader programme — we
say this plainly): the identical base model and decoder, same prompts, same
constrained-JSON grammar, same 2,048-token budget, greedy decoding.

| TEST (8 cases) | Base | + SFT adapter |
| --- | ---: | ---: |
| Complete, valid responses | 1/8 | 8/8 |
| Matched complete frozen references | 0/8 | 6/8 |
| Current-decision agreement | 1/8 | 8/8 |

On the DEV diagnostic set the adapted model got all eight decisions right
but only 2/8 complete reference agreements — a gap we publish beside the TEST
result rather than bury: the model is more consistent at choosing the
decision than at composing the full typed answer.

## Was it just formatting?

Fair question. Seven of eight base responses burned their entire budget on
runaway recursive nesting — 70-plus nested "group" nodes, never once
emitting an actual action. So the completion improvement is real, but did
the model only learn to stop rambling?

To answer that, we shipped the 16 TEST answers — with arm labels, scores and
references stripped — to a blinded review against the source documents. The
reviewer could see every answer in full, including truncated ones, and
judged substance: does the decision follow from the evidence? Are operations
distinguished correctly? Are alternative routes preserved? Do citations
actually support the claims?

Result, joined to arms only after the review was frozen:

| Advisory outcome | Base | + SFT |
| --- | ---: | ---: |
| Acceptable | 1 | 7 |
| Minor revision | 0 | 1 |
| Unacceptable | 7 | 0 |

Two findings make this more informative than a simple win.

The adapted model's one "minor revision": on an invoice case it picked a
legitimate remedy (correct the invoice to quote an approved PO) but omitted
the alternative route (a recorded Procurement exception). A real defect —
incomplete alternative coverage — left in the report as-is. We did not
retrain to clean up the headline.

And the blinded review found that the base's truncated answers weren't just
malformed — several contained substantive errors *before* the nesting
started: treating a recorded-complete control as blocked because a PDF
wasn't attached, inventing attachment requirements the governing policy
never states, reopening controls a collective completion statement covers.
"It only learned formatting" is not an adequate description of these
answers.

But we resist the stronger claim too. We can't cleanly decompose the gain
into formatting versus reasoning — the base struggled badly with this
particular recursive interface, and we ran no experiment isolating that
factor. The defensible conclusion: fine-tuning improved the complete
model-plus-decoder system under the tested conditions, and the improvement
included better evidence-relative decisions in the saved responses. Not
"we proved a general increase in reasoning ability independent of the
interface."

And the evaluation is small: eight TEST cases across five families,
previously investigated in the wider research programme — excluded from this
training trajectory, but not untouched material. On the eight DEV cases, the
adapted model matched complete references on only 2/8 (though 8/8 decisions).
The TEST–DEV gap — fluent decision selection, weaker full-answer composition
— remains open. No significance claim is made from reused cases.

## Two numbers, one honesty

Why report both 6/8 and 7/8? Because they answer different questions. 6/8
asked: did the entire typed response match the frozen reference, exactly?
7/8 asked: was the response substantively acceptable under the supplied
evidence and task?

One answer failed the first but passed the second: it cited only the one
policy directly governing the special condition, where the reference
expected the general onboarding policy too. The review contract didn't
require exhaustive source lists — so the exact-match endpoint was stricter
than the task warrants. We kept the original score, recorded the semantic
finding separately, and noted the endpoint as a candidate refinement for
future task versions. Nothing retrofitted.

Keeping both numbers is a strength. It shows an evaluation can distinguish
an overly strict reference match from a genuine task defect.

## The honest ledger

What this establishes: a small, typed-action specialist can be trained
locally in minutes; it completes what it starts; and a blinded source-based
review found its answers substantively supported on seven of eight held-out
TEST cases, with the eighth needing one alternative-coverage correction.

What it doesn't: general competence beyond this task; a safe executor;
performance on real (non-synthetic) evidence; superiority to reinforcement
learning — GRPO was never actually trained here, so no such comparison
exists. And the review was model-advisory, from a reviewer who had
participated in the design discussion — a source-based finding, not fully
independent corroboration, and no human certification of anything.

## The takeaway

Our failed GRPO pilot wasn't wasted. It was a measurement audit that told us
our target behaviour wasn't yet explicit enough to optimise. Making it
explicit — typed operations, machine-checkable identities, an answer-blind
grammar, a reference-relative comparator — turned one 14-minute supervised
run into something we can actually reason about: matched arms, frozen
references, a blinded review, and disagreements that tell us about the
measurement rather than hiding it.

The sequence is the method: audit what your reward can see, make the
behaviour teachable, run the simple baseline, and review the answers against
sources — not against your answer key alone.

Before you push optimisation pressure on a model, make sure your target
behaviour is explicit enough to teach and evaluate. Then test a simple
training baseline. That's the whole lesson, and this time the experiment
agreed with it.