# Legacy research prototype — disabled

## Status

The original compile, train, evaluate, route, and adapter-chat workflow is
retained only to preserve the history of the experiment. It is not a supported
or scientifically valid workflow.

Every corresponding `emmlx` command and convenience script fails closed.
There is no override flag.

## What the prototype attempted

The prototype explored a local MLX lifecycle shaped like:

```text
governed records
  -> document reconstruction and continuation views
  -> question/answer alignment
  -> general replay ("Recover-lite")
  -> LoRA adapter
  -> lexical evaluation
```

It also generated per-domain datasets and exposed a lexical domain router.
Those components helped identify the requirements for the revised programme,
but their outputs cannot support research or deployment claims.

## Why it is invalid

### Split leakage

The legacy compiler splits question paraphrases after generation and reuses
the same row in train, validation, and test when only one or two examples are
available. Superficial template variation does not create an independent
evaluation question family.

### Rejected storage substrate

The legacy trainer targets only `q_proj` and `v_proj`, trains only 8–24 layers,
and uses rank primarily as a laptop convenience. It omits MLP projections and
does not implement the rank × exposure experiment.

### Incorrect MLX scale interpretation

MLX-LM applies `scale` directly to the LoRA update. The legacy presets use
20–32 as the multiplier rather than the intended PEFT-style `alpha / rank`
equivalent.

### Fixed iterations

Training length is a fixed preset, not a declared number of validated views
and semantic exposures per fact. Methods are not compute/exposure matched.

### Invalid answer scoring

The legacy evaluator passes answers using substring containment, token F1, or
keyword coverage. It cannot reliably reject wrong numbers, currencies,
comparators, dates, negations, unsupported exceptions, or fabricated
provenance.

### Unvalidated routing and recovery

The lexical router has no accepted routing benchmark. General replay is a
training intervention, not IAR-style post-hoc recovery. Neither may be
described as a solved deployment capability.

## Disabled interfaces

The following names remain recognizable so a user receives an explicit
explanation rather than a missing-command error:

```text
emmlx compile
emmlx train
emmlx evaluate
emmlx route
emmlx chat
scripts/run_demo.sh
scripts/run_ablation.sh
```

They all terminate before legacy compiler, trainer, evaluator, router, or
inference code executes.

## Replacement requirements

The legacy path must not be re-enabled. A replacement requires:

1. verified frozen evaluation assets and explicit temporal evaluation dates;
2. split-contract validation before training-view generation;
3. exact, lexical, and semantic-neighbour leakage enforcement;
4. exposure-derived training schedules and all-linear rank experiments;
5. answer-blind generation followed by typed/provenance/semantic grading;
6. general-capability and retention guardrails;
7. versioned, classified artifacts and promotion/rejection evidence.

See [`ROADMAP.md`](ROADMAP.md) and
[`adr/0001-purpose-hardware-and-first-boundary.md`](adr/0001-purpose-hardware-and-first-boundary.md).
