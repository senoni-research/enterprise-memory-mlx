# M4 Max training runbook

This runbook starts with a cheap end-to-end check, then runs the meaningful
4B QLoRA ablation. The repository detects unified memory automatically; the
40-core GPU does not require a separate setting.

## 1. Bootstrap

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
emmlx doctor
```

Confirm that the report says `Apple Silicon: True`, `MLX-LM installed: True`,
and shows the expected M4 Max chip and unified-memory amount.

## 2. Compile the synthetic knowledge

```bash
emmlx compile
```

This creates deterministic datasets and a knowledge snapshot manifest under
`artifacts/`. One deliberately restricted sample is excluded; this verifies
the default governance gate.

## 3. Smoke-test the entire training path

```bash
emmlx train --stage vanilla --preset smoke
emmlx train --stage all --preset smoke
emmlx evaluate --suite domain --adapter artifacts/adapters/vanilla
emmlx evaluate --suite domain --adapter artifacts/adapters/recover
```

The smoke model is only a plumbing check. Do not interpret its knowledge score
as evidence for or against the method.

## 4. Clear smoke adapters

```bash
rm -rf artifacts/adapters artifacts/configs artifacts/registry artifacts/eval
```

The compiled datasets and manifest can remain because they are independent of
the selected base model.

## 5. Run the meaningful 4B experiment

```bash
./scripts/run_ablation.sh auto
```

`auto` selects a conservative preset from the detected unified memory. The
first meaningful comparison is:

1. base model;
2. vanilla QA-only QLoRA;
3. Inject → Align → Recover-lite QLoRA.

Review the generated Markdown reports in `artifacts/eval/results/`. The main
questions are whether staged training improves held-out domain answers over
vanilla SFT, whether every prior formulation remains accessible, and whether
general-task behaviour regresses.

## 6. Train a modular domain adapter

```bash
emmlx train --domain engineering --stage all --preset auto
emmlx route "What approvals are needed for a high-risk production release?"
```

Repeat for another domain only after the global experiment works. The router
is intentionally conservative and returns `context_required` below its
threshold.

## Memory fallback

Use the smaller preset first:

```bash
emmlx train --stage all --preset m4max-quick
```

The quick preset uses the 4B 4-bit model, batch size 1, eight adapted layers,
shorter sequences and gradient checkpointing. If macOS memory pressure remains
high, reduce `max_seq_length` in the generated YAML under
`artifacts/configs/` and invoke it directly:

```bash
mlx_lm.lora --config artifacts/configs/inject.yaml
```

Keep the same reduction for all stages so the ablation remains comparable.

## Replace the sample corpus

Before introducing company material:

1. duplicate the synthetic JSONL files outside Git history;
2. use only stable, broadly shared, revocable-at-source records;
3. obtain expert-written held-out questions;
4. retain `restricted` and `secret` exclusions;
5. review the manifest before starting training;
6. treat the adapter as sensitive data.

The adapter is a compiled artefact. A policy change should produce a new
knowledge snapshot, a newly trained adapter and a complete regression report;
it should not silently mutate the old adapter.
