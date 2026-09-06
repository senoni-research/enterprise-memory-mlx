# ADR 0002: Model-upgrade exploratory v1

Status: accepted for one bounded local exploratory run
Date: 5 September 2026

## Decision

Open `model-upgrade-exploratory/v1` as a new experiment. It does not amend,
rescue, or promote the September 1 candidate. The old adapter, outputs, stopping
rule, and `human_review_pending` status remain unchanged. Disabled legacy
compile/train/evaluate/route/chat paths remain disabled.

The experiment separately measures:

1. generator capability: Qwen3.8-27B versus the matched new Qwen3-4B control;
2. acquisition: each adapter versus its own unadapted base;
3. practical competitiveness: adapted Qwen3.8-27B versus direct full/oracle
   context and the frozen experimental BM25 control.

The immutable protocol is
`knowledge/model_upgrade_exploratory/v1/protocol.json`. Measured generation may
begin only after the source-only curriculum, hybrid target coverage, finite
backward pass, parameter update, adapter save/reload, template behavior, memory,
and local Gemma structured-output checks pass.

## Models

- Generator/trainable candidate:
  `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
- Matched small-model control:
  `mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
- Frozen advisory verifier:
  `mlx-community/gemma-4-31b-it-8bit@f5f3dc92ab4af76724c36c21eb6bedadb3a851be`

All runtime work after snapshot acquisition is local. Large models are loaded
sequentially in separate processes.

## Evidence boundary

Gemma labels have:

- `reviewer_kind: model`
- `verification_status: single_local_judge_advisory`
- `human_approved: false`
- `promotion_eligible: false`
- `usable_for_judge_certification: false`

Deterministic factual and provenance failures remain authoritative. Passing the
prospective screen authorizes only one matched seed-43 repeat.
