# Exact-memory QLoRA: training 35k-token rows on a 128 GB Mac

How we made a 4-bit Qwen3-4B adapter update on complete company-review
examples without shortening the evidence, changing the loss, or leaving
the laptop.

This note is written for a technical audience and can be published as-is.
It describes a **runtime implementation change**. The training objective,
approved targets, model identity and LoRA settings were not altered to
make the run fit.

## The problem we actually had

The hardware target for this repository is one M4 Max with 128 GB of
unified memory. That is a lot of RAM for a laptop and not a lot of RAM
for long-context training.

The experiment asked a 4-bit Qwen3-4B Instruct model
(`mlx-community/Qwen3-4B-Instruct-2507-4bit`) to imitate five
owner-approved review answers. The longest complete chat row is
**35,406 tokens**. Almost all of that length is the evidence packet.
Only **1,912 tokens** are the assistant answer we supervise.

The first stock MLX-LM training path died on the first backward pass of
that row:

```text
[METAL] Command buffer execution failed: Insufficient Memory
```

That failure was about **this configured execution path**, not about
whether Qwen4B or supervised learning can work. Shortening the evidence
or dropping the long case would have been a different experiment.

Installed versions, recorded with `importlib.metadata.version` rather
than `mlx.__version__`:

| Package | Version |
|---|---|
| mlx | 0.32.2 |
| mlx-metal | 0.32.2 |
| mlx-lm | 0.31.3 |

Official MLX 0.31.x Metal training selects **unfused attention** and
falls back for attention gradients. The installed 0.32.2 build still
materialized a full-length score matrix on the stock path. We treated
that as a fact to measure, not a reason to upgrade dependencies or
change inference.

## Two tensors, not “the model is too big”

A 4-bit 4B model with rank-16 QLoRA on all 36 layers is not what
exhausted memory. Two activations were.

### 1. The attention score matrix

Unfused causal attention at training time builds a score tensor of
shape `[heads, queries, keys]`. For the longest row, after the usual
next-token shift:

```text
32 × 35,405 × 35,405 × 2 bytes ≈ 75 GiB
```

That is **one layer**, scores only, before softmax, values, residuals,
or backward. Thirty-six layers and a backward pass make the naive
picture impossible on 128 GB.

Grouped-query attention (32 query heads, 8 key/value heads) does not
save you if the implementation still broadcasts to the query-head
count before the `QK^T`.

### 2. The full-vocabulary logits

Stock `default_loss` projects every hidden state to the tokenizer
vocabulary, then masks the loss. For this model:

```text
35,405 × 151,936 × 2 bytes ≈ 10 GiB
```

Ten gigabytes of logits is survivable by itself. Combined with the
score matrix, optimizer state and the backward graph, it is not the
place to be generous. And it is wasted: only the assistant span is
supervised.

## What we did not do

These would have made the run smaller and the experiment different:

- truncate the evidence or the target
- drop the long case
- reduce layers, rank, or sequence length
- use a sliding window
- detach keys and values
- replace causal attention with an approximate kernel
- silently inherit a warmup/cosine schedule
- change the inference path

The transformer still sees the **original causal input**. The loss is
still token-average cross-entropy over the same assistant tokens,
including the first supervised token and the end-of-answer marker.
Gradients still flow through the prompt.

## Trick 1: score 256 queries at a time, keep every allowed key

Each query block of 256 still attends to **all originally permitted
keys and values**. The mask uses **absolute positions**, not a
rectangular “this block is causal if we pretend it starts at zero”.

For a query at position `i`, the allowed keys remain `j ≤ i`. A block
that starts at 512 and ends at 768 is masked with those absolute
indices. The last block can be short; that uneven tail is required,
not optional.

One block of scores is about:

```text
32 × 256 × 35,405 × 2 bytes ≈ 0.54 GiB
```

A Python loop over blocks is not a memory proof. The implementation
**checkpoints each block** so backward recomputes that block’s scores
instead of retaining every block’s `QK^T` at once.

Grouped-query mapping, Q/K RMSNorm, RoPE and the usual scale stay
exactly where the Qwen3 module already puts them. We only replace the
training-time `scaled_dot_product_attention` call. Generation still
uses the stock MLX kernel and KV cache.

## Trick 2: project the assistant span, not the prompt

Hidden state `i` predicts token `i + 1`. If the prompt is `P` tokens
and the full row is `L` tokens, the supervised hidden span is:

```text
start = P - 1
end   = L - 1          # exclusive
count = L - P
```

For the long case: `P = 33,494`, `L = 35,406`, `count = 1,912`.
The first selected hidden index is **33,493**, not 33,494. Treating
`P` as the first hidden index would drop the first assistant token
and supervise one token too far to the right.

The full sequence still goes through every transformer block. After
the final norm we gather those 1,912 hidden states, then apply the
**original tied, quantized** output projection. We never compute a
35k-by-vocab logit tensor and slice it afterwards.

Projection plus cross-entropy runs in chunks of 256 tokens. Chunk
losses are **summed in float32**, then divided by the total supervised
count. Averaging chunk means would overweight a short final chunk.

## Trick 3: checkpoint the layer class once

MLX-LM’s `grad_checkpoint(layer)` replaces `__call__` on `type(layer)`.
Calling it once per layer instance, 36 times, nested the wrapper 36
deep. That is not “more checkpointing”. It is a larger graph and a
larger working set.

The corrected installer wraps each relevant layer class **once**,
marks the wrapper, and refuses a second wrap in the same process.
Preflight and the measured trainer share that wrapper lifetime. They
do not independently wrap the same class again.

This was a defect in our first retry, not a property of Qwen or of
supervised learning.

## Smaller contracts that still matter

**Do not pad every row to the global longest length.** A 8,183-token
example is padded only to the next 32-token alignment. VALIDATION.md
already records that padding a short row to an unused 2,048 cap blew
Metal’s graph resource count. A 35,406 unused cap would have been
worse.

**Do not supervise padding.** MLX-LM’s `default_loss` numbers target
positions from one. If the length you pass it is the unpadded token
count, a padded row can include the first padding token in the loss.
That is a loss-contract defect. We report it separately rather than
calling a pad-excluding implementation “bitwise identical” to the
stock helper.

**Do not trust an unevaluated lazy graph.** A forward that has not
been `mx.eval`’d and synchronized is not a successful pass.

## How we knew the math was the same

Before the long retry we ran disposable short-sequence checks on the
real quantized, tied-output 4B model, where the stock path still
fits.

Paired comparisons, with identical seeds and dropout:

- scalar masked loss and supervised-token count
- attention outputs and Q/K/V gradients, including an uneven final
  query block
- every trainable adapter gradient
- one AdamW step from the same initial state
- a **nonzero** LoRA-B state, so zero-initialized adapters cannot hide
  a missing gradient path

We used declared float16/float32 tolerances, not a claim of bitwise
identity. Adapter gradients matched to numerical zero on the short
rows. The first loss mismatch we saw was float16 reduction of chunk
sums; casting per-token loss to float32 before the sum closed it.

Those checks are runtime tests. They are not extra training examples
and not measured optimizer updates.

## What the optimized preflight showed

On the same five frozen rows, same rank-16 QLoRA, same constant
`5e-5` AdamW, same seed 42:

| Step | Case | Tokens | Loss |
|---|---|---:|---:|
| 1 | DEV-004 | 35,406 | 1.477 |
| 2 | DEV-002 | 8,183 | 1.493 |
| 3 | DEV-004 | 35,406 | 1.146 |

Loss and gradients were finite. 504 adapter tensors changed. The
disposable adapter saved and reloaded. DEV-004 still contributed
exactly 1,912 supervised tokens.

The preflight adapter was then discarded. The measured 40-update run
reloads the untouched pinned base and starts a fresh adapter.

MLX’s reported peak allocator figure on the long row was larger than
physical RAM. That number is not “we used 183 GB of wired memory”.
The useful fact is narrower: the stock path raised
`kIOGPUCommandBufferCallbackErrorOutOfMemory` on the first backward
pass; the exact-memory path completed that pass and kept training.

## Implementation identity

In this repository the training-only helpers live in
`src/enterprise_memory_mlx/learning_mechanics_exact.py`.

| Knob | Value |
|---|---|
| Implementation id | `qwen4b-learning-mechanics-exact-memory/v1` |
| Query block size | 256 |
| Projection chunk size | 256 |
| Checkpointing | once per transformer block class |
| Inference | stock MLX-LM, unpatched |

Scientific settings that did **not** change: model and revision, rank
16, scale 2.0, dropout 0.05, all 36 layers, all seven attention/MLP
projections, constant `5e-5`, weight decay 0.01, batch size 1, seed
42, eight epochs, forty measured updates.

## What this does not mean

This is a laptop engineering result about **exact long-context QLoRA**.

It does not show that the adapter generalizes, that the 4B model is a
qualified teacher, that the pack is training-eligible, or that the
method is ready to promote. Those claims require different evidence.

It also does not show that 128 GB is comfortable. The long row is
still a multi-tens-of-minutes backward pass. The point is that the
limit was two materializations, and both had exact replacements.

## If you take one sentence

Keep the full causal context. Supervise the answer. Never build the
`L × L` score matrix or the `L × vocab` logit tensor if the math does
not require them.
