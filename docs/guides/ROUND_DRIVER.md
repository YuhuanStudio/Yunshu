# Round driver (Qwen3.5 family): one packed forward per step

Status: stage 1 behind `YUNSHU_ROUND_DRIVER` (experimental). Replaces, once validated on
Qwen3.8-27B, both the upstream-`BatchGenerator` shared batch and the single-request speculative
lane for dense Qwen3.5-family models, and the stage-1 fused-prefill capture hack.

## Why

Upstream mlx-vlm's `BatchGenerator` runs one `_prompt_batch` at a time, prefill and decode as
separate forwards, speculative batches that cannot take joins, and MTP drafter state for one
batch. Measured consequences on 27B (M5 Max):

- MMLU-Pro 300 b8 (long reasoning, 8 in flight): Yunshu 139 tok/s (drafts only when a request
  is alone), TensorFold MTP parallel-8 159, Splash 223.
- A 16K prompt arriving while 4 rows decode stalls them to 0.7 tok/s for the whole prefill.
- 8 concurrent 1K prompts: the last TTFT is ~10 s (prefill one request at a time).

## Shape

Every request is a **row** with its *own* single-row caches (a `KVCache` per attention layer,
an `ArraysCache` per GatedDeltaNet layer, and a `KVCache` for its MTP head). A **step** builds
one packed forward over **segments**:

- decode rows: `1 + d` tokens (the pending token plus `d` drafts, `d >= 0`);
- prefill chunks: fixed 64-token spans of waiting prompts (absolute positions from the prompt
  start), as many as the step's token budget allows, several prompts at once.

Per layer, everything with weights runs **once** over the packed tokens (norms, attention and
GDN in/out projections, MLP, the LM head for rows that need logits); only the sequence mixers
run per segment on the row's own caches (attention over its KV; the GDN conv + recurrence on its
state). Decode is bandwidth bound, so the decode rows ride along the prefill tokens the step
computes anyway (Sarathi / vLLM chunked prefill), and draft rows ride along the decode rows.

## Lossless: what holds per row

A row's arithmetic must not depend on which rows share the step, how many drafts the others
verify, or whether it drafts at all:

- **Projections** (every target linear and the LM head): TensorFold's lane matmul
  (`kernels/tensorfold/lane_qmm.py`, MIT), whose per-row bits follow the weight shape, not the
  row count, for 1..128 rows; larger calls are cut into 128-row pieces (`kernels/lane_linear.py`).
- **Decode / verify attention** (`T <= 8` tokens): the ragged token-tile kernel over the row's
  own `KVCache` buffer, whose per-token bits do not depend on `T` (capacity padded to the
  kernel's 64-key window, see `ragged_kv.dense_lane_attention`).
- **GDN**: per row, on its own state; decode/verify rows always run the recorded-history kernel
  under an upstream speculative cache transaction, so a draft window and plain decode take the
  same per-token recurrence; rejected tokens roll back by the transaction's commit (KV trimmed,
  GDN state replayed to the accepted length).
- **Prefill**: fixed 64-token chunks, each its own segment (attention SDPA and GDN kernel over
  exactly that chunk), so a prompt's bits do not depend on what it was packed with.
- Norms, activations, embedding: per token.

So for greedy rows: **spec on == spec off, and a row alone == the same row in any batch**
(tested bitwise on a 4-bit Qwen3.5-0.8B: `tests/unit/test_round_driver.py`). Sampled rows use
their own seeded key per drawn position and never draft; rows with logits processors
(grammar/JSON, penalties) never draft.

## Drafting: cost-aware, per row, every step

Each row keeps its MTP head's KV current every step (the head runs on the kept positions'
hidden states and next tokens: one extra decoder layer per committed token), so any row can
start drafting at any step — no "enter speculation with missing history".

Draft lengths come from TensorFold's allocation rule (`engine/allocate.py`, MIT): each row's
`j`-th draft lands with probability `prod(acceptance rate at depth <= j)` (per-row per-depth
EMA), and rows are granted drafts greedily by marginal expected tokens while
`expected tokens / (forward ms(rows) + overhead)` improves. `forward ms(rows)` is measured
online by total packed rows. Alone, a row drafts deep; at 8 rows, drafts stop when a wider
forward costs more than it lands — no fixed row-count threshold.

## Scheduling

Admit → prefill chunks under a token budget (small while rows decode, large when none do) →
decode rows every step (never starved) → commit, stop checks, MTP head update → emit tokens.
Cancellation and abandonment are checked between steps.

## Not yet (stage 2+)

- APC prefix reuse and checkpoints, images / mRoPE prompts, thinking budgets, int8 KV, MoE:
  such requests keep the existing runner path until moved.
- DFlash2 drafting (block drafts need the target's layer taps; same allocation).
- Tree drafts.
