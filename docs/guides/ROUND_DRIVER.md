# Round driver (Qwen3.5 family): packed rows, per-row invariant, multi-row drafting

Status: stage 1 behind `YUNSHU_ROUND_DRIVER` (experimental, off). Once validated on Qwen3.8-27B
(`scripts/research/validate_round_driver.sh`) it replaces, for dense Qwen3.5-family models, both
the upstream-`BatchGenerator` shared batch and the single-request speculative lane, and the flag
is deleted.

## Why

Upstream mlx-vlm's `BatchGenerator` runs one `_prompt_batch` at a time, speculative batches that
cannot take joins, and MTP drafter state for one batch. Measured on 27B (M5 Max):

- MMLU-Pro 300 b8 (long reasoning, 8 in flight): Yunshu 139 tok/s (drafts only while a request is
  alone), TensorFold MTP parallel-8 159, Splash 223.
- 8 concurrent 1K prompts: the last TTFT is ~10 s (prefill one request at a time).

## Shape

Every request is a **row** with its *own* single-row caches (a `KVCache` per attention layer, an
`ArraysCache` per GatedDeltaNet layer, and a `KVCache` for its MTP head). Each **step** is one
packed forward of one kind:

- **decode step**: every decoding row's window, `1 + d` tokens (the pending token plus `d >= 0`
  drafts);
- **prefill step**: fixed 512-token chunks of waiting prompts (absolute spans from the prompt
  start), several prompts in one forward, up to 2048 tokens when no row decodes and 512 while rows
  decode.

While rows decode and prompts wait, prefill and decode steps alternate one-to-one; the prefill
budget sets the trade between the waiting prompt's TTFT and the decoding rows' rate. (Packing the
decode rows *into* the prefill forward measured no better than alternating separate forwards at the
same chunk size on Qwen3.8-27B (2026-09-29) — because prefill is compute
bound and the only shared saving is one weight read; so steps stay single-kind.)

Per layer, everything with weights runs once over the packed tokens (norms, attention and GDN
in/out projections, MLP, then one LM-head call over the rows that need logits); only the sequence
mixers run per segment on the row's own caches (attention over its KV; the GDN conv + recurrence on
its state).

## Lossless: what holds per row

A row's arithmetic does not depend on which rows share the step, how many drafts the others
verify, or whether it drafts at all:

- **Projections** (every target linear and the LM head): TensorFold's lane matmul
  (`kernels/tensorfold/lane_qmm.py`, MIT) — per-row bits follow the weight shape, not the row
  count, for 1..128 rows; wider calls are cut into 128-row pieces (`kernels/lane_linear.py`;
  tested 1..300 rows, 4/5/8-bit, including the narrow GDN `in_proj_a/b`).
- **Decode / verify attention** (`T <= 8` tokens): the ragged token-tile kernel over the row's own
  `KVCache` buffer, per-token bits independent of `T` (capacity padded to 64-key windows).
- **GDN**: per row on its own state; every decode window runs under an upstream speculative cache
  transaction, and a rejected suffix rolls back by the transaction's commit (KV trimmed, GDN state
  at the kept length).
- **Prefill**: fixed chunks, each its own segment, so a prompt's bits do not depend on what it was
  packed with.
- Norms, activations, embedding: per token.

For greedy rows: **spec on == spec off, and a row alone == the same row in any batch or join
order**, with drafts all accepted or all rejected, and with the thinking budget's forced tokens
(`tests/unit/test_round_driver.py` on a random 4-bit Qwen3.5; `scripts/research/
sweep_round_driver.py` on a real checkpoint). Sampled rows draw with their own seeded key and never
draft; rows with logits processors (grammar / JSON, penalties) or logprobs never draft.

## Drafting: the MTP head for every greedy row, cost-aware depth

Each draftable row keeps its MTP head's KV current every step: the head absorbs every committed
position with the *target's* hidden state (prompt chunks as they prefill, then each step's kept
window), so the head's cache is the same whether or not the row drafted, and any row can start
drafting at any step. Chain entries from drafting are temporary and trimmed before the next absorb.

Draft depth per row comes from TensorFold's allocation rule (`round_driver/allocate.py`, MIT): the
`j`-th draft lands with probability `prod(per-depth acceptance EMA)`, rows get drafts greedily by
marginal expected tokens, and the step keeps the allocation that maximizes expected committed
tokens per `(forward ms at that many packed rows + chain ms x deepest chain)`, both measured online.
Alone, a row drafts as deep as it pays; at 8 rows, drafts stop where a wider forward costs more than
it lands — no row-count thresholds.

## Not yet (stage 2)

- Batched mixers: decode rows' attention and GDN run one call per row per layer; one launch per
  layer for all rows needs slot-buffer caches (the ragged KV cache's layout for rows' KV and GDN
  state) — the main per-row overhead.
- APC prefix reuse and checkpoints, image / audio prompts (mRoPE), int8 KV, MoE: those requests stay
  on the upstream path.
- DFlash2 drafting (block drafts from the target's layer taps; same allocation).
- Tree drafts.
