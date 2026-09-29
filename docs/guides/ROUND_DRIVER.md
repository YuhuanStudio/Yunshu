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

Every request is a **row**. A row prefills on its own single-row caches (a `KVCache` per attention
layer, an `ArraysCache` per GatedDeltaNet layer, a `KVCache` for its MTP head); when its prompt is
done it **joins the decode batch**, where all decoding rows share slot buffers (`round_driver/
batch.py`):

- attention K/V: `[S, HKV, CAP, D]` per attention layer, the row owning slot `s`, keys `0 .. n - 1`
  (the join copies the row's prefill keys in; `S` doubles and `CAP` grows in 512-key steps);
- GDN state and conv window: batch arrays `[B, Hv, Dv, Dk]` / `[B, K - 1, C]` per layer in decode
  order (a join concatenates, a finish takes; a step's kernel output is the next step's input);
- the MTP head's KV: the same slot buffers, its own slot per drafting row.

Each **step** is one forward of one kind:

- **decode step**: every decoding row's window, `1 + d` tokens (the pending token plus `d >= 0`
  drafts), right-padded to the longest window `T`; every weight-bearing op sees `B * T` rows (lane
  matmuls are flat in the row count up to 128 rows), padded positions carry a copy of the row's last
  token and write keys past its length / are skipped by the recurrence;
- **prefill step**: fixed 512-token chunks of waiting prompts (absolute spans from the prompt
  start), several prompts in one forward, up to 2048 tokens when no row decodes and 512 while rows
  decode.

While rows decode and prompts wait, prefill and decode steps alternate one-to-one; the prefill
budget sets the trade between the waiting prompt's TTFT and the decoding rows' rate. (Packing the
decode rows *into* the prefill forward measured no better than alternating separate forwards at the
same chunk size on Qwen3.8-27B (2026-09-29) — because prefill is compute
bound and the only shared saving is one weight read; so steps stay single-kind.)

Per decode layer everything is one launch for all rows: norms, projections and MLP over the packed
`B * T` rows; attention writes every row's keys with one scatter and reads them with one
ragged-attention launch (`kernels/ragged_attention.py`, token-tile kernel: per-token bits do not
depend on the window length, `lengths = n + T` so a shorter window's padded tail only reads garbage
it never returns); the GDN conv is one batched conv and the recurrence one launch
(`kernels/gdn_rows.py`, upstream's step kernel per (row, head, value dim) with a per-row length).

## Lossless: what holds per row

A row's arithmetic does not depend on which rows share the step, how many drafts the others
verify, or whether it drafts at all:

- **Projections** (every target linear and the LM head): TensorFold's lane matmul
  (`kernels/tensorfold/lane_qmm.py`, MIT) — per-row bits follow the weight shape, not the row
  count, for 1..128 rows; wider calls are cut into 128-row pieces (`kernels/lane_linear.py`;
  tested 1..300 rows, 4/5/8-bit, including the narrow GDN `in_proj_a/b`).
- **Decode / verify attention** (`T <= 8` tokens): the ragged token-tile kernel over the row's slot,
  per-token bits independent of `T` and of the other rows (capacity a multiple of 64 keys).
- **GDN**: one launch over the rows' states, each row's arithmetic that of the upstream step kernel
  for its own tokens. The kernel also writes the state after each real token but the last
  (`hist`); a row that keeps only part of its window continues from `hist[kept - 1]` and the conv
  window from its position `kept` (`DecodeBatch.commit`); KV needs only its length set.
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

## Not yet

- APC prefix reuse and checkpoints, image / audio prompts (mRoPE), int8 KV, MoE: those requests stay
  on the upstream path.
- DFlash2 drafting (block drafts from the target's layer taps; same allocation).
- Tree drafts.
