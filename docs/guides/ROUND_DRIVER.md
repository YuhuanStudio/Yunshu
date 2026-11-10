# Round driver (every mlx-vlm model): packed rows, per-row invariant, multi-row drafting


> Console docs: the maintained, translated version of this page is `/docs/developers/round-driver` (source `frontend/docs/developers/round-driver.mdx`). This file stays as the repository-side reference read by tooling.
Status: the round driver serves every request of every mlx-vlm model, through three model layers: the packed
Qwen3.5 layer (dense and MoE Qwen3.5-family, `forward.py`), the packed Flash-Next layer (Qwen4, `flashnext.py`),
and the generic layer (`generic.py`, the model's own upstream forward, one call per decoding row; no drafter unless opted in, see below) for every other family
and for Apple Paravirtual Metal. Speculation (checkpoint MTP head or DFlash2 drafter) runs on the packed layers;
with `YUNSHU_VLM_DRAFT=off` the driver still serves every request, without drafting, and that is the identity
reference (spec on == spec off, token for token). DFlash rows share a fair 32-node verify budget
(`allocate.fair_nodes`) and verify through `verify.py` (packed windows; every matmul through `lane_linear`);
scheduling uses `serving.work_scheduler.decode_quantum`.

The motivation numbers below describe earlier prototype runs, not the release 0.1.5 cross-engine snapshot. See [current measured comparisons](../BENCHMARKS.md#where-yunshu-stands).

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
- **prefill step**: waiting prompts' *atoms* (512-token chunks on the grid from the prompt start,
  cut at the prompt's checkpoints and end): one atom while rows decode, up to 4096 tokens when no
  row decodes. A prompt's consecutive full atoms run as ONE segment (its matmuls see 4096 rows), a
  partial atom (checkpoint cut, tail) as a segment of its own.

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

## APC prefix cache

The driver uses the same `APCManager` as the default runner, through mlx-vlm's `APCCoordinator`
(checkpoint strategy: hybrid GDN + KV state cannot be sliced, so the reuse unit is a snapshot at a
prefix boundary). A row looks up the longest stored checkpoint of its prompt when it is added and
starts from the restored caches (`row.done` = the restored length); while it prefills it stores
snapshots at the coordinator's checkpoint lengths (every 2048 tokens up to the resident entry budget,
and the prompt minus its last token, so a repeated prompt recomputes one token). Rows that draft
also snapshot the MTP head's KV (appended to the cache list) so a restored row drafts from the first
step; those entries are keyed apart (`extra_hash` salt) from rows without a head, and from the
upstream runner's entries.

Prefill numerics depend on the atom a token is in, so the atom plan is a function of the prompt
alone: `chunk`-token atoms on the grid from position 0, cut at the prompt's own checkpoints and its
end. A row restored at a checkpoint on the grid continues with the atoms a cold prefill runs, and a
repeated prompt has the same plan both times, so hit == miss token for token
(`tests/unit/test_round_driver_apc.py`). A checkpoint from a prompt with a different plan (a prefix
that ends off the grid) is still valid state; the continuation just follows this prompt's own plan.
How many full atoms one step runs together (512 while decoding rows wait, 4096 idle) never changes a
token's bits: stock quantized matmul rows do not depend on a multiple-of-512 row count, prefill
attention runs in `chunk`-token query blocks on the absolute grid, and the narrow GDN
`in_proj_a/b` take the lane kernel (their stock kernel follows the row count). Measured on 27B: the
same prompt as 512 / 1024 / 2048 / 4096-token spans gives identical hidden states and caches over all
64 layers (`scripts/research/probe_span_bits.py`).

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
- **Prefill**: atoms (see the APC section), each segment with its own matmul calls, so a prompt's bits
  do not depend on what it was packed with. Projections of full-atom segments run MLX's stock
  quantized matmul on the layer's weight untiled from the lane layout (`LaneLinear.prefill`,
  transient: a few layers' copies at a time, ~140 ms per step on 27B), not the 128-row lane pieces
  (~25% slower at prefill sizes). A partial atom of up to 256 tokens (a prompt's tail, the token after
  a checkpoint: a repeated prompt's whole prefill) uses the lane kernel: no untiling, so a repeated
  32K document's first token comes ~100 ms sooner.
- Norms, activations, embedding: per token.

For greedy rows: **spec on == spec off, and a row alone == the same row in any batch or join
order**, with drafts all accepted or all rejected, and with the thinking budget's forced tokens
(`tests/unit/test_round_driver.py` on a random 4-bit Qwen3.5; `scripts/research/
sweep_round_driver.py` on a real checkpoint). Sampled rows draw with `keyed_sampling.KeyedSampler` (seed, generation index): the token at index `g` is a pure
function of the row's logits, seed and `g`, so a draft is kept exactly when serial sampling would have drawn it
and a sampled row's stream is the same alone, batched, drafting or not. XTC rows never draft. Other processors
draft when they are windowable (pure token masks, at most one frequency / presence penalty, repetition penalty),
and logprobs and response-format guides draft too; each window row is processed on its own (`driver.windowable`,
`_processors_draft`).

## Drafting: the MTP head for every greedy or keyed-sampled row, cost-aware depth

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

## Step pipeline

A decode step is one host read: the window's tokens are built on the device from the previous step's
(still lazy) drafts; the MTP head runs over every window position inside the verify graph (early
absorb: its greedy readout is the next first draft whichever position the row keeps, only `used`
decides which); the host then walks acceptance and queues the next chain with `async_eval` before
building the next step's graph, so the chain runs on the GPU while the host builds. Cost samples
subtract the queued chain (measured every 16th chain synchronously). A lone drafting row reads out
through the lane's reduced draft vocabulary (`draft_vocab.py`); with several rows the full head reads
out. Single request, Qwen3.8-27B: 75.5 -> 87.3 tok/s against 93 for the default lane. The remaining
~6% is not the verify forward (the lane matmul is faster than the packed kernels there) and not host
time between steps; it has not been isolated.

## Not yet

- Phase B performance/identity qualification on 27B, including 32K concurrency.

## Flash-Next (Qwen4) rows

Text requests of Qwen3.8-Flash-Next (HC + QSA + GDN + MoE + PLE) are driver rows too, alone or with others, every row
drafting with the checkpoint's native MTP head (`round_driver/flashnext.py`). The driver keeps span planning, APC, sampling,
cancel and events; the model side differs:

- every row keeps its own caches (`QSAKVCache` with the QSA indexer keys and block summaries, `ArraysCache` with the GDN state,
  the conv window and the PLE token history), so per-row state is isolated by construction;
- a decode step packs the rows' windows (the pending token and at most one draft) into one forward: the fused HC and router /
  expert kernels run once over the pack (at most 16 positions, then the next pack), PLE / GDN / QSA run per row on the row's cache,
  and each row commits through upstream's speculative cache transactions. Each row gets the bits of the same window verified alone,
  so solo == concurrent and spec on == spec off;
- prefill is the model's own forward per span; spans are the absolute 2048 grid plus the row's checkpoints (Flash-Next prefill is not
  split-invariant), so an APC hit runs the spans a cold prompt did. The prompt-tail replay carries the QSA indexer rows and block summaries and the PLE / GDN state with the KV rows;
- DFlash2 rows (the default when a matching drafter is on disk) keep the Qwen3.5 batch's per-row context rings and batched drafter forward, draft a chain of the drafter's best candidates per round (fair share of a 16-position pack) and verify it through the same packed step with the five-layer feature capture;
- image prompts ride as prompt embeddings + mRoPE positions with a per-row decode offset; explicit cache plans (cache seams) steer the
  row's own APC coordinator (every family); a finished MTP-head row stores its head KV beside the target's for the next turn;
- sampled requests with logprobs draft on MTP head rows too: every window row's log-softmax is reduced on its own (`row_logprobs`), so a window reports the bits of the plain steps;
- not on the driver yet: int8 KV (stays bf16), deeper MTP chains.

## Qwen3.5-MoE (`qwen3_5_moe`) rows

The sparse variant of the Qwen3.5 layer set takes the same packed layer, MTP and DFlash drafting included. Attention, GDN, the
lane projections and the drafters are the dense model's; the feed-forward is the upstream verifier's `_feed_forward`, which for a
sparse block runs the router, the routed experts (one `gather_qmm` call per token and expert, so a token's rows do not depend on
the other tokens of the call) and the shared expert. Prefill spans use the model's own block call. The packed `forward._mlp`
(research / tests) runs a sparse block once per segment. Arithmetic gates on a real MoE checkpoint (solo == concurrent, spec on ==
off) are GPU checks, not CPU-test claims.

## Generic layer (every other mlx-vlm family)

A model that has no packed layer (Gemma, the Qwen-VL / Omni families, ...) is served by the same driver through
`round_driver/generic.py`, on the model's own upstream forward and the stock arithmetic: prefill runs per row on the row's own cache;
every decoding row makes the same upstream call a lone request makes, on its own cache, one call per row per round
(`GenericBatch.forward`). Batched stock kernels (projections at M=rows, attention over left padding) are not row-invariant, so batching
them changed a row's output beside other rows; per-row calls keep solo == concurrent, greedy and seeded.

- known gap: concurrent throughput of small generic families is below the old (non-identical) batched path
  (Gemma-4 E2B 4-bit burst4 about 196 vs 268 tok/s, E4B bf16 about 50 vs 98; solo is unchanged). Batching decode rows through one
  upstream forward is not row-invariant (the first difference is the projection at M=rows, then attention over left padding), and
  MLX's own batch-axis call is bit-exact against the single-row kernel only for quantized layers, not dense bf16 (measured; the
  coordination overhead then cost more than the shared launch saved). The fix direction is a row-invariant multi-row kernel
  bitwise equal to MLX's M=1 gemv / qmv (so a lone row keeps the stock path).
- prefill is the stock single-row recipe: chunks with `n_to_process`, then the last prompt token alone with `logits_to_keep=1`; the
  inputs are the runner's prompt features (`get_input_embeddings` of the prompt, or `prepare_media`'s), sequence-aligned kwargs cut
  with the span, so images, audio and video ride as they do upstream (mRoPE models keep their per-row state: `_rope_deltas`, `_position_ids`);
- spans are the absolute 2048 grid plus the row's checkpoints and `apc_prefill_stride` is the grid (a hit runs the spans a cold prompt did);
- rows keep their own caches (no batch cache, no `extract`);
- no drafter by default and no verify kernels (the opt-in n-gram, draft-LM and Gemma-4 assistant drafters are described below): every row decodes one token a round through its own processors, guide and sampler; APC, cancel,
  memory admission, telemetry and the next-turn store are the driver's and the runner's;
- KV stays bf16 (`YUNSHU_KV_PRECISION` is the Qwen3.5 layer's option) and the prompt-tail replay is not kept;
- hosts whose lane arithmetic self-test fails (older chips) and the paravirtual-Metal VM also run this layer (plain decode).

## Phase B adapter (WIP, still opt-in)

With a bounded-context DFlash2 drafter, the existing driver captures target layer taps during prefill, retains the drafter's trailing window, and keeps each row's target KV/GDN caches as lane transactions. Every decode step reserves each row's root and fairly splits a shared 32-node budget; the remainder rotates between rows. DFlash proposals share one batched drafter forward, while rings, RoPE positions and convolution sequences remain independent. CopyDrafter chains reuse the same budget. Target projections, MLP and the output head run over packed nodes; attention and GDN keep per-sequence state. Each row accepts using argmax or position-keyed Gumbel, then commits its own path through the existing tree replay.

Invariant tool-call guides draft in free text, cut a round at grammar entry, and use masked one-token steps in a constrained body. Logprob reporting uses float32 readout. Other processors, media and int8 KV keep the driver's existing handling for them. Flash-Next native MTP admission is unfinished: `verify.native_chain_forward` packs HC, target projections, router and routed/shared experts, with native per-row QSA/GDN/PLE transactions. Its small-array parity checks include QSA boundaries, PLE history and partial commit/abort; these are not a real-checkpoint or serving verdict. DFlash target APC checkpoints share the driver's target-only layout; proposal rings are reconstructed from the available suffix after a cache hit. The experiment remains off by default. No 27B speed or merge verdict is claimed here.

Verification: `yv ab --base <sha> --cand <sha> --suite phaseb-pilot --label phaseb-pilot`; then `phaseb-identity` (4K/32K matched solo, burst2/4, stagger, greedy/seeded, spec on/off) and `phaseb-speed` (three interleaved quiet repetitions, also agent defaults). Standard long, APC and serve-27b gates remain required.

`YUNSHU_JUMP_FORWARD=1` batches a guide's singleton token continuation into one
target window on packed Qwen3.5, Flash-Next and generic rows. Planning restores
the guide before emission; only emitted tokens advance it. Runs stop at grammar
entry, stop-token and length boundaries. The default remains off: generic upstream
T>1 forwards may round differently from repeated T=1 forwards; their reply
checkpoints use a separate namespace from the default. Thinking-budget
rows and processors without a window contract retain single-token decoding.

`yv ab --base BASE --cand CAND --suite jump_forward` measures the existing
JSON-schema family workload with three interleaved option-off/option-on pairs,
requiring output digest/token-count equality and schema validation. A second
long-literal schema requires forced-window engagement; the ordinary schema may
have no singleton runs. It reports client decode tok/s; it is separate from the unconstrained
`identity` suite and the GPU bit-equality tests.

## n-gram / suffix speculation on the generic layer

A request with `spec_decode: true` (or `YUNSHU_NGRAM_DEFAULT=1`) on a generic-layer model drafts from
its own context: after each emit the row's prompt plus emitted tokens propose the continuation of the
longest earlier match (`YUNSHU_SPEC_PROPOSER` = `ngram` or `suffix`; the proposers are the text engine's,
unchanged), the driver feeds `[pending, *drafts]` as one upstream window and keeps the prefix whose
top-1 equals the draft (`round_driver/ngram.py`). Rejected positions are cut with `cache.trim`; a row
that keeps missing backs off for up to 8 steps.

Guarantee (the old text path's): every kept token is the model's own top-1 over that T>1 window. It is
not bitwise equal to one-token decode on stock kernels, which is why it is a documented opt-in and its
reply checkpoints use their own APC namespace. Scope: greedy rows without guide, thinking budget,
logprobs or prompt logprobs; windowable processors only; one decoding row at a time (rows of a batch cache
would each need their own cut: concurrent rows decode plain); models whose cache cannot trim (rotating
window, recurrent state) decode plain and log it once. Packed Qwen3.5 and Flash-Next rows never take this
drafter: their windows are batch-invariant verifies (MTP, DFlash copy drafts), so spec on equals spec off
there and a `spec_decode` request changes nothing.

`yv ab --base BASE --cand CAND --suite ngram --model <generic model>` runs spec off against on on the
candidate tree (repetitive 1.5K and 32K-token prompts, one prose prompt), requiring engagement on the
repetitive cases and reporting decode tok/s and whether the text was exactly equal; `--suite ngram-plain`
requires the drafter to stay off and the two arms identical (packed layers, rotating caches).

## Draft LM and Gemma-4 assistant on the generic layer

Two model drafters plug in where the n-gram drafter does (`round_driver/draftlm.py`; same window, same
verify, same lone-row / trimmable-cache / greedy scope, same APC namespace and back-off):

- `YUNSHU_DRAFT_MODEL=<mlx-lm checkpoint>`: a separate draft LM (same vocabulary as the target) drafts up to
  5 tokens greedily from its own per-row KV cache.
- `YUNSHU_GEMMA4_ASSISTANT=<assistant dir>` on a Gemma-4 target: the assistant drafts from the target's normed
  hidden state at the last kept position and the target's own shared K/V (the generic layer returns that
  hidden state for a lone row that asks for it).

Both are used for requests with `spec_decode: true`; the assistant wins over a draft LM when both are set,
and either wins over n-gram. Gemma-4's sliding-window cache rotates: once the context nears the window
(512) a window can no longer be trimmed and the row decodes plain (the old text route stopped there too).
Sampled (temperature > 0) rows decode plain on this layer.
