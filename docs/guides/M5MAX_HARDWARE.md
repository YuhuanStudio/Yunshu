# M5 Max hardware map for Yunshu

What each hardware unit of an M5 Max (40-core GPU, 128 GB, macOS 27, MLX 0.32.2) delivers when measured,
what Yunshu uses of it, and which proposals survive a roofline check. Probes live in
`scripts/research/hw/` (each writes JSONL); numbers below are medians from one machine.
Every GPU, ANE and heavy-CPU probe ran under the shared GPU lock, because these units share one
LPDDR5X memory system and one run disturbs another.

## Capability table

| Unit | Measured peak | Yunshu today | Gap |
|---|---|---|---|
| Memory bandwidth (spec 614 GB/s) | GPU read 560 GB/s (91%), copy 530, write 450; SLC-resident 1-64 MB buffers re-read at 500-850 GB/s; CPU read 37 GB/s, copy 140 GB/s | Decode streams 15.7 GB of weights per step | Stock decode reads 507 GB/s (91% of the best read kernel): the floor is 28 ms and the step is 31 ms |
| GPU tensor units (NAX) | 65 TFLOPS half/bf16, 119 TOPS int8 (raw matmul2d); MLX dense bf16 62 TFLOPS, qmm 4-bit 58 TFLOPS at M>=512; without NAX 15.6 TFLOPS | NAX qmm in MLX (prefill), NAX-packed 4-bit kernel for decode and verify | Prefill at ~750 tok/s is 27B x 2 FLOP x 750 = 40 TFLOPS: 65% of the qmm peak. int8 tensor path has 2x headroom and is unused |
| GPU launch / sync | eval round trip 105-150 us; 2-4 us per dependent kernel inside one eval; encode 1 us/op | ~2500 graph nodes, 500 quantized matmuls, 80 custom kernels per decode step | Launch cost is hidden: see mx.compile below |
| ANE | 39-op 27B block, hidden 5120: 6.4 ms (T=64), 118 ms fp16 / 31 ms 4-bit palettized (T=512), 63 ms 4-bit (T=1024); see ANE section | Embedding co-processor only (`ane_embedding.py`) | Prefill on ANE beside decode: 250 tok/s at -8..11% decode; see ANE prefill section |
| CPU | 6 super + 12 perf cores, SME2 (svl 64 B); Accelerate sgemm 2.5-2.7 TFLOPS fp32 (M>=128) but 28 GB/s at M=2..8 and 265 GB/s at M=1; argmax over 248K vocab 76 us, top-50 1.4 ms, softmax 0.33 ms; thread hand-off 7 us | Detokenize, grammar, sampling glue | Nothing in decode is CPU-bound |
| Wired memory | `iogpu.wired_limit_mb` = 124518 (default policy); `mx.set_wired_limit` changes no step time or variance | `set_wired_limit` when the model is over half of the recommended set | Nothing to gain: no page faults observed (0 major) |
| Power | `lowpowermode 0`, AC power, no thermal or performance warning recorded; 8 s decode loops hold 22.3 ms +-0.3 ms | none | Nothing measured to gain |

## Findings by unit

### Memory

Read bandwidth saturates above ~1 GB footprints: 533 GB/s at 1 GB, 560 GB/s at 8 GB. A 27B decode step
reads ~15.7 GB, so the same read kernel needs 28 ms; stock mlx-vlm decode takes 31.05 ms (507 GB/s), the
batch-invariant NAX-packed engine path 37.8 ms with invariance on (31.6 ms with it off).
Per-projection, the packed 4-bit kernel reaches 415-466 GB/s on the big MLP shapes and 130-140 GB/s on
the 5120x1024 K/V projections (2.9 MB each, launch and tail dominated), 270 GB/s on 6144x5120.
The small GDN `in_proj_a/b` (5120 x 48, 35 layers) take 11-16 us each for 0.1 MB.
Buffers up to ~64 MB re-read at 500+ GB/s from the system-level cache (batched dispatch), which is
what makes 1-2 layer microbenchmarks look faster than a whole step: cycle through all layers' weights
when timing.

Roofline: decode tok/s <= BW / bytes-per-token. 560 GB/s / 15.7 GB = 35.7 tok/s plain, i.e. 28 ms.
Only fewer bytes (more accepted tokens per read, lower bits) or sharing reads across rows moves it.

### GPU compute and the neural accelerators

Dense matmul on the NAX path: bf16/fp16 62 TFLOPS at 4096^2 (fp32 30-37, non-NAX 15.6). A 4-bit qmm
reaches 58 TFLOPS from M=512 upward. Decode-shaped qmm stays weight-bandwidth-bound through M=4 (550 GB/s
of weights for 4/5/8-bit); at M=8 the stock kernel drops to 180-260 GB/s (a kernel switch, not a
hardware limit), at M=16-32 the NAX kernel is at 300-400 GB/s, and past M~64 it is compute-bound: M=64
34-50 TFLOPS, M=128 48-52. The crossover where compute matches weight streaming is
M ~ 2 x (TFLOPS x 1e12) / (BW x 8 / bits ...): for 4-bit weights (0.5 B/param) it is
65e12 / (560e9 / 0.5) ~ 58 rows. Verify at 2-8 rows is far below it, so one weight read serves up to
~50 rows for free: the reason speculative and multi-row decode pay.
Raw matmul2d peak is 119 TOPS for int8 inputs (65 for half): an int8-activation path (W4A8 / W8A8)
could double compute-bound prefill in principle, at a quantization-quality cost.

### Launch and sync

Round trip of a trivial eval is 105-150 us; a chain of dependent tiny kernels inside one eval costs
2-4 us each. `MLX_MAX_OPS_PER_BUFFER` (4 to 400), `MLX_MAX_MB_PER_BUFFER` and `MLX_METAL_FAST_SYNCH`
change neither microbenchmarks nor the 64-layer decode proxy (22.3 ms in every setting).
A real 27B decode step builds ~2480 graph nodes (497 quantized matmuls, 80 custom kernels, 305 RMSNorm,
321 reshapes) in 1.7 ms of CPU time, which pipelines behind the previous step's GPU work (`async_eval`).
Ablation on stock mlx-vlm: replacing every quantized matmul with a stub takes the step from 31.1 to 6.8 ms;
stubbing the GDN kernel saves 1.0 ms and attention 0.6 ms (at 512 tokens of context). Everything that is
not a weight matmul costs ~6.8 ms of which launch overhead is part.
`mx.compile` on a Qwen3.5 layer: MLP block 309 -> 307 us, GDN layer 514 -> 504 us (2%), attention layer
unchanged (compiling fuses only elementwise chains, and MLX already fuses the SwiGLU gate). Elementwise
chains do speed up (163 -> 152 us single, 20 -> 4.7 us batched) but those are ~7% of the non-matmul 6.8 ms.
MLX has no indirect command buffers or graph replay to exploit; the launch gap is not the bottleneck.

Roofline: a step cannot go below its weight-read time (28 ms); the ~3 ms above it is where fewer
dispatches could help, at most 10%, and MLX gives no mechanism for it beyond hand-fused kernels.

### ANE

Built with coremltools 9 in an isolated venv (`/Volumes/P5Plus/yunshu-test-envs/ane`); the ANE ran the
whole block (`MLComputePlan`: 36-39 of 39 ops on the ANE) for fp16, int8 and 4-bit palettized weights.
Int4 linear-quantized weights fall back to the GPU; palettization is the 4-bit format the ANE takes.

| workload (hidden 5120) | tokens | fp16 | int8 | palettized 4-bit |
|---|---|---|---|---|
| 5120x17408 linear | 1..16 | 1.3-1.4 ms | 1.3-1.4 ms | 0.77-1.1 ms |
| DFlash2-size block (32/8x128 attn, MLP 17408) | 1 / 8 / 16 | 4.2 / 4.2 / 4.2 ms | 4.1 / 3.2 / 3.2 | 2.2 / 2.2 / 2.3 |
| 5 such blocks (DFlash2 drafter depth) | 8 | 27.6 (CPU fallback) | 15.6 | 11.0 |
| 27B-shaped block (24/4x256 with q gate) | 64 | 6.4 | | 3.5 |
| | 128 | 9.3 | | 9.1 |
| | 256 | 61 | 31 | 15.9 |
| | 512 | 119 | | 31 |
| | 1024 | 235 | 121 | 64 |

ANE throughput on a 27B block at T=1024: ~16 K tok/s per block fp16 -> 4.4 K tok/s for a 64-layer
model if every layer were this block, in practice about 1/4 of the GPU's prefill (GPU ~750 tok/s for
the whole 64-layer model = 48 K tok/s-layers, ANE 4-bit 16 K tok/s-layers per block time of 64 ms).
Concretely: a full 64-layer 27B prefill would take ~64 x 64 ms = 4.1 s per 1024 tokens on the ANE
(250 tok/s) against 1.4 s on the GPU (750 tok/s). The ANE compute-bound rate is therefore about a third
of the GPU's when both run alone.

**ANE next to GPU decode** (`bg_gpu_concurrency.py`, GPU decode proxy of 11.5 GB per step, 22.3 ms alone,
ANE in a separate process):

| ANE load | ANE rate alone -> with GPU | GPU decode alone -> with ANE |
|---|---|---|
| DFlash-size block fp16, M=8 | 238 -> 235 calls/s (-1%) | 22.3 -> 29.0 ms (-23%) |
| same, palettized 4-bit | 445 -> 436 (-2%) | 22.3 -> 25.1 ms (-11%) |
| 5 blocks int8 | 66 -> 65 (-2%) | 22.3 -> 26.6 ms (-16%) |
| 5 blocks palettized | 95 -> 92 (-2.5%) | 22.3 -> 25.2 ms (-12%) |

The ANE keeps ~98% of its rate while the GPU decodes, but it takes 11-23% of the GPU's decode rate
through the shared memory system: every ANE call streams its weights (170-1500 MB per call) from DRAM,
and the fp16 model streams the most. So the ANE is not free bandwidth. Prefill-shaped ANE results with
the GPU decode running are in the "ANE prefill" section below.

### CPU

Accelerate uses SME2 for large sgemm: 2.5-2.7 TFLOPS fp32 at M=1024-4096, but only 28 GB/s effective
weight streaming for M=2..8 (a small-M dispatch path with no SME win) and 265 GB/s for M=1 (matvec).
Sampling glue over the 248,320-token vocabulary: argmax 76 us, top-50 (argpartition) 1.4 ms, softmax
0.33 ms, full sort 2.8 ms per row (x16 rows: 1.2 / 19 / 5 / 44 ms). Hand-off between threads: 7 us.
CPU memory traffic also slows the GPU: a fp32 gemv loop on 1 core dropped GPU decode by 22%, on 4
cores by 23% (the CPU took 37 GB/s of 560), two SME gemm threads by 7%.
Conclusion: sampling on the CPU is cheap for one row (argmax is 76 us, hidden behind the next step
under `async_eval`), impractical for top-k/sort over 16 rows (up to 44 ms), and the CPU cannot draft
usefully (a drafter block reads 300 MB+; at 28-265 GB/s that is 1.2-10 ms per layer).

### Wired memory and residency

The default wired limit is 124.5 GB. `mx.set_wired_limit(max recommended)` before and after a 300-step
decode run gave the same distribution: median 31.24 vs 31.25 ms, p99 32.7 vs 32.0 ms, zero major page
faults either way. `mx.set_cache_limit(0)` made decode slower (median 32.05 ms, more minor faults
from re-allocation). Weights stay resident on a 128 GB machine; there is nothing here to win, and it
matters only on machines where the model approaches the recommended working set.

### Power

`pmset -g`: `lowpowermode 0`, on AC, no thermal or CPU performance warning recorded. Sustained
8-second decode loops hold 22.3 ms with p95 22.5 ms (stream proxy) and 31.1 ms p95 31.6 ms (27B),
so no throttle is visible. High Power mode is a System Settings choice this project does not change;
if a user sees drift under long sessions, they can enable it (Settings -> Battery -> Energy Mode) and
run `scripts/research/hw/decode_proxy_step.py` before and after.

## Prefill and concurrent decode: why a running request collapses

Two GPU command queues do not overlap prefill and decode. With a decode-shaped loop and a prefill loop on
separate MLX streams and threads (`gpu_stream_overlap.py`), both slow down and decode gets no priority:
against a 128-token prefill chunk decode ran at 34% of its solo rate, against 512 tokens 21%, 2048
tokens 8%. Interleaving on one queue (one decode step then one prefill chunk) keeps decode steps at
their normal 22.8 ms, but each token then waits for a whole chunk: decode rate ratio 0.77 / 0.35 / 0.12
for 128 / 512 / 2048-token chunks. A chunk of 2048 tokens takes 160 ms in the proxy (compute-bound, ~54
TFLOPS), a 512-token chunk 41 ms. Nothing in the hardware runs them side by side: the GPU is bandwidth-bound
in decode and compute-bound in prefill, but both fill every core, and the tensor units are shared.

On the real 27B round driver (one MTP row decoding at 129 tok/s, an 8K-token prompt arrives):

| prefill chunk | decode during prefill | prompt TTFT alone | prompt TTFT beside decode | worst decode gap |
|---|---|---|---|---|
| 512 (old fixed span) | 6.4 tok/s | 10.9 s | 12.1 s | 780 ms |
| 128 | 24.2 tok/s | 11.0 s | 15.5 s | 255 ms |

Smaller spans gave 3.8x more decode rate while prefilling, at 28% longer TTFT for the arriving prompt
(prefill work per step falls with the span because each chunk rereads the weights). The prompt's
tokens are identical to running it alone (`b_tokens_equal_alone`), and the same for both chunk sizes.
Now `YUNSHU_ROUND_PREFILL_CHUNK` sets it (default 512, unchanged behaviour).

## Proposals, ranked by expected gain per effort

Each has a one-line roofline argument; proposals that fail it are listed last.

1. **Prefill chunk as a policy (done).** Prefill and decode contend for the same compute, so decode
   rate during a prefill ~ chunk_time^-1: a 128-token span costs 12-15 ms per 27B step vs 60-160 ms for
   512-2048. Measured 6.4 -> 24 tok/s for the running request. Next: adapt the chunk to the number of
   decoding rows and their deadlines (small chunk while a request streams, 2048 when idle). Effort low.
2. **Keep decode kernels at the bandwidth line.** Roofline 28 ms/step at 560 GB/s; stock is 31 ms
   (507 GB/s), the invariant NAX-packed path 37.8 ms with invariance on. The big MLP/GDN projections
   run 415-466 GB/s, the small K/V and GDN gate projections (130-140 GB/s and 10 GB/s) are 5-10% of
   bytes but ~25% of projection time: fuse them into one launch (Q/K/V, or gate+qkv) to reach the
   line. Expected -1 to -3 ms/step (3-8% decode). Effort medium; already in the other agents' scope.
3. **Wider verify / multi-row decode.** One weight read serves up to ~58 rows before compute binds
   (4-bit, 65 TFLOPS / 560 GB/s), and measured qmm at M=1..4 costs the same 91-101 us as M=1: decode
   with 4-8 drafts per step multiplies tokens per read, no new hardware needed. Ceiling: acceptance
   (MTP block 6 at 88 tok/s vs 32 plain). Already in progress (round driver).
4. **Int8-activation prefill (W4A8/W8A8) on the tensor units.** Compute-bound prefill at ~750 tok/s
   uses 40 of 65 TFLOPS; int8 matmul2d peaks at 119 TOPS, so the compute floor could fall ~1.8x, but a
   qmm reading 4-bit weights and quantizing activations changes outputs (lossy): only as a user
   option, and only if the GPU is compute-bound, which it is above M~64. Expected TTFT -25% at best.
   Effort high, lossy.
5. **Batched sequence mixers in the round driver.** Per-row attention/GDN launches at 8 rows are ~2.5
   us/kernel x 100 x rows: 2 ms at 8 rows against a 40-ms step (5%). Effort medium.
6. **ANE prefill beside GPU decode (lossy option).** Measured above: 250 tok/s prefill for the whole
   model while decode keeps 89-92%; roofline holds only for prompts arriving during a stream, and the
   numerics differ from MLX 4-bit. Effort very high.
7. **Sampling glue: leave on GPU.** argmax over 248K is 76 us on CPU and fully hidden; top-k/sort at
   16 rows costs 19-44 ms on CPU, so it stays GPU-side (`YUNSHU_GPU_SAMPLER`). No action.

Dropped, with the reason:

- **MTP/DFlash drafter on the ANE overlapped with GPU verify.** Round n+1's drafts depend on round n's
  accepted tokens, so drafting and verify are sequentially dependent; and the ANE call streams
  170-1500 MB of weights, taking 11-23% of GPU decode bandwidth (measured above). Fails both tests.
- **`mx.compile` on decode layers.** 2% at best on a GDN layer, 0% on attention and MLP: the MLX graph
  is already fused where it matters and dispatch hides behind GPU time. Fails the roofline: the 3 ms
  above the 28 ms floor is 10% at most.
- **CPU drafting or CPU sampling overlap.** A drafter layer reads 300 MB at 28-265 GB/s (1-10 ms) while
  the CPU's own traffic costs the GPU 7-23% of decode; sampling is already 76 us hidden.
- **Wired-memory tuning, `MLX_MAX_OPS_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`, High Power mode.** No effect
  measured (equal medians and variance; no page faults; no throttle).
- **Second command queue for prefill.** Two queues time-share the GPU with no priority; decode fell
  to 8-34% of its solo rate. Chunking on one queue (proposal 1) dominates it.

## ANE prefill next to GPU decode

A 27B-shaped block (hidden 5120, 39 ops) on the ANE at prefill sizes, in a separate process, while the GPU
runs the 22.3 ms/step decode proxy (`session_ane_prefill.sh`, `bg_gpu_concurrency.py`, 8 s each, under the lock):

| ANE block | ANE rate alone -> with GPU decode | GPU decode step alone -> with ANE |
|---|---|---|
| T=512, palettized 4-bit | 16.3 -> 16.0 K tok/s per layer (-1.6%) | 22.36 -> 24.31 ms (-7.7%) |
| T=1024, palettized 4-bit | 16.0 -> 15.8 K (-1.3%) | 22.29 -> 25.10 ms (-10.7%) |
| T=1024, fp16 | 4.2 -> 4.2 K (0%) | 22.27 -> 25.46 ms (-11.1%) |

Read as a whole model: 64 layers at 16 K tok/s per layer is ~250 tok/s of prefill on the ANE (~4 s per
1K tokens; an 8K prompt ~33 s) while decode keeps ~89-92% of its rate. The GPU-chunked alternative
(proposal 1) prefills the same 8K prompt in 15.5 s but leaves decode at 24 tok/s of 129 (19%).

Verdict: the ANE is the only unit that prefills while decode keeps almost its full rate, so it is a
real, if narrow, opportunity: a long prompt arriving while another request streams (agent loops).
It is not free (the ANE's weight streaming costs the GPU ~8-11%), it is 3x slower than the idle GPU
(so never for a cold prompt on an idle engine), and the block measured here is only a shape proxy. A real
path needs Core ML per-layer packages for the full-attention and GDN layers with KV / recurrent state
in and out, fp16 activations and palettized weights that differ numerically from the MLX 4-bit affine
weights the decode uses (so the prompt's KV would not match GPU prefill: a lossy user option, not a
default). Effort very high; ranked last among the survivors.

Roofline: ANE prefill rate = 16 K tok/s-layers / 64 = 250 tok/s, decode loses the ~0.1 of bandwidth the
ANE streams; gain exists only when decode and a prompt are concurrent.
