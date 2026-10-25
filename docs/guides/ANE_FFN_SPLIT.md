# Neural Engine FFN split (`YUNSHU_ANE_FFN_SPLIT=splash`)

An optional, **lossy** prefill speed-up for dense Qwen3.5-family models (the 27B). Each
prompt step's dense FFN is split by intermediate channel: the GPU runs about three quarters of
the channels, the Apple Neural Engine (ANE) the rest, at the same time, and the GPU adds the
ANE's partial down projection to its own. It is a port of the scheme in
[Splash](https://github.com/incoai/splash) 1.4.0 (Apache-2.0; see `THIRD_PARTY_NOTICES.md`).
Off by default.

```
YUNSHU_ANE_FFN_SPLIT=splash yunshu serve -m <27B checkpoint>
```

## What it does

- The ANE runs a **W8A8** program: int8 weights (rotated by a block Hadamard transform, one scale
  per row, requantized by the GPU from the quantized planes one layer ahead into two
  IOSurface-backed weight sets) and int8 activations (rotated, one scale per token). The program is
  compiled and loaded through the private `AppleNeuralEngine` client, one function per 128-row
  step from 512 to 2048 rows. Private selectors are checked at load; anything missing or changed
  means the GPU runs alone.
- GPU and ANE share IOSurface memory without a copy. A Metal shared event inside the GPU command
  stream tells the ANE its inputs are packed; the ANE signals the same event when its partial sums
  are written, and the GPU's join kernel waits on it. No CPU round trip.
- The share and the least step that takes the split are calibrated on this Mac at first start (a
  few seconds once the program is compiled, remembered in `YUNSHU_ANE_FFN_CACHE`, default
  `~/.yunshu/ane_cache/ffn`), with Splash's timing model. A split that later loses to the GPU
  alone, or any ANE failure, stops it for the life of the process; the step is restored from a
  snapshot of the caches and rerun on the GPU.
- Only single-row prompt steps with enough rows take the split. Decode, speculative verify and
  short tails run as without it. Prefix-cache states computed with the split are kept apart from
  GPU-only states by the kernel id.

## Cost

| | |
|---|---|
| Accuracy | Lossy. Splash's own check: the ANE's part differs from the GPU's by about 2-3% RMS (measured 2.6-3.0% on random rows, 1.8-2.6% on real prompt rows of layers 0-63; 0.3-1.1% of the whole FFN output). Outputs and logprobs differ slightly from `off`. |
| Memory | about 3.0 GiB resident beyond the model (the GPU's copy of the leading down-projection channels, the ANE surfaces and the GPU slices' scales) |
| Start | the program compiles once per OS build (about 30 s cold); later starts load it in a few seconds |

## Settings

`YUNSHU_ANE_FFN_SPLIT` (`off`, `splash`), `YUNSHU_ANE_FFN_SHARE` (fixed fraction, skips the
calibration), `YUNSHU_ANE_FFN_MIN_ROWS` (with a fixed share), `YUNSHU_ANE_FFN_CACHE`,
`YUNSHU_ANE_FFN_TRACE` (per-step ANE and gap timings in the log). All need a restart.

## Measured (Qwen3.8-27B oQ4e, M5 Max, cold prompt, quiet runs)

| prompt | GPU only | split, calibrated | split, share 0.24 from 512 rows |
|---:|---:|---:|---:|
| 1K | 1.009 s | 1.011 s (+0.2%) | 0.911 s (-9.7%) |
| 8K | 7.476 s | 6.902 s (-7.7%) | 6.831 s (-8.6%) |
| 32K | 32.873 s | 31.230 s (-5.0%) | 30.921 s (-5.9%) |
| 128K | 190.0 s | | 183.3 s (-3.5%) |

Medians of 3 interleaved runs (128K: 2). Greedy output of the first tokens is identical in every arm.
The gain is a fixed saving per FFN layer, so it shrinks as attention grows with the context.

Splash 1.4.0 on the same machine, its own ANE split on against off: 1K cold TTFT 0.831 s against
0.938 s (-11.4%), 32K 30.78 s against 34.46 s (-10.7%). With the split on, Yunshu and Splash land on
the same time at 32K (30.9 s and 30.8 s); Yunshu's GPU-only path is already 5% faster, so the
relative gain is smaller.

Share and floor: Splash's calibration (ported as is) picks a share of 0.21-0.24 here and, on this
engine's noisier timings, a least step of 1,550-2,050 rows, so 1K prompts stay on the GPU. The
split layer measured in calibration is 16-17 ms against 18.9 ms GPU-alone per 2048-row layer
(Splash's own fixture: 15.5 against 20.0). `YUNSHU_ANE_FFN_SHARE=0.24` with
`YUNSHU_ANE_FFN_MIN_ROWS=512` took the split down to 512 rows and was the best of the shares tried
(8K cold: 0.09 -> -1.2%, 0.15 -> -4.4%, 0.21 -> -7.7%, 0.26 -> -6.1%, 0.32 -> +5.2%).

Concurrency (a 21K-token newcomer joins a decoding request, 3 interleaved runs each): the newcomer's
TTFT 25.07 s -> 23.42 s (-6.6%); the running request's decode rate while the newcomer prefills stays
11-12 tok/s (11.3 -> 11.7; the scheduler gives decode a fixed slice per prefill atom), and 98 tok/s
before and 78 tok/s after in both arms.

Accuracy on real activations (post-RMSNorm rows of a 1,024-token prompt, layers 0-63 sampled): RMS
0.09 (layer 0) to 1.16 (layer 63), maximum 3 to 52. The ANE's part differs from the GPU's by 1.8-2.6%
RMS (0.3-1.1% of the whole FFN output).

Reproduce: `scripts/research/anenative_calibrate.sh`, `anenative_gate1.sh --arm off|splash`,
`anenative_conc.sh`, `anenative_real_acts.sh`, `anenative_splash_ref.sh` (all through `gpuq`).

## Why not lossless

A Neural Engine share in fp16 with exact dequantized weights would keep the arithmetic close to the
GPU's, but the ANE runs fp16 matmuls about 3x slower per channel than the W8A8 program (native, 2048
rows: 4,352 channels take 38 ms in fp16 against about 12.7 ms for 4,096 W8A8 channels), so it can
take only about 8% of the channels. That saves at most 1.6 ms of a 19 ms layer before the 2-3 ms of
hand-off latency measured in the engine, and fp16 on the ANE has an absolute error floor at the small
activations of the early layers (the Core ML study measured 60% error at input scale 0.05 and 2.7% at
0.5; real post-norm RMS is 0.09-0.3 in layers 0-16). There is no lossless variant.
