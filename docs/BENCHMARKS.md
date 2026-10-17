# Benchmarks

Release 0.1.5 snapshot on Apple M5 Max, 128 GB unified memory. Main is in the 0.1.6
development cycle; these measurements are pinned to `a0661800f6668e79c44ae52351a5aef10e7389d4`.
The [published evidence tables](reports/RELEASE_015_BENCHMARKS.md) preserve the supplied
release015 numbers, ranges, versions, missing cells and provenance (2026-10-13–14).

## Where Yunshu stands

Yunshu does not lead every engine on every axis. The following are observed snapshot
comparisons, not statistically established rankings: Yunshu has two valid long-context
reps, reused competitors one, and only one code/agentic prompt per depth.
Derived gaps are rounded against Yunshu. Three or more interleaved repetitions and more prompts are needed for a durable lead claim.

| Axis | Yunshu | Comparison and gap | Evidence rows |
|---|---|---|---|
| 27B decode at 32K / 64K | oQ4e 134.400 / 117.550 tok/s | Best competitor Splash 116.200 / 92.300; observed advantage 15.66% / 27.35% (different weights) | [L32, L64](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| 27B APC at 128K | 131071 cached tokens in both reps; warm 0.593 s | TensorFold reports 0 cached tokens, warm 247.575 s in its one measured run | [L128](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| Lossless speculation | 18 spec-on/off cells, zero digest mismatches | Scoped identity evidence, not a competitor speed ranking | [Identity](reports/RELEASE_015_BENCHMARKS.md#identity-evidence) |
| 27B cold TTFT at 32K / 64K / 128K | 34.590 / 79.195 / 199.440 s | Splash 31.259 / 74.192 / 198.108 s; Yunshu takes 3.331 / 5.003 / 1.332 s longer (10.66% / 6.75% / 0.68%) | [L32–L128](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| 27B immediate follow-up at 32K / 64K / 128K | 2.958 / 4.022 / 6.452 s | TensorFold 0.335 / 0.451 / 253.910 s (128K cache failed); llama.cpp DFlash2 0.420 / 0.516 / 0.719 s. Yunshu takes 7.05 / 7.80 / 8.98 times llama.cpp's time | [L32–L128](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| 27B warm TTFT at 32K / 64K / 128K | 0.182 / 0.329 / 0.593 s | llama.cpp DFlash2 0.095 / 0.126 / 0.194 s (Yunshu +0.087 / +0.203 / +0.399 s); Splash 0.143 / 0.169 / 0.220 s (Yunshu +0.039 / +0.160 / +0.373 s) | [L32–L128](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| 27B decode at 128K | oQ4e 77.950 tok/s | Splash 89.900 and TensorFold 82.400; Yunshu 13.30% / 5.41% slower. Yunshu TF q4g64 is 98.700, so the conclusion depends on weights | [L128](reports/RELEASE_015_BENCHMARKS.md#27b-long-context) |
| Flash-Next decode at 32K / 64K | 67.050 / 56.750 tok/s | llama.cpp MTP 74.000 / 69.300; Yunshu 9.40% / 18.11% slower, cross-weights | [F32, F64](reports/RELEASE_015_BENCHMARKS.md#flash-next-long-context) |
| Flash-Next follow-up at 32K / 64K | 2.474 / 2.918 s | llama.cpp MTP 0.282 / 0.312 s; Yunshu +2.192 / +2.606 s | [F32, F64](reports/RELEASE_015_BENCHMARKS.md#flash-next-long-context) |

## Long-context results first

Each table row in the evidence includes cold TTFT, decode at depth, warm TTFT, immediate
follow-up TTFT, cached tokens, ranges and valid rep counts.

- [27B: 32K, 64K, 128K](reports/RELEASE_015_BENCHMARKS.md#27b-long-context).
- [Flash-Next: 32K, 64K](reports/RELEASE_015_BENCHMARKS.md#flash-next-long-context); 128K is unmeasured in this snapshot.
- [Secondary 1K/8K ten-prompt corpus](reports/RELEASE_015_BENCHMARKS.md#secondary-corpus).

## Method and limits

Greedy long decode requests produce 2048 tokens; immediate follow-up produces 256.
The secondary corpus uses ten prompts and 1024 output tokens. Long rows report median
[min, max]; corpus rows report mean / median [min, max] and paired ratio mean ± sample
standard deviation. Do not combine those different statistics.

TensorFold 1.0.7 and Splash 1.3.0 compare against Yunshu's TF affine q4g64 arm;
oMLX 125ca681 compares against oQ4e. llama.cpp b11565 (`2f160ba59`) uses GGUF and
has cross-weights comparisons only. The 32K/64K oQ4e advantage above is a product
comparison: the same-weights TF q4g64 Yunshu arm at 64K is 86.400 tok/s, behind
TensorFold's 92.100 and Splash's 92.300. Quantization and greedy output affect speed.

Only complete, validated records qualify. Failed, short or contended runs do not
count as success. Quiet gpuq admission and foreign CPU/swap checks are required for
timing jobs; rejected attempts remain in the retained source ledger. No fresh GPU
measurement is implied by this documentation refresh.

Reproduction tools: `scripts/research/bench_engines.py`, `scripts/dev/yv`, and
`docs/guides/VERIFY.md`. [Accuracy](guides/ACCURACY.md) describes the scoped correctness
gates. Memory, concurrency and general model quality are separate axes; the decode
table does not establish a lead on them.
