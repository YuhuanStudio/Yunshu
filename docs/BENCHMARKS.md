# Benchmarks


> Console docs: the maintained, translated version of this page is `/docs/guides/benchmarks` (source `frontend/docs/guides/benchmarks.mdx`). This file stays as the repository-side reference read by tooling.
Apple M5 Max, 128 GB unified memory. Yunshu development source pinned to
`5ac5452c3d82f43a2208d34730cd35136bd73fa0`. Competitor releases were checked on
2026-10-19; this is a dated snapshot, not a promise to track future releases.
The [0.1.5 snapshot](reports/RELEASE_015_BENCHMARKS.md) remains unchanged as history.

Greedy, single-user, same release015 code/agentic prompt corpus at 32K/64K/128K.
Cold and warm requests require exactly 2048 output tokens, immediate follow-up 256.
Every numerical row below has three qualifying interleaved repetitions in its own
cohort. Values are median [min,max]; cohorts are not pooled. First valid attempt per
repetition is selected, with complete output, cold/warm text identity, engaged mode
and clean quiet-gpuq receipts. Missing, short, parity-failed and contended attempts
remain failures. Only one prompt per depth: this does not establish general quality
or workload performance. Different weights and quantizations affect generated text
and speed; the Yunshu TF4 arm uses the same checkpoint as native TensorFold.

Sticky host swap was present (about 8–10 GiB), so these are not zero-swap results.
Each cell records before/after vm.swapusage and pageouts; admission rejects swap
contention and memory-window growth/pageouts above 64 MiB. Older memory-only
rejections were revalidated against that budget without overwriting receipts;
real gpuq contention remains excluded. No other owned engine was resident at admission.
Peak host delta includes reclamation effects and is not process allocation or model
size. RSS and phys_footprint are reported separately; mmap weights limit either metric.

## Where Yunshu stands

| Axis / cohort | 32K | 64K | 128K |
|---|---|---|---|
| 27B decode tok/s: Yunshu oQ4e / TensorFold 1.1.0 | 135.2 / 200.4 | 118.0 / 150.6 | 75.5 / 116.7 |
| Same-checkpoint decode tok/s: Yunshu TF4 / TensorFold | 142.2 / 200.4 | 84.1 / 150.6 | 94.8 / 116.7 |
| 27B cold TTFT s: Yunshu oQ4e / TensorFold | 34.640 / 34.887 | 79.288 / 84.463 | 200.094 / 241.684 |
| 27B warm TTFT s: Yunshu oQ4e / TensorFold | 0.178 / 0.152 | 0.314 / 0.213 | 0.573 / 241.252 |
| 27B immediate follow-up TTFT s: Yunshu oQ4e / TensorFold | 2.961 / 0.228 | 3.906 / 0.284 | 6.235 / 247.820 |
| Flash-Next decode tok/s: Yunshu / shared-MTP llama.cpp | 66.7 / 60.4 | 56.7 / 55.0 | 24.0 / 42.3 |

TensorFold leads measured 27B decode at all three depths, including the same-checkpoint
comparison. Yunshu has lower cold TTFT; native TensorFold has lower warm/follow-up
TTFT at 32K/64K but reports zero cached tokens at 128K. llama.cpp has lower warm and
follow-up TTFT than Yunshu at every depth. Flash-Next 128K decode is a Yunshu weakness.
Splash/oMLX and other unqualified engines are not counted as wins or losses.

## 27B — TensorFold 1.1.0 and llama.cpp b11721

Yunshu arms use DFlash2; native TensorFold explicitly engages DFlash2. llama.cpp MTP
and DFlash2 configurations remain separate. The GGUF comparison is cross-quantization.


[Full ranges, versions, memory and failure coverage](reports/RELEASE_016_BENCHMARKS_2026-10-19.md). All 78 console screenshots now show the qualified 40K–73K MTP session (21 replies, median 1,611 tokens); real late-session rates around 19–20 tok/s remain visible. See the dated report for capture provenance and limits.

TensorFold 1.1.0's same-round suffix-copy graft contributes to its decode lead:
a clean three-repetition 512-token diagnostic measured copy48/copy0 medians of
234.37/112.12, 141.83/95.02 and 85.68/69.37 tok/s at 32K/64K/128K,
with identical paired raw output. This diagnostic does not replace the 2048-token
standings; see the dated report for turn/cache limits.
