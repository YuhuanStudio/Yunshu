# Benchmarks: where the numbers come from

The headline tables are in the [README](../README.md#performance). Every number there comes from
one of the dated runs below; the raw JSONL stays with the maintainers (it is not part of the repo),
and the harnesses under `scripts/research/` reproduce each run on your own machine.
[PERF_TREND.md](reports/PERF_TREND.md) is the append-only log of the long-running KPI snapshots.
This page repeats no numbers, so it cannot drift from them.

All current runs are on one M5 Max (128 GB) with Qwen3.8-27B (`Jundot/Qwen3.8-27B-oQ4e-mtp`)
unless the directory says otherwise.

| Run | What it measured |
|---|---|
| 2026-09-29-ragged-idle | ragged KV on/off on an idle GPU: MTP lane decode by output type at 1K / 32K / 131K with parity, speed sweep, repeated short-context runs, concurrency |
| 2026-09-29-ragged-lane | ragged KV (slot buffers + lane kernel): MMLU-Pro soak, lane parity fix |
| 2026-09-29-ragged-default | validation after ragged KV became the default (27B matrix, QA, concurrency; Gemma-4 keeps stock caches) |
| 2026-09-29-tool-format | capability matrix after the tool-call format dispatch (Gemma-4, Qwen3.8) |
| 2026-09-29-fused-prefill, 2026-09-28-fairness | decode of other rows while a long prompt prefills (chunked / fused prefill; both removed) |
| 2026-09-29-yunshu-dflash | DFlash2 speed sweep before the DFlash fixes |
| 2026-09-28-tensorfold | TensorFold 0.3.6.1 (MTP and DFlash2): matrix, speed sweep, MMLU-Pro soak |
| 2026-09-28-matrix | cross-engine capability matrix (34 checks), TTFT / cache reuse, decode by output type, MMLU-Pro soaks, the 1K–200K speed sweep, batch speculative decode |
| 2026-09-28-ragged | ragged per-row KV cache (bf16 / int8): matrix, concurrency, mixed QA, MMLU-Pro soak |
| 2026-09-28-kvquant | upstream quantized KV vs bf16 (speed, concurrency, mixed QA) |
| 2026-09-28-parity | speculative on == off token parity at 1.2K and 16.5K context |
| 2026-09-28-families | other model families on the runner (Gemma-4, GLM-OCR, audio chat) |
| 2026-09-28-settings | re-validation after the settings registry change |
| 2026-09-28-ane | decode under concurrent prefill; oMLX's ANE option |
| 2026-09-28-engines | how the other engines were set up for comparison |
| 2026-09-28-omni | Qwen3-Omni multi-turn check on upstream mlx-vlm |
| 2026-09-28-qwen38, 2026-09-27-qwen38 | earlier Qwen3.8 prefix-cache (APC) and runtime studies |

## Reproducing

The harnesses are in `scripts/research/`. Each one prints its usage with `--help` and appends JSONL:

| Script | Measures |
|---|---|
| `bench_engine_matrix.py` | capability checks + TTFT / cache reuse against any OpenAI-compatible server |
| `bench_context_batch.py` | the 1K–200K prompt sweep and b1–b8 concurrency (unique prompts, no cache hits) |
| `soak_mmlu_pro.py` | MMLU-Pro accuracy + long-run stability (8 in flight, 16384 max tokens, `reasoning_effort=medium`) |
| `probe_concurrency.py` | aggregate decode under N concurrent streams |
| `bench_mixed_load.py` | background decode rate while a long prompt prefills |

Comparisons against other engines use the same checkpoint, the same client script and a separate
port for each server. A result only counts if it was measured. Estimates are not recorded.
