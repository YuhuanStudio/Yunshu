# Yunshu v0.1.4 cross-engine benchmark snapshot

Release: `8c099150f9aeb084658f05e1f5a27747388f8aae` (tag resolved 2026-10-08 08:04, Asia/Taipei).
Snapshot in progress: **unmeasured/rejected cells are unknown**. Startup pilots are excluded from rankings.
Historical observations remain in [PERF_TREND](reports/PERF_TREND.md) under their original builds and limitations.

## Method

M5 Max, 128 GB. One benchmark engine resident at a time, own instance on ports 18990–18999, main-checkout gpuq,
p0, owner/label prefix `snapshot014`. Three independent server-session repetitions alternate engines within each
context/corpus group. Timing cells require quiet admission; contended evidence is rejected and retried.

One frozen corpus, greedy sampling. Complete reference input **including instructions and chat template** is exactly 1024/8192/32768/65536/131072 tokens
under the Jundot 27B tokenizer, asserted before sending. For 128K, content is 131061 tokens plus 11 template tokens. Actual
server `prompt_tokens` are recorded and checked for the same-checkpoint engines. Prose/code 128K corpus SHA256:
`50023b280f09b6aeab89cd5fc84b5dd03cf3a196894b965f0a756357a18c13d9` /
`d106cb1f3bdeacd5f26ba1ac7ec3165e5cb1b8beca57a8fa589721bf057d8044`.

Cold = first corpus request in a fresh instance after two tiny startup requests; loading time is separate. Cold/repeated
replies must reach 2048 tokens; follow-up appends the previous answer and a continuation request (256-token reply).
TTFT ends at the first meaningful streamed content/reasoning/tool delta; decode excludes TTFT. Full-hit requires
reported `cached_tokens >= prompt_tokens - 1`; absent proof means unknown, with repeated-request TTFT reported separately.

Long QA is yv's ten deterministic station/passcode needles at 32K/64K/128K, with every question fitted to the exact content
budget. Concurrency: 2/4 simultaneous 32K prompts, 2048-token replies, deterministic cohort headers preventing cross-trial
cache reuse. Effective throughput includes prefill; per-request TTFT/decode are separate. Agentbench: same 20 opencode
tasks and CLI, generation forced to temperature 0/top_p 1; API/tool-contract failures are listed explicitly.

Memory: both process-tree physical-footprint and RSS accounting sums, sampled every two seconds from startup; idle is 30 seconds after
the group. Short spikes can be missed and shared pages are not deduplicated. File-backed/GPU allocations can be accounted differently by native engines. Each group owns a fresh server. Gaps are
excess latency/memory or throughput deficit versus the best; ranking requires three clean session repetitions.
Hypotheses below are possible causes, not profiler-established findings.

## Engine configurations

| engine | version / isolated environment | requested configuration | target weights |
|---|---|---|---|
| yunshu-new | v0.1.4 frozen lock; snapshot014-yunshu | release defaults; normal-layout DFlash2 discovery; no tuning override | Jundot/Qwen3.8-27B-oQ4e-mtp |
| tf-new | TensorFold 0.6.1; tensorfold-0.6.1 | explicit DFlash2, own snapshots, update check off | same oQ4e |
| mlxlm | mlx-lm 0.32.0 / MLX 0.32.3; snapshot014-mlxlm | stock server; prompt/decode concurrency 8; thinking off | same oQ4e |
| omlx | v0.7.0; omlx-bench/venv | DFlash2; own base/model/SSD paths; concurrency 8; 96 GiB guard; hot 8GB / SSD 20GB | same oQ4e |
| splash | 1.1.0 packaged native engine | own port; DFlash2; **BF16 KV**, disk cache off, reasoning none | different: prepared Splash target |
| llamacpp | 836d57176dc699a726c55418e4f96b8ca628e1bf; llamacpp/build | Metal all layers, Flash Attention, MTP, unquantized KV, thinking off; context capacity per slot | different: UD-Q4_K_M GGUF + Q4_0 MTP head |
| mtplx | 2.12.0; mtplx/.venv | turbo/native-MTP (documented fastest verify profile, context fallback); KV quant off; own cache | same oQ4e |

Python environments live under `/Volumes/P5Plus/yunshu-test-envs/`. The user's oMLX app and pre-existing Splash services
are never managed. Splash/GGUF are product comparisons within the 27B family: different target quantizations confound
quality, size and speed. Requested lossless paths are checked for engagement; cross-engine bit identity is not claimed.

## Reproduce

Use Python 3.13+ from the installed main venv for CPU scripts. Generate the corpus with
`scripts/research/gen_snapshot_prompts.py`, then run `scripts/dev/yv ab --base 8c099150f9aeb084658f05e1f5a27747388f8aae
--cand HARNESS_COMMIT_SHA --suite snapshot --label snapshot014-released --priority 0 --detach`, with
`GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq`, `GPUQ_OWNER=snapshot014`, and `YV_GPUQ` set to the main checkout's gpuq.
Both refs must be full, distinct intended commit SHAs; check the first line of `yv.log`. Resume with the same command.
Use `yv status/wait RUN_DIRECTORY`; aggregate promoted evidence with
`bench_snapshot.py aggregate --yv-run RUN_DIRECTORY`. Only rc 0, terminal complete, valid schema/token budgets,
expected engaged mode and clean contention evidence enter tables. Raw evidence/job IDs remain attached to the verdict.

# Metric tables

Cells are median (min-max) n=<reps>; TTFT in s, decode in tok/s, memory in GiB.

## Cold TTFT

| ctx / kind | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 1K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 1K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Warm full-hit TTFT (cached_tokens >= prompt_tokens - 1)

| ctx / kind | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 1K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 1K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Warm repeated-request TTFT (cache coverage varies)

| ctx / kind | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 1K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 1K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Follow-up turn TTFT

| ctx / kind | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 1K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 1K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Decode tok/s (2048-token reply, cold request)

| ctx / kind | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 1K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 1K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 32K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K prose | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K code | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Long-context recall (needle, correct / asked)

| ctx | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 32K | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 64K | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 128K | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

## Concurrency effective throughput (prefill included), tok/s (32K prompts, 2048-token replies)

| n | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 2 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| 4 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |

## Concurrency mean per-request TTFT, seconds (32K prompts, 2048-token replies)

| n | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 2 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| 4 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |

## Concurrency mean per-request decode, tok/s (32K prompts, 2048-token replies)

| n | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| 2 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| 4 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |

## Sampled peak memory (process-tree physical footprint, GiB)

| group | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| d1k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d8k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d32k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d64k-prose | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d64k-code | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d128k-prose | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d128k-code | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| c2c4 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n32k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n64k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n128k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |

## Idle memory after 30 seconds (process-tree physical footprint, GiB)

| group | yunshu-new | tf-new | splash | omlx | mtplx | mlxlm | llamacpp |
|---|---|---|---|---|---|---|---|
| d1k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d8k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d32k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d64k-prose | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d64k-code | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d128k-prose | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| d128k-code | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| c2c4 | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n32k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n64k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |
| n128k | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown | unknown; gap unknown |

## Gap hypotheses (unproven; applies to each positive gap in the corresponding engine/metric)

| engine | metric | hypothesis |
|---|---|---|
| yunshu-new | ttft | Hybrid-state checkpoint restore, prefix lookup and first-token dispatch may dominate warm TTFT; cold TTFT includes prefill projections and GDN recurrence. |
| yunshu-new | decode | DFlash acceptance, verify block cost and prompt-copy opportunities vary between prose/code and context lengths. |
| yunshu-new | memory | Retained APC snapshots, allocator pools and speculative scratch may explain peak/idle differences. |
| tf-new | ttft | Snapshot selection/restore and prefill kernel scheduling may account for differences. |
| tf-new | decode | DFlash verify kernels, proposal acceptance and scheduler overhead may account for differences. |
| tf-new | memory | Snapshot retention and allocator release policy may account for differences. |
| splash | ttft | Native prefill and GDN-state/cache scheduling may account for differences; BF16 KV is deliberately used here. |
| splash | decode | Native Metal verification and DFlash proposal batching may account for differences; target weights differ. |
| splash | memory | Native buffer arenas and prefix retention may account for differences; different target quantization confounds comparisons. |
| omlx | ttft | Paged prefix-cache lookup, hybrid state restore and SSD/hot-cache transitions may account for differences. |
| omlx | decode | DFlash acceptance and continuous-batch scheduling may account for differences. |
| omlx | memory | Hot/SSD cache policy, page pools and concurrent-request guard may account for differences. |
| mtplx | ttft | Native-MTP hybrid prefill and SessionBank snapshot restore may account for differences. |
| mtplx | decode | Native MTP acceptance, compiled verification routing and per-step state work may account for differences. |
| mtplx | memory | Repaged KV and SessionBank retention may account for peak/idle differences. |
| mlxlm | ttft | Generic prefill kernels and hybrid-cache prefix reuse limits may account for differences. |
| mlxlm | decode | Autoregressive decoding evaluates the target each token; speculative engines amortize target verification over accepted proposals. |
| mlxlm | memory | No speculative drafter reduces resident weights; prompt cache and allocator retention still contribute. |
| llamacpp | ttft | GGUF kernels, graph scheduling and slot prefix reuse may account for differences; weights differ from oQ4e. |
| llamacpp | decode | Native MTP proposal acceptance, GGUF kernel layout and Metal graph dispatch may account for differences. |
| llamacpp | memory | Preallocated per-slot KV and graph buffers may account for differences; GGUF weight sizes differ. |

Weights differ for splash (own Splash quantization) and llamacpp (UD-Q4_K_M GGUF), see SETUP.md.
