# Benchmarks: methods, results and limits

Measurements below are snapshots, not scores for the latest release. Latest release:
**v0.1.2 (2026-10-02)**; v0.1.1 was released on 2026-09-29. Changes after v0.1.2 are
**unreleased main**. Unless stated otherwise: M5 Max, 128 GB, Qwen3.8-27B
`Jundot/Qwen3.8-27B-oQ4e-mtp`. Raw JSONL and captured agent prompts stay private;
the harnesses are public. The [append-only measurement log](reports/PERF_TREND.md)
keeps historical observations, including losses. Do not pool different builds or corpora.

## Single-request comparison (tfbench, 2026-10-02)

Source: [PERF_TREND](reports/PERF_TREND.md), recorded by `4e1338c4`.
Same checkpoint, greedy, 256 generated tokens, GPU-queue serialized; each cell is the
median of three independent server sessions. Code and prose prompts at each context
are separate cells. `scripts/research/tfbench.py` records wire TTFT, decode rate,
response digests and `x_yunshu` speculative counters; cold, repeated and edited-tail
requests are distinct. TTFT is arrival to first meaningful streamed output; decode
excludes prefill. Aggregate concurrent rates are not single-request rates.

**The Yunshu build was that morning's main; its SHA was not captured. It actually used
MTP**, confirmed by `Speculative decoding: mtp` in the log: the isolated HOME hid the
DFlash2 drafter from automatic discovery. TensorFold explicitly used DFlash2. This is
a measured deployment comparison with a draft-mode mismatch, not a same-drafter A/B,
and it predates the prefill, wide-verify and prompt-copy merges below.

| Context / output | TensorFold 0.6.1 decode tok/s / cold TTFT s | TensorFold 0.3.6.1 decode tok/s / cold TTFT s | Yunshu main (MTP) decode tok/s / cold TTFT s |
|---|---|---|---|
| 1K code | 140.5 / 1.2 | 134.7 / 1.2 | 69.3 / 1.4 |
| 1K prose | 72.4 / 1.2 | 75.6 / 1.2 | 52.9 / 1.4 |
| 8K code | 80.6 / 8.4 | 80.7 / 8.9 | 60.2 / 11.1 |
| 8K prose | 67.4 / 8.5 | 68.9 / 9.0 | 57.5 / 11.1 |
| 32K code | 91.2 / 39.4 | 86.4 / 42.5 | 58.7 / 47.2 |
| 32K prose | 55.1 / 39.4 | 58.7 / 41.3 | 46.3 / 47.7 |

TensorFold leads in every cell above. At concurrency 2 / 4 / 8, aggregate decode was
79.8 / 129.6 / 164.2 tok/s (TF 0.6.1), 80.8 / 131.1 / 168.6 (TF 0.3.6.1), and
39.8 / 73.2 / 112.6 (Yunshu MTP). Warm agent replay: TF 0.6.1 105–159 tok/s,
Yunshu 38–90, medians for four recorded opencode request bodies; cold TTFT was similar.
These replay rates are not whole-agent task success rates.

## Later main measurements (unreleased, 2026-10-02)

| Change / workload | Before | After | Provenance |
|---|---|---|---|
| Cold TTFT, 8K | 11.0 s | 8.57 s | prefill merge `a4d71bc8` |
| Cold TTFT, 32K | 47.5 s | 38.3 s | prefill merge `a4d71bc8` |
| MTP prompt-copy, 32K code turn 2 | 63 tok/s | 100–138 tok/s | PERF_TREND, `c666be70`, merged `bb4895ca` |
| MTP prompt-copy, 8K code turn 2 | 62 tok/s | 86 tok/s | same |
| MTP prompt-copy, 1K code cold | 72 tok/s | 83 tok/s | same |

The prefill figures come from the dated merge measurement record (`a4d71bc8`), not
from a fresh run for this documentation update; the public record does not give repeat
counts or intervals. Harnesses: `scripts/research/prefill_profile.py` (forward time by
operation class), `tfbench.py` (HTTP cold TTFT), `apc_restore_identity.py` (cold vs partial
and full restore, token and logprob identity). Stock matmul for large prefill chunks and
the chunked GDN core change cold-prefill numerics relative to the old path. Prefill
settings are part of APC keys and disk namespaces; this is not bit identity across builds.

Prompt-copy: greedy output digests matched off/on for every recorded cell. Prose was
unchanged within ±3% single-run noise; editing replay 0004 went 92 → 106 tok/s, the
other three replays were unchanged. These are workload-specific observations, not a
universal speedup. Use `scripts/research/agentic/replay_traffic.py` for recorded traffic.
Wide invariant verify is implemented up to 32 rows (`a4566d5a`), but there is no updated
full cross-engine battery after these merges. Wider tree drafting remains off by default.

## Splash: historical exploratory comparison

Source: PERF_TREND, **2026-09-27**, M5 Max / 128 GiB, 3,323-token prompt, single runs,
non-exclusive GPU. These are not medians or release scores.

| Engine / condition | Cold / repeated TTFT s |
|---|---|
| Yunshu direct VLMEngine (build SHA not recorded) | 3.511 / 3.512 |
| mlx-vlm 0.7.3, APC off | 3.464 / 3.459 |
| Splash 1.1.0, its own quantized model, INT8 KV | 3.221 / 0.130 |

Splash was ahead here. Different weights and cache configuration prevent attributing
this difference to the engine alone. No new Splash comparison followed the main merges.
The old README's MMLU-Pro and decode leaderboard lacked numeric provenance in this
page or PERF_TREND and has been removed rather than promoted to current results.

## Agentic coding (partial, 2026-10-02)

Source: PERF_TREND, recorded by `2815311c`; method:
[AGENTIC_BENCH](guides/AGENTIC_BENCH.md). Real pinned agent CLIs, isolated homes,
local server, hidden-test grading, 20-task target with repeats. Failed runs remain in
the denominator. The available snapshots differ; do not interpret their medians as
paired speedups. `prod2` and `prod3` have no results.

| Snapshot | Agent | Pass / runs | Rate (Wilson 95%) | Wall median / p90 s | Cache hit | Decode median tok/s | TTFT median s | Timeouts |
|---|---|---|---|---|---|---|---|---|
| prod, `c4e2b244+`, 2026-09-30 | opencode | 34 / 41 | 83% (69–91%) | 337 / 1200 | 83.6% | 22.2 | 0.9 | 5 |
| prod4, `d225f16c`, measured 2026-10-02 | Claude Code | 14 / 15 | 93% (70–99%) | 93 / 309 | 89.2% | 60.8 | 1.9 | 0 |
| prod4, `d225f16c`, measured 2026-10-02 | opencode | 9 / 10 | 90% (60–98%) | 112 / 137 | 84.2% | 56.9 | 0.8 | 0 |

All recorded rows had 0 API errors, malformed calls and leaked call markup. Prod4 covers
10 of 20 tasks with 1–3 repeats each; neither agent cell is complete. Codex and TensorFold
agentic cells have no published result. Snapshot `d225f16c` is included in v0.1.2; the
measurements are not evidence for the later unreleased changes.

## Accuracy: three distinct tiers

Source and complete conditions: [ACCURACY](guides/ACCURACY.md), 2026-10-02 results
recorded by `2815311c`. Tier 1 is teacher-forced KLD / top-1 / perplexity, Tier 2 is
fixed-prompt greedy divergence, Tier 3 is paired task evaluation. Invariant verify vs
invariant one-token decode can be bit-exact without matching stock MLX's reductions.
The recorded Tier 2 plain-runner vs speculative-path gate failed (26/50 matches);
inside the invariant path and round driver the spec-on/off gate holds. Do not claim
unqualified stock-output identity or infer task equivalence from a short soak.

| Paired evaluation | Status / n | Stock mlx-vlm | Yunshu defaults | Delta points (95% CI) | McNemar p |
|---|---|---|---|---|---|
| GSM8K | final, 1319 | 97.12% | 97.65% | +0.53 [+0.04, +1.02] | 0.065 |
| MMLU-Pro | partial, 917 / 2000 | 83.97% | 83.53% | -0.44 [-1.44, +0.57] | 0.523 |
| IFEval | partial, 179 / 541 | 90.50% | 88.27% | -2.23 [-6.01, +1.54] | 0.388 |
| BFCL / needle | pending | — | — | — | — |

Same checkpoint, greedy, thinking on / medium except BFCL and needle. Pair only items
completed in both arms; reference transport failures bias the partial subsets. Reference
error attempts: GSM8K 1319, MMLU-Pro 168, IFEval 6; Yunshu: 0, 16, 1 respectively.
The guide gives attempts, unresolved items and truncation counts. None of these rows
establishes an accuracy improvement; partial rows are not benchmark-wide scores.
Scripts: `scripts/research/accuracy/kld.py`, `greedy_div.py`, `paired_eval.py`.

## APC storage tiers (unreleased main, 2026-10-02)

Source: [KV_CACHE_MATRIX](guides/KV_CACHE_MATRIX.md), measurement record `d00f61d7`,
identity record `63b00e16`, merged `57272ed7`. Three growing sessions, 27 greedy requests,
20 s idle gaps, TB4 SSD unless stated; prompt-token cache fraction differs from request
hit rate. These are replay summaries, not repeated-run confidence intervals.

| Configuration | Prompt tokens from cache | TTFT mean / p50 / p90 / max s | Peak RSS GiB |
|---|---|---|---|
| 4 GiB RAM only | 31.7% | 21.2 / 17.7 / 39.8 / 44.7 | 4.7 |
| 4 GiB RAM + SSD, disk supersede | 86.3% | 6.3 / 6.0 / 9.1 / 13.9 | 5.7 |
| 4 GiB + lossless WARM + SSD | 86.3% | 6.2 / 6.2 / 8.9 / 13.9 | 9.0 |
| 4 GiB + int8 WARM + SSD (lossy) | 86.3% | 7.2 / 6.0 / 14.7 / 19.0 | 6.3 |
| 4 GiB + internal SSD > TB4 > simulated HDD | 85.7% | 5.6 / 5.7 / 8.9 / 14.0 | 5.4 |
| 4 GiB + TB4 > simulated HDD | 81.3% | 10.4 / 7.7 / 19.2 / 28.3 | 5.8 |

Lossless WARM did not improve this workload and stays off; int8 / int4 are lossy options.
Tier restores with the same cached length matched all-RAM text: 24 HOT, 131 SSD,
21 lossless-WARM hits and 7 cold requests, none different. Disk supersede left 6 files /
10.8 GiB instead of 46 / 63 GiB with the same hits. HDD / NAS results are throttled
simulations, not measurements on real HDD / NAS hardware.
Scripts: `scripts/research/apc_audit/{tier_roofline,tier_breakeven,tier_capacity_model,
session_replay,tier_report}.py` and `queue_tier4.sh`. The guide distinguishes codec
rooflines from through-server restore time; do not quote the former as HTTP TTFT.

## Reproducing and recording a result

Run GPU work through `scripts/dev/gpuq` with a unique label and `GPUQ_OWNER`; never
compete with the serving process. See [SERVE_AND_DEVELOP](guides/SERVE_AND_DEVELOP.md).
Unit tests and lint run directly. The harnesses use local checkpoint paths / private
corpora: inspect them and prepare those inputs before running, rather than treating
them as portable one-command benchmarks.

```bash
uv run --no-sync python scripts/research/tfbench.py --help
uv run --no-sync python scripts/research/agentic/run_agentic.py --help
uv run --no-sync python scripts/research/accuracy/paired_eval.py --help
```

For tfbench, actual dispatch is `--part decode` or `--part ca` (concurrency + agent
replay); the module header's standalone `conc` / `agent` examples are stale. For Yunshu
DFlash, pass `--env YUNSHU_VLM_DRAFT=/absolute/path/to/drafter` and verify the engaged
mode in the log. Pin both engine SHAs, dependencies, checkpoint fingerprint, prompt
corpus, seed, sampling, context, output budget and cache state. Use at least three
interleaved runs for a new speed claim. Require rc 0, expected outputs, and the
harness's terminal record (`part_done` for tfbench); interrupted parts are not results.
Compare complete digests for lossless A/Bs and record failures, timeout and truncation
counts alongside accuracy. No benchmark or server was started for this docs update.


## 0.1.3 draft measurements (2026-10-03)

M5 Max, Jundot/Qwen3.8-27B-oQ4e-mtp; each comparison uses the same checkpoint.
These experiments use distinct source snapshots and workloads. Do not multiply
improvements or compare their rates across rows. The append-only
[PERF_TREND](reports/PERF_TREND.md) retains full runs, engaged modes, output checks
and rejected experiments.

| Change / metric | Before → after | Source in PERF_TREND / receipt |
|---|---|---|
| Complete JSON warm decode, AR → DFlash | 23.4 → 111.4 tok/s | Oct 3 constrained speculation; `1003-125849-00-cspec-complete-json-tool-quiet-r3-1254` |
| Complete tool-call warm decode, AR → DFlash | 23.0 → 77.7 tok/s | Same receipt; complete schema-valid output, tokens equal |
| Follow-up TTFT, 8K code / 32K code, MTP | 569 → 512 ms / 900 → 721 ms | Oct 3 native singleton KV capacity; `1003-145700-00-prefill4-http-combo8k-1456` / `http-combo32k-1456` |
| DFlash cold code decode, 1K / 8K / 32K | 97.74 → 109.48 / 74.50 → 82.19 / 72.74 → 82.56 tok/s | Oct 3 DFlash2 greedy prompt-copy islands; `1003-161523-00-wide4-timing-bundle-1615` |
| MTP copy cap 8 → 16, 8K code turn 2 | 106.1 → 129.2 tok/s | Oct 3 prompt-copy maximum; `1003-110949-00-wide3-copy-cap-bindfix-1111` |

All above are medians of three interleaved clean repetitions, with matching token
digests. JSON/tool runs also verify complete schema-valid outputs and warm cache
hits. Prompt copying has workload-dependent tradeoffs: cap 16 lowered measured
prose rates by 0.2–1.0%; DFlash copy-island prose changes ranged −0.05% to +4.23%.
The follow-up experiment leaves 32K prose/code TTFT 56/51 ms behind TensorFold;
8K gaps are 1/7 ms. No claim of universal superiority follows from these rows.

Prompt-cache single-flight is also landed, but its performance summary in merge
`85f6ae01` is not yet recorded in PERF_TREND. Its numerical claim is withheld here
until the underlying measurement is added to the canonical log.
