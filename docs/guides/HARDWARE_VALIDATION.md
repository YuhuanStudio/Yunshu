# Hardware validation plan

Apple Silicon support is a product target, not evidence that every chip/macOS
combination has been tested. Hosted `macos-14` jobs in CI do not prove a particular
M-series GPU generation or real Metal availability. Current workflows are in
`.github/workflows/ci.yml` and `release.yml`; no new runner is provisioned here.

## Planned tiers

| Tier | Trigger | Checks | Evidence to retain |
|---|---|---|---|
| Fast | Each change | Linux lint/build; model-free public-contract and CLI tests where dependencies permit | Source SHA, dependency lock and test counts |
| Correctness | Trusted PR or pre-release | Apple Silicon unit suite; tiny-model API/cache/constraint smoke | Chip, memory, macOS/Python/MLX versions, checkpoint revision, passed/failed/skipped |
| Hardware coverage | Scheduled or pre-release | M1, M2, M3, M4, M5; oldest supported and current macOS, with available hosts | Explicit tested/untested cells; output/cache/logprob parity |
| Performance | Controlled scheduled run | Same-checkpoint decode and cold/warm/turn-2 TTFT; three interleaved repetitions | Raw receipts, engaged mode, output lengths, digests, medians and regressions |

A maintainer must inventory available hosts and supported macOS minimums before
turning this plan into required CI. Do not infer GPU coverage from runner names,
and do not run untrusted PR code on a personal model host. Hardware unavailable
for a release must be listed as untested, rather than marked passed.

Benchmark thresholds need a stable noise baseline before becoming gates. Keep
correctness failures separate from performance noise and infrastructure failures;
retain failed cases instead of silently retrying them out of the report. See
[benchmark methodology](../BENCHMARKS.md) and [accuracy evidence](ACCURACY.md).
