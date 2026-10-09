# Apple Silicon host telemetry

`YUNSHU_TELEMETRY=on` enables a local 1 Hz background sampler. It requires no root,
never imports MLX, and makes native calls only on `yunshu-host-telemetry`, outside
the inference executor. `YUNSHU_TELEMETRY_INTERVAL_S` changes the interval; restart
the server after changing either setting. Default is off: the M5 27B A/B (1K and 32K, 5 reps) showed identical decode speed within noise but a small, consistent follow-up TTFT cost at 1K prose (+1.8%, about 11 ms) and a borderline -0.4% decode at 32K code, so it is opt-in. No samples leave the machine.

The implementation derives IOReport and HID access from
[pierre427/mlx2](https://github.com/pierre427/mlx2), Apache-2.0, revision
`92ada04cc7a59c3263e214b0a1127924fedd52db`; see THIRD_PARTY_NOTICES.md and vendor.json.

## Host API and CLI

`GET /v1/yunshu/host` uses the existing console permission (`can_manage_models`).
It preserves `thermal`, `power`, `memory`, and `memory_pressure` (15-second cache)
and adds `telemetry` with an independent 1 Hz cache:

```json
{
  "telemetry": {
    "state": "ok",
    "sampled_at": 1791400000.0,
    "interval_s": 1.0,
    "watts": {"cpu": 5.0, "gpu": 30.0, "ane": 0.0, "dram": 2.0, "package": 37.0},
    "gpu": {"frequency_mhz": 900.0, "active_ratio": 0.75},
    "temperature": {"state": "ok", "die_max_c": 70.0, "die_mean_c": 65.0, "battery_c": null},
    "reasons": {}
  }
}
```

Power is interval energy divided by elapsed seconds. Package watts is the sum of
the reported CPU/GPU/ANE/DRAM channels, not wall-plug power. GPU MHz is weighted
by active DVFS residency; active ratio includes idle residency in its denominator.
The frequency table is accepted only if it matches contiguous P1..Pn states.
Missing counters are null with reasons, never invented zeroes. Unavailable,
disabled, or stale telemetry returns `state: unknown` with `reason`. Partial
power coverage returns `state: partial`; temperature can independently be unknown.

`yunshu top` watches these cached fields using the configured API key.
`yunshu top --once` prints one table; `yunshu --json top` emits one JSON snapshot.
Unknown values remain visible. The CLI does not sample hardware itself.

## Request energy estimates

`x_yunshu.energy` and `/v1/yunshu/requests/recent` expose schema
`yunshu.energy.v1`. `RunStats.energy` is populated by the gateway off the MLX
thread. Each `prefill` / `decode` object reports `joules`, `joules_per_token`,
`gpu_watts_mean`, `coverage_ratio`, and `extrapolated_s`. Prefill uses uncached
prompt tokens; decode uses generated tokens. The prefill-end timestamp, or first
token when unavailable, separates phases. Queue time is excluded.

The method `host_window_gpu_plus_dram` integrates sampled GPU+DRAM power, clipping
intervals at phase boundaries. The unfinished tail uses the latest power for at
most two sampling intervals. Short phases are estimates at sampler resolution.
Missing coverage, stale readings, and absent phase clocks return null joules with
a reason. CPU and ANE are reported separately and excluded from request energy.
History is bounded to 7,200 intervals; requests older than that can lose coverage.

These are host energy windows including background activity and idle power.
Concurrent requests have overlapping windows; they are not independently metered
and summing receipts can count the same host energy more than once. Receipts
explicitly identify this limitation. Use serial benchmark cells for efficiency
comparisons. Cancellation reports only observed engine timing, not a forecast.

## Prometheus and benchmarks

The existing authenticated `/metrics` scrape adds:

- `yunshu_gpu_watts`, `yunshu_package_watts`
- `yunshu_gpu_frequency_mhz`, `yunshu_gpu_active_ratio`
- `yunshu_die_temperature_celsius` (maximum die temperature)
- `yunshu_request_energy_joules_total{phase="prefill|decode"}` (session sum of estimates)

Unknown gauges are omitted. The request counter is updated once at request
completion, never on scrapes. It has the overlapping-window caveat above.

tfbench request cells retain the complete receipt plus decode `joules_per_token`
and `gpu_watts_mean`. yv speed verdict cells report their per-arm medians under
`efficiency`; they do not decide speed acceptance. Engines without this extension
return null efficiency. Enable telemetry on both arms for efficiency comparisons.
For sampler overhead, compare an explicitly off baseline with an explicitly on
candidate, using pinned distinct commit SHAs and at least three interleaved reps
at 1K and 32K. A positive speed verdict alone does not prove sensors worked;
check the real-run host evidence too.

`yv ab --suite telemetry --base <sha> --cand <sha> --label telemetry-host --priority -1`
runs the real sensor/receipt probe as a candidate-only correctness stage (no quiet
timing admission). `--suite telemetry,smoke,identity,apc,speed --ctx 1024,32768
--reps 3` combines the sensor and overhead gates. Small-model pilots use spec off;
27B probes require the logged MTP mode.

## macOS 27 power-counter compatibility

On macOS 27, Energy Model CPU/ANE counters can freeze without Apple's entitlement.
Yunshu uses macmon's driver-qualified CLPC scalar IOReport catalog for macOS 27
(qualified upstream on 27.0/27.0.1), preferring valid CLPC CPU/GPU/ANE deltas over
legacy values. Unknown driver/OS catalogs are never guessed. CPU/ANE values on
macOS 27+ are unknown with a reason when no valid qualified CLPC delta exists;
other valid Energy Model domains remain available. This avoids reporting a
frozen zero as measured power. All descriptors, samples and registry objects are
released. No root, Apple entitlement, SMC access or MLX calls are needed.

The catalog and descriptor format derive from
[vladkens/macmon](https://github.com/vladkens/macmon), MIT, revision
`7df49f55d9a1b9072e31fc8ba991abda84593563`; the full MIT notice is retained.
CPU channels ending in `CPU Energy` and ANE/DRAM channel families are aggregated;
a malformed unit or negative delta invalidates that domain. Die temperature
prefers HID `tdie` sensors, falling back to `pACC/eACC/GPU MTR Temp Sensor` names
when no tdie values exist, without mixing sensor families.

`--suite preflight,telemetry-tiny,telemetry,smoke,identity,apc,speed` runs a pinned
M5 0.8B pilot before the 27B sensor gate, stopping on the first failure. The probe
collects the closing interval after short requests. Sampling timestamps are taken
before CF parsing, so phase windows remain contiguous; uncovered sub-millisecond
phases never become invented zero-joule readings. For the overhead decision use
`--speed-tol 0`: consistent costs beyond measured noise select default off.

`--suite telemetry-overhead` selects the entire ladder with 1K/32K prose cells,
three interleaved reps, a zero fixed speed tolerance and per-context jobs. Native
pilots use 10-minute timeouts; individual timing cells fit the bounded gpuq short
lane, keeping no more than four cells pending together without raising priority.
