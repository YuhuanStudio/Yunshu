# docs/ layout

Yunshu is a fast, local, single-node LLM / VLM inference engine for Apple Silicon.
Qwen3.8-27B is the first fully tuned model; other modalities are supported capabilities.
Latest release: v0.1.4 (2026-10-08); main contains the unreleased 0.1.5 cycle. Benchmark pages
label later main measurements as unreleased. Start at the top-level [README](../README.md).

## Start here

- [Installation and quickstart](../README.md#quickstart)
- [Model support and validation](guides/MODEL_SUPPORT.md)

## Use and configure

- [guides/CLI.md](guides/CLI.md): first run, models, monitoring, JSON and shell completion

- [guides/FIRST_RUN.md](guides/FIRST_RUN.md): install, choose a cached or small model, readiness, first chat/API request and upgrades
- [guides/CLIENTS.md](guides/CLIENTS.md): connecting OpenAI / Anthropic SDKs, coding agents, Open
  WebUI
- [guides/SERVICE.md](guides/SERVICE.md): running in the background (launchd), uninstalling
- [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
- [guides/DECISIONS.md](guides/DECISIONS.md): typed decisions and supported checkpoints
- [guides/EVALS.md](guides/EVALS.md): local eval definitions, runs and graders
- [guides/WEB_SEARCH.md](guides/WEB_SEARCH.md): search/fetch configuration and privacy
- [guides/TAVILY.md](guides/TAVILY.md): Tavily-compatible retrieval and research
- [guides/TELEMETRY.md](guides/TELEMETRY.md): local host power/thermal/memory sampling
- [guides/API_SURFACE.md](guides/API_SURFACE.md) and [guides/API_EXTENSIONS.md](guides/API_EXTENSIONS.md): per-route coverage and Yunshu extensions
- [CONSOLE.md](CONSOLE.md): build and use the source console
- [API.md](API.md): endpoints
- [CONFIGURATION.md](CONFIGURATION.md): every `YUNSHU_*` setting (generated)
- [guides/ACCURACY.md](guides/ACCURACY.md): distribution, greedy and paired task evidence
- [guides/KV_CACHE_MATRIX.md](guides/KV_CACHE_MATRIX.md): current APC tiers and measured tradeoffs
- [guides/AGENT_COMPAT.md](guides/AGENT_COMPAT.md): coding-agent feature evidence
- [BENCHMARKS.md](BENCHMARKS.md): where each published number comes from and how to reproduce it

## Contribute and maintain

- [Contributing](../CONTRIBUTING.md) and [security/privacy](../SECURITY.md)
- [Roadmap and RFC process](ROADMAP.md)
- [Hardware validation plan](guides/HARDWARE_VALIDATION.md) and [M5 Max notes](guides/M5MAX_HARDWARE.md)
- [Auxiliary scheduling](guides/AUXILIARY_SCHEDULING.md) and [upstream tracking](guides/UPSTREAM_TRACKING.md)

[RELEASING.md](../RELEASING.md) (cutting a release),
[guides/RELEASE_GATE.md](guides/RELEASE_GATE.md) (the end-to-end acceptance run a release must pass) and
[guides/RELEASE_READINESS.md](guides/RELEASE_READINESS.md) (the outward-facing checklist).

```
docs/
  reports/
    PERF_TREND.md       ← absolute-KPI trend log (append-only, honest; records regressions)
    perf_history/       ← time-named KPI snapshots (perf_<UTC>.json)
    img/                ← trend charts
  guides/               # hand-written guides and reference notes
    CLIENTS.md, SERVICE.md, TROUBLESHOOTING.md, RELEASE_READINESS.md, RELEASE_GATE.md, VERIFY.md
    KV_CACHE_MATRIX.md, PROMPT_CACHING_APIS.md, ROUND_DRIVER.md
  results/              # raw bench JSON artifacts (historical; framework_comparison etc.)
  archive/
    mlx_vlm_omni_audio/ ← real omni/audio findings (speech-to-speech capability)
    legacy_vlm_loop/    ← how the deleted pre-runner VLM generation loop worked (+ its KV-prefix notes)
    reports/omlx_scheduler_gaps.md  ← competitive notes on oMLX
```

The old machine-generated reports (REPORT.md, REGRESSION_REPORT.md, COVERAGE_MATRIX.md) and the
wave-narrative VALIDATION_REPORT were removed in the refocus — they were stale or unreliable. `PERF_TREND.md`
is the one honest, append-only performance log; regenerate a snapshot with:

```bash
PYTHONPATH=. uv run python scripts/perf_history.py both
```
