# docs/ layout

```
docs/
  reports/
    PERF_TREND.md       ← absolute-KPI trend log (append-only, honest; records regressions)
    perf_history/       ← time-named KPI snapshots (perf_<UTC>.json)
    img/                ← trend charts
  guides/               # hand-written reference notes
    KV_CACHE_MATRIX.md, VLM_TEXT_KV_PREFIX.md, PROMPT_CACHING_APIS.md
  results/              # raw bench JSON artifacts (historical; framework_comparison etc.)
  archive/
    mlx_vlm_omni_audio/ ← real omni/audio findings (relevant to the omni refocus)
    reports/omlx_scheduler_gaps.md  ← competitive notes on oMLX
```

The old machine-generated reports (REPORT.md, REGRESSION_REPORT.md, COVERAGE_MATRIX.md) and the
wave-narrative VALIDATION_REPORT were removed in the refocus — they were stale or unreliable. `PERF_TREND.md`
is the one honest, append-only performance log; regenerate a snapshot with:

```bash
PYTHONPATH=. uv run python scripts/perf_history.py both
```
