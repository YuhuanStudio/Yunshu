# scripts/ layout

Sub-scripts are grouped by role (Wave 688 reorg). Only the **orchestrators** live at
the top level; everything they invoke lives in a subfolder.

```
scripts/
  regression.py        # ONE entry point — runs every section, writes the unified report
  perf_history.py      # append-only absolute-KPI snapshots + PERF_TREND (shared lib + CLI)
  report.py            # unified regression+perf report generator (Markdown + charts)
  coverage_matrix.py   # living coverage map (auto-derives GATED set from regression.py)

  verify/      # verify_*.py — correctness GATES (must PASS for GO). Real-model + protocol.
  bench/       # bench_*.py, sweep_*.py, fair_bench, profile_engine_loop + framework/loop
               #   helpers (_fw_*.py drive each framework; _bench_*.py drive engine-loop/oMLX)
  realmodel/   # test_*.py — real-model smoke/integration drivers (test_all_modalities, …)
  validate/    # validate_*.py, run_phase0_validation, run_real_validation_gen, soak_test
  tools/       # quality_comparison, extract_mtp_weights, launch_mesh, release_prep,
               #   update_progress, roofline
  _archive/    # dead / one-off experiments kept for history (Lance/DeltaNet velocity,
               #   _diag_*, old benchmark_model/benchmark_paper, superseded bench.py)
```

## Conventions

- **Run from the repo root** with `PYTHONPATH=.`. The harness invokes every section
  with `cwd=<repo root>`, so scripts use repo-relative paths (`./models`, `reference/…`).
- A script that needs the repo root computes it from its own depth, e.g. a file at
  `scripts/bench/x.py` uses `Path(__file__).resolve().parent.parent.parent`.
- The oMLX comparison env is **`.venvs/omlx`** (not /tmp): `OMLX_PYTHON=.venvs/omlx/bin/python`.

## Common entry points

```bash
PYTHONPATH=. uv run python scripts/regression.py --tier smoke|standard|full
PYTHONPATH=. OMLX_PYTHON=.venvs/omlx/bin/python uv run python scripts/regression.py --tier full
PYTHONPATH=. uv run python scripts/perf_history.py both     # snapshot + render trend
PYTHONPATH=. uv run python scripts/report.py                # unified report
PYTHONPATH=. uv run python scripts/coverage_matrix.py       # coverage map
```
