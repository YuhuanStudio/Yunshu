"""Aggregate benchmark runner — drives every bench/<name>/run.py harness.

Backs the `just bench-all` recipe (the recipe referenced this script but it did
not exist, so `just bench-all` hard-failed). Discovers each immediate
``bench/<name>/run.py``, runs its module-level ``run_*`` entrypoint, collects
the returned report, and writes a combined ``bench/report_all.json``.

Run: PYTHONPATH=. uv run python bench/run_all.py
"""

import importlib.util
import inspect
import json
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent


def _discover() -> list[Path]:
    """Return each bench/<name>/run.py (excluding this aggregator)."""
    return sorted(
        p for p in _BENCH_ROOT.glob("*/run.py") if p.parent.name != "__pycache__"
    )


def _load_runner(run_py: Path):
    """Import a run.py and return its primary ``run_*`` callable (or None)."""
    spec = importlib.util.spec_from_file_location(
        f"_bench_{run_py.parent.name}", run_py
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Prefer a `run_*` function; fall back to `main`.
    candidates = [
        fn
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("run_") and fn.__module__ == module.__name__
    ]
    if candidates:
        return candidates[0]
    return getattr(module, "main", None)


def run_all() -> dict:
    runners = _discover()
    print(f"=== Yunshu bench-all: {len(runners)} suite(s) ===\n")

    reports: dict[str, object] = {}
    for run_py in runners:
        name = run_py.parent.name
        print(f"── {name} " + "─" * max(0, 40 - len(name)))
        fn = _load_runner(run_py)
        if fn is None:
            print(f"  SKIP: no run_*/main entrypoint in {run_py}")
            reports[name] = {"status": "skipped", "reason": "no entrypoint"}
            continue
        t0 = time.perf_counter()
        try:
            # Call with defaults only — every runner exposes keyword args with
            # sensible defaults, so a no-arg call is the portable contract.
            result = fn()
            reports[name] = {
                "status": "ok",
                "elapsed_s": time.perf_counter() - t0,
                "report": result,
            }
        except Exception as e:  # one suite failing must not abort the rest
            print(f"  FAIL: {type(e).__name__}: {e}")
            reports[name] = {
                "status": "error",
                "elapsed_s": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}",
            }
        print()

    out = _BENCH_ROOT / "report_all.json"
    combined = {"suites": list(reports.keys()), "reports": reports}
    with open(out, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    ok = sum(1 for r in reports.values() if isinstance(r, dict) and r.get("status") == "ok")
    print(f"=== bench-all done: {ok}/{len(reports)} ok → {out} ===")
    return combined


if __name__ == "__main__":
    run_all()
