import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tl",
    Path(__file__).resolve().parents[2] / "scripts/research/round3_prefill_timeline.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_summarize():
    s = m.summarize([{"kind": "decode", "dur": 0.5}, {"kind": "decode", "dur": 0.25}])
    assert s == {"decode": {"n": 2, "s": 0.75, "tokens": 0}}


def test_probe_does_not_rebind_the_profiled_slice_body():
    """A later ``body = <prompt text>`` rebound the closure the profiler wrapper
    calls, so every slice raised and the probe hung in iter_tokens."""
    import ast
    from pathlib import Path

    src = Path("scripts/research/round3_prefill_timeline.py").read_text()
    names = [
        t.id
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert names.count("slice_body") == 1 and "body" not in names


def test_onoff_wrapper_routes_a_lone_request_to_the_driver(monkeypatch):
    import runpy
    import sys

    from yunshu_engine import vlm_batch_runner

    monkeypatch.setattr(vlm_batch_runner, "DRIVER_MIN_CONCURRENCY", 2)
    monkeypatch.setattr(runpy, "run_path", lambda *a, **k: None)
    sys.path.insert(0, "scripts/research")
    try:
        import round3_onoff

        round3_onoff.main()
    finally:
        sys.path.remove("scripts/research")
    assert vlm_batch_runner.DRIVER_MIN_CONCURRENCY == 1
