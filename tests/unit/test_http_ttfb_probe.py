import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "http_ttfb_ab", ROOT / "scripts/research/http_ttfb_ab.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_summarize_and_compare_group_by_version(tmp_path, capsys):
    for name, ver, base in (("a", "1.0", 1.0), ("b", "2.0", 2.0)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "fastapi": ver,
                    "starlette": "9",
                    "times_ms": [base + i / 100 for i in range(10)],
                }
            )
        )
    rows = m.compare([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert set(rows) == {"fastapi 1.0/starlette 9", "fastapi 2.0/starlette 9"}
    assert rows["fastapi 2.0/starlette 9"]["median_ms"] > 2.0


def test_probe_measures_real_sse_bytes_in_process():
    import asyncio

    out = asyncio.run(m.measure(2, warmup=1))
    assert len(out["times_ms"]) == 2 and all(t > 0 for t in out["times_ms"])
