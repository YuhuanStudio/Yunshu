import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "scripts/research"


def load():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "tsl", ROOT / "tavily_stage_latency.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_and_summarize():
    m = load()
    t = m.parse_timing("serp;dur=12.500, fetch;dur=3.000")
    assert t == {"serp": 12.5, "fetch": 3.0}
    s = m.summarize([t, {"serp": 20.0}])
    assert s["serp"]["n"] == 2 and s["fetch"]["n"] == 1


async def test_fixture_run_reports_every_depth():
    import argparse

    m = load()
    rows = await m.run(
        argparse.Namespace(live=False, reps=2, queries=None, depths=list(m.DEPTHS))
    )
    assert [r["depth"] for r in rows] == list(m.DEPTHS)
    assert all("serp" in r["stages"] for r in rows)
