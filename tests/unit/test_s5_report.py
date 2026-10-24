import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts/research/agentic/s5_report.py"
_s = importlib.util.spec_from_file_location("s5r", _p)
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)


def run(agent, passed, last):
    return {
        "type": "run",
        "agent": agent,
        "task": "t",
        "repeat": 1,
        "passed": passed,
        "wall_s": 10,
        "prompt_tokens": 100,
        "cached_tokens": 50,
        "artifacts": "a",
        "request_log": [
            {"ttft_s": 1.0, "finish_reason": last[0], "tool_calls": last[1]}
        ],
    }


def test_wilson_and_taxonomy():
    lo, hi = m.wilson(19, 20)
    assert 0.74 < lo < 0.78 and 0.98 < hi <= 1.0
    rep = m.report(
        {
            "codex": [
                run("codex", True, ("tool_calls", 1)),
                run("codex", False, ("stop", 0)),
            ]
        }
    )
    assert "stalled-turn" in rep and "50%" in rep
