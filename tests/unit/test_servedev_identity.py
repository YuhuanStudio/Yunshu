import importlib.util
from pathlib import Path

from yunshu_engine import settings

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "servedev_identity", REPO / "scripts/research/servedev_identity.py"
)
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def row(ids, lps=None, finish="length", ctx=1024, seed=1):
    return {
        "ids": ids,
        "logprobs": lps or [-0.1] * len(ids),
        "finish": finish,
        "ctx": ctx,
        "seed": seed,
    }


def run(*rows):
    return {"complete": True, "rows": list(rows)}


def test_equal_runs_pass():
    assert si.compare_runs(run(row([1, 2])), run(row([1, 2])))["ok"]


def test_token_divergence_reports_position():
    r = si.compare_rows(row([1, 2, 3]), row([1, 9, 3]))
    assert not r["ok"] and r["first_token_divergence"] == 1


def test_logprob_only_difference_fails():
    r = si.compare_rows(row([1, 2], [-0.1, -0.2]), row([1, 2], [-0.1, -0.2000001]))
    assert r["tokens_equal"] and not r["logprobs_equal"] and not r["ok"]


def test_length_and_incomplete_fail_closed():
    assert not si.compare_rows(row([1, 2]), row([1, 2, 3]))["ok"]
    assert not si.compare_rows(row([1]), {"ids": [], "logprobs": [], "finish": None})[
        "ok"
    ]
    assert not si.compare_runs({"complete": False, "rows": []}, run(row([1])))["ok"]
    assert not si.compare_runs(run(row([1], ctx=1)), run(row([1], ctx=2)))["ok"]


def test_configs_use_registered_knobs_and_cover_the_list():
    for env, _mode in si.CONFIGS.values():
        for k in env:
            assert k in settings.REGISTRY, k
    for base, cands in si.KNOBS.values():
        assert base in si.CONFIGS and all(c in si.CONFIGS for c in cands)


def test_prompts_are_deterministic_and_cover_both_contexts():
    a, b = si.build_prompts(), si.build_prompts()
    assert a == b and {p["ctx"] for p in a} == {1024, 32768}


def test_handshake503_judge():
    spec2 = importlib.util.spec_from_file_location(
        "servedev_handshake503", REPO / "scripts/research/servedev_handshake503.py"
    )
    hs = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(hs)
    good503 = {"status": 503, "code": "research_pause_timeout", "elapsed_s": 0.6}
    assert hs.judge(good503, {"status": 200, "tokens": 5})[0]
    assert not hs.judge({"status": 200, "tokens": 5}, {"status": 200, "tokens": 5})[0]
    assert not hs.judge(good503, {"status": 200, "tokens": 0})[0]
    assert not hs.judge({**good503, "elapsed_s": 30}, {"status": 200, "tokens": 5})[0]
