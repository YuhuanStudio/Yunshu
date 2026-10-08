import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from servedev import bandit, replay, shapes  # noqa: E402
from servedev.loop import STAGES, Loop  # noqa: E402


def test_shapes_reject_content_and_skip_incomplete():
    with pytest.raises(ValueError):
        shapes.from_event({"prompt_tokens": 5, "completion_tokens": 1, "messages": "x"})
    assert (
        shapes.from_event(
            {"prompt_tokens": 5, "completion_tokens": 3, "cancelled": True}
        )
        is None
    )
    s = shapes.from_event(
        {"prompt_tokens": 5000, "completion_tokens": 3, "cached_tokens": 1000}
    )
    assert s and s.bucket == 16384
    got = shapes.sample([s, shapes.Shape(10, 2)], 6, seed=1)
    assert {x.bucket for x in got} == {1024, 16384}
    assert shapes.sample([s], 3, 1) == shapes.sample([s], 3, 1)


def test_synthetic_prefix_shared_by_cached_fraction():
    s = shapes.Shape(1000, 4, cached_tokens=500)
    a = shapes.synthetic_messages(s, 1)[0]["content"]
    b = shapes.synthetic_messages(s, 2)[0]["content"]
    assert a != b and a[:200] == b[:200]


def fake_send(ids, ttft=100.0, tps=50.0):
    return lambda body: {
        "ids": list(ids),
        "finish": "stop",
        "ttft_ms": ttft,
        "decode_tps": tps,
        "mem_gb": 20.0,
    }


def test_replay_lossless_token_diff_fails_closed():
    sh = [shapes.Shape(100, 4)] * 3
    base = replay.run_arm(fake_send([1, 2, 3]), sh)
    ok = replay.compare(base, replay.run_arm(fake_send([1, 2, 3], ttft=80), sh))
    assert ok["ok"] and ok["ttft_delta_pct"] < 0
    bad = replay.compare(base, replay.run_arm(fake_send([1, 2, 4]), sh))
    assert not bad["ok"] and "changed tokens" in bad["reason"]
    assert replay.compare(base, base[:2])["ok"] is False


def test_replay_incomplete_run_raises():
    with pytest.raises(replay.ReplayError):
        replay.run_arm(
            lambda b: {"ids": [], "finish": "stop", "ttft_ms": 1, "decode_tps": 1},
            [shapes.Shape(10, 2)],
        )


def test_bandit_bans_nonidentical_and_promotes_with_margin(tmp_path):
    arms = bandit.arms_from({"K": ["a", "b", "c"]})
    base, b, c = arms
    bd = bandit.Bandit(arms, base, min_n=3, margin=0.03)
    assert bd.choose() == base
    for _ in range(3):
        bd.observe(base, 1.0)
        bd.observe(b, 1.10)
    bd.observe(c, 5.0, identical=False)  # fastest but changes tokens: banned
    assert c in bd.banned and bd.best() == b
    assert c not in [bd.choose() for _ in range(20)]
    led = bandit.Ledger(tmp_path / "l.jsonl")
    led.promote(dict(b), dict(base), {"gain": 0.1})
    assert led.active() == dict(b)
    led.revert_last("canary regression")
    assert led.active() == dict(base)
    with pytest.raises(LookupError):
        led.revert_last("again")


def make_loop(tmp_path, results):
    calls = []
    st = {n: (lambda c, n=n: results[n]) for n in STAGES}
    return Loop(
        tmp_path / "log",
        st,
        lambda c: calls.append(("p", c)),
        lambda c: calls.append(("r", c)),
    ), calls


GOOD = {"pass": True, "complete": True}


def test_loop_promotes_only_when_all_pass(tmp_path):
    lp, calls = make_loop(tmp_path, {n: GOOD for n in STAGES})
    assert lp.run("abc").outcome == "promoted" and calls == [("p", "abc")]


@pytest.mark.parametrize("stage", STAGES)
def test_loop_rolls_back_on_fail_or_incomplete(tmp_path, stage):
    for bad in ({"pass": True}, {"pass": False, "complete": True, "reason": "x"}):
        lp, calls = make_loop(
            tmp_path, {n: (bad if n == stage else GOOD) for n in STAGES}
        )
        r = lp.run("abc")
        assert (
            r.outcome == "rolled_back"
            and r.failed_stage == stage
            and calls == [("r", "abc")]
        )
    events = [
        json.loads(x)["event"] for x in (tmp_path / "log").read_text().splitlines()
    ]
    assert events[0] == "start" and "rolled_back" in events


def test_loop_stage_crash_rolls_back(tmp_path):
    def boom(c):
        raise RuntimeError("x")

    st = {n: (boom if n == "canary" else (lambda c: GOOD)) for n in STAGES}
    rolled = []
    r = Loop(tmp_path / "l", st, lambda c: None, rolled.append).run("z")
    assert r.outcome == "rolled_back" and rolled == ["z"]
