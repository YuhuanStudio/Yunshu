"""Statistics and scorers of the Tier 3 paired evaluation harness."""

import importlib.util
import math
import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts/research/accuracy/paired_eval.py"
_spec = importlib.util.spec_from_file_location("paired_eval", _p)
pe = importlib.util.module_from_spec(_spec)
sys.modules["paired_eval"] = pe
_spec.loader.exec_module(pe)


def test_mcnemar_known_values():
    assert pe.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(pe.mcnemar_exact(5, 5), 1.0)
    assert math.isclose(pe.mcnemar_exact(0, 8), 2 * 0.5**8)
    assert math.isclose(pe.loss_p(8, 0), 0.5**8)
    assert pe.loss_p(0, 8) == 1.0


def test_paired_stats():
    ref = [True] * 90 + [False] * 10
    cand = [True] * 85 + [False] * 15  # 5 lost
    s = pe.paired_stats(ref, cand)
    assert s["b_ref_only"] == 5 and s["c_cand_only"] == 0
    assert math.isclose(s["delta"], -0.05)
    lo, hi = s["ci95"]
    assert lo < -0.05 < hi < 0.0
    assert math.isclose(s["discordant"], 0.05)


def test_gsm8k_score():
    b = pe.BENCHES["gsm8k"]
    item = {"gold": "1234"}
    assert b.score(item, {"content": "<think>x</think>so\nAnswer: 1,234"})["correct"]
    assert b.score(item, {"content": "we get 1234."})["correct"]
    assert not b.score(item, {"content": "Answer: 12"})["correct"]


def _tc(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_bfcl_score():
    b = pe.BENCHES["bfcl"]
    item = {
        "gold": [
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ]
    }
    good = [
        _tc("spotify_play", '{"artist": "maroon 5", "duration": 15}'),
        _tc("spotify_play", '{"artist": "Taylor Swift", "duration": 20.0}'),
    ]
    assert b.score(item, {"tool_calls": good})["correct"]
    assert not b.score(item, {"tool_calls": good[:1]})["correct"]
    bad = [_tc("spotify_play", '{"artist": "x", "duration": 15}')] * 2
    assert not b.score(item, {"tool_calls": bad})["correct"]
    assert b.score(item, {"tool_calls": [_tc("spotify_play", "{oops")]})["parse_fail"]
    assert b.score(item, {"content": "text"})["parse_fail"]


def test_bfcl_optional_and_schema():
    gold = {"f": {"x": [1], "y": ["", 2]}}
    assert pe._call_matches(("f", {"x": 1}), gold)  # y optional
    assert not pe._call_matches(("f", {"x": 2}), gold)
    s = pe._schema({"type": "dict", "properties": {"a": {"type": "float"}}})
    assert s["type"] == "object" and s["properties"]["a"]["type"] == "number"


def test_needle_score():
    b = pe.BENCHES["needle"]
    multi = {"kind": "multivalue", "gold": ["1111111", "2222222", "3333333"]}
    assert b.score(multi, {"content": "1111111, 2222222 and 3333333"})["correct"]
    assert not b.score(multi, {"content": "1111111, 2222222"})["correct"]
    single = {"kind": "single", "gold": ["1111111"]}
    assert b.score(single, {"content": "1,111,111"})["correct"]
    assert not b.score(single, {"content": "1111111 or 2222222"})["correct"]
