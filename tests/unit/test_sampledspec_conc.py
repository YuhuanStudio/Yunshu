import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import sampledspec_conc as c  # noqa: E402


def test_arm_order_alternates():
    assert c.arm_order(0) == ("base", "cand")
    assert c.arm_order(1) == ("cand", "base")


def test_plans():
    p = c.plans("uneven", 900, 12)
    assert p[0][2] == 900 and p[1][2] == 300 and p[1][1] == 0
    assert c.plans("stagger", 900, 12)[1][1] == 12


def test_split_rates():
    a = dict(times=[i * 0.1 for i in range(100)], ttft=1.0)
    b = dict(times=[0.5 + i * 0.1 for i in range(30)], ttft=0.4)
    r = c.split_rates(a, b)
    assert r["overlapped"] and r["a_after_tokens"] > 0
    assert abs(r["a_overlap_tps"] - 10) < 0.5 and abs(r["a_after_tps"] - 10) < 0.5
    assert r["b_ttft"] == 0.4


def test_digest_stable():
    assert c.digest("x", [["a", -1.0]]) == c.digest("x", [["a", -1.0]])
    assert c.digest("x", []) != c.digest("y", [])
