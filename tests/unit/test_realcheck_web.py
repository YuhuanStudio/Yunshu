"""scripts/research/realcheck_web.py: the verdict fails closed (CPU only, no server)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import realcheck_web as rw  # noqa: E402


def test_verdict_fails_closed():
    want = tuple(rw.CHECKS)
    ok = {n: {"status": "pass"} for n in want}
    assert rw.verdict(ok, want) == []
    assert rw.verdict({}, want) == [f"{n}: never ran" for n in want]
    bad = {**ok, "fetch_page": {"status": "fail", "detail": "boom"}}
    assert rw.verdict(bad, want) == ["fetch_page: boom"]


def test_checks_reject_a_missing_search_call():
    class R:
        output = []

    class Responses:
        def create(self, **k):
            return R()

    class Oa:
        responses = Responses()

    class C:
        oa = Oa()
        model = "m"

    try:
        rw.responses_search(C())
    except AssertionError as e:
        assert "non-stream output" in str(e)
    else:
        raise AssertionError("an empty output must fail")
