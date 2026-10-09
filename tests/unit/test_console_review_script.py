import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "console_review", Path(__file__).parents[2] / "scripts/research/console_review.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules["console_review"] = m
spec.loader.exec_module(m)


def test_plan_is_deterministic_bursty_and_covers_every_kind():
    a = m.build_plan(19, 540)
    assert [x.at for x in a] == [x.at for x in m.build_plan(19, 540)]
    assert {x.kind for x in a} >= {"short", "p8k", "long", "tool", "cancel", "fail"}
    assert sum(x.kind == "fail" for x in a) == 1
    assert max(x.at for x in a) < 540
    gaps = [b.at - c.at for c, b in zip(a, a[1:], strict=False)]
    assert max(gaps) > 15  # idle gaps between bursts


def test_prefix_families_repeat_so_there_are_hits_and_misses():
    big = [x for x in m.build_plan(19, 900) if x.kind in ("p8k", "p32k")]
    assert any(x.params["hit"] for x in big) and any(not x.params["hit"] for x in big)


def test_strip_meta_drops_text_but_keeps_token_counts():
    out = m.strip_meta(
        {
            "id": "r",
            "prompt": "secret",
            "prompt_tokens": 5,
            "x": [{"content": "c", "ttft_ms": 3}],
        }
    )
    assert out == {"id": "r", "prompt_tokens": 5, "x": [{"ttft_ms": 3}]}


def test_sheet_indices_keep_ends_and_cap():
    assert m.sheet_indices(5) == [0, 1, 2, 3, 4]
    s = m.sheet_indices(100)
    assert s[0] == 0 and s[-1] == 99 and len(s) <= 24
