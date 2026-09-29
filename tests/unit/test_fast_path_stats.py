"""FastPathStats: the mlx-lm fast path feeds the same RunStats the gateway reads."""

import asyncio
import time

from yunshu_engine.fast_path_stats import FastPathStats
from yunshu_gateway import x_yunshu


def test_phases_and_x_yunshu_fields():
    ev = asyncio.Event()
    fp = FastPathStats(ev, prompt_tokens=1000)
    st = ev.run_stats
    assert st.phase == "queued"
    fp.admit(cached_tokens=200, to_prefill=800)
    assert st.phase == "prefill" and st.cached_tokens == 200
    fp.progress(400, 800)
    assert (st.prefill_done, st.prefill_total) == (400, 800)

    info = x_yunshu.RequestInfo("r1", "POST", "/v1/chat/completions")
    info.gen = type("G", (), {"stats": st})()
    prog = x_yunshu.progress_payload(info)
    assert prog["phase"] == "prefill" and prog["percent"] == 50.0
    assert prog["processed_tokens"] == 600  # cached tokens count as processed

    time.sleep(0.02)
    fp.token(1)
    assert st.phase == "decode" and st.prefill_done == 800
    time.sleep(0.02)
    fp.token(5)
    fp.finish("stop")
    assert st.phase == "done" and st.generated == 5

    stats = x_yunshu.build_stats(info, {"prompt_tokens": 1000, "completion_tokens": 5})
    assert stats["ttft_ms"] is not None and stats["ttft_ms"] > 0
    assert stats["prefill_ms"] > 0 and stats["prefill_tps"] > 0
    assert stats["decode_tps"] > 0 and stats["cached_tokens"] == 200


def test_slotted_event_does_not_break():
    class Slotted:
        __slots__ = ()

    fp = FastPathStats(Slotted(), prompt_tokens=3)
    fp.admit(0, 3)
    fp.token(1)
    assert fp.stats.generated == 1


def test_no_cancel_event():
    fp = FastPathStats(None, prompt_tokens=3)
    fp.admit(0, 3)
    fp.progress(3, 3)
    assert fp.stats.prefill_done == 3
