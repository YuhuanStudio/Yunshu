from types import SimpleNamespace

from yunshu_gateway.x_yunshu import RequestInfo, live_decode_rate


def _info(generated, t_first=0.0, t_last=100.0):
    info = RequestInfo(request_id="r", method="POST", path="/v1/chat/completions")
    st = SimpleNamespace(generated=generated, t_first=t_first or 0.001, t_last=t_last)
    info.gen = SimpleNamespace(stats=st)
    return info


def test_live_rate_follows_a_slowdown_instead_of_the_lifetime_average():
    info = _info(1)
    # 100 s at 40 tok/s, then a peer arrives and the row drops to 10 tok/s.
    t = 0.0
    live_decode_rate(info, t)
    for _ in range(100):
        t += 1.0
        info.gen.stats.generated += 40
        live_decode_rate(info, t)
    for _ in range(8):
        t += 1.0
        info.gen.stats.generated += 10
        rate = live_decode_rate(info, t)
    lifetime = (info.gen.stats.generated - 1) / t
    assert lifetime > 35  # what the console used to show
    assert 9 <= rate <= 12  # what the row is actually doing now


def test_first_call_falls_back_to_lifetime_and_finished_rows_decay():
    info = _info(101, t_first=1.0, t_last=3.0)
    assert live_decode_rate(info, 3.0) == 50.0
    t = 3.0
    for _ in range(8):
        t += 1.0
        rate = live_decode_rate(info, t)
    assert rate == 0.0
