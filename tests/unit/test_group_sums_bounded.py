"""The verify kernels' group-sum table must not pin every layer output (Metal resource limit)."""

import importlib
import threading

import pytest


@pytest.mark.parametrize("mod", ["omlx", "omlx_q4"])
def test_register_group_sums_is_bounded(mod):
    q = importlib.import_module(f"yunshu_engine.kernels.{mod}.qwen35_verify_qmm")
    out = {}

    def run():
        xs = [object() for _ in range(q.GROUP_SUMS_MAX * 3)]
        for x in xs:
            q.register_group_sums(x, 1)
        out["n"] = len(q._ROUTE_ARMED.sums)
        out["last"] = q.group_sums_for(xs[-1])
        out["first"] = q.group_sums_for(xs[0])
        q.clear_group_sums()

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert out["n"] <= q.GROUP_SUMS_MAX
    assert out["last"] == 1 and out["first"] is None


def test_driver_step_clears_group_sums(monkeypatch):
    from yunshu_engine.round_driver import driver

    calls = []
    monkeypatch.setattr(driver, "_clear_verify_state", lambda: calls.append(1))

    class Fake:
        def _step_impl(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        driver.RoundDriver.step(Fake())
    assert calls == [1]
