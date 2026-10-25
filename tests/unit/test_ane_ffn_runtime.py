"""CPU tests of the ANE FFN split's serving hook: layer order, validation, restore and rerun."""

import pytest

from yunshu_engine import ane_ffn_runtime as rt_mod


class FakeX:
    ndim = 3

    def __init__(self, rows, name="bfloat16"):
        self.shape = (1, rows, 8)
        self.dtype = type("D", (), {"name": name})()

    def reshape(self, *shape):
        return self


class FakeOut:
    def reshape(self, *shape):
        return self


class FakeSplit:
    def __init__(self, layers=3, fail_finish=False, minimum=512):
        self.layers = [None] * layers
        self.share = 0.235
        self.minimum_rows = minimum
        self.served_ms = 0.0
        self.reason = "boom"
        self.fail_finish = fail_finish
        self.calls = []
        self._retired = False

    def splits(self, rows):
        return rows >= self.minimum_rows and not self._retired

    def begin(self):
        self.calls.append("begin")
        return True

    def add(self, index, x, residual, rows):
        self.calls.append(("add", index, rows))
        return FakeOut()

    def finish(self):
        self.calls.append("finish")
        if self.fail_finish:
            self._retired = True
            return False
        return True

    def retired(self):
        return self._retired

    def close(self):
        pass


class Mlp:
    def __init__(self, index):
        self._ane_index = index


def run_layers(rt, rows):
    outs = []
    for i in range(len(rt.split.layers)):
        outs.append(rt.run(Mlp(i), FakeX(rows)))
    return outs


def test_split_chunk_runs_every_layer_and_finishes_once():
    rt = rt_mod.Runtime(FakeSplit(), "r")
    outs = run_layers(rt, 1024)
    assert all(o is not None for o in outs)
    assert rt.split.calls == ["begin", ("add", 0, 1024), ("add", 1, 1024), ("add", 2, 1024), "finish"]
    assert rt.chunks_split == 1 and rt.chunks_gpu == 0


def test_short_or_batched_or_wrong_dtype_chunks_stay_on_the_gpu():
    rt = rt_mod.Runtime(FakeSplit(minimum=796), "r")
    assert run_layers(rt, 700) == [None, None, None]
    assert rt.chunks_gpu == 1 and rt.split.calls == []
    wrong = rt_mod.Runtime(FakeSplit(), "r")
    assert wrong.run(Mlp(0), FakeX(1024, "float16")) is None
    x = FakeX(1024)
    x.shape = (2, 1024, 8)
    assert wrong.run(Mlp(0), x) is None
    assert wrong.run(object(), FakeX(1024)) is None


def test_out_of_order_layer_falls_back_without_calling_the_split():
    rt = rt_mod.Runtime(FakeSplit(), "r")
    assert rt.run(Mlp(0), FakeX(1024)) is not None
    assert rt.run(Mlp(2), FakeX(1024)) is None  # layer 1 skipped
    rt.abandon()
    assert rt.split.calls[-1] == "finish"


def test_failed_validation_raises_and_guard_restores_and_reruns(monkeypatch):
    split = FakeSplit(fail_finish=True)
    rt = rt_mod.Runtime(split, "r")
    monkeypatch.setattr(rt_mod, "_RUNTIME", rt)

    class Cache:
        def __init__(self):
            self.offset = 5
            self.cache = ["a", "b"]

    cache = Cache()
    runs = []

    def step():
        runs.append(rt.engaged)
        cache.offset += 7
        cache.cache = ["x", "y"]
        if rt.engaged:
            run_layers(rt, 1024)
        return "ok"

    assert rt_mod.guarded([cache], step) == "ok"
    assert runs == [True, False]  # rerun on the GPU alone
    assert (cache.offset, cache.cache) == (12, ["x", "y"])  # second run's own writes
    assert rt.failures == 1 and rt.engaged is False


def test_guard_is_transparent_without_runtime_or_after_retirement(monkeypatch):
    monkeypatch.setattr(rt_mod, "_RUNTIME", None)
    assert rt_mod.guarded([], lambda: 3) == 3
    split = FakeSplit()
    split._retired = True
    monkeypatch.setattr(rt_mod, "_RUNTIME", rt_mod.Runtime(split, "r"))
    assert rt_mod.guarded([], lambda: 4) == 4
    assert rt_mod.kernel_suffix().startswith("+ane-ffn")
    monkeypatch.setattr(rt_mod, "_RUNTIME", None)
    assert rt_mod.kernel_suffix() == ""


def test_setting_is_registered_off_by_default():
    from yunshu_engine import settings

    assert settings.REGISTRY["YUNSHU_ANE_FFN_SPLIT"].default == "off"
    with pytest.raises(Exception):  # noqa: B017
        settings.REGISTRY["YUNSHU_ANE_FFN_SPLIT"].choices.index("bogus")
