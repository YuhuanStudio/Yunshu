"""Preemptible prefill: a background request can stop between decoder-layer groups."""

import threading
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.kernels import preempt  # noqa: E402
from yunshu_engine.vlm_batch_runner import VLMBatchRunner  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


class _Layer:
    def __call__(self, x):
        return x + 1


def _run(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _wrapped_layers(n):
    cls = type("L", (), {"__call__": preempt.wrap(_Layer.__call__)})
    return [cls() for _ in range(n)]


def test_outside_a_scope_the_wrapper_changes_nothing():
    out = _run(_wrapped_layers(8), mx.zeros((2,)))
    assert out.tolist() == [8.0, 8.0]


def test_scope_evaluates_every_k_layers_and_keeps_the_numbers():
    ev = threading.Event()
    with preempt.scope(ev, every=4) as state:
        out = _run(_wrapped_layers(8), mx.zeros((2,)))
    assert out.tolist() == [8.0, 8.0]
    assert (state.layers, state.evals) == (8, 2)


def test_cancel_stops_at_the_next_layer_group():
    ev = threading.Event()
    calls = []

    class Layer:
        def __call__(self, x):
            calls.append(1)
            if len(calls) == 5:
                ev.set()  # a request arrives during the second group
            return x + 1

    cls = type("L", (), {"__call__": preempt.wrap(Layer.__call__)})
    with preempt.scope(ev, every=4), pytest.raises(preempt.PreemptedError):
        _run([cls() for _ in range(16)], mx.zeros((2,)))
    assert (
        len(calls) == 8
    )  # stopped at the end of the group that saw the cancel, not at 16


def test_no_cancel_no_raise():
    ev = threading.Event()
    with preempt.scope(ev, every=2):
        _run(_wrapped_layers(6), mx.zeros((1,)))


def test_install_is_idempotent_and_supported_reflects_it():
    pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer

    before = Qwen3_5DecoderLayer.__call__
    try:
        assert preempt.install() is True
        once = Qwen3_5DecoderLayer.__call__
        assert preempt.install() is True
        assert Qwen3_5DecoderLayer.__call__ is once
        assert preempt.supported()
    finally:
        Qwen3_5DecoderLayer.__call__ = before


def test_runner_scopes_only_a_lone_preemptible_job(monkeypatch):
    monkeypatch.setattr(preempt, "supported", lambda: True)
    seen = []

    class Gen:
        def next(self):
            seen.append(preempt._SCOPE.get())
            return "ok"

    ev = threading.Event()
    ev.preemptible = True
    rehearsal = SimpleNamespace(cancel_event=ev)
    plain = SimpleNamespace(cancel_event=threading.Event())

    assert (
        VLMBatchRunner._next_preemptible(
            SimpleNamespace(gen=Gen(), jobs={1: rehearsal})
        )
        == "ok"
    )
    assert seen[-1] is not None and seen[-1].cancel_event is ev
    VLMBatchRunner._next_preemptible(SimpleNamespace(gen=Gen(), jobs={1: plain}))
    assert seen[-1] is None
    # shared with another job: a preempt exception would corrupt the other row
    VLMBatchRunner._next_preemptible(
        SimpleNamespace(gen=Gen(), jobs={1: rehearsal, 2: plain})
    )
    assert seen[-1] is None


def test_preempted_error_reports_the_layer_group():
    ev = threading.Event()
    ev.set()
    with preempt.scope(ev, every=4), pytest.raises(preempt.PreemptedError) as info:
        _run(_wrapped_layers(16), mx.zeros((1,)))
    assert (info.value.layers, info.value.group) == (4, 1)
