"""vLLM 0bfeb8d0e re-check (DFlash must not lose / misalign the last prefix block on hybrids).

Yunshu stores token-exact checkpoints with a one-token guard for every drafter, and the DFlash
window is captured after the chunk ending exactly at the checkpoint was appended, so nothing is
dropped. What was wrong is the opposite: ``store_checkpoint`` captured a window for every forward
boundary, including ones that are never stored (split-invariant prefill calls it every stride) and
for opted-out requests, churning the small window budget. Only stored checkpoints capture now.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import KVCache  # noqa: E402

from tests.unit.test_apc_manager import _coordinator, _mgr  # noqa: E402
from yunshu_engine import dflash_apc_window  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def _kv(n):
    c = KVCache()
    c.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    return [c]


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dflash_apc_window,
        "capture",
        lambda tokens, extra, prompt_end=False: calls.append((len(tokens), prompt_end)),
    )
    return calls


def _setup(**kw):
    m = _mgr(prefill_alignment=64, **kw)
    c = _coordinator(m)
    ids = list(range(1, 700))
    c._current_ids = tuple(ids)
    c._split_lengths = True
    c._store_lengths = frozenset({512, 640})
    return m, c, ids


def test_forward_boundary_that_is_not_stored_captures_no_window(captured):
    _, c, ids = _setup()
    assert c.store_checkpoint(ids[:256], _kv(256)) is False  # a split boundary only
    assert captured == []


def test_stored_checkpoint_captures_its_window(captured):
    _, c, ids = _setup()
    c.defer_checkpoint_stores = False
    assert c.store_checkpoint(ids[:640], _kv(640)) is True
    assert captured == [(640, False)]


def test_prompt_end_checkpoint_is_flagged_as_prompt_end(captured):
    _, c, ids = _setup()
    c.defer_checkpoint_stores = False
    c._current_ids = tuple(ids[:641])
    assert c.store_checkpoint(ids[:640], _kv(640)) is True
    assert captured == [(640, True)]


def test_opted_out_request_captures_no_window(captured):
    _, c, ids = _setup()
    c.set_store_off(ids)
    assert c.store_checkpoint(ids[:640], _kv(640)) is False
    assert captured == []
