"""Flash-Next (QSA) rows restored from the SSD tier decode as their cold run (MTPLX #635).

Conversation B shares a prefix with a stored conversation A. B's lookup restores A's checkpoint
from the SSD tier and prefills only B's own suffix; the QSA indexer must then select blocks from
B's restored keys, never from A's. Output must equal a cold run of B, token for token.
"""

import random
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_round_driver_flashnext import make_tiny  # noqa: E402
from yunshu_engine.apc_manager import SpillDiskStore, YunshuAPCManager  # noqa: E402
from yunshu_engine.round_driver.driver import RoundDriver  # noqa: E402
from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_and_prefill_checkpoints(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("YUNSHU_NEXT_TURN_PREFILL", "off")
    yield
    mx.set_default_device(previous)


class _Stop:
    def __init__(self):
        self.eos = set()

    def add_eos_token_ids(self, ids):
        self.eos |= set(ids or [])

    def __call__(self, token):
        return int(token) in self.eos


def _model():
    lm, _ = make_tiny()
    return SimpleNamespace(
        language_model=lm, config=SimpleNamespace(image_token_index=None)
    )


def _processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=_Stop()))


def _run(runner, ids):
    stats = RunStats()
    out = list(runner.iter_tokens(ids, max_tokens=6, stats=stats, allow_draft=False))
    return out, stats


def _cold_runner(model):
    runner = VLMBatchRunner(model, _processor())
    runner.driver = RoundDriver(model, chunk=2048, stop_tokens=set())
    return runner


def _cached_runner(model, mgr):
    runner = VLMBatchRunner(model, _processor(), apc_manager=mgr, apc_semantic_hash=0)
    runner.driver = RoundDriver(model, apc=mgr, chunk=2048, stop_tokens=set())
    return runner


def _manager(tmp_path, max_entries):
    disk = SpillDiskStore(tmp_path, namespace="qsa", num_workers=1, max_bytes=1 << 30)
    return YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 128},
        max_entries=max_entries,
    )


def test_partial_prefix_of_a_stored_qsa_entry_is_never_restored(tmp_path):
    """MTPLX #635 needs a restore below a longer QSA entry's end. Yunshu's match rule refuses it:
    a QSA checkpoint is not trimmable, so only its whole stored length can be restored."""
    from mlx_vlm import apc as upstream

    rnd = random.Random(7)

    def toks(n):
        return [rnd.randrange(3, 60) for _ in range(n)]

    a = toks(260)
    b = a[:140] + toks(
        120
    )  # shares a's first 140 tokens: no stored checkpoint of a ends at 140
    model = _model()

    mgr = _manager(tmp_path, max_entries=1)
    runner = _cached_runner(model, mgr)
    _run(runner, a)
    entries = [e for e in mgr._exact_cache.values() if len(e.token_ids) > 140]
    assert entries, "A's checkpoint is not in RAM to inspect"
    for e in entries:
        trimmable = upstream._dense_checkpoint_trimmable(
            e.prompt_cache, len(e.token_ids)
        )
        assert trimmable is False
        assert (
            upstream._checkpoint_match_len(tuple(b), e.token_ids, 200, 16, trimmable)
            == 0
        )
    _run(
        runner, toks(200)
    )  # an unrelated conversation pushes A out of RAM to the SSD tier
    mgr.disk.flush()
    out_b, st_b = _run(runner, b)
    assert st_b.cached_tokens == 0, st_b

    out_cold, _ = _run(_cold_runner(model), b)
    assert out_b == out_cold


def test_ssd_restore_of_the_stored_conversation_decodes_as_cold(tmp_path):
    rnd = random.Random(11)
    a = [rnd.randrange(3, 60) for _ in range(260)]
    a2 = a + [rnd.randrange(3, 60) for _ in range(40)]
    model = _model()

    mgr = _manager(tmp_path, max_entries=1)
    runner = _cached_runner(model, mgr)
    _run(runner, a)
    _run(runner, [rnd.randrange(3, 60) for _ in range(200)])
    mgr.disk.flush()
    out_a2, st = _run(runner, a2)
    assert st.cache_tier == "ssd" and st.cached_tokens >= 128, st
    out_cold, _ = _run(_cold_runner(model), a2)
    assert out_a2 == out_cold
