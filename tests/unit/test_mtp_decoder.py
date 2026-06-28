"""Tests for MTP decoder pure logic functions."""

from unittest.mock import MagicMock

from yunshu_engine.mtp_decoder import (
    MTPConfig,
    MTPStats,
    _get_eos_ids,
    _restore_cache,
    _snapshot_cache,
)


class TestMTPConfig:
    def test_defaults(self):
        cfg = MTPConfig()
        assert cfg.max_tokens == 256
        assert cfg.cooldown_on_reject is False
        assert cfg.fastmtp_top_k == 0
        assert cfg.use_n_confirmed is True

    def test_custom(self):
        cfg = MTPConfig(max_tokens=512, cooldown_on_reject=True, fastmtp_top_k=32768)
        assert cfg.max_tokens == 512
        assert cfg.cooldown_on_reject is True
        assert cfg.fastmtp_top_k == 32768


class TestMTPStats:
    def test_defaults(self):
        s = MTPStats()
        assert s.accepts == 0
        assert s.rejects == 0
        assert s.cooldowns == 0
        assert s.tokens_generated == 0
        assert s.total_cycles == 0


class TestGetEosIds:
    def test_single_eos(self):
        tok = MagicMock()
        tok.eos_token_id = 2
        assert _get_eos_ids(tok) == {2}

    def test_list_eos(self):
        tok = MagicMock()
        tok.eos_token_id = [2, 3]
        assert _get_eos_ids(tok) == {2, 3}

    def test_none_eos(self):
        tok = MagicMock()
        tok.eos_token_id = None
        assert _get_eos_ids(tok) == set()

    def test_no_eos_attr(self):
        tok = MagicMock(spec=[])
        assert _get_eos_ids(tok) == set()


class TestSnapshotRestoreCache:
    def test_snapshot_and_restore_arrays(self):
        cache = [MagicMock()]
        cache[0].cache = [MagicMock(), MagicMock()]
        assert hasattr(cache[0], "cache")
        snap = _snapshot_cache(cache)
        assert snap[0][0] == "arrays"
        assert len(snap[0][1]) == 2

    def test_snapshot_and_restore_kv(self):
        cache = [MagicMock(spec=["offset"])]
        cache[0].offset = 10
        snap = _snapshot_cache(cache)
        assert snap[0][0] == "kv"
        assert snap[0][1] == 10

    def test_snapshot_unknown(self):
        cache = [MagicMock(spec=["other"])]
        snap = _snapshot_cache(cache)
        assert snap[0][0] is None

    def test_restore_arrays(self):
        cache = [MagicMock()]
        arr1, arr2 = MagicMock(), MagicMock()
        cache[0].cache = [arr1, arr2]
        snap = _snapshot_cache(cache)

        # Modify cache
        cache[0].cache = [MagicMock()]

        # Restore
        _restore_cache(cache, snap)
        assert len(cache[0].cache) == 2
        assert cache[0].cache[0] is arr1

    def test_restore_kv(self):
        cache = [MagicMock(spec=["offset"])]
        cache[0].offset = 10
        snap = _snapshot_cache(cache)

        cache[0].offset = 20
        _restore_cache(cache, snap)
        assert cache[0].offset == 10
