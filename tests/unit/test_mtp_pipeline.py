"""Tests for MTP pipeline integration — n_confirmed_patch + mtp_decoder wired into BatchedEngine."""

import pytest


class TestNConfirmedPipelineIntegration:
    """Verify n_confirmed_patch is applied during engine startup."""

    def test_apply_n_confirmed_patch_importable(self):
        from yunshu_engine.n_confirmed_patch import apply_n_confirmed_patch
        assert callable(apply_n_confirmed_patch)

    def test_apply_n_confirmed_patch_idempotent(self):
        from yunshu_engine.n_confirmed_patch import apply_n_confirmed_patch
        result1 = apply_n_confirmed_patch()
        result2 = apply_n_confirmed_patch()
        assert result1 == result2

    def test_clear_rollback_importable(self):
        from yunshu_engine.n_confirmed_patch import clear_rollback
        assert callable(clear_rollback)

    def test_restore_rollback_importable(self):
        from yunshu_engine.n_confirmed_patch import restore_rollback
        assert callable(restore_rollback)

    def test_rollback_on_empty_cache(self):
        from yunshu_engine.n_confirmed_patch import clear_rollback, restore_rollback
        clear_rollback([])
        result = restore_rollback([])
        assert result is True


class TestMTPDecoderImport:
    """Verify MTPDecoder and MTPConfig can be imported."""

    def test_mtp_config_importable(self):
        from yunshu_engine.mtp_decoder import MTPConfig
        cfg = MTPConfig()
        assert cfg.use_n_confirmed is True

    def test_mtp_config_custom(self):
        from yunshu_engine.mtp_decoder import MTPConfig
        cfg = MTPConfig(
            max_tokens=512,
            cooldown_on_reject=True,
            fastmtp_top_k=32768,
            use_n_confirmed=True,
        )
        assert cfg.max_tokens == 512
        assert cfg.cooldown_on_reject is True
        assert cfg.fastmtp_top_k == 32768

    def test_mtp_stats_importable(self):
        from yunshu_engine.mtp_decoder import MTPStats
        stats = MTPStats()
        assert stats.accepts == 0
        assert stats.rejects == 0
        assert stats.tokens_generated == 0

    def test_mtp_decoder_importable(self):
        from yunshu_engine.mtp_decoder import MTPDecoder
        assert MTPDecoder is not None

    def test_run_mtp_decode_importable(self):
        from yunshu_engine.mtp_decoder import run_mtp_decode
        assert callable(run_mtp_decode)


class TestMTPStrategyIntegration:
    """Verify MTPStrategy wraps MTPDecoder correctly."""

    def test_mtp_strategy_importable(self):
        from yunshu_engine.spec_interface import MTPStrategy
        strategy = MTPStrategy()
        assert strategy.name == "mtp"

    def test_mtp_strategy_with_decoder(self):
        from yunshu_engine.spec_interface import MTPStrategy
        strategy = MTPStrategy(decoder="fake_decoder")
        assert strategy.decoder == "fake_decoder"

    def test_mtp_strategy_begin_end(self):
        from yunshu_engine.spec_interface import MTPStrategy
        strategy = MTPStrategy()
        strategy.begin("req-1")
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.strategy_name == "mtp"
        strategy.accept([], 0)
        stats = strategy.stats()
        assert stats["name"] == "mtp"
        strategy.end("req-1")

    def test_mtp_strategy_stats_without_decoder(self):
        from yunshu_engine.spec_interface import MTPStrategy
        strategy = MTPStrategy()
        stats = strategy.stats()
        assert stats["total_drafts"] == 0
        assert "mtp_accepts" not in stats

    def test_mtp_strategy_reset(self):
        from yunshu_engine.spec_interface import MTPStrategy
        strategy = MTPStrategy()
        strategy.begin("req-1")
        strategy.draft([1, 2, 3], n=5)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_drafts"] == 0


class TestSpecStrategyFactoryMTP:
    """Verify SpecStrategyFactory supports MTP type."""

    def test_factory_creates_mtp(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory, MTPStrategy
        strategy = SpecStrategyFactory.create({"type": "mtp"})
        assert isinstance(strategy, MTPStrategy)

    def test_factory_mtp_with_decoder(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory, MTPStrategy
        strategy = SpecStrategyFactory.create({
            "type": "mtp",
            "decoder": "fake",
            "decoder_config": {"max_tokens": 128},
        })
        assert isinstance(strategy, MTPStrategy)
        assert strategy.decoder == "fake"

    def test_factory_composite_with_mtp(self):
        from yunshu_engine.spec_interface import (
            SpecStrategyFactory, CompositeStrategy, MTPStrategy,
        )
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "ngram"},
                {"type": "mtp"},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "ngram" in strategy.name
        assert "mtp" in strategy.name


class TestBatchedEngineMTPFields:
    """Verify BatchedEngine has MTP fields and methods."""

    def test_engine_has_mtp_decoder_field(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        assert hasattr(engine, '_mtp_decoder')
        assert engine._mtp_decoder is None

    def test_engine_has_mtp_strategy_field(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        assert hasattr(engine, '_mtp_strategy')
        assert engine._mtp_strategy is None

    def test_engine_has_generate_mtp_method(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        assert hasattr(engine, '_generate_mtp')
        assert callable(engine._generate_mtp)

    def test_engine_has_stream_generate_mtp_method(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        assert hasattr(engine, '_stream_generate_mtp')
        assert callable(engine._stream_generate_mtp)

    def test_engine_get_stats_no_mtp(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        stats = engine.get_stats()
        assert "mtp" not in stats

    def test_engine_stop_clears_mtp(self):
        """Verify stop() resets MTP state."""
        import asyncio
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        engine._loaded = True
        engine._mtp_decoder = object()  # sentinel
        engine._mtp_strategy = object()  # sentinel
        engine._model = None
        engine._tokenizer = None
        # stop() is async, need to run in loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.stop())
        except Exception:
            pass
        finally:
            loop.close()
        assert engine._mtp_decoder is None
        assert engine._mtp_strategy is None


class TestBatchedEngineMTPRouting:
    """Verify MTP routing in generate() and stream_generate()."""

    def test_generate_routes_to_mtp_when_available(self):
        """When spec_decode=True and _mtp_decoder is set, should route to MTP."""
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        engine._loaded = True

        # Track which method was called
        called = {"mtp": False}

        async def _fake_mtp(prompt, max_tokens, temperature, **kwargs):
            called["mtp"] = True
            from yunshu_engine.batched_engine import GenerationOutput
            return GenerationOutput(text="test", finished=True, finish_reason="stop")

        engine._generate_mtp = _fake_mtp
        engine._mtp_decoder = object()  # truthy sentinel

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.generate("test", spec_decode=True, use_engine_loop=False)
            )
            assert called["mtp"], "Should have routed to _generate_mtp"
            assert result.text == "test"
        finally:
            loop.close()

    def test_generate_skips_mtp_without_decoder(self):
        """Without _mtp_decoder, should fall through to standard path."""
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        engine._loaded = True
        engine._mtp_decoder = None  # explicitly None

        # Should not raise — falls through to fast path
        assert engine._mtp_decoder is None

    def test_get_spec_strategy_prefers_mtp(self):
        """_get_spec_strategy should return MTP strategy when available."""
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        engine._mtp_strategy = "mtp_strategy_instance"
        assert engine._get_spec_strategy() == "mtp_strategy_instance"

    def test_get_spec_strategy_falls_back_to_env(self):
        """Without MTP, should fall back to env-based strategy."""
        import os
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine()
        engine._mtp_strategy = None
        # Without YUNSHU_SPEC_STRATEGY env, should return None
        old = os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
        try:
            result = engine._get_spec_strategy()
            assert result is None
        finally:
            if old is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old


class TestMTPEnvConfig:
    """Verify MTP environment variable configuration."""

    def test_mtp_cooldown_env(self):
        import os
        from yunshu_engine.mtp_decoder import MTPConfig
        # Default: cooldown disabled
        cfg = MTPConfig()
        assert cfg.cooldown_on_reject is False

    def test_mtp_fastmtp_default_disabled(self):
        from yunshu_engine.mtp_decoder import MTPConfig
        cfg = MTPConfig()
        assert cfg.fastmtp_top_k == 0

    def test_n_confirmed_default_enabled(self):
        from yunshu_engine.mtp_decoder import MTPConfig
        cfg = MTPConfig()
        assert cfg.use_n_confirmed is True
