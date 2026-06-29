"""Tests for KV Prefix Cache integration.

Tests:
- compute_block_hash() chain hashing correctness
- compute_prompt_hashes() for various prompt lengths
- KVCacheManager.allocate_for_prefill() with cache hits and misses
- KVCacheManager.cache_completed_blocks() registration
- KVCacheManager.free_request() block freeing
- Prefix cache hit across two requests with same prefix
- Memory estimation via compute_num_blocks()
- Hit rate tracking in KVCacheManager
- EngineCore KV cache wiring
- BatchedEngine model architecture extraction
- Admin /kv-cache endpoint
"""

from yunshu_kv.hash import compute_block_hash, compute_prompt_hashes
from yunshu_kv.manager import (
    KVCacheConfig,
    KVCacheManager,
    PrefixMatch,
    compute_num_blocks,
)

# ── Chain Hashing Correctness ──


class TestChainHashCorrectness:
    """Tests for compute_block_hash() chain hashing properties."""

    def test_deterministic_same_inputs(self):
        """Same parent + tokens + extra_keys produces same hash."""
        tokens = [10, 20, 30, 40]
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(None, tokens)
        assert h1 == h2

    def test_different_parent_produces_different_hash(self):
        """Different parent hash changes the result (chain property)."""
        tokens = [1, 2, 3, 4]
        h_no_parent = compute_block_hash(None, tokens)
        h_with_parent = compute_block_hash(42, tokens)
        assert h_no_parent != h_with_parent

    def test_different_tokens_produces_different_hash(self):
        """Different token IDs produce different hashes."""
        h1 = compute_block_hash(None, [1, 2, 3, 4])
        h2 = compute_block_hash(None, [4, 3, 2, 1])
        assert h1 != h2

    def test_extra_keys_change_hash(self):
        """Extra keys (model hash) affect the hash."""
        tokens = [1, 2, 3, 4]
        h1 = compute_block_hash(None, tokens, extra_keys=())
        h2 = compute_block_hash(None, tokens, extra_keys=(99,))
        assert h1 != h2

    def test_string_extra_key(self):
        """String extra keys are mixed into the hash."""
        tokens = [1, 2, 3]
        h1 = compute_block_hash(None, tokens, extra_keys=("model_a",))
        h2 = compute_block_hash(None, tokens, extra_keys=("model_b",))
        assert h1 != h2

    def test_chain_order_matters(self):
        """Hash(A->B) != Hash(A) then separately Hash(B).

        Chain hashing: the parent hash is mixed in, so the order of
        blocks in the chain affects the hash.
        """
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]

        h_a = compute_block_hash(None, tokens_a)
        h_b_from_a = compute_block_hash(h_a, tokens_b)

        h_b_standalone = compute_block_hash(None, tokens_b)

        assert h_b_from_a != h_b_standalone

    def test_parent_zero_equivalent_to_none(self):
        """Parent hash of 0 is the same as None (both pack as 0)."""
        tokens = [1, 2, 3]
        h_none = compute_block_hash(None, tokens)
        h_zero = compute_block_hash(0, tokens)
        # Both pack 0 for parent, so they should be equal
        assert h_none == h_zero


# ── compute_prompt_hashes ──


class TestComputePromptHashes:
    """Tests for compute_prompt_hashes() with various prompt lengths."""

    def test_exact_multiple_of_block_size(self):
        """256 tokens / 64 block_size = 4 hashes."""
        tokens = list(range(256))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 4

    def test_remainder_tokens_not_hashed(self):
        """Tokens that don't fill a complete block are not hashed."""
        tokens = list(range(200))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 3  # 192 / 64 = 3, remainder 8 not hashed

    def test_shorter_than_one_block(self):
        """Fewer tokens than block_size produces no hashes."""
        tokens = list(range(10))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 0

    def test_empty_prompt(self):
        """Empty token list produces no hashes."""
        hashes = compute_prompt_hashes([], block_size=64)
        assert len(hashes) == 0

    def test_exactly_one_block(self):
        """Exactly block_size tokens produce exactly one hash."""
        tokens = list(range(64))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 1

    def test_chain_property(self):
        """Hashes are chained: hash[i] depends on hash[i-1]."""
        tokens = list(range(128))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 2

        # Verify manually
        h0 = compute_block_hash(None, tokens[0:64])
        h1 = compute_block_hash(h0, tokens[64:128])
        assert hashes[0] == h0
        assert hashes[1] == h1

    def test_different_prompts_different_hashes(self):
        """Different prompts produce different hash chains."""
        tokens1 = list(range(64))
        tokens2 = list(range(64, 128))
        h1 = compute_prompt_hashes(tokens1, block_size=64)
        h2 = compute_prompt_hashes(tokens2, block_size=64)
        assert h1 != h2

    def test_extra_keys_propagate(self):
        """Extra keys are mixed into each block hash."""
        tokens = list(range(64))
        h1 = compute_prompt_hashes(tokens, block_size=64, extra_keys=())
        h2 = compute_prompt_hashes(tokens, block_size=64, extra_keys=(42,))
        assert h1 != h2


# ── KVCacheManager.allocate_for_prefill ──


class TestAllocateForPrefill:
    """Tests for KVCacheManager.allocate_for_prefill() with hits and misses."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_first_request_no_cache_hit(self):
        """First request always has zero matched tokens."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert match.num_matched_tokens == 0
        assert len(match.unmatched_token_ids) == 64
        assert table.num_blocks == 4  # 64/16

    def test_allocate_for_short_prompt(self):
        """Prompt shorter than block_size still gets one block."""
        mgr = self._make_manager()
        tokens = [1, 2, 3]
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert match.num_matched_tokens == 0
        assert table.num_blocks == 1  # ceil(3/16) = 1

    def test_allocate_returns_prefix_match_dataclass(self):
        """allocate_for_prefill returns PrefixMatch with correct fields."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert isinstance(match, PrefixMatch)
        assert hasattr(match, "matched_blocks")
        assert hasattr(match, "num_matched_tokens")
        assert hasattr(match, "unmatched_token_ids")

    def test_different_model_hash_no_hit(self):
        """Different model_hash prevents cache hits."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens, model_hash=1)
        mgr.cache_completed_blocks(table1, tokens, model_hash=1)
        mgr.free_request(table1)

        # Same tokens but different model hash
        table2, match2 = mgr.allocate_for_prefill(tokens, model_hash=2)
        assert match2.num_matched_tokens == 0


# ── KVCacheManager.cache_completed_blocks ──


class TestCacheCompletedBlocks:
    """Tests for KVCacheManager.cache_completed_blocks() registration."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_caches_full_blocks(self):
        """cache_completed_blocks registers full blocks in prefix cache."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 4  # 4 full blocks

        # Verify blocks are now in hash lookup
        blocks = table.get_blocks()
        for b in blocks:
            assert b.block_hash is not None

    def test_does_not_cache_partial_block(self):
        """Incomplete final block is not cached."""
        mgr = self._make_manager()
        tokens = list(range(50))  # 3 full blocks (48) + 2 remaining
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 3

        blocks = table.get_blocks()
        assert blocks[0].block_hash is not None
        assert blocks[1].block_hash is not None
        assert blocks[2].block_hash is not None
        # Last block is partial, may or may not be cached depending on implementation

    def test_no_double_caching(self):
        """Already cached blocks are not re-cached."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached1 = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached1 == 2

        cached2 = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached2 == 0  # already cached

    def test_caching_disabled(self):
        """With enable_caching=False, no blocks are cached."""
        config = KVCacheConfig(block_size=16, enable_caching=False)
        mgr = KVCacheManager(config, num_blocks=200)
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 0


# ── KVCacheManager.free_request ──


class TestFreeRequest:
    """Tests for KVCacheManager.free_request() block freeing."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_frees_all_blocks(self):
        """free_request returns all blocks to the free pool."""
        mgr = self._make_manager()
        initial_free = mgr.num_free_blocks

        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)
        assert mgr.num_free_blocks == initial_free - 4

        mgr.free_request(table)
        assert mgr.num_free_blocks == initial_free

    def test_clears_block_table(self):
        """free_request empties the block table."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert table.num_blocks > 0
        mgr.free_request(table)
        assert table.num_blocks == 0

    def test_shared_blocks_ref_counted(self):
        """Shared blocks (prefix cache hit) are not freed until all refs drop."""
        mgr = self._make_manager()

        # Request 1: allocate and cache
        tokens1 = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)

        # Request 2: hits prefix cache (shares blocks with table1)
        tokens2 = list(range(64)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 64

        # Free request 1 — shared blocks should still be alive
        mgr.free_request(table1)

        # Free request 2
        mgr.free_request(table2)

        # All blocks back to free pool
        assert mgr.num_free_blocks == 199  # 200 - 1 (null block)


# ── Prefix Cache Hit Across Requests ──


class TestPrefixCacheHitAcrossRequests:
    """Tests for prefix cache reuse across two requests with same prefix."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_full_prefix_hit(self):
        """Second request fully reuses first request's prefix."""
        mgr = self._make_manager()

        tokens1 = list(range(64))
        table1, match1 = mgr.allocate_for_prefill(tokens1, model_hash=42)
        assert match1.num_matched_tokens == 0

        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Same tokens, should fully hit
        tokens2 = list(range(64)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 64
        assert len(match2.unmatched_token_ids) == 16
        mgr.free_request(table2)

    def test_partial_prefix_hit(self):
        """Second request partially matches prefix."""
        mgr = self._make_manager()

        tokens1 = list(range(48))  # 3 full blocks
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # First 48 tokens match, rest don't
        tokens2 = list(range(48)) + list(range(100, 132))  # 80 tokens total
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 48
        assert len(match2.unmatched_token_ids) == 32
        mgr.free_request(table2)

    def test_no_prefix_hit_different_content(self):
        """Different prompts produce no prefix hit."""
        mgr = self._make_manager()

        tokens1 = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Completely different tokens
        tokens2 = list(range(200, 264))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 0
        mgr.free_request(table2)

    def test_chain_hash_breaks_on_mismatch(self):
        """If block N doesn't match, block N+1 can't match (chain hash)."""
        mgr = self._make_manager()

        # First request: tokens 0-31 (2 blocks)
        tokens1 = list(range(32))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Second request: tokens 0-15 match block 0, but tokens 16-31 differ
        # Block 0 matches (tokens 0-15), block 1 doesn't (tokens 16-31 differ)
        # So only 1 block (16 tokens) matches
        tokens2 = list(range(16)) + list(range(200, 216)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 16  # only block 0 matches
        mgr.free_request(table2)

    def test_three_requests_incremental_prefix(self):
        """Three requests with incrementally longer shared prefixes."""
        mgr = self._make_manager()

        # Request 1: 32 tokens (2 blocks)
        tokens1 = list(range(32))
        t1, _ = mgr.allocate_for_prefill(tokens1, model_hash=0)
        mgr.cache_completed_blocks(t1, tokens1, model_hash=0)
        mgr.free_request(t1)

        # Request 2: same 32 tokens + 16 more (3 blocks total)
        tokens2 = list(range(48))
        t2, m2 = mgr.allocate_for_prefill(tokens2, model_hash=0)
        assert m2.num_matched_tokens == 32
        mgr.cache_completed_blocks(t2, tokens2, model_hash=0)
        mgr.free_request(t2)

        # Request 3: same 48 tokens + 16 more
        tokens3 = list(range(64))
        t3, m3 = mgr.allocate_for_prefill(tokens3, model_hash=0)
        assert m3.num_matched_tokens == 48
        mgr.free_request(t3)


# ── Memory Estimation via compute_num_blocks ──


class TestComputeNumBlocks:
    """Tests for compute_num_blocks() memory estimation."""

    def test_typical_m3_ultra_config(self):
        """192GB UMA, 40GB weights, 32L/8KV/128d model."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        # 192 GB UMA, 40 GB model weights
        n = compute_num_blocks(config, 192 * 2**30, 40 * 2**30)
        assert n > 0
        # bytes per block = 64 * 32 * 8 * 128 * 2 * 2 = 8,388,608 = 8MB
        # available = (192 - 40) * 0.85 = 129.2 GB
        # blocks = 129.2 GB / 8 MB ≈ 16,691
        assert 15000 < n < 18000

    def test_small_model_large_uma(self):
        """Small model on large UMA yields many blocks."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=12,
            num_kv_heads=4,
            head_dim=64,
            dtype_bytes=2,
        )
        n = compute_num_blocks(config, 192 * 2**30, 2 * 2**30)
        assert n > 50000

    def test_zero_layers_yields_zero(self):
        """Zero layers means zero bytes per block."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=0,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n = compute_num_blocks(config, 192 * 2**30, 0)
        assert n == 0

    def test_activation_reserve_reduces_blocks(self):
        """Higher activation_reserve_ratio reduces available blocks."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n_low = compute_num_blocks(
            config, 192 * 2**30, 40 * 2**30, activation_reserve_ratio=0.1
        )
        n_high = compute_num_blocks(
            config, 192 * 2**30, 40 * 2**30, activation_reserve_ratio=0.5
        )
        assert n_low > n_high

    def test_large_weights_reduce_blocks(self):
        """Larger model weights leave less room for KV cache."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n_small_weights = compute_num_blocks(config, 192 * 2**30, 10 * 2**30)
        n_large_weights = compute_num_blocks(config, 192 * 2**30, 100 * 2**30)
        assert n_small_weights > n_large_weights


# ── Hit Rate Tracking ──


class TestHitRateTracking:
    """Tests for hit rate tracking in KVCacheManager."""

    def test_initial_hit_rate_zero(self):
        """Fresh manager has 0.0 hit rate."""
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)
        assert mgr.hit_rate == 0.0
        assert mgr._total_lookups == 0
        assert mgr._total_hits == 0

    def test_hit_rate_after_miss(self):
        """First request (all misses) has 0.0 hit rate.

        The lookup loop breaks at the first miss (chain hash), so
        a fresh cache with 4-block prompt does 1 lookup (miss on first block).
        """
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)

        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)
        # Loop breaks at first miss: 1 lookup, 0 hits
        assert mgr._total_lookups == 1
        assert mgr._total_hits == 0
        assert mgr.hit_rate == 0.0
        mgr.free_request(table)

    def test_hit_rate_after_hit(self):
        """Second request with prefix hit updates hit rate.

        Request 1: 64 tokens (4 blocks), first hash misses -> 1 lookup, 0 hits.
        Request 2: 80 tokens (5 block hashes checked), first 4 hit then break
        on 5th which doesn't exist -> 5 lookups, 4 hits.
        Total: 6 lookups, 4 hits = 66.7% hit rate.
        """
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)

        tokens1 = list(range(64))
        t1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(t1, tokens1, model_hash=42)
        mgr.free_request(t1)

        # After request 1: 1 lookup (first hash miss, break), 0 hits
        assert mgr._total_lookups == 1
        assert mgr._total_hits == 0

        tokens2 = list(range(64)) + list(range(100, 116))
        t2, m2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert m2.num_matched_tokens == 64

        # Request 2: checks 5 hashes (4 hit + 1 miss on 5th), then break
        assert mgr._total_lookups == 1 + 5
        assert mgr._total_hits == 4
        assert abs(mgr.hit_rate - 4 / 6) < 0.001
        mgr.free_request(t2)


# ── EngineCore KV Cache Wiring ──


class TestEngineCoreKVWiring:
    """Tests for EngineCore KV cache configuration and wiring."""

    def test_paged_kv_disabled_by_default(self):
        """EngineCore without enable_paged_kv uses base Scheduler."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.scheduler import Scheduler

        config = EngineCoreConfig()
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        assert isinstance(core.scheduler, Scheduler)
        assert core._kv_manager is None
        assert core.get_kv_cache_stats() == {"enabled": False}

    def test_paged_kv_enabled_creates_paged_scheduler(self):
        """EngineCore with enable_paged_kv creates PagedScheduler."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.paged_scheduler import PagedScheduler

        config = EngineCoreConfig(
            enable_paged_kv=True,
            kv_block_size=16,
            kv_num_blocks=100,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        assert isinstance(core.scheduler, PagedScheduler)
        assert core._kv_manager is not None

    def test_kv_stats_with_paged_enabled(self):
        """get_kv_cache_stats returns meaningful stats when paged KV is on."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig

        config = EngineCoreConfig(
            enable_paged_kv=True,
            kv_block_size=16,
            kv_num_blocks=100,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        stats = core.get_kv_cache_stats()
        assert stats["enabled"] is True
        assert stats["block_size"] == 16
        assert stats["total_blocks"] == 100
        assert stats["free_blocks"] == 99  # -1 for null block
        assert "usage" in stats
        assert "hit_rate" in stats
        assert "active_block_tables" in stats

    def test_engine_core_config_arch_fields(self):
        """EngineCoreConfig has model architecture fields."""
        from yunshu_engine.engine_core import EngineCoreConfig

        config = EngineCoreConfig(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_num_blocks=1024,
        )
        assert config.num_layers == 32
        assert config.num_kv_heads == 8
        assert config.head_dim == 128
        assert config.kv_num_blocks == 1024


# ── BatchedEngine Model Architecture Extraction ──


class TestBatchedEngineArchExtraction:
    """Tests for BatchedEngine._extract_model_arch()."""

    def test_extracts_from_standard_config(self):
        """Extracts num_layers, num_kv_heads, head_dim from HuggingFace config."""
        from yunshu_engine.batched_engine import BatchedEngine

        class FakeConfig:
            num_hidden_layers = 32
            num_key_value_heads = 8
            hidden_size = 4096
            num_attention_heads = 32

        class FakeModel:
            config = FakeConfig()

        result = BatchedEngine._extract_model_arch(FakeModel())
        assert result["num_layers"] == 32
        assert result["num_kv_heads"] == 8
        assert result["head_dim"] == 128  # 4096 / 32

    def test_returns_empty_for_none_model(self):
        """Returns empty dict for None model."""
        from yunshu_engine.batched_engine import BatchedEngine

        result = BatchedEngine._extract_model_arch(None)
        assert result == {}

    def test_returns_empty_for_model_without_config(self):
        """Returns empty dict when model has no config."""
        from yunshu_engine.batched_engine import BatchedEngine

        class BareModel:
            pass

        result = BatchedEngine._extract_model_arch(BareModel())
        assert result == {}

    def test_returns_empty_for_incomplete_config(self):
        """Returns empty dict when config is missing required fields."""
        from yunshu_engine.batched_engine import BatchedEngine

        class PartialConfig:
            num_hidden_layers = 32
            # missing num_key_value_heads

        class PartialModel:
            config = PartialConfig()

        result = BatchedEngine._extract_model_arch(PartialModel())
        assert result == {}

    def test_handles_args_attribute(self):
        """Also checks model.args as fallback (some MLX models use args)."""
        from yunshu_engine.batched_engine import BatchedEngine

        class FakeArgs:
            num_hidden_layers = 24
            num_key_value_heads = 4
            hidden_size = 2048
            num_attention_heads = 16

        class FakeModel:
            args = FakeArgs()

        result = BatchedEngine._extract_model_arch(FakeModel())
        assert result["num_layers"] == 24
        assert result["num_kv_heads"] == 4
        assert result["head_dim"] == 128

    def test_batched_engine_kv_stats_no_engine(self):
        """get_kv_cache_stats returns prefix cache stats when no engine is loaded."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine()
        stats = engine.get_kv_cache_stats()
        assert "prefix_cache" in stats
        assert stats["prefix_cache"]["entries"] == 0


# ── Helper ──


def _fake_executor():
    """Create a fake executor that runs sync functions immediately."""
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=1)


# ── Memory Pressure Eviction ──


class TestMemoryPressureEviction:
    """Test C12: memory-pressure-driven KV cache eviction."""

    def test_evict_under_pressure_no_entries(self):
        """evict_under_pressure returns 0 when cache is empty."""
        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=64)
        evicted = cache.evict_under_pressure(threshold_pct=50.0)
        assert evicted == 0

    def test_evict_under_pressure_no_mlx_metal(self, monkeypatch):
        """evict_under_pressure returns 0 when device_info lacks working set size."""
        import mlx.core as mx

        monkeypatch.setattr(mx, "device_info", lambda: {})
        monkeypatch.setattr(mx, "get_active_memory", lambda: 0)

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=64)

        # Add a dummy entry
        tokens = mx.array([1, 2, 3, 4, 5] * 32)  # 160 tokens > min_prefix
        cache.add(tokens, [])

        evicted = cache.evict_under_pressure(threshold_pct=50.0)
        assert evicted == 0

    def test_evict_under_pressure_under_threshold(self, monkeypatch):
        """evict_under_pressure does not evict when utilization is below threshold."""
        import mlx.core as mx

        monkeypatch.setattr(
            mx,
            "device_info",
            lambda: {
                "max_recommended_working_set_size": 100_000_000,
            },
        )
        monkeypatch.setattr(mx, "get_active_memory", lambda: 50_000_000)  # 50% util

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=64)
        tokens = mx.array([1, 2, 3, 4, 5] * 32)
        cache.add(tokens, [])

        evicted = cache.evict_under_pressure(threshold_pct=85.0)
        assert evicted == 0
        assert cache.size == 1

    def test_evict_under_pressure_above_threshold(self, monkeypatch):
        """evict_under_pressure evicts LRU entries when utilization exceeds threshold."""
        import mlx.core as mx

        call_count = [0]

        def mock_get_active():
            # First call returns high utilization, subsequent calls drop
            call_count[0] += 1
            if call_count[0] <= 2:
                return 90_000_000  # 90% utilization
            return 70_000_000  # 70% — below threshold-5

        monkeypatch.setattr(
            mx,
            "device_info",
            lambda: {
                "max_recommended_working_set_size": 100_000_000,
            },
        )
        monkeypatch.setattr(mx, "get_active_memory", mock_get_active)
        monkeypatch.setattr(mx, "clear_cache", lambda: None)

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=64)

        # Add multiple entries
        for i in range(5):
            tokens = mx.array([i + 1] * 64)
            cache.add(tokens, [])

        assert cache.size == 5
        evicted = cache.evict_under_pressure(threshold_pct=85.0)
        assert evicted >= 1
        assert cache.size < 5

    def test_evict_under_pressure_max_evict_cap(self, monkeypatch):
        """evict_under_pressure caps eviction at 25% of entries per call."""
        import mlx.core as mx

        monkeypatch.setattr(
            mx,
            "device_info",
            lambda: {
                "max_recommended_working_set_size": 100_000_000,
            },
        )
        monkeypatch.setattr(mx, "get_active_memory", lambda: 90_000_000)
        monkeypatch.setattr(mx, "clear_cache", lambda: None)

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=64)

        # Add 20 entries
        for i in range(20):
            tokens = mx.array([i + 1] * 64)
            cache.add(tokens, [])

        evicted = cache.evict_under_pressure(threshold_pct=85.0)
        # Should evict at most 5 (25% of 20)
        assert evicted <= 5
        assert cache.size >= 15


class TestWarmTierClearAndHybridNoTrim:
    """594: clear() must reset _warm_flags; hybrid no_trim reuse."""

    @staticmethod
    def _mk_cache(n, layers=2):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        out = []
        for _ in range(layers):
            c = KVCache()
            c.keys = mx.zeros((1, 2, n, 4))
            c.values = mx.zeros((1, 2, n, 4))
            c.offset = n
            out.append(c)
        return out

    def test_clear_resets_warm_flags_no_crash(self):
        """clear() then add() with WARM enabled must not raise IndexError."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=8, hot_limit=1, min_prefix_length=32)
        for i in range(4):
            cache.add(mx.array([i + 1] * 64), self._mk_cache(64))
        assert len(cache._warm_flags) == len(cache._prompts)
        cache.clear()
        assert cache._warm_flags == []
        # Re-add after clear — pre-fix this raised IndexError in _maybe_demote.
        for i in range(4):
            cache.add(mx.array([i + 10] * 64), self._mk_cache(64))
        assert len(cache._warm_flags) == len(cache._prompts)

    def test_no_trim_mode_returns_only_full_prefix(self):
        """no_trim mode reuses an entry only when the WHOLE entry is a prefix
        of the query (trim=0); a partial match must be rejected."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=16, min_prefix_length=32)
        cache._no_trim_mode = True
        base = list(range(1, 257))  # 256-token boundary prefix
        cache.add(mx.array(base), self._mk_cache(256))
        # Query that EXTENDS the stored prefix → trim=0 full-prefix reuse.
        q_ext = mx.array(base + [900, 901, 902])
        snap, remaining, matched = cache.get(q_ext)
        assert matched == 256
        assert remaining == 3
        assert snap is not None
        # Query that DIVERGES inside the stored prefix → no reuse (would trim).
        q_div = mx.array(base[:200] + [777] * 100)
        snap2, _, matched2 = cache.get(q_div)
        assert snap2 is None
        assert matched2 == 0

    def test_hash_chain_strict_prefix_multiturn_no_crash(self):
        """a longer query whose FULL cached prefix is a strict prefix
        (the canonical multi-turn / shared-system-prompt case) drove the hash-chain
        collision guard at kv_prefix_cache.py, where a raw `mx.array != mx.array`
        inside an `if` raised ``ValueError: Only length-1 arrays ...``. get() has no
        try/except, so it propagated out — and via the engine call site it was
        caught and silently fell back to a full prefill, defeating the cache on its
        primary use case. Must return the hit, not raise."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=16, min_prefix_length=32)
        base = list(range(1, 129))  # 128 tokens = 2 full blocks (turn-1 context)
        cache.add(mx.array(base), self._mk_cache(128))
        # turn-2: identical 128-token prefix + 50 new tokens → hash-chain hit with
        # best_length == len(cached) < len(query) → the crashing guard branch.
        query = mx.array(base + list(range(1000, 1050)))
        snap, remaining, matched = cache.get(query)  # pre-fix: raised ValueError
        assert snap is not None, "strict-prefix multi-turn hit lost to crash-fallback"
        assert matched == 128
        assert remaining == 50


class TestSlidingWindowSSDBypass:
    """: gemma-4 (sliding-window RotatingKVCache) was routed through the
    whole-snapshot hybrid SSD tier (built for Qwen3.5 linear-attn), spilling
    923 MB / restoring 0 every run. The two static classifiers and the eviction
    spill gate must: (1) keep recurrent ArraysCache → hybrid-snapshot SSD;
    (2) keep plain KVCache → per-block SSD; (3) skip BOTH for RotatingKVCache."""

    @staticmethod
    def _kvcache(n=64, layers=2):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        out = []
        for _ in range(layers):
            c = KVCache()
            c.keys = mx.zeros((1, 2, n, 4))
            c.values = mx.zeros((1, 2, n, 4))
            c.offset = n
            out.append(c)
        return out

    @staticmethod
    def _rotating(n=64, layers=2):
        """A RotatingKVCache-like layer: real .keys plus the rotating-buffer
        attrs (max_size / _idx) that distinguish it from a plain KVCache."""
        import mlx.core as mx

        class _Rot:
            def __init__(self):
                self.keys = mx.zeros((1, 2, n, 4))
                self.values = mx.zeros((1, 2, n, 4))
                self.offset = n
                self.max_size = 32
                self.keep = 0
                self._idx = n

        return [_Rot() for _ in range(layers)]

    @staticmethod
    def _recurrent(layers=2):
        """An ArraysCache-like recurrent layer: no per-token .keys tensor."""

        class _Arr:
            keys = None
            values = None

        return [_Arr() for _ in range(layers)]

    def test_classifiers(self):
        from yunshu_engine.kv_prefix_cache import KVPrefixCache as K

        # plain KVCache: block-decomposable, NOT recurrent
        kv = self._kvcache()
        assert K._is_block_decomposable(kv) is True
        assert K._has_recurrent_layer(kv) is False
        # sliding-window RotatingKVCache: NEITHER (in-RAM reuse only)
        rot = self._rotating()
        assert K._is_block_decomposable(rot) is False
        assert K._has_recurrent_layer(rot) is False
        # recurrent ArraysCache: recurrent, NOT block-decomposable
        rec = self._recurrent()
        assert K._is_block_decomposable(rec) is False
        assert K._has_recurrent_layer(rec) is True
        assert K._is_block_decomposable([]) is False

    def _spy_caches(self):
        saved_blocks, saved_snaps = [], []

        class _SSD:
            def has_block(self, h):
                return False

            def save_block(self, **kw):
                saved_blocks.append(kw)

        class _Hybrid:
            def has(self, k):
                return False

            def save(self, k, c, n):
                saved_snaps.append((k, n))

        return _SSD(), _Hybrid(), saved_blocks, saved_snaps

    def test_rotating_spills_nothing(self):
        """Evicting a sliding-window entry writes to NEITHER SSD tier."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        c = KVPrefixCache(max_entries=16, min_prefix_length=1)
        ssd, hyb, blocks, snaps = self._spy_caches()
        c._ssd_cache, c._hybrid_ssd, c._no_trim_mode = ssd, hyb, True
        c.add(mx.array(list(range(1, 65))), self._rotating(64))
        c._remove_entry(0)
        assert blocks == [], "RotatingKVCache must not spill per-block"
        assert snaps == [], "RotatingKVCache must not spill whole-snapshot"

    def test_recurrent_spills_whole_snapshot(self):
        """A truly-recurrent entry still uses the hybrid whole-snapshot tier."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        c = KVPrefixCache(max_entries=16, min_prefix_length=1)
        ssd, hyb, blocks, snaps = self._spy_caches()
        c._ssd_cache, c._hybrid_ssd, c._no_trim_mode = ssd, hyb, True
        c.add(mx.array(list(range(1, 65))), self._recurrent())
        c._remove_entry(0)
        assert len(snaps) == 1, "recurrent must spill the whole snapshot"
        assert blocks == []

    def test_plain_kvcache_spills_per_block(self):
        """A plain KVCache entry still uses the per-block SSD tier."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        c = KVPrefixCache(max_entries=16, min_prefix_length=1)
        ssd, hyb, blocks, snaps = self._spy_caches()
        c._ssd_cache, c._hybrid_ssd = ssd, hyb  # no_trim_mode False (trimmable)
        c.add(mx.array(list(range(1, 65))), self._kvcache(64))
        c._remove_entry(0)
        assert len(blocks) >= 1, "plain KVCache must spill per-block"
        assert snaps == []


class TestScopedSSDDir:
    """the SSD cache dir is namespaced per model so two models that
    share a prompt prefix never collide on content-hash-keyed KV blocks on disk
    (cross-model KV corruption fix)."""

    def _scoped(self, cache_dir, model):
        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        return KVPrefixCache._scoped_ssd_dir(cache_dir, model)

    def test_different_models_get_different_dirs(self):
        a = self._scoped("/tmp/kv", "Qwen2.5-0.5B")
        b = self._scoped("/tmp/kv", "gemma-4-27b")
        assert a != b
        # Both live under the same base.
        import os

        assert os.path.dirname(a) == os.path.dirname(b) == os.path.expanduser("/tmp/kv")

    def test_same_model_is_stable(self):
        assert self._scoped("/tmp/kv", "Qwen2.5-0.5B") == self._scoped(
            "/tmp/kv", "Qwen2.5-0.5B"
        )

    def test_empty_model_uses_base(self):
        import os

        assert self._scoped("/tmp/kv", "") == os.path.expanduser("/tmp/kv")

    def test_names_sanitizing_identically_still_disambiguated(self):
        # Two HF ids that sanitize to the same safe suffix must differ via digest.
        a = self._scoped("/tmp/kv", "org/model")
        b = self._scoped("/tmp/kv", "org-model")
        assert a != b

    def test_path_like_model_id_is_filesystem_safe(self):
        d = self._scoped("/tmp/kv", "/Volumes/P5/models/Qwen2.5-0.5B-4bit")
        # No stray path separators from the model id leak into the leaf name.
        import os

        leaf = os.path.basename(d)
        assert "/" not in leaf and leaf
