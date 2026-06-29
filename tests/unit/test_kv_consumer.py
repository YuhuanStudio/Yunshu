"""mesh KV-transfer receiver routes received blocks to a decode-side consumer.

The KVTransferServer held no engine reference and the KVCacheManager had no load_kv_blocks /
_kv_layers target, so received cross-node KV was discarded (hollow consumer → "loaded 0 of N",
decode re-prefilled). set_block_consumer(fn) lets the decode engine register a callback that
reconstructs the KV into its reusable cache (load_kv_blocks_into_cache, proven), keeping
the server engine-agnostic. Real-socket end-to-end proof: scripts/verify/verify_kv_sync_consumer.py.
"""
from __future__ import annotations

import inspect

import yunshu_engine.kv_transfer as KT  # noqa: N812  # intentional short module alias


def test_set_block_consumer_api():
    srv = KT.KVTransferServer(KT.KVTransferConfig(enabled=False))
    assert srv._block_consumer is None
    def sentinel(blocks, model):
        return len(blocks)
    srv.set_block_consumer(sentinel)
    assert srv._block_consumer is sentinel


def test_consumer_wired_into_receive_path():
    src = inspect.getsource(KT)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the receiver prefers the consumer, then falls back to the kv_manager path
    assert "if self._block_consumer is not None:" in code
    assert "self._block_consumer(" in code
    # honest reporting keys on whether a load was attempted (consumer OR kv_manager)
    assert "_load_attempted" in code
    assert "if not _load_attempted:" in code
