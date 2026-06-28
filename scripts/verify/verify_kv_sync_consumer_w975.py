"""W975 — verify the mesh KV-transfer receiver delivers blocks to a decode-side consumer.

The audit found the KVTransferServer receiver was a "hollow consumer": it held no engine
reference and the KVCacheManager had no load_kv_blocks / _kv_layers target, so received
cross-node KV blocks were discarded ("loaded 0 of N blocks") and the decode node re-prefilled.
W975 adds set_block_consumer(fn) — the decode engine registers a callback that reconstructs
the received KV into ITS reusable cache (via load_kv_blocks_into_cache, the W970-proven
primitive), keeping the mesh server engine-agnostic.

This drives the REAL client→server network path over a loopback socket and proves: the
consumer fires with the received blocks (layer_data intact), and the transfer reports the
consumer's REAL loaded count honestly (COMPLETED when >0, FAILED when the consumer loads 0).
Combined with W970 (received blocks → reconstructed cache → generate_with_kv == greedy), this
closes the cross-node KV-reuse path at the transport level. Synthetic blocks (no model) — fast.

Run:  PYTHONPATH=. uv run python scripts/verify/verify_kv_sync_consumer_w975.py
"""
import asyncio
import socket

from yunshu_engine.kv_transfer import (
    KVBlockData,
    KVTransferClient,
    KVTransferConfig,
    KVTransferServer,
    TransferStatus,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _blocks():
    return [
        KVBlockData(block_hash=0xABCDEF, token_count=3, layer_data={0: b"k0v0data", 1: b"k1v1"}),
        KVBlockData(block_hash=0x123456, token_count=2, layer_data={0: b"blk2"}),
    ]


async def _round_trip(consumer):
    port = _free_port()
    server = KVTransferServer(KVTransferConfig(enabled=True, listen_port=port))
    if consumer is not None:
        server.set_block_consumer(consumer)
    await server.start()
    try:
        client = KVTransferClient(KVTransferConfig(
            enabled=True, remote_host="127.0.0.1", remote_port=port,
        ))
        res = await client.send_blocks(_blocks(), model_name="m", total_tokens=5, layer_count=2)
        return res
    finally:
        await server.stop()


async def main():
    # ── 1. consumer fires with the received blocks; transfer reports its real count ──
    seen = {}

    def consumer(blocks, model_name):
        seen["blocks"] = blocks
        seen["model"] = model_name
        return len(blocks)  # "loaded all"

    res = await _round_trip(consumer)
    assert res.status == TransferStatus.COMPLETED, res.status
    assert res.blocks_transferred == 2, f"reported {res.blocks_transferred}, expected 2"
    assert seen.get("model") == "m", seen
    got = seen.get("blocks") or []
    assert len(got) == 2, f"consumer got {len(got)} blocks"
    # layer_data survived the wire intact
    assert got[0].layer_data == {0: b"k0v0data", 1: b"k1v1"}, got[0].layer_data
    assert got[1].layer_data == {0: b"blk2"}, got[1].layer_data
    print(f"consumer fired: {len(got)} blocks, model={seen['model']!r}, "
          f"reported COMPLETED/{res.blocks_transferred}")

    # ── 2. a consumer that loads 0 is reported HONESTLY as FAILED (no silent re-prefill) ──
    # The server NAKs (ack=0) because nothing loaded, so the client sees a rejected transfer
    # rather than a false success — exactly the W846/W975 honesty contract (no silent
    # re-prefill while stats over-report success).
    res0 = await _round_trip(lambda blocks, model_name: 0)
    assert res0.status == TransferStatus.FAILED, f"loaded-0 must FAIL, got {res0.status}"
    assert res0.error, "a failed transfer must carry an error"
    print(f"loaded-0 honestly reported FAILED: {res0.error!r}")

    # ── 3. NO consumer = pure relay (wire OK, reported as transferred — pre-W975 behavior) ──
    res_relay = await _round_trip(None)
    assert res_relay.status == TransferStatus.COMPLETED, res_relay.status
    print(f"no-consumer relay still COMPLETED ({res_relay.blocks_transferred} blocks)")

    print("\nW975 mesh KV-transfer consumer wiring (blocks→consumer over real socket, honest "
          "report): PASS")


if __name__ == "__main__":
    asyncio.run(main())
