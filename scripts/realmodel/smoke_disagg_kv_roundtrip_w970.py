"""W970 — prove disaggregated KV reuse survives SERIALIZATION (the cross-node primitive).

The disagg /prefill→/decode split only delivers value if the prefilled KV cache can cross a
network and still be reused on the decode node WITHOUT re-prefilling. The audit found the HTTP
wire never serializes KV (decode re-prefills). Before wiring the wire, prove the foundation:

  live KV cache --extract_kv_blocks_from_cache--> bytes (the wire) --load_kv_blocks_into_cache-->
  fresh cache --generate_with_kv--> text   ==   reference greedy text

If this holds, serialized KV is genuinely reusable and lossless (the cross-node primitive is
real; the rest is plumbing). If it diverges, THAT is the real bug. Runs single-process with a
tiny real model — loopback, no second machine needed.

Run:  PYTHONPATH=. uv run python scripts/realmodel/smoke_disagg_kv_roundtrip_w970.py
"""
import asyncio

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_engine.external_prefill import ExternalPrefiller
    from yunshu_engine.kv_transfer import (
        extract_kv_blocks_from_cache,
        load_kv_blocks_into_cache,
        _tensor_to_bytes,
        _bytes_to_tensor,
    )
    from mlx_lm.models.cache import make_prompt_cache

    eng = BatchedEngine(MODEL)
    await eng.start()
    tok = eng._tokenizer
    prompt = "The quick brown fox jumps over"
    ids = tok.encode(prompt)

    # ── Prefill node: produce live KV cache + first-token logits ──
    pf = ExternalPrefiller(eng._model, tok)
    res = pf.prefill(ids)
    assert res.kv_cache is not None and res.last_logits is not None, "prefill must produce cache+logits"

    # ── Serialize across the "wire" (exactly what would be sent to a decode node) ──
    blocks = extract_kv_blocks_from_cache(res.kv_cache, ids)
    logits_bytes = _tensor_to_bytes(res.last_logits)
    n_blocks = len(blocks)
    total_bytes = sum(sum(len(b) for b in blk.layer_data.values()) for blk in blocks) + len(logits_bytes)
    print(f"serialized: {n_blocks} block(s), {total_bytes} bytes over the wire")
    assert n_blocks > 0, "extract produced no blocks — serialization is broken"

    # ── Decode node: reconstruct a FRESH cache from the bytes (no access to the live cache) ──
    recon_cache = make_prompt_cache(eng._model)
    loaded = load_kv_blocks_into_cache(recon_cache, blocks)
    recon_logits = _bytes_to_tensor(logits_bytes)
    print(f"reconstructed: loaded {loaded} block(s) into a fresh cache")
    assert loaded == n_blocks, f"loaded {loaded} of {n_blocks} blocks — KV did not survive the wire"

    # ── Decode from the RECONSTRUCTED cache (this is what a real decode node would do) ──
    out_recon = await eng.generate_with_kv(
        recon_cache, ids, recon_logits, max_tokens=24, temperature=0.0,
    )

    # ── Reference: plain greedy generate from the raw prompt (re-prefill path) ──
    ref = await eng.generate(prompt, max_tokens=24, temperature=0.0)

    print("[reconstructed-KV reuse]:", repr(out_recon.text[:90]))
    print("[reference greedy]      :", repr(ref.text[:90]))

    assert out_recon.text == ref.text, (
        "SERIALIZED-KV reuse diverged from reference greedy — the cross-node KV primitive is "
        f"LOSSY/broken:\n  recon: {out_recon.text!r}\n  ref:   {ref.text!r}"
    )
    print("primitive round-trip: PASS")

    # ── END-TO-END through the ACTUAL prefill WIRE (_serialize/_deserialize_prefill_result) ──
    # This is exactly what crosses the network between a remote prefill node and the decode
    # node. Prove the wire carries reusable KV (the W970 disagg fix), not the old no-op.
    from yunshu_engine.external_prefill import (
        PrefillResult, _serialize_prefill_result, _deserialize_prefill_result,
    )
    res2 = pf.prefill(ids)  # fresh prefill (the previous cache was consumed by the round-trip)
    pr = PrefillResult(token_ids=ids, num_tokens=len(ids), kv_cache=res2.kv_cache,
                       last_logits=res2.last_logits)
    wire = _serialize_prefill_result(pr)
    got = _deserialize_prefill_result(wire)  # decode node receives this
    assert got.kv_blocks, "WIRE DROPPED KV — the disagg no-op is NOT fixed"
    assert got.last_logits is not None, "wire dropped first-token logits"
    print(f"wire carried {len(wire)} bytes incl. KV blocks + logits")
    wire_cache = make_prompt_cache(eng._model)
    assert load_kv_blocks_into_cache(wire_cache, got.kv_blocks) > 0
    out_wire = await eng.generate_with_kv(wire_cache, got.token_ids, got.last_logits,
                                          max_tokens=24, temperature=0.0)
    print("[wire-reconstructed reuse]:", repr(out_wire.text[:90]))
    assert out_wire.text == ref.text, (
        "decode through the real prefill WIRE diverged from reference — disagg fix is broken:\n"
        f"  wire: {out_wire.text!r}\n  ref:  {ref.text!r}"
    )
    await eng.stop()
    print("\nW970 disagg KV reuse (primitive + real /prefill wire → decode reuse == greedy): PASS")


if __name__ == "__main__":
    asyncio.run(main())
