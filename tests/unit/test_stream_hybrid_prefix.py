"""Streaming hybrid (recurrent + attention) models reuse prefixes losslessly.

Drives the real ``BatchedEngine._stream_generate_fast`` on a tiny random
``qwen3_next`` model (ArraysCache + KVCache, not trimmable) with a real
``KVPrefixCache``: the second request must hit a trim=0 boundary snapshot,
stream the same text as a cache-less run, and no post-generation entry (key
longer than the cache) may be stored.
"""

import asyncio

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm")

from mlx_lm.models import qwen3_next  # noqa: E402

from yunshu_engine.batched_engine import BatchedEngine  # noqa: E402
from yunshu_engine.kv_prefix_cache import KVPrefixCache, cache_length  # noqa: E402

VOCAB = 64
BLOCK = 16


class _Detok:
    def __init__(self, tok):
        self._tok = tok
        self.reset()

    def reset(self):
        self.tokens = []
        self._emitted = 0

    def add_token(self, t):
        self.tokens.append(int(t))

    def finalize(self):
        pass

    @property
    def last_segment(self):
        full = self._tok.decode(self.tokens)
        seg = full[self._emitted :]
        self._emitted = len(full)
        return seg


class _Tok:
    eos_token_id = VOCAB - 1

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(c) % (VOCAB - 4)) for c in text]

    def decode(self, ids, **_):
        return "".join(f"<{i}>" for i in ids)

    @property
    def detokenizer(self):
        return _Detok(self)


def _model():
    mx.random.seed(0)
    args = qwen3_next.ModelArgs(
        model_type="qwen3_next",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=64,
        mlp_only_layers=[0, 1, 2, 3],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=512,
        head_dim=16,
        full_attention_interval=2,
    )
    m = qwen3_next.Model(args)
    mx.eval(m.parameters())
    return m


def _engine(model, cache):
    eng = object.__new__(BatchedEngine)
    eng._model = model
    eng._tokenizer = _Tok()
    eng.model_name = "tiny-hybrid"
    eng._adaptive_spec = None
    eng._kv_manager = None
    eng._kv_prefix_cache = cache
    eng._hybrid_prefix_block = BLOCK
    eng._kv_quant_bits = None
    eng._kv_quant_group_size = 64
    eng._kv_quant_start = 0
    eng._lora_manager = None
    eng._mem_pressure_threshold = 0
    eng._lookahead_reasoning = None
    eng._preprocessor_registry = None
    return eng


def _stream(eng, prompt):
    async def run():
        text, last = "", None
        async for out in eng._stream_generate_fast(
            prompt=prompt, max_tokens=12, temperature=0.0
        ):
            text += out.text or ""
            last = out
        return text, last

    r = asyncio.run(run())
    assert r[1].finish_reason != "error", r[1].error
    return r


def test_streaming_hybrid_reuses_prefix_losslessly():
    model = _model()
    p1 = "".join(chr(97 + i % 13) for i in range(70))
    p2 = p1 + "xyzxyz"
    assert len(_Tok().encode(p1)) >= 2 * BLOCK

    cache = KVPrefixCache(min_prefix_length=BLOCK)
    eng = _engine(model, cache)
    _stream(eng, p1)
    text2, out2 = _stream(eng, p2)

    ref_text, _ = _stream(_engine(model, None), p2)

    assert out2.cached_tokens > 0, out2
    assert text2 == ref_text
    assert cache._prompts
    for key, kv in zip(cache._prompts, cache._caches, strict=True):
        assert len(key) == cache_length(kv)
