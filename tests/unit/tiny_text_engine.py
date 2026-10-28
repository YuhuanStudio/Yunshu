"""A real BatchedEngine text fast path over a tiny random Llama (CPU, no download).

Lets a unit test run ``_generate_fast`` / ``_stream_generate_fast`` end to end
(mlx-lm ``generate_step``, logits processors, sampler, logprobs, prefix tracker)
and compare outputs before/after a host-side change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

VOCAB = 64


class FakeTokenizer:
    bos_token = None
    bos_token_id = None
    eos_token_id = 1
    eos_token_ids = {1}
    vocab_size = VOCAB

    def encode(self, text, add_special_tokens=True, **kw):
        return [2 + (ord(c) % 60) for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)

    def apply_chat_template(
        self, msgs, tokenize=False, add_generation_prompt=True, **kw
    ):
        return "".join(m["content"] for m in msgs)

    @property
    def detokenizer(self):
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

        return NaiveStreamingDetokenizer(self)

    def __len__(self):
        return VOCAB


def make_engine(seed: int = 0):
    import mlx.core as mx
    from mlx_lm.models import llama

    from yunshu_engine.batched_engine import BatchedEngine

    mx.random.seed(seed)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=VOCAB,
        max_position_embeddings=512,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    eng = BatchedEngine(model_name="tiny")
    eng._loaded = True
    eng._model = model
    eng._tokenizer = FakeTokenizer()
    eng._kv_prefix_cache = MagicMock()
    eng._kv_prefix_cache.get.return_value = (None, None, 0)
    eng._check_memory_guard = lambda p, mt, **kw: (None, mt)
    return eng


async def collect_stream(eng, **kw):
    """Run ``_stream_generate_fast``: (text, final GenerationOutput). Per-token logprob
    entries of every chunk are gathered on ``final.logprobs`` (the engine yields them
    chunk by chunk)."""
    text, last, lps = "", None, []
    async for out in eng._stream_generate_fast(**kw):
        text += out.new_text or ""
        lps.extend(out.logprobs or [])
        last = out
    last.logprobs = lps or None
    return text, last
