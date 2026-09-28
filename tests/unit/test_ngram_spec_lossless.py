"""The n-gram speculative path must be lossless for greedy decoding.

Runs the real ``BatchedEngine._generate_ngram_spec`` on a tiny random llama
(no checkpoint) whose greedy continuation repeats, so drafts are proposed and
accepted, and compares the tokens with plain greedy ``generate_step``. This is
the evidence behind routing ``spec_decode=true`` / ``YUNSHU_NGRAM_DEFAULT=1``
to the n-gram path (an older base loop duplicated/dropped tokens).
"""

import asyncio

import mlx.core as mx
import pytest

mlx_lm = pytest.importorskip("mlx_lm")

from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models import llama  # noqa: E402

from yunshu_engine.batched_engine import BatchedEngine  # noqa: E402
from yunshu_engine.ngram_proposer import NgramConfig, NgramProposer  # noqa: E402
from yunshu_engine.spec_draft_verifier import SpecDraftVerifier  # noqa: E402

VOCAB = 48


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
    def text(self):
        return self._tok.decode(self.tokens)

    @property
    def last_segment(self):
        full = self.text
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


def _tiny_model(seed):
    mx.random.seed(seed)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=VOCAB,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


def _engine(model):
    eng = object.__new__(BatchedEngine)
    eng._model = model
    eng._tokenizer = _Tok()
    eng.model_name = "tiny-llama"
    eng._ngram_proposer = NgramProposer(NgramConfig(max_n=3, k=4, mode="lps"))
    eng._adaptive_spec = None
    eng._kv_manager = None
    eng._kv_prefix_cache = None
    eng._kv_quant_bits = None
    eng._kv_quant_group_size = 64
    eng._kv_quant_start = 0
    eng._lora_manager = None
    eng._mem_pressure_threshold = None
    eng._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}
    eng._spec_draft_verifier = SpecDraftVerifier(track_stats=True)
    return eng


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ngram_spec_matches_plain_greedy(seed):
    model = _tiny_model(seed)
    tok = _Tok()
    prompt = "abcabcabcabc"
    ids = tok.encode(prompt)
    n = 48
    ref = []
    for token, _ in generate_step(
        mx.array(ids), model, max_tokens=n, sampler=lambda x: mx.argmax(x, axis=-1)
    ):
        t = int(token)
        if t == tok.eos_token_id:
            break
        ref.append(t)
    eng = _engine(model)
    out = asyncio.run(
        eng._generate_ngram_spec(prompt=prompt, max_tokens=n, temperature=0.0)
    )
    assert out.error is None
    assert out.text == tok.decode(ref)
    # The path really speculated (otherwise this proves nothing).
    assert eng._ngram_stats["accepted"] > 0, eng._ngram_stats
