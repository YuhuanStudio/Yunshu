"""DFlash speculative greedy output == plain greedy output, bit-exact, on the
serving kernels (batch-invariant verify + ragged lane attention).

There is no DFlash drafter for a small Qwen3.5, so the drafter is a randomly
initialized DFlash2 model sized for Qwen3.5-0.8B (quantized to 4 bits in
memory so the batch-invariant projections apply) whose proposals are replaced
by an *oracle*: the plain run's next tokens with one token corrupted in most
rounds. That drives the real DFlash round loop (context capture, window trim,
verify with captured layers, rollback of KV and GDN state) through full,
partial and zero acceptance, which a random drafter alone would not.

    uv run pytest tests/integration/test_dflash_spec_parity.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

# A local Qwen3.5 checkpoint (e.g. Qwen3.5-0.8B-MLX-bf16); the test skips
# when YUNSHU_PARITY_MODEL is unset.
MODEL = Path(os.environ.get("YUNSHU_PARITY_MODEL", "")).expanduser()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not mx.metal.is_available(), reason="needs an Apple GPU"),
    pytest.mark.skipif(
        not (MODEL / "config.json").exists(),
        reason="set YUNSHU_PARITY_MODEL to a local Qwen3.5 checkpoint directory",
    ),
]

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(4000)
)


def _random_dflash2(target_config) -> object:
    import mlx.nn as nn
    from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

    layers = int(target_config.num_hidden_layers)
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": int(target_config.hidden_size),
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": int(target_config.vocab_size),
            "layer_types": ["sliding_attention", "sliding_attention"],
            # A small window so prompts > 64 tokens exercise the context trim.
            "sliding_window": 65,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_target_layers": layers,
            "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "dflash_config": {
                "block_size": 8,
                "mask_token_id": 248070,
                "target_layer_ids": [2, layers // 2, layers - 3],
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 32,
                "selector_top_k": 16,
            },
        }
    )
    drafter = DFlash2DraftModel(cfg)
    mx.random.seed(0)
    drafter.update(
        nn.utils.tree_map(
            lambda p: mx.random.normal(p.shape, dtype=mx.bfloat16) * 0.02,
            drafter.parameters(),
        )
    )
    return drafter


class _Oracle:
    """Replace the drafter's proposals with the plain run's tokens, corrupting
    one position in two of every three rounds."""

    def __init__(self, drafter):
        self.drafter = drafter
        self.real = drafter.draft_block_greedy
        self.ref: list[int] = []
        self.pos: int | None = None
        self.round = 0
        self.accepted_rounds = 0
        drafter.draft_block_greedy = self

    def reset(self, ref: list[int]) -> None:
        self.ref, self.pos, self.round = ref, None, 0

    def __call__(self, last_bonus, hidden, cache, block_size, sampler, token_dtype):
        real = self.real(last_bonus, hidden, cache, block_size, sampler, token_dtype)
        mx.eval(real)  # the real drafter still runs (context path under test)
        self.pos = 1 if self.pos is None else self.pos + int(hidden.shape[1])
        n = int(block_size) - 1
        draft = [
            self.ref[min(self.pos + i, len(self.ref) - 1)] if self.ref else 0
            for i in range(n)
        ]
        if n and self.round % 3:
            k = self.round % n
            draft[k] = (draft[k] + 1) % 248000
        self.round += 1
        return mx.array([draft], dtype=token_dtype)


@pytest.fixture(scope="module")
def engine():
    import mlx.nn as nn
    from mlx_vlm import load

    from yunshu_engine import dflash_context
    from yunshu_engine.kernels import batch_invariant, omlx, ragged_kv

    omlx.apply()
    model, processor = load(str(MODEL))
    lm = model.language_model
    dflash_context.install(lm)
    nn.quantize(
        lm,
        group_size=64,
        bits=4,
        class_predicate=lambda _p, mod: (
            isinstance(mod, nn.Linear) and mod.weight.shape[-1] % 64 == 0
        ),
    )
    batch_invariant.install(lm, model=model, packed=False)
    batch_invariant.set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    drafter = _random_dflash2(lm.args)
    oracle = _Oracle(drafter)
    try:
        yield model, processor, drafter, oracle
    finally:
        ragged_kv.set_dense_lane(False)
        batch_invariant.set_active(False)


def _generate(engine, prompt: str, max_tokens: int, block: int) -> list[int]:
    from mlx_vlm.generate.ar import BatchGenerator

    from yunshu_engine.mrope import clear_rope_state

    model, processor, drafter, _ = engine
    tok = processor.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(text, add_special_tokens=False)
    gen = BatchGenerator(
        model.language_model,
        processor,
        max_tokens=max_tokens,
        draft_model=drafter if block else None,
        draft_kind="dflash" if block else None,
        draft_block_size=block or None,
        greedy_sampling=True,
        compute_logprobs=False,
    )
    clear_rope_state(model)
    kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
    (uid,) = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])
    out: list[int] = []
    try:
        while True:
            _, responses = gen.next()
            done = False
            for r in responses:
                if r.uid != uid:
                    continue
                if r.token is not None:
                    out.append(int(r.token))
                done = done or r.finish_reason is not None
            if done:
                return out
    finally:
        gen.close()


@pytest.fixture
def lane_widths(monkeypatch):
    """Query widths the lane kernel served (proof the routing engaged)."""
    from yunshu_engine.kernels import ragged_kv

    seen: list[int] = []
    orig = ragged_kv.dense_lane_attention

    def spy(queries, cache, scale):
        out = orig(queries, cache, scale)
        if out is not None:
            seen.append(int(queries.shape[2]))
        return out

    monkeypatch.setattr(ragged_kv, "dense_lane_attention", spy)
    return seen


@pytest.mark.parametrize("context", [0, 1500])
def test_dflash_greedy_equals_plain_greedy(engine, lane_widths, context):
    tok = engine[1].tokenizer
    oracle = engine[3]
    prompt = "Explain in detail how a refrigerator works."
    if context:
        filler = tok.encode(FILLER, add_special_tokens=False)[:context]
        prompt = tok.decode(filler) + "\n\nSummarize the readings above, then " + prompt
    plain = _generate(engine, prompt, 96, 0)
    assert len(plain) > 16
    for block in (4, 8):
        oracle.reset(plain)
        lane_widths.clear()
        spec = _generate(engine, prompt, 96, block)
        assert oracle.round > 3  # the DFlash round loop ran
        assert max(lane_widths) > 1  # verify ran on the lane kernel
        first = next(
            (i for i, (a, b) in enumerate(zip(spec, plain, strict=False)) if a != b),
            None,
        )
        assert spec == plain, f"block {block}: first difference at token {first}"
