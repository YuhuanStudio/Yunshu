"""Speculative (MTP) greedy output == plain greedy output, bit-exact, with the
speculative lane's decode and verify attention on the ragged kernels.

Small real model (Qwen3.5-0.8B with its MTP head), quantized to 4 bits in
memory so the batch-invariant projections apply; ``ragged_kv.set_dense_lane``
routes decode (T=1) and verify (T<=8) attention through the same per-row
kernel. Prompts: a short one and one with >1024 tokens of context (past MLX's
first SDPA plan switch). Point ``YUNSHU_PARITY_MODEL`` at another Qwen3.5
checkpoint with an MTP head to run elsewhere.

    uv run pytest tests/integration/test_ragged_spec_parity.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

MODEL = Path(
    os.environ.get(
        "YUNSHU_PARITY_MODEL", "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16"
    )
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not mx.metal.is_available(), reason="needs an Apple GPU"),
    pytest.mark.skipif(not (MODEL / "config.json").exists(), reason=f"no {MODEL}"),
]

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(4000)
)


def _drafter(path: Path):
    """The checkpoint's MTP head: from the shards when indexed, else from a
    separate ``mtp-weights.safetensors``."""
    import yunshu_engine.mlxvlm_mtp as m

    extra = path / "mtp-weights.safetensors"
    if not extra.exists():
        return m._load_drafter_in_memory(str(path))
    weights = mx.load(str(extra))
    orig = m._load_mtp_head_tensors
    m._load_mtp_head_tensors = lambda _p: {
        k.removeprefix("mtp."): v for k, v in weights.items()
    }
    try:
        return m._load_drafter_in_memory(str(path))
    finally:
        m._load_mtp_head_tensors = orig


@pytest.fixture(scope="module")
def engine():
    import mlx.nn as nn
    from mlx_vlm import load

    from yunshu_engine.kernels import batch_invariant, omlx, ragged_kv

    omlx.apply()
    model, processor = load(str(MODEL))
    lm = model.language_model
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
    try:
        yield model, processor, _drafter(MODEL)
    finally:
        ragged_kv.set_dense_lane(False)
        batch_invariant.set_active(False)


def _generate(engine, prompt: str, max_tokens: int, block: int) -> list[int]:
    from mlx_vlm.generate.ar import BatchGenerator

    from yunshu_engine.mrope import clear_rope_state

    model, processor, drafter = engine
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
        draft_kind="mtp" if block else None,
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
def test_mtp_greedy_equals_plain_greedy(engine, lane_widths, context):
    tok = engine[1].tokenizer
    prompt = "Explain in detail how a refrigerator works."
    if context:
        filler = tok.encode(FILLER, add_special_tokens=False)[:context]
        prompt = tok.decode(filler) + "\n\nSummarize the readings above, then " + prompt
    plain = _generate(engine, prompt, 96, 0)
    assert len(plain) > 16 and set(lane_widths) == {1}
    for block in (3, 6):
        lane_widths.clear()
        spec = _generate(engine, prompt, 96, block)
        assert max(lane_widths) > 1  # verify ran on the lane kernel
        first = next(
            (i for i, (a, b) in enumerate(zip(spec, plain, strict=False)) if a != b),
            None,
        )
        assert spec == plain, f"block {block}: first difference at token {first}"
