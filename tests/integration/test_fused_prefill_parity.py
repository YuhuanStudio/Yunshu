"""Fused chunked prefill leaves greedy output unchanged (small real model).

Qwen3.5-0.8B cast to float32 (stock KV caches): two rows decode while a ~1.7K
token prompt prefills, once with separate prefill forwards and once fused at a
64-token budget, on one thread with the same schedule. Every request's greedy
tokens must match (in bf16 the packed matmul shapes round differently, the
same class of difference batch composition already causes; see
``scripts/research/check_fused_prefill.py``).

    uv run pytest tests/integration/test_fused_prefill_parity.py -q
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


def test_fused_prefill_greedy_parity():
    from mlx_vlm import load

    from yunshu_engine import fused_prefill
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    model, proc = load(str(MODEL))
    model.set_dtype(mx.float32)
    tok = proc.tokenizer

    def ids(text):
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok.encode(s, add_special_tokens=False)

    a = ids("Write a long story about a lighthouse keeper.")
    b = ids("Explain how a CPU cache works, in detail.")
    c = ids(
        "".join(
            f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n"
            for i in range(100)
        )
        + "\nWhich sensor had the highest reading?"
    )

    def run(budget):
        r = VLMBatchRunner(model, proc)
        r.stop_tokens = {tok.eos_token_id}
        r.fused_prefill_tokens = budget
        ga = r.iter_tokens(a, max_tokens=40)
        out = {"a": [next(ga)]}
        gb = r.iter_tokens(b, max_tokens=40)
        out["b"] = [next(gb)]
        for _ in range(3):
            out["a"].append(next(ga))
            out["b"].append(next(gb))
        gc = r.iter_tokens(c, max_tokens=20)
        out["c"] = list(gc)
        out["a"] += list(ga)
        out["b"] += list(gb)
        return out

    ref = run(0)
    before = fused_prefill.stats()["steps"]
    got = run(64)
    assert fused_prefill.stats()["steps"] - before >= len(c) // 64 - 2
    assert got == ref
