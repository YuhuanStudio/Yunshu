from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.dflash_draft import (  # noqa: E402
    install_quantized_context,
    install_selector_readout,
)


def projection(width=64, bits=8):
    return nn.QuantizedLinear.from_linear(
        nn.Linear(64, width, bias=False), group_size=64, bits=bits
    )


def drafter(projections):
    return SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=SimpleNamespace(k_proj=k, v_proj=v))
            for k, v in projections
        ],
        _project_context_kv=lambda h: None,
    )


def test_quantized_context_matches_private_projections_and_keeps_shared_head():
    # Unequal K/V widths exercise split boundaries rather than equal chunks.
    pairs = [(projection(64), projection(128)), (projection(128), projection(64))]
    draft = drafter(pairs)
    target = projection()
    draft.lm_head = target
    weight = target.weight
    hidden = mx.random.normal((1, 4, 64))
    expected = [(k(hidden), v(hidden)) for k, v in pairs]
    assert install_quantized_context(draft)
    actual = draft._project_context_kv(hidden)
    for reference, result in zip(expected, actual, strict=True):
        for a, b in zip(reference, result, strict=True):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5, rtol=1e-5)
    assert draft.lm_head is target and target.weight is weight
    method = draft._project_context_kv
    assert install_quantized_context(draft) and draft._project_context_kv is method


@pytest.mark.parametrize("mixed", [True, False])
def test_unsupported_projections_keep_the_original_fallback(mixed):
    pairs = (
        [(projection(), projection(bits=4))]
        if mixed
        else [(nn.Linear(64, 64), nn.Linear(64, 64))]
    )
    draft = drafter(pairs)
    method = draft._project_context_kv
    assert not install_quantized_context(draft)
    assert draft._project_context_kv is method


def test_greedy_dflash2_proposals_use_the_subclass_selector():
    draft = SimpleNamespace(
        candidate_selector=SimpleNamespace(select=lambda *args: None),
        draft_block=lambda *args: "trained selector",
        draft_block_greedy=lambda *args: "inherited unary readout",
    )
    assert install_selector_readout(draft)
    assert draft.draft_block_greedy(123) == "trained selector"
    assert install_selector_readout(draft)


def test_non_selector_drafter_keeps_its_greedy_readout():
    draft = SimpleNamespace(draft_block_greedy=lambda *args: "greedy")
    assert not install_selector_readout(draft)
    assert draft.draft_block_greedy() == "greedy"
