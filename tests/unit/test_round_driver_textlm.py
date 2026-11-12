"""Round driver, generic layer over an mlx-lm text checkpoint (``round_driver/textlm.py``, T7).

A tiny random llama (mlx-lm's own model class, CPU float32, no download) is wrapped by the adapter and
driven on the generic layer. The tests pin what T7 promises: a row's greedy tokens equal mlx-lm's
``generate_step`` on the same model, concurrent rows equal their solo runs, the cache is mlx-lm's own, the
call refuses what it does not carry, and a checkpoint on disk loads to the same model.
"""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from yunshu_engine.round_driver import driver as drv  # noqa: E402
from yunshu_engine.round_driver import textlm  # noqa: E402
from yunshu_engine.round_driver.layer import GenericLayer, language_model  # noqa: E402

VOCAB = 64
ARGS = dict(
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


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_model(seed: int = 0):
    from mlx_lm.models import llama

    mx.random.seed(seed)
    model = llama.Model(llama.ModelArgs(**ARGS))
    mx.eval(model.parameters())
    return model


def prompt(n: int, salt: int = 1) -> list[int]:
    return [4 + (salt * 7 + 5 * i) % (VOCAB - 4) for i in range(n)]


def request(adapter, ids, n=8, handle=None):
    feats = adapter.get_input_embeddings(
        mx.array(ids, dtype=mx.int32)[None], None, mask=None
    )
    return drv.Request(
        ids,
        n,
        handle=handle if handle is not None else tuple(ids[:3]),
        prompt_kwargs=feats.to_dict(),
        draft=False,
    )


def make(adapter, chunk=8):
    return drv.RoundDriver(
        adapter,
        layer=GenericLayer(language_model(adapter)),
        stop_tokens=set(),
        chunk=chunk,
        apc=None,
    )


def run(d, reqs, join_at=None):
    """Drive ``d`` to idle; each request joins at its step in ``join_at`` (default: all at once)."""
    join_at = join_at or {}
    out: dict = {}
    pending = list(reqs)
    steps = 0
    while d.busy() or pending:
        for r in list(pending):
            if join_at.get(r.handle, 0) <= steps:
                d.add(r)
                pending.remove(r)
        for e in d.step():
            out.setdefault(e.handle, []).append(e.token)
        steps += 1
        assert steps < 500
    return out


def reference(model, ids, n):
    """mlx-lm's own greedy generation of ``n`` tokens (``generate_step``, argmax sampler)."""
    from mlx_lm.generate import generate_step

    got = []
    for tok, _lp in generate_step(
        mx.array(ids, dtype=mx.int32), model, stream=mx.cpu, max_tokens=n
    ):
        got.append(int(tok))
    return got


def test_rows_equal_mlx_lm_generate_step_greedy():
    model = tiny_model()
    adapter = textlm.TextModelAdapter(model)
    for ids in (prompt(20), prompt(5, salt=3)):
        d = make(adapter)
        r = request(adapter, ids, n=10)
        got = run(d, [r])[r.handle]
        assert got == reference(model, ids, 10)


def test_concurrent_rows_equal_their_solo_runs():
    model = tiny_model()
    adapter = textlm.TextModelAdapter(model)
    a, b = prompt(20), prompt(7, salt=2)
    solo_a = run(make(adapter), [request(adapter, a, n=12, handle="a")])["a"]
    solo_b = run(make(adapter), [request(adapter, b, n=12, handle="b")])["b"]
    together = run(
        make(adapter),
        [request(adapter, a, n=12, handle="a"), request(adapter, b, n=12, handle="b")],
    )
    assert together["a"] == solo_a
    assert together["b"] == solo_b
    # a row that joins while another is decoding gets the same tokens too
    staggered = run(
        make(adapter),
        [request(adapter, a, n=12, handle="a"), request(adapter, b, n=12, handle="b")],
        join_at={"b": 4},
    )
    assert staggered["a"] == solo_a
    assert staggered["b"] == solo_b


def test_make_cache_is_mlx_lm_s_cache_per_layer():
    from mlx_lm.models.cache import KVCache

    adapter = textlm.TextModelAdapter(tiny_model())
    cache = adapter.language_model.make_cache()
    assert len(cache) == ARGS["num_hidden_layers"]
    assert all(isinstance(c, KVCache) for c in cache)


def test_logits_to_keep_keeps_the_last_rows_and_extra_kwargs_are_refused():
    model = tiny_model()
    lm = textlm.TextLM(model)
    ids = mx.array(prompt(6), dtype=mx.int32)[None]
    full = lm(ids, cache=lm.make_cache())
    last = lm(ids, cache=lm.make_cache(), logits_to_keep=1)
    assert last.shape == (1, 1, VOCAB)
    assert mx.allclose(last, full[:, -1:], atol=1e-6).item()
    with pytest.raises(ValueError, match="not carried"):
        lm(ids, cache=lm.make_cache(), rope_deltas=mx.zeros((1, 1), dtype=mx.int32))


def test_media_inputs_are_refused_and_embeddings_are_the_token_embedding():
    model = tiny_model()
    adapter = textlm.TextModelAdapter(model)
    ids = mx.array(prompt(4), dtype=mx.int32)[None]
    with pytest.raises(ValueError, match="no vision tower"):
        adapter.get_input_embeddings(ids, mx.zeros((1, 3, 8, 8)), mask=None)
    embeds = adapter.get_input_embeddings(ids, None, mask=None).to_dict()[
        "inputs_embeds"
    ]
    assert embeds.shape == (1, 4, ARGS["hidden_size"])
    assert mx.array_equal(embeds, model.model.embed_tokens(ids)).item()


def test_a_checkpoint_on_disk_loads_to_the_same_model(tmp_path):
    from mlx.utils import tree_flatten

    model = tiny_model(seed=5)
    (tmp_path / "config.json").write_text(json.dumps(ARGS))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    ids = prompt(9)
    mine = textlm.TextModelAdapter(model)
    want = run(make(mine), [request(mine, ids, n=6, handle="x")])["x"]
    loaded = textlm.load_text_model(tmp_path)
    got = run(make(loaded), [request(loaded, ids, n=6, handle="x")])["x"]
    assert got == want
