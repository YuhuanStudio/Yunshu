"""An exact repeat writes back the stored final block instead of running its forward."""

import types

import pytest

mx = pytest.importorskip("mlx.core")
ar = pytest.importorskip("mlx_vlm.generate.ar")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache  # noqa: E402

from yunshu_engine import cache_prefill, prompt_end_replay  # noqa: E402
from yunshu_engine.kernels import gdn_prefill  # noqa: E402

BASE, ROWS, HEADS, DIM = 64, 5, 2, 4


@pytest.fixture(autouse=True)
def cpu_arrays(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(gdn_prefill, "split_enabled", lambda: False)
    prompt_end_replay.clear()
    yield
    prompt_end_replay.clear()
    mx.set_default_device(device)


class Model:
    def __init__(self):
        self.calls = 0
        self.language_model = types.SimpleNamespace(_rope_deltas=mx.zeros((1, 1)))

    def __call__(self, ids, cache=None, inputs_embeds=None, **kwargs):
        self.calls += 1
        k = mx.ones((1, HEADS, ROWS, DIM)) * (self.calls + 1)
        for c in cache:
            if isinstance(c, ArraysCache):
                c.cache = [mx.full((1, 3), self.calls + 10.0), mx.full((1, 2), 7.0)]
            else:
                c.update_and_fetch(k, k + 1)
        return types.SimpleNamespace(
            logits=mx.full((1, 1, 6), self.calls + 0.5),
            hidden_states=[mx.full((1, ROWS, 3), 2.0 + self.calls)],
            shared_kv_states=None,
        )


class Speculative:
    kwargs = {"capture_layer_ids": [1]}
    chunks: list = []

    def finish(self, output):
        return output


def make_cache():
    kv = BatchKVCache([0])
    k = mx.arange(HEADS * BASE * DIM, dtype=mx.float32).reshape(1, HEADS, BASE, DIM)
    kv.update_and_fetch(k, k)
    arrays = ArraysCache(2)
    arrays.cache = [mx.zeros((1, 3)), mx.zeros((1, 2))]
    return [arrays, kv, arrays.__class__(2)]


def make_batch(model, ids=tuple(range(BASE + ROWS)), **overrides):
    install = cache_prefill.install
    install()
    cls = ar.PromptProcessingBatch
    batch = object.__new__(cls)
    cache = make_cache()
    cache[2].cache = [mx.zeros((1, 3)), mx.zeros((1, 2))]
    batch.__dict__.update(
        model=model,
        uids=[0],
        logits_processors=[None],
        _right_pad_per_row=None,
        draft_model=object(),
        draft_kind="dflash",
        _prompt_kwargs={},
        _inputs_embeds=mx.zeros((1, ROWS, 3)),
        prompt_cache=cache,
        _speculative_prefill=Speculative(),
        _apc_meta=[{"full_input_ids": list(ids), "extra_hash": 3}],
        _processed_prompt_columns=0,
    )
    batch.__dict__.update(overrides)
    return batch


@pytest.fixture
def fake_generate(monkeypatch):
    def generate(self, *args, **kwargs):
        output = self.model(
            None, cache=self.prompt_cache, inputs_embeds=self._inputs_embeds
        )
        output = self._speculative_prefill.finish(output)
        gen = types.SimpleNamespace(
            prompt_cache=self.prompt_cache,
            logits=output.logits,
            hidden=output.hidden_states,
        )
        self.prompt_cache = []
        return gen

    cache_prefill.install()
    monkeypatch.setattr(ar.PromptProcessingBatch.__mro__[1], "generate", generate)


def state(gen):
    out = []
    for c in gen.prompt_cache:
        if isinstance(c, ArraysCache):
            out.append([a.tolist() for a in c.cache])
        else:
            out.append(
                (c.keys[..., : c._idx, :].tolist(), c.values[..., : c._idx, :].tolist())
            )
    return out, gen.logits.tolist(), [h.tolist() for h in gen.hidden]


def test_repeat_replays_the_stored_block(fake_generate):
    model = Model()
    first = make_batch(model).generate()
    assert model.calls == 1 and prompt_end_replay.stats()["stored"] == 1
    again = make_batch(model).generate()
    assert model.calls == 1, "the forward must not run again"
    assert prompt_end_replay.stats()["replayed"] == 1
    assert state(again) == state(first)


def test_replay_restores_the_real_model_and_offsets(fake_generate):
    model = Model()
    make_batch(model).generate()
    batch = make_batch(model)
    gen = batch.generate()
    assert batch.model is model
    kv = gen.prompt_cache[1]
    assert kv._idx == BASE + ROWS and int(kv.offset[0]) == BASE + ROWS


@pytest.mark.parametrize(
    "change",
    [
        {"ids": tuple(range(1, BASE + ROWS + 1))},
        {"draft_kind": None, "draft_model": None},
        {"_prompt_kwargs": {"pixel_values": mx.zeros((1,))}},
        {"uids": [0, 1]},
    ],
)
def test_other_requests_run_the_forward(fake_generate, change):
    model = Model()
    make_batch(model).generate()
    make_batch(model, **change).generate()
    assert model.calls == 2
    assert prompt_end_replay.stats()["replayed"] == 0


def test_logits_processors_still_replay(fake_generate):
    # processors run on the returned logits after the forward, so a replayed block feeds them the same input
    model = Model()
    make_batch(model, logits_processors=[[lambda _ctx, logits: logits]]).generate()
    make_batch(model, logits_processors=[[lambda _ctx, logits: logits]]).generate()
    assert model.calls == 1
    assert prompt_end_replay.stats()["replayed"] == 1


def test_different_tail_length_runs_the_forward(fake_generate):
    model = Model()
    make_batch(model).generate()
    make_batch(model, _inputs_embeds=mx.zeros((1, ROWS + 1, 3))).generate()
    assert model.calls == 2


def test_missing_rope_state_runs_the_forward(fake_generate):
    model = Model()
    make_batch(model).generate()
    model.language_model._rope_deltas = None
    make_batch(model).generate()
    assert model.calls == 2


def test_capacity_evicts_oldest(fake_generate, monkeypatch):
    monkeypatch.setattr(prompt_end_replay, "MAX_ENTRIES", 2)
    model = Model()
    for i in range(3):
        make_batch(model, ids=tuple(range(i, BASE + ROWS + i))).generate()
    assert prompt_end_replay.stats()["entries"] == 2
    make_batch(model).generate()  # the oldest was evicted
    assert model.calls == 4


def test_stored_kv_rows_do_not_pin_the_context_buffer(fake_generate):
    model = Model()
    make_batch(model).generate()
    entry = next(iter(prompt_end_replay._STORE.values()))
    keys = next(p[0] for tag, p in entry.slots if tag == "k")
    assert keys.shape == (1, HEADS, ROWS, DIM)
    assert entry.nbytes < 4096
