"""The opt-in jump-forward loop prefills a long prompt in chunks and reads only the last position's logits
(swift-lm 3615fe4: the vocabulary projection of prompt positions nobody reads is never computed)."""

from tests.unit.test_jump_forward_stop import (
    _engine,
    _FakeConstraint,
    _FakeModel,
    _FakeTok,
)
from yunshu_engine import engine_fast


class _Recorder(_FakeModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lengths = []

    def __call__(self, x, cache=None):
        self.lengths.append(int(x.shape[1]))
        return super().__call__(x, cache)


def _run(monkeypatch, n_prompt, step, n_chunks=0):
    monkeypatch.setattr(engine_fast, "_JUMP_PREFILL_STEP", step)
    id2str = {0: "a", 1: "b"}
    # the scripted model advances one entry per forward: the first n_chunks entries belong to prefill chunks
    model = _Recorder(seq=[1] * n_chunks + [0, 1], vocab=2)
    eng = _engine(model, _FakeTok(id2str))
    con = _FakeConstraint(all_ids=list(id2str), forced="")
    out = eng._jump_forward_generate_sync(
        list(range(10, 10 + n_prompt)), con, max_tokens=2, eos_ids=[]
    )
    return model.lengths, out


def test_long_prompt_runs_chunks_then_the_last_token_alone(monkeypatch):
    lengths, (text, ids, _, _) = _run(monkeypatch, n_prompt=11, step=4, n_chunks=3)
    # 10 head tokens in chunks of 4, 4, 2, then the last token alone, then one decode forward
    assert lengths[:4] == [4, 4, 2, 1]
    assert sum(lengths[:3]) == 10
    assert ids == [0, 1] and text == "ab"


def test_short_prompt_stays_one_forward(monkeypatch):
    lengths, (_, ids, _, _) = _run(monkeypatch, n_prompt=4, step=4)
    assert lengths[0] == 4 and ids == [0, 1]
