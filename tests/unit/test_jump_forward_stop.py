"""jump-forward decode loop (opt-in YUNSHU_JUMP_FORWARD=1) mishandled
stop sequences and forced-continuation truncation.

BUG 1 (HIGH): the loop ran to max_tokens / EOS and only post-hoc truncated `text` at
  the first stop substring — gen_ids was left untouched, so the caller's
  completion_tokens=len(gen_ids) over-counted every post-stop token and finish_reason
  was reported "length" instead of "stop". Fixed by detecting the stop INCREMENTALLY
  (like _generate_fast): break at the token that completes the stop, return a stop_hit
  flag, and have the caller report finish_reason="stop" with an accurate token count.
BUG 2 (MED): when a forced continuation overshot max_tokens, fids was truncated to the
  remaining room but constraint.advance(fc) advanced the FSM by the FULL forced string
  (desyncing the constraint from gen_ids), and a wasted model() forward still ran.
  Fixed by advancing only by the emitted (truncated) text and skipping the final forward.

The loop only touches self._model / self._tokenizer, so it's exercised directly with
controllable fakes (allowed=all → apply_json_constraint is a no-op → the fake model's
argmax drives the token sequence deterministically).
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from yunshu_engine.batched_engine import BatchedEngine


class _FakeTok:
    def __init__(self, id2str):
        self.id2str = id2str
        self.str2id = {v: k for k, v in id2str.items()}

    def decode(self, ids):
        return "".join(self.id2str.get(i, "") for i in ids)

    def encode(self, s, add_special_tokens=False):
        return [self.str2id[ch] for ch in s]


class _FakeConstraint:
    """allowed=all (no masking); records advances; scriptable forced_continuation."""

    def __init__(self, all_ids, forced=""):
        self._all_ids = list(all_ids)
        self._forced = forced
        self.advanced = []

    def get_allowed_tokens(self, tok, gen_ids):
        return self._all_ids

    def advance(self, text):
        self.advanced.append(text)

    def forced_continuation(self):
        return self._forced


class _FakeModel:
    """Returns logits whose last-row argmax follows a scripted token sequence."""

    def __init__(self, seq, vocab):
        self.seq = seq
        self.vocab = vocab
        self.calls = 0

    def make_cache(self):
        return []

    def __call__(self, x, cache=None):
        nxt = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        seqlen = x.shape[1]
        out = mx.full((1, seqlen, self.vocab), -10.0)
        out[0, -1, nxt] = 10.0
        return out


def _engine(model, tok):
    eng = BatchedEngine.__new__(BatchedEngine)
    eng._model = model
    eng._tokenizer = tok
    return eng


def test_bug1_stop_truncates_token_count_and_sets_stop_hit():
    # vocab: 0='a' 1='b' 2=']' 3='c'
    id2str = {0: "a", 1: "b", 2: "]", 3: "c"}
    tok = _FakeTok(id2str)
    # model emits a, b, ], c, c, ...  → with stop="]" it must break AT ']'
    model = _FakeModel(seq=[0, 1, 2, 3, 3], vocab=4)
    con = _FakeConstraint(all_ids=list(id2str), forced="")
    eng = _engine(model, tok)

    text, ids, n_fwd, stop_hit = eng._jump_forward_generate_sync(
        [99], con, max_tokens=20, eos_ids=[], stop=["]"]
    )
    assert stop_hit is True
    # gen_ids stops at the ']' token — 3 tokens (a,b,]), NOT a run to max_tokens=20
    assert ids == [0, 1, 2]
    # text is truncated at the stop substring
    assert text == "ab"
    # caller's finish_reason logic: stop_hit -> "stop", count is accurate (3, not 20)
    fr = "length" if (not stop_hit and len(ids) >= 20) else "stop"
    assert fr == "stop"


def test_bug1_no_stop_runs_to_max_tokens_length():
    id2str = {0: "a", 1: "b"}
    tok = _FakeTok(id2str)
    model = _FakeModel(seq=[0, 1, 0, 1, 0], vocab=2)
    con = _FakeConstraint(all_ids=list(id2str), forced="")
    eng = _engine(model, tok)
    text, ids, n_fwd, stop_hit = eng._jump_forward_generate_sync(
        [99],
        con,
        max_tokens=4,
        eos_ids=[],
        stop=["zzz"],  # never matches
    )
    assert stop_hit is False
    assert len(ids) == 4
    fr = "length" if (not stop_hit and len(ids) >= 4) else "stop"
    assert fr == "length"


def test_bug2_forced_truncation_advances_by_emitted_text_only():
    # vocab: 0='a' 1='X' 2='Y' 3='Z'
    id2str = {0: "a", 1: "X", 2: "Y", 3: "Z"}
    tok = _FakeTok(id2str)
    model = _FakeModel(seq=[0, 0, 0], vocab=4)
    con = _FakeConstraint(all_ids=list(id2str), forced="XYZ")  # 3 forced tokens
    eng = _engine(model, tok)
    # max_tokens=2: emit 'a' (1), then forced "XYZ" has room=1 -> truncate to "X"
    text, ids, n_fwd, stop_hit = eng._jump_forward_generate_sync(
        [99], con, max_tokens=2, eos_ids=[], stop=None
    )
    assert ids == [0, 1]  # a + X (truncated), capped at max_tokens
    assert text == "aX"
    # the FSM was advanced by the EMITTED text only — never the full "XYZ"
    assert con.advanced == ["a", "X"]
    assert "XYZ" not in con.advanced
    # the wasted final forward was skipped: only the single pre-loop forward ran
    assert model.calls == 1


def test_caller_finish_reason_uses_stop_hit():
    # lock in the caller's finish_reason wiring (in _generate_fast, source-guarded)
    src = inspect.getsource(BatchedEngine._generate_fast)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_jf_stop," in code and ") = await _jf_loop.run_in_exec" in code
    assert "not _jf_stop and len(_jf_ids) >= max_tokens" in code
