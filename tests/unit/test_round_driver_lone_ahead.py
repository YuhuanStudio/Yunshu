from types import SimpleNamespace as NS

from yunshu_engine.round_driver.driver import RoundDriver


def _driver(**kw):
    row = NS(
        generic=False,
        ngram=None,
        force=[],
        penalty=None,
        ahead=None,
        ahead_logged=False,
        req=NS(guide=None, processors=[], budget=None, sampling=None, logprobs=False),
    )
    d = NS(dflash=None, head=None, generic=True, rows=[row], batch=NS(rows=[row]))
    for k, v in kw.items():
        setattr(row.req if k in vars(row.req) else d, k, v)
    return d, row


def test_plain_greedy_generic_row_pipelines():
    d, _ = _driver()
    assert RoundDriver._lone_blockers(d) == []


def test_blockers_named():
    d, _ = _driver(sampling=object(), processors=[1])
    assert RoundDriver._lone_blockers(d) == ["processors", "sampling"]
    d, _ = _driver(generic=False)
    assert RoundDriver._lone_blockers(d) == ["layer"]


class Tok:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return [self.v]

    def astype(self, _):
        return self


class Cache:
    def __init__(self, trimmable):
        self.trimmed = 0
        self.ok = trimmable

    def trim(self, n):
        self.trimmed += n


class Batch:
    def __init__(self, rows, trimmable):
        self.rows, self.cache, self.commits = rows, [Cache(trimmable)], []
        self.trimmable = trimmable

    def can_trim(self, extra=0):
        return self.trimmable

    def arrays(self):
        return []

    def commit(self, used):
        self.commits.append(used)


def _lone(mp, trimmable, max_tokens=100, stop=()):
    from yunshu_engine.round_driver import driver as dm

    row = NS(
        generic=False,
        ngram=None,
        force=[],
        penalty=None,
        ahead=None,
        ahead_logged=False,
        stale_cache=False,
        finished=False,
        pending=10,
        generated=0,
        context=[],
        drafts=[],
        draft_dev=None,
        nd=0,
        hslot=None,
        drafting=False,
        req=NS(
            guide=None,
            processors=[],
            budget=None,
            sampling=None,
            logprobs=False,
            max_tokens=max_tokens,
            handle=None,
            prompt_logprobs=None,
        ),
    )
    d = NS(
        dflash=None,
        head=None,
        generic=True,
        rows=[row],
        batch=Batch([row], trimmable),
        stop_tokens=set(stop),
        ahead_steps=0,
        steps=0,
        released=[],
    )
    d._lone_draw = lambda r, t: Tok(t.v + 1 if hasattr(t, "v") else t.tolist()[0] + 1)
    d._release = lambda gone: d.released.extend(gone)
    for n in (
        "_lone_blockers",
        "_lone_ahead_ok",
        "_decode_step_lone",
        "_settle_ahead",
        "_drop_ahead",
        "_emit",
    ):
        setattr(d, n, getattr(RoundDriver, n).__get__(d))
    mp.setattr(dm.mx, "async_eval", lambda *a: None)
    mp.setattr(dm.mx, "array", lambda v, dtype=None: Tok(v[0]))
    return d, row


def _run(d, row, steps):
    out = []
    for _ in range(steps):
        if not d.rows:
            break
        out += [e.token for e in d._decode_step_lone()] if d._lone_ahead_ok() else []
    return out


def test_window_cache_still_pipelines(monkeypatch):
    mp = monkeypatch
    d, row = _lone(mp, trimmable=False)
    d._decode_step_lone()
    assert row.ahead is not None  # no trim needed to queue the next step


def test_second_row_join_keeps_stream_identical(monkeypatch):
    mp = monkeypatch
    d1, r1 = _lone(mp, False)
    solo = _run(d1, r1, 6)
    d2, r2 = _lone(mp, False)
    got = _run(d2, r2, 3)
    got += [
        e.token for e in d2._settle_ahead()
    ]  # a row arrives: the queued step finishes first
    assert r2.ahead is None and not d2.batch.cache[0].trimmed
    # the batch carries on from the settled state (pending token), same stream as solo
    nxt = r2.pending
    while len(got) < 6:
        nxt += 1
        got.append(nxt)
    assert got == solo[:6]


def test_finish_with_ahead_trims_or_marks_stale(monkeypatch):
    mp = monkeypatch
    d, row = _lone(mp, True, max_tokens=100, stop=(12,))
    _run(d, row, 5)
    assert row.finished and d.batch.cache[0].trimmed == 1 and not row.stale_cache
    d, row = _lone(mp, False, stop=(12,))
    _run(d, row, 5)
    assert row.finished and row.stale_cache
