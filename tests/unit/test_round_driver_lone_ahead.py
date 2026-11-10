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
