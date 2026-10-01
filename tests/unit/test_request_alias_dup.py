from yunshu_engine.request_tracker import RequestTracker, current_request_id


def _reg(t, rid, cid):
    tok = current_request_id.set(cid)
    try:
        return t.register(rid, "m")
    finally:
        current_request_id.reset(tok)


def test_first_finishing_keeps_second_alias():
    t = RequestTracker()
    _reg(t, "eng-1", "client")
    _reg(t, "eng-2", "client")
    t.unregister("eng-1")
    assert t.resolve("client") == "eng-2"
    assert t.cancel("client") is True


def test_alias_holder_finishing_falls_back_to_other():
    t = RequestTracker()
    _reg(t, "eng-1", "client")
    _reg(t, "eng-2", "client")
    t.unregister("eng-2")
    assert t.resolve("client") == "eng-1"
    t.unregister("eng-1")
    assert t.resolve("client") == "client"
