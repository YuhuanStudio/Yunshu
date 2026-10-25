from yunshu_engine.serving.work_scheduler import decode_quantum


def test_tail_share_default_and_override(monkeypatch):
    monkeypatch.delenv("YUNSHU_DECODE_TAIL_SHARE", raising=False)
    assert decode_quantum(2.0, early=False) == 1.0
    monkeypatch.setenv("YUNSHU_DECODE_TAIL_SHARE", "1.0")
    assert decode_quantum(2.0, early=False) == 2.0
    assert decode_quantum(2.0, early=True) == 2.0  # early rows keep DECODE_SHARE
    monkeypatch.setenv("YUNSHU_DECODE_TAIL_SHARE", "0.25")
    assert decode_quantum(2.0, early=False) == 0.5
