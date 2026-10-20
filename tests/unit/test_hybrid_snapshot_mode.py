"""LoRA requests never enter the hybrid boundary-snapshot path (its keys carry no adapter)."""

from yunshu_engine.engine_fast import hybrid_snapshot_mode


def _mode(**over):
    kw = dict(
        trimmable=False,
        seed_bypass=False,
        auxiliary=False,
        lora_adapter=None,
        prefix_cache=object(),
    )
    kw.update(over)
    return hybrid_snapshot_mode(**kw)


def test_hybrid_snapshot_mode():
    assert _mode()
    assert not _mode(lora_adapter="adapter")
    assert not _mode(trimmable=True)
    assert not _mode(seed_bypass=True)
    assert not _mode(auxiliary=True)
    assert not _mode(prefix_cache=None)
