"""Special requests retain the original DFlash verifier when trees are selected."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from yunshu_engine import tree_verify as tv  # noqa: E402


@pytest.mark.parametrize(
    "request_kind", ["grammar", "target_hidden_adapter", "sampled"]
)
def test_dflash_tree_preserves_special_request_verifier(monkeypatch, request_kind):
    from yunshu_engine import dflash_tree, mtp_lane

    monkeypatch.setattr(dflash_tree, "supported", lambda *_: True)
    monkeypatch.setattr(tv, "lane_ready", lambda *_: True)
    monkeypatch.setitem(
        mtp_lane._STATE, "guide", object() if request_kind == "grammar" else None
    )
    draft = SimpleNamespace()
    if request_kind == "target_hidden_adapter":
        draft.prepare_target_hidden = lambda *_: None
    model = SimpleNamespace()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        yield 17, "original verifier"

    result = list(
        dflash_tree.dflash_tree_rounds(
            model,
            draft,
            [],
            None,
            first_bonus=11,
            max_tokens=2,
            sampler=None,
            greedy_sampling=request_kind != "sampled",
            _original=original,
        )
    )
    assert result == [(17, "original verifier")]
    assert len(calls) == 1
    assert calls[0][0] == (model, draft, [], None)
    assert calls[0][1]["first_bonus"] == 11
