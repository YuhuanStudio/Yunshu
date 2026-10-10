from yunshu_engine import settings, tree_verify


def test_profile_flag_is_off_by_default_and_tick_accumulates(monkeypatch):
    assert settings.get("YUNSHU_DEBUG_TREE_PROFILE") is False
    assert tree_verify._profile_on() is False
    monkeypatch.setattr(tree_verify.mx, "eval", lambda *_: None)
    tree_verify.PROFILE.clear()
    tree_verify.PROFILE["rounds"] = 0
    t = tree_verify._tick("mlp_ms", [], 0.0)
    assert tree_verify.PROFILE["mlp_ms"] > 0 and t > 0
