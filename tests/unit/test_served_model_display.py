import yunshu_gateway.engine as eng
from yunshu_gateway.x_yunshu import served_model


def test_single_model_mode_shows_the_served_model_not_the_client_alias(monkeypatch):
    monkeypatch.setattr(eng, "get_model_manager", lambda: None)
    monkeypatch.setattr(eng, "get_display_model_id", lambda: "Qwen3.8-27B-oQ4e-mtp")
    assert served_model("m") == "Qwen3.8-27B-oQ4e-mtp"


def test_multi_model_mode_keeps_the_requested_id(monkeypatch):
    monkeypatch.setattr(eng, "get_model_manager", lambda: object())
    monkeypatch.setattr(eng, "get_display_model_id", lambda: "other")
    assert served_model("org/real-id") == "org/real-id"
