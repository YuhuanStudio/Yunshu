"""A model that fails to load leaves the server up, with a stated reason."""

from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app, describe_load_error


def test_describe_missing_path():
    msg = describe_load_error("/x/y", FileNotFoundError("No such file or directory"))
    assert "/x/y" in msg and "not found" in msg


def test_describe_truncated_weights():
    exc = RuntimeError("[load_safetensors] invalid data offsets; incomplete download")
    assert "yunshu pull" in describe_load_error("m", exc)


def test_describe_out_of_memory():
    assert "not enough memory" in describe_load_error("m", MemoryError("out of memory"))


def test_ready_reports_reason_when_load_failed():
    from yunshu_gateway.engine import set_engine

    set_engine(None)
    app = create_app()
    app.state.load_error = "m: RuntimeError: boom (see the server log)"
    r = TestClient(app).get("/health/ready")
    assert r.status_code == 503
    assert r.json()["reason"].endswith("(see the server log)")


def test_rate_limit_is_off_by_default():
    from yunshu_engine import settings

    assert settings.get("YUNSHU_RATE_LIMIT_RPM") == 0
