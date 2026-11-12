"""MLX's Metal live-buffer maximum (``Resource limit (499000) exceeded``) is a 507 with restart
guidance, never an opaque 500; a plain byte shortage keeps its own answer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from yunshu_engine.exceptions import is_resource_count_failure
from yunshu_gateway import files_store
from yunshu_gateway.main import create_app

MLX_MSG = "[metal::malloc] Resource limit (499000) exceeded."


def test_detects_the_mlx_live_buffer_limit_only():
    assert is_resource_count_failure(RuntimeError(MLX_MSG))
    assert not is_resource_count_failure(RuntimeError("out of memory"))
    assert not is_resource_count_failure(
        RuntimeError("Resource limit (5) exceeded")
    )  # no metal


def test_live_buffer_limit_on_a_route_is_507_with_restart_guidance(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()

    def boom(*_a, **_k):
        raise RuntimeError(MLX_MSG)

    monkeypatch.setattr(files_store.FileStore, "list", boom)
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        r = c.get("/v1/files")
    files_store.reset_store()
    assert r.status_code == 507
    err = r.json()["error"]
    assert err["code"] == "gpu_resource_limit"
    assert "restart the server" in err["message"]
