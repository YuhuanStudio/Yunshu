"""routers.admin_routes mounts the console management routers under /v1."""

from fastapi import FastAPI

from yunshu_gateway.routers import admin_cache, admin_logs, admin_models, admin_routes


def _paths(app: FastAPI) -> set[str]:
    return set(app.openapi()["paths"])


def test_register_mounts_every_management_router_under_v1():
    app = FastAPI()
    admin_routes.register(app)
    mounted = _paths(app)
    assert mounted
    for mod in (admin_models, admin_cache, admin_logs):
        sub = FastAPI()
        sub.include_router(mod.router)
        for path in _paths(sub):
            assert "/v1" + path in mounted, path
    assert all(p.startswith("/v1/") for p in mounted)


def test_register_is_deterministic_for_a_fresh_app():
    a, b = FastAPI(), FastAPI()
    admin_routes.register(a)
    admin_routes.register(b)
    assert _paths(a) == _paths(b)
