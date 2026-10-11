"""Dump the gateway OpenAPI schema and WebSocket routes without loading a model (CPU only).

Writes frontend/generated/routes.json (gitignored): every route the gateway registers, as
"METHOD /path", for the docs route check (frontend/scripts/docs-routes.mjs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from yunshu_gateway.main import create_app  # noqa: E402


def walk(routes):
    for r in routes:
        if hasattr(r, "effective_candidates"):
            yield from walk(r.effective_candidates())
            continue
        orig = getattr(r, "original_route", r)
        path = getattr(r, "path", "") or getattr(orig, "path", "")
        if "WebSocket" in type(orig).__name__:
            yield "WS", path
        else:
            for m in getattr(orig, "methods", None) or ():
                if m not in ("HEAD", "OPTIONS"):
                    yield m, path


def collect_routes() -> list[str]:
    """Every "METHOD /path" the gateway registers, WebSockets as ``WS``."""
    return sorted({f"{m} {p}" for m, p in walk(create_app().routes)})


def main() -> None:
    out = ROOT / "frontend" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    routes = collect_routes()
    (out / "routes.json").write_text(json.dumps(routes, indent=1) + "\n")
    print(f"{len(routes)} routes")


if __name__ == "__main__":
    main()
