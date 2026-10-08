"""Dump the gateway OpenAPI schema and WebSocket routes without loading a model (CPU only).

Writes site/generated/openapi.json and site/generated/routes.json.
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


def main() -> None:
    app = create_app()
    out = ROOT / "site" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    (out / "openapi.json").write_text(json.dumps(schema, indent=1, sort_keys=True) + "\n")
    routes = sorted({f"{m} {p}" for m, p in walk(app.routes)})
    (out / "routes.json").write_text(json.dumps(routes, indent=1) + "\n")
    print(f"{len(schema['paths'])} openapi paths, {len(routes)} routes")


if __name__ == "__main__":
    main()
