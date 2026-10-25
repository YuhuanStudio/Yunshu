"""Export the real gateway schema at build time, without MLX or app lifespan.

An import guard makes CPU-only generation enforceable even on Apple Silicon.
No inference backend, model or server is initialized.
"""

from __future__ import annotations

import importlib.abc
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm", "mlx_vlm", "mlx_audio"}:
            raise ImportError(f"OpenAPI export must not import {fullname}")
        return None


def export_schema() -> dict:
    # model_manager imports mlx.core for method bodies; these schema-only stubs
    # expose no numerical operations. Any attempt to execute MLX fails loudly.
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    sys.meta_path.insert(0, NoMLX())
    from yunshu_gateway.main import create_app

    schema = create_app().openapi()
    # A relative root URL sends playground requests to the console's own proxy.
    schema["servers"] = [{"url": "/", "description": "Console same-origin proxy"}]
    return schema


def main() -> None:
    out = ROOT / "frontend" / "generated" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = export_schema()
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    print(
        f"OpenAPI: {len(schema['paths'])} paths; schema-only MLX stub; no backend loaded"
    )


if __name__ == "__main__":
    main()
