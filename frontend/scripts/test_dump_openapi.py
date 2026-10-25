"""The documentation build must reflect real routes and never load MLX."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("dump_openapi.py")


def test_export_schema_without_gpu():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    schema = json.loads((SCRIPT.parents[1] / "generated" / "openapi.json").read_text())
    assert schema["openapi"].startswith("3.")
    assert "post" in schema["paths"]["/v1/chat/completions"]
    assert "post" in schema["paths"]["/v1/messages"]
    assert "get" in schema["paths"]["/v1/models"]
    assert schema["servers"] == [
        {"url": "/", "description": "Console same-origin proxy"}
    ]
    assert "schema-only MLX stub" in result.stdout


def test_import_guard_rejects_accidental_real_backend_import():
    spec = importlib.util.spec_from_file_location("docs_dump", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("mlx", "mlx.core", "mlx_lm", "mlx_vlm.models", "mlx_audio"):
        try:
            module.NoMLX().find_spec(name)
        except ImportError:
            pass
        else:
            raise AssertionError(f"Real backend unexpectedly permitted: {name}")
