"""Freeze one speculative benchmark's Python source across server sessions."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))
from gpuq_contention import was_contended  # noqa: E402


def refuse_contended(output: Path) -> bool:
    """A forced admission after quiet timeout must not waste a whole sweep."""
    if not was_contended():
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as out:
        out.write(
            json.dumps(
                {
                    "complete": True,
                    "success": False,
                    "contended": True,
                    "reason": "CPU contention at admission; measurement skipped",
                }
            )
            + "\n"
        )
    print("CPU contention at admission: skipped measurement; rerun quietly", flush=True)
    return True


def freeze(output: Path) -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[2]
    source = output.with_name(output.stem + "-source")
    shutil.copytree(
        root / "python", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    digest = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return source, {
        "head": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python_sha256": digest.hexdigest(),
        "source": str(source),
    }
