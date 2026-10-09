"""Regenerate docs/images/console/<lang>/*.webp from the console (Vite dev server) in all three languages.

    node frontend/node_modules/vite/bin/vite.js ... &   # or any running console
    python scripts/docs/console_shots.py http://127.0.0.1:3994

Runs console_shots.mjs once per language (PNG into a temp dir), converts each PNG to WebP (Pillow) and
steps the quality down until the file is at most 300 KB.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGS = {"en": "en-US", "zh-CN": "zh-CN", "zh-TW": "zh-TW"}
MAX_BYTES = 300 * 1024


def to_webp(png: Path, out: Path) -> int:
    from PIL import Image  # noqa: PLC0415

    im = Image.open(png).convert("RGB")
    for q in (88, 80, 72, 64, 56, 48, 40):
        im.save(out, "WEBP", quality=q, method=6)
        if out.stat().st_size <= MAX_BYTES:
            return q
    # still too big: halve the pixel count (the 2x screenshots are larger than the page needs)
    im = im.resize((im.width * 3 // 4, im.height * 3 // 4))
    im.save(out, "WEBP", quality=60, method=6)
    if out.stat().st_size > MAX_BYTES:
        raise SystemExit(f"{out.name} is {out.stat().st_size} bytes after shrinking")
    return -1


def main(base: str) -> int:
    for lang, locale in LANGS.items():
        with tempfile.TemporaryDirectory(prefix="console-shots-") as tmp:
            env = {**os.environ, "LOCALE": locale, "FAST": "1"}
            subprocess.run(["node", "scripts/docs/console_shots.mjs", base, tmp], cwd=ROOT, env=env, check=True)
            dest = ROOT / "docs/images/console" / lang
            dest.mkdir(parents=True, exist_ok=True)
            for png in sorted(Path(tmp).glob("*.png")):
                q = to_webp(png, dest / (png.stem + ".webp"))
                print(f"{lang}/{png.stem}.webp q={q} {(dest / (png.stem + '.webp')).stat().st_size // 1024} KB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3994"))
