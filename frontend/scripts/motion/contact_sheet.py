"""Contact sheets for motion evidence: one PNG per scenario/theme/viewport, `before` row over `after` row,
N frames each, evenly spaced over the part of the recording where something changes (a still picture
before the first change is skipped). Usage:

    python contact_sheet.py <shots-dir> [--n 8] [--window-ms 720]

Reads <shots>/<label>/<scenario>/<theme>-<vp>/f_<ms>.jpg, writes <shots>/sheets/<scenario>-<theme>-<vp>.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


def frames(d: Path) -> list[tuple[int, Path]]:
    return sorted((int(p.stem.split("_")[1]), p) for p in d.glob("f_*.jpg"))


def first_change(fr: list[tuple[int, Path]]) -> int:
    """Index of the first frame that differs visibly from the opening frame."""
    base = Image.open(fr[0][1]).convert("L").resize((96, 54))
    for i, (_, p) in enumerate(fr[1:], 1):
        diff = ImageChops.difference(base, Image.open(p).convert("L").resize((96, 54)))
        if max(diff.getdata()) > 24:
            return max(0, i - 1)
    return 0


def pick(fr: list[tuple[int, Path]], n: int, window: int | None) -> list[tuple[int, Path]]:
    i0 = first_change(fr)
    t0 = fr[i0][0]
    t1 = fr[-1][0] if window is None else min(fr[-1][0], t0 + window)
    out = []
    for k in range(n):
        t = t0 + (t1 - t0) * k / max(1, n - 1)
        cand = [f for f in fr if f[0] <= t + 1]
        out.append(cand[-1] if cand else fr[0])
    return out


def sheet(rows: list[tuple[str, list[tuple[int, Path]]]], width: int, out: Path) -> None:
    th = None
    tiles = []
    for _, picked in rows:
        r = []
        for t, p in picked:
            im = Image.open(p).convert("RGB")
            h = round(im.height * width / im.width)
            th = h
            r.append((t, im.resize((width, h))))
        tiles.append(r)
    pad, lab, head = 4, 18, 22
    cols = max(len(r) for r in tiles)
    W = 70 + cols * (width + pad)
    H = head + len(rows) * (th + lab + pad)
    canvas = Image.new("RGB", (W, H), (24, 24, 27))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), out.stem, fill=(230, 230, 230))
    for ri, (name, _) in enumerate(rows):
        y = head + ri * (th + lab + pad)
        d.text((6, y + th // 2), name, fill=(250, 250, 250))
        for ci, (t, im) in enumerate(tiles[ri]):
            x = 70 + ci * (width + pad)
            canvas.paste(im, (x, y))
            d.text((x + 3, y + th + 3), f"{t} ms", fill=(190, 190, 190))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    root = Path(sys.argv[1])
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    window = int(sys.argv[sys.argv.index("--window-ms") + 1]) if "--window-ms" in sys.argv else 720
    labels = [l for l in ("before", "after") if (root / l).is_dir()]
    cases = sorted({p.relative_to(root / l).parts[:2] for l in labels for p in (root / l).glob("*/*") if p.is_dir()})
    for scenario, tag in cases:
        rows = []
        for l in labels:
            d = root / l / scenario / tag
            fr = frames(d) if d.is_dir() else []
            if len(fr) > 1:
                long = scenario in {"skeleton-arrival", "live-1hz", "engine-restart", "playground-streaming"}
                rows.append((l, pick(fr, n, None if long else window)))
        if rows:
            vp = int(tag.rsplit("-", 1)[1])
            sheet(rows, 190 if vp == 390 else 300, root / "sheets" / f"{scenario}-{tag}.png")
            print("sheet", scenario, tag)


if __name__ == "__main__":
    main()
