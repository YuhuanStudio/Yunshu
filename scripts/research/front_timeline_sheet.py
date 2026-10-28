"""Console frames next to the engine timeline (CPU only; reads a console_review.py run directory).

    python scripts/research/front_timeline_sheet.py RUN_DIR VIEWPORT PAGE [--cells 12]

For a page recorded by review-capture.mjs (frames f-*.jpg, shifts.json with the wall-clock `times` of
each frame) and the run's replay.jsonl (engine /status every 250 ms, each line with its wall clock),
writes RUN_DIR/timelines/<viewport>-<page>.jpg: a row of evenly spaced frames, and under it the
engine's active requests (step line) and live decode tok/s (bars) over the same window, with a tick
under every frame at the moment it was taken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_replay(path: Path) -> list[dict]:
    """[{wall, active, decode_tps}] from replay.jsonl status lines that carry a wall clock."""
    rows = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        st = rec.get("status")
        if not st or "wall" not in rec:
            continue
        rows.append(
            {
                "wall": rec["wall"],
                "active": (st.get("requests") or {}).get("active") or 0,
                "decode_tps": (st.get("throughput") or {}).get("live_decode_tps")
                or 0.0,
            }
        )
    return rows


def pick(n_frames: int, cells: int) -> list[int]:
    """Evenly spaced frame indices, always including the first and the last (pure)."""
    if n_frames <= cells:
        return list(range(n_frames))
    return sorted({round(i * (n_frames - 1) / (cells - 1)) for i in range(cells)})


def window(rows: list[dict], t0: float, t1: float) -> list[dict]:
    """Replay rows inside the frame window (pure)."""
    return [r for r in rows if t0 <= r["wall"] <= t1]


def main() -> int:
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("viewport")
    ap.add_argument("page")
    ap.add_argument("--cells", type=int, default=12)
    a = ap.parse_args()
    run = Path(a.run)
    pdir = run / a.viewport / a.page
    shifts = json.loads((pdir / "shifts.json").read_text())
    times = shifts.get("times") or []
    frames = sorted(pdir.glob("f-*.jpg"))
    if not times or len(times) != len(frames):
        print(
            f"FAIL {pdir}: {len(times)} frame times for {len(frames)} frames (older capture?)"
        )
        return 1
    rows = window(load_replay(run / "replay.jsonl"), times[0] / 1000, times[-1] / 1000)
    if not rows:
        print("FAIL no engine samples inside the frame window")
        return 1
    idx = pick(len(frames), a.cells)
    thumbs = [Image.open(frames[i]) for i in idx]
    cw = 300
    ch = int(thumbs[0].height * cw / thumbs[0].width)
    cols = min(len(thumbs), 6)
    rws = (len(thumbs) + cols - 1) // cols
    plot_h = 140
    W = cols * cw
    sheet = Image.new("RGB", (W, rws * ch + plot_h + 20), (20, 20, 24))
    for k, im in enumerate(thumbs):
        sheet.paste(im.resize((cw, ch)), ((k % cols) * cw, (k // cols) * ch))
    d = ImageDraw.Draw(sheet)
    top = rws * ch + 10
    t0, t1 = times[0] / 1000, times[-1] / 1000
    x = lambda t: int((t - t0) / (t1 - t0) * (W - 1))  # noqa: E731
    max_tps = max(r["decode_tps"] for r in rows) or 1.0
    max_act = max(r["active"] for r in rows) or 1
    for r in rows:
        h = int(r["decode_tps"] / max_tps * (plot_h - 20))
        d.line(
            [(x(r["wall"]), top + plot_h), (x(r["wall"]), top + plot_h - h)],
            fill=(70, 130, 200),
        )
    prev = None
    for r in rows:
        y = top + plot_h - int(r["active"] / max_act * (plot_h - 20))
        if prev is not None:
            d.line([prev, (x(r["wall"]), y)], fill=(240, 180, 60), width=2)
        prev = (x(r["wall"]), y)
    for i in idx:
        d.line(
            [
                (x(times[i] / 1000), top + plot_h),
                (x(times[i] / 1000), top + plot_h + 8),
            ],
            fill=(255, 255, 255),
        )
    d.text(
        (4, top),
        f"active requests (max {max_act}, orange) / decode tok/s (max {max_tps:.0f}, blue)",
        fill=(220, 220, 220),
    )
    out = run / "timelines"
    out.mkdir(exist_ok=True)
    path = out / f"{a.viewport}-{a.page}.jpg"
    sheet.save(path, quality=80)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
