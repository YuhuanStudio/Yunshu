"""video.py — generate a video from a text prompt (Wan 2.x / LTX-2), on-device.

Yunshu runs Apple-MLX video-diffusion models locally. Point it at a Wan or LTX
model, POST a prompt, get back an MP4 — text-to-video, or image-to-video with
``--image``.

─────────────────────────────────────────────────────────────────────────────
1. Install the video extra and serve a video model. A model whose name contains
   "wan" / "ltx" / "video" is routed to the video engine automatically:

       uv sync --extra video
       uv run yunshu serve -m /path/to/Wan2.2-TI2V-5B-mlx --port 8000

2. Run this (text-to-video):

       uv run --with requests python examples/video.py "a fluffy cat in a garden"

   Image-to-video (animate a still):

       uv run --with requests python examples/video.py "gentle wind" --image photo.jpg

Writes video_out.mp4. Video diffusion is heavy — a few-second clip is a minute or
two on a 5B model. Pass fewer --frames / --steps (or a smaller --width/--height)
to go faster while you experiment.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import base64
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "prompt",
        nargs="?",
        default="a fluffy ginger cat walking slowly across a sunny garden",
        help="text description of the video",
    )
    ap.add_argument("--url", default="http://localhost:8000", help="server base URL")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument(
        "--frames", type=int, default=49, help="frame count (Wan needs 4n+1); ~16 fps"
    )
    ap.add_argument("--steps", type=int, default=20, help="denoising steps")
    ap.add_argument("--image", help="source image path for image-to-video (I2V)")
    ap.add_argument("--out", default="video_out.mp4")
    args = ap.parse_args()

    body: dict = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "response_format": "mp4",
    }
    if args.image:
        with open(args.image, "rb") as f:
            body["image"] = base64.b64encode(f.read()).decode()

    mode = "image-to-video" if args.image else "text-to-video"
    print(
        f'{mode}: "{args.prompt}"\n'
        f"  {args.width}x{args.height}, {args.frames} frames, {args.steps} steps "
        f"— this can take a minute or two…",
        flush=True,
    )
    t0 = time.time()
    try:
        r = requests.post(f"{args.url}/v1/video/generations", json=body, timeout=1800)
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"could not connect to {args.url} — is the server running?"
        ) from None

    if r.status_code == 503:
        raise SystemExit(
            "\n✗ No video model is loaded. Serve one (name must contain wan/ltx/video) "
            "and install the extra:\n\n"
            "    uv sync --extra video\n"
            "    uv run yunshu serve -m /path/to/Wan2.2-TI2V-5B-mlx --port 8000\n"
        )
    r.raise_for_status()

    item = r.json()["data"][0]
    if not item.get("video"):
        raise SystemExit(f"server returned no video (method={item.get('method')})")
    with open(args.out, "wb") as f:
        f.write(base64.b64decode(item["video"]))
    dt = time.time() - t0
    print(
        f"\n✓ wrote {args.out} — {item['num_frames']} frames "
        f"{item['width']}x{item['height']} @ {item['fps']}fps "
        f"in {dt:.0f}s (method: {item['method']})"
    )


if __name__ == "__main__":
    main()
