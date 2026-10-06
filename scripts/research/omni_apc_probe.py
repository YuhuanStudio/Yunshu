"""Why does a repeated image not hit the prefix cache on Qwen3-Omni (a repeated audio clip does)?

Runs the processor twice on the same image and prints, per output key, whether the two runs are
identical and what the APC salt (mlx_vlm.apc.semantic_extra_hash) is each time. Usage:
    omni_apc_probe.py MODEL_DIR OUT_JSON
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path


def png(size=224, rgb=(200, 30, 30)) -> bytes:
    def chunk(tag, data):
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def digest(v):
    import numpy as np

    try:
        a = np.asarray(v)
        return [list(a.shape), str(a.dtype), float(np.abs(a.astype("float64")).sum())]
    except Exception as e:  # noqa: BLE001
        return f"{type(v).__name__}: {e}"


def main(model_dir: str, out: str) -> int:
    from mlx_vlm import apc
    from mlx_vlm.utils import load_processor, prepare_inputs

    proc = load_processor(Path(model_dir))
    d = tempfile.mkdtemp()
    p = Path(d) / "a.png"
    p.write_bytes(png())
    runs = []
    for _ in range(2):
        raw = prepare_inputs(
            proc,
            images=[str(p)],
            audio=None,
            prompts="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What colour?<|im_end|>\n<|im_start|>assistant\n",
            image_token_index=None,
            add_special_tokens=True,
        )
        pv = raw.get("pixel_values")
        runs.append(
            {
                "keys": sorted(raw),
                "digests": {k: digest(v) for k, v in raw.items()},
                "image_hash": apc.hash_image_payload(pixel_values=pv)
                if pv is not None
                else None,
                "pixel_values_is_none": pv is None,
            }
        )
    same = {
        k: runs[0]["digests"].get(k) == runs[1]["digests"].get(k)
        for k in runs[0]["digests"]
    }
    res = {
        "runs": runs,
        "same": same,
        "hash_equal": runs[0]["image_hash"] == runs[1]["image_hash"],
        "complete": True,
    }
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(
        json.dumps(
            {
                "same": same,
                "hash_equal": res["hash_equal"],
                "pv_none": runs[0]["pixel_values_is_none"],
            }
        ),
        flush=True,
    )
    return 0 if res["hash_equal"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
