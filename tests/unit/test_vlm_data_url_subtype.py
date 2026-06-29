"""a subtype-less data: image URL must not crash with IndexError.

`data:image;base64,<...>` (valid base64 image, no MIME subtype) passed _extract_images'
`data:image` dispatch but then hit header.split("/")[1] in _save_base64_image, raising
IndexError → a generic 500. Default the extension to png when the subtype is absent.
"""

from __future__ import annotations

import asyncio
import base64
import threading

from yunshu_engine.vlm_engine import VLMEngine


def _decode_ext(data_url: str) -> str:
    eng = VLMEngine.__new__(VLMEngine)
    eng._temp_files = []  # registry used by _register_temp_file
    eng._temp_files_lock = threading.Lock()
    path = asyncio.run(eng._save_base64_image(data_url))
    # the extension is derived from the path suffix
    return path.rsplit(".", 1)[-1]


def test_subtype_less_data_url_defaults_to_png():
    # 1x1 png
    png = base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"
        )
    ).decode()
    path = _decode_ext(f"data:image;base64,{png}")
    assert path in ("png",)


def test_normal_data_url_still_works():
    png = base64.b64encode(b"\x89PNG\r\n").decode()
    assert _decode_ext(f"data:image/jpeg;base64,{png}") in ("jpg", "jpeg", "png")
