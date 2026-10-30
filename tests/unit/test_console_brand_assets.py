"""The console's brand asset set (favicon, touch icon, manifest) is served by the console process."""

import json
import shutil
import struct
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_console.app import ConsoleStaticFiles

PUBLIC = Path(__file__).resolve().parents[2] / "frontend" / "public"


def _client(tmp_path):
    console = tmp_path / "console_static"
    console.mkdir()
    (console / "index.html").write_text("<main>shell</main>", encoding="utf-8")
    for name in (
        "favicon.ico",
        "apple-touch-icon.png",
        "manifest.json",
        "web-app-manifest-192x192.png",
        "web-app-manifest-512x512.png",
    ):
        shutil.copy(PUBLIC / name, console / name)
    app = FastAPI()
    app.mount("/console", ConsoleStaticFiles(console), name="console")
    return TestClient(app)


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_favicon_and_touch_icon_resolve(tmp_path):
    client = _client(tmp_path)
    ico = client.get("/console/favicon.ico")
    assert ico.status_code == 200
    # The shared YuhuanStudio favicon.ico (same bytes as Yunxin / YunUI, pinned in brand-icons.mjs) is a
    # 32 px PNG, which every browser reads under the .ico name.
    assert _png_size(ico.content) == (32, 32)
    # The shared YuhuanStudio icon set is raster only (no favicon.svg since the brand change).
    assert client.get("/console/favicon.svg").status_code == 404
    assert _png_size(client.get("/console/apple-touch-icon.png").content) == (180, 180)


def test_manifest_icons_exist_with_their_declared_size(tmp_path):
    client = _client(tmp_path)
    manifest = client.get("/console/manifest.json")
    assert manifest.status_code == 200
    data = json.loads(manifest.text)
    assert data["name"] == "Yunshu" and data["display"] == "standalone"
    assert data["start_url"].startswith("/console/") and data["scope"] == "/console/"
    raster = [i for i in data["icons"] if i["type"] == "image/png"]
    assert {i["sizes"] for i in raster} == {"192x192", "512x512"}
    for icon in data["icons"]:
        assert icon["src"].startswith("/console/")
        response = client.get(icon["src"])
        assert response.status_code == 200, icon["src"]
        if icon["type"] == "image/png":
            side = int(icon["sizes"].split("x")[0])
            assert _png_size(response.content) == (side, side)
