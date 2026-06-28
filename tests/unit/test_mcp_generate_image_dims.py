"""a grounded MCP JSON-RPC hunt found the /v1/mcp dispatch substantially correct
(notification vs id:null, id echo, error codes, model isolation, tool-throw→isError all
right). One real client-facing bug + a sibling, both schema-vs-behavior mismatches:

- generate_image advertised width/height integer params (default 512) in its inputSchema
  but READ an unadvertised `size` "WxH" string and IGNORED width/height → a spec-conformant
  client sending {"width":512,"height":512} silently always got 1024x1024.
- synthesize_speech advertised voice default "A cheerful female voice" while the tool
  defaults to "alloy" — the advertised default was never used.

Fix: honor width/height (size string kept as a backward-compat fallback, default 512 to
match the schema); align the voice advertised default to the actual "alloy".
"""
from __future__ import annotations

import asyncio

from yunshu_engine.image_engine import ImageGenEngine
from yunshu_gateway.routers import mcp


class _CapImg:
    def __init__(self):
        self.kw = None

    async def generate_image(self, **kwargs):
        self.kw = kwargs
        return b"\x89PNG\r\n\x1a\n"


def _run(args, monkeypatch):
    eng = ImageGenEngine.__new__(ImageGenEngine)
    cap = _CapImg()
    eng.generate_image = cap.generate_image
    import types
    entry = types.SimpleNamespace(is_loaded=True, engine=eng)
    mgr = types.SimpleNamespace(list_entries=lambda: [entry])
    from yunshu_gateway import engine as eng_mod
    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: mgr)
    resp = asyncio.run(mcp._tool_generate_image(args, 1))
    return cap.kw, resp


def test_honors_advertised_width_height(monkeypatch):
    kw, resp = _run({"prompt": "a cat", "width": 512, "height": 768}, monkeypatch)
    assert kw["width"] == 512 and kw["height"] == 768  # NOT 1024x1024


def test_size_string_is_backward_compat_fallback(monkeypatch):
    kw, _ = _run({"prompt": "a cat", "size": "640x480"}, monkeypatch)
    assert kw["width"] == 640 and kw["height"] == 480


def test_default_matches_schema_512(monkeypatch):
    kw, _ = _run({"prompt": "a cat"}, monkeypatch)
    assert kw["width"] == 512 and kw["height"] == 512


def test_tools_list_schema_consistent_with_behavior():
    resp = asyncio.run(mcp._handle_tools_list({}, 1))
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    gi = tools["generate_image"]["inputSchema"]["properties"]
    assert gi["width"]["default"] == 512 and gi["height"]["default"] == 512
    # voice advertised default now matches the tool's actual "alloy"
    sv = tools["synthesize_speech"]["inputSchema"]["properties"]["voice"]
    assert sv["default"] == "alloy"
