"""Render captured agent request bodies to prompt token ids with the server's own path.

The gateway routers convert an Anthropic / Chat Completions / Responses body to engine
messages; ``VLMEngine`` templates and tokenizes them. This drives the real routers through
FastAPI's TestClient against a VLMEngine that has a tokenizer but no model, and records the
tokens the engine would have prefilled (CPU only).

    from render_bodies import Renderer
    r = Renderer("/path/to/model")
    out = r.render(body_dict)            # {"ids": [...], "text": "...", "messages": [...]}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YUNSHU_AUTH_DISABLED", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))


class _CapturedError(Exception):
    pass


def kind_of(body: dict) -> str:
    if "input" in body and "messages" not in body:
        return "responses"
    if "system" in body or "stop_sequences" in body:
        return "anthropic"
    for m in body.get("messages", []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for b in c
        ):
            return "anthropic"
    return "chat"


class Renderer:
    def __init__(self, model_path: str):
        from mlx_lm.utils import load_config, load_tokenizer

        from yunshu_engine.vlm_engine import VLMEngine

        recorded: list[dict] = []

        class _Eng(VLMEngine):
            def _record(self, messages, kwargs):
                tpl = self._request_template_extra(kwargs)
                think = self._default_enable_thinking(
                    kwargs.get("enable_thinking"),
                    constrained=kwargs.get("json_schema") is not None,
                )
                text = self._format_prompt(
                    messages, enable_thinking=think, template_extra=tpl
                )
                ids = self._tokenize_with_cache(
                    messages, enable_thinking=think, template_extra=tpl
                ).tolist()
                recorded.append({"ids": ids, "text": text, "messages": messages})
                raise _CapturedError()

            async def generate(self, prompt=None, messages=None, **kw):
                self._record(prompt if prompt is not None else messages, kw)

            async def generate_stream(self, prompt=None, messages=None, **kw):
                self._record(prompt if prompt is not None else messages, kw)
                yield None

        eng = _Eng(model_path)
        eng._tokenizer = load_tokenizer(Path(model_path))
        eng._config = load_config(Path(model_path))
        eng._model = object()
        eng._running = True
        self.engine = eng
        self.recorded = recorded
        from fastapi.testclient import TestClient

        from yunshu_gateway.engine import set_engine
        from yunshu_gateway.main import create_app

        set_engine(eng)
        self._client = TestClient(create_app(), raise_server_exceptions=False)
        self._client.__enter__()

    def render(self, body: dict) -> dict:
        body = dict(body)
        body["stream"] = False
        kind = kind_of(body)
        path = {
            "anthropic": "/v1/messages",
            "responses": "/v1/responses",
            "chat": "/v1/chat/completions",
        }[kind]
        self.recorded.clear()
        hdr = {"x-api-key": "k", "anthropic-version": "2023-06-01"}
        self._client.post(path, json=body, headers=hdr)
        if not self.recorded:
            raise RuntimeError(f"no engine call captured for {kind} body")
        return self.recorded[-1]


def load_body(p: Path) -> dict:
    return json.loads(Path(p).read_text())


if __name__ == "__main__":
    r = Renderer(sys.argv[1])
    for f in sys.argv[2:]:
        out = r.render(load_body(Path(f)))
        print(f, len(out["ids"]))
