"""CPU test of the EmbeddingGemma 2 route check (route_checks_media.embed_gemma2_served) against a
canned good server: request shapes (object items with data URIs, vLLM messages, task, dimensions,
bad-input 400s) and the reference comparison. A server that ignores `task` or returns the wrong
audio vector must fail."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "python"))

import egemma2_cases as ec  # noqa: E402
import route_checks as rc  # noqa: E402
import route_checks_media as m  # noqa: E402

from yunshu_gateway.routers.embeddings import messages_to_item  # noqa: E402


def _blob(x):
    if isinstance(x, str) and x.startswith("data:"):
        return hashlib.sha1(base64.b64decode(x.split(",", 1)[1])).hexdigest()
    if isinstance(x, str) and len(x) < 300 and Path(x).is_file():
        return hashlib.sha1(Path(x).read_bytes()).hexdigest()
    if isinstance(x, list):
        return [_blob(i) for i in x]
    return x


def key(item, task=None):
    if isinstance(item, str):
        item = {"text": item}
    return (
        json.dumps({k: _blob(v) for k, v in item.items()}, sort_keys=False) + f"|{task}"
    )


def vec(k, dim=768):
    import numpy as np

    rng = np.random.default_rng(int(hashlib.sha1(k.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def mix(a, b, w):
    import numpy as np

    v = np.array(a) * (1 - w) + np.array(b) * w
    return (v / np.linalg.norm(v)).tolist()


class Fake:
    """Canned model: pseudo-random unit vector per (item, task); the speech clip is near its
    transcript and the red picture near its caption, like the real model."""

    def __init__(self, media, ignore_task=False):
        self.media, self.ignore_task = media, ignore_task
        sp, red = key({"audio": media["speech"]}), key({"image": media["red"]})
        self.near = {
            sp: key(ec.SPEECH),
            red: key("a red circle on a white background"),
        }

    def v(self, item, task=None):
        k = key(item, None if self.ignore_task else task)
        out = vec(k)
        kk = key(item)  # the neighbours ignore the task
        if kk in self.near:
            out = mix(out, vec(self.near[kk]), 0.9)
        return out

    def __call__(self, req: httpx.Request) -> httpx.Response:
        j = json.loads(req.content) if req.content else {}
        p = req.url.path

        def js(d, s=200):
            return httpx.Response(s, json=d)

        if p == "/v1/embeddings":
            task = j.get("task")
            if task not in (None, "SearchQuery", "Document"):
                return js(
                    {
                        "error": {
                            "message": "unknown task",
                            "type": "invalid_request_error",
                        }
                    },
                    400,
                )
            if "messages" in j:
                items = [messages_to_item(j["messages"])]
            else:
                items = j["input"] if isinstance(j["input"], list) else [j["input"]]
            text0 = items[0].get("text") if isinstance(items[0], dict) else items[0]
            bad = (
                (isinstance(text0, str) and len(text0) > 50000)
                or (
                    isinstance(text0, str)
                    and text0.count("<|image|>")
                    and "image" not in items[0]
                )
                or (
                    isinstance(items[0], dict)
                    and any(
                        str(items[0].get(k, "")).endswith("AAAA")
                        for k in ("audio", "image")
                    )
                )
            )
            if bad:
                return js(
                    {
                        "error": {
                            "message": "bad input",
                            "type": "invalid_request_error",
                        }
                    },
                    400,
                )
            data = []
            for i, it in enumerate(items):
                x = self.v(self._ref_form(it), task)
                if j.get("dimensions"):
                    import numpy as np

                    x = np.array(x[: j["dimensions"]])
                    x = (x / np.linalg.norm(x)).tolist()
                data.append({"object": "embedding", "index": i, "embedding": x})
            return js({"object": "list", "data": data, "model": "m",
                       "usage": {"prompt_tokens": 2000 * len(items), "total_tokens": 2000 * len(items)}})  # fmt: skip
        if p == "/v1/score":
            a = vec(key(j["text_1"]))
            sims = [
                m.cosine(a, mix(a, vec(key(t)), 0.0 if "kitten" in t else 1.0))
                for t in j["text_2"]
            ]
            return js({"data": [{"index": i, "score": s} for i, s in enumerate(sims)]})
        if p == "/api/embed":
            return js({"embeddings": [self.v(t) for t in j["input"]]})
        return js({}, 404)

    @staticmethod
    def _ref_form(it):
        return it


def build(ignore_task=False, wrong_audio=False):
    import tempfile

    d = tempfile.mkdtemp(prefix="eg2t-")
    media = ec.make_media(d)
    fake = Fake(media, ignore_task)
    if wrong_audio:
        fake.near.pop(key({"audio": media["speech"]}))
    http = httpx.Client(transport=httpx.MockTransport(fake), base_url="http://srv")
    cs = ec.cases(media)

    def ref(items):
        return [fake.v(i) for i in items]

    ctx = rc.Ctx(
        url="http://srv", token="", model="m", kind="embed2", http=http,
        oa=openai.OpenAI(base_url="http://srv/v1", api_key="x", http_client=http, max_retries=0),
        an=None, shared={},
        fixtures={"embed_reference": ref, "egemma2_media": d,
                  "embed_reference_task": lambda t, task: fake.v(t, task)},
    )  # fmt: skip
    return ctx, cs


def test_good_server_passes_every_assertion():
    ctx, _ = build()
    rc.REGISTRY["embed_gemma2_served"].fn(ctx)
    assert ctx.notes["egemma2_min_cos"] > 0.999


def test_server_that_ignores_task_fails():
    ctx, _ = build(ignore_task=True)
    with pytest.raises(rc.Fail, match="task"):
        rc.REGISTRY["embed_gemma2_served"].fn(ctx)


def test_audio_that_is_not_near_its_transcript_fails():
    ctx, _ = build(wrong_audio=True)
    with pytest.raises(rc.Fail, match="audio"):
        rc.REGISTRY["embed_gemma2_served"].fn(ctx)


def test_wire_items_use_data_uris_and_keep_text(tmp_path):
    media = ec.make_media(str(tmp_path))
    w = m.egemma2_wire_item(
        {"text": "a", "image": [media["red"]], "video": [[media["frame0"]]]}
    )
    assert w["text"] == "a" and w["image"][0].startswith("data:image/png;base64,")
    assert w["video"][0][0].startswith("data:image/png;base64,")
    assert re.match(
        r"data:audio/wav;base64,",
        m.egemma2_wire_item({"audio": media["tone"]})["audio"],
    )
