"""Absent sampling parameters take the served model's generation_config, then the API default."""

import json
import re
from pathlib import Path

import pytest

from yunshu_gateway import sampling_defaults as sd
from yunshu_gateway.routers.anthropic import AnthropicMessage, AnthropicMessagesRequest
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest
from yunshu_gateway.routers.responses import ResponsesRequest

# Qwen3.8-27B-oQ4e-mtp/generation_config.json
QWEN = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    def make(gc):
        d = tmp_path / "m"
        d.mkdir(exist_ok=True)
        if gc is not None:
            (d / "generation_config.json").write_text(json.dumps(gc))
        monkeypatch.setattr(sd, "model_path_for", lambda model=None: str(d))
        return d

    return make


def _build(kind, **kw):
    if kind == "chat":
        return ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], **kw
        )
    if kind == "completions":
        return CompletionRequest(model="m", prompt="hi", **kw)
    if kind == "responses":
        return ResponsesRequest(model="m", input="hi", **kw)
    return AnthropicMessagesRequest(
        model="m", messages=[AnthropicMessage(role="user", content="hi")], **kw
    )


KINDS = ["chat", "completions", "responses", "anthropic"]


@pytest.mark.parametrize("kind", KINDS)
def test_absent_takes_generation_config(kind, model_dir):
    model_dir(QWEN)
    req = _build(kind)
    assert req.temperature is None and req.top_p is None and req.top_k is None
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k) == (1.0, 0.95, 20)
    assert (req.min_p, req.repetition_penalty) == (0.0, 1.0)


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_wins_even_when_equal_to_old_default(kind, model_dir):
    model_dir(QWEN)
    req = _build(kind, temperature=0.7, top_p=1.0, top_k=0)
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k) == (0.7, 1.0, 0)


@pytest.mark.parametrize("kind", KINDS)
def test_greedy_stays_greedy(kind, model_dir):
    model_dir(QWEN)
    req = _build(kind, temperature=0.0)
    sd.resolve_sampling(req)
    assert req.temperature == 0.0 and req.top_k == 20


@pytest.mark.parametrize("kind", KINDS)
def test_missing_key_falls_back_to_api_default(kind, model_dir):
    model_dir({"top_k": 40})
    req = _build(kind)
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k) == (1.0, 1.0, 40)


@pytest.mark.parametrize("gc", [None, {}, "junk"])
@pytest.mark.parametrize("kind", KINDS)
def test_no_or_empty_generation_config(kind, gc, model_dir):
    model_dir(gc)
    req = _build(kind)
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k, req.min_p) == (1.0, 1.0, 0, 0.0)


def test_penalties_only_when_defined(model_dir):
    model_dir({"repetition_penalty": 1.05, "presence_penalty": 1.5})
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.repetition_penalty, req.presence_penalty) == (1.05, 1.5)
    assert req.frequency_penalty == 0.0


def test_invalid_values_ignored(model_dir):
    model_dir({"temperature": -1, "top_p": 7, "top_k": 2.5, "min_p": True})
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k, req.min_p) == (1.0, 1.0, 0, 0.0)


def test_resolve_is_idempotent_and_reads_are_cached(model_dir, monkeypatch):
    d = model_dir(QWEN)
    sd._CACHE.clear()
    reads = []
    orig = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: reads.append(self) or orig(self, *a, **k),
    )
    for _ in range(3):
        req = _build("chat")
        sd.resolve_sampling(req)
        sd.resolve_sampling(req)
    assert len(reads) == 1
    (d / "generation_config.json").write_text(json.dumps({"temperature": 0.3}))
    req = _build("chat")
    sd.resolve_sampling(req)
    assert req.temperature == 0.3


def test_sources_reported(model_dir):
    model_dir(QWEN)
    rep = sd.fill_sampling(
        {"temperature": 0.2, "top_p": None, "min_p": None}, sd.model_path_for()
    )
    assert rep["temperature"] == {"value": 0.2, "source": "request"}
    assert rep["top_p"] == {"value": 0.95, "source": "generation_config"}
    assert rep["min_p"] == {"value": 0.0, "source": "api_default"}


def test_per_model_in_multi_model_mode(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, t in ((a, 0.6), (b, 0.9)):
        d.mkdir()
        (d / "generation_config.json").write_text(json.dumps({"temperature": t}))

    class Entry:
        def __init__(self, p):
            self.model_path = str(p)

    class Mgr:
        def get_entry(self, mid):
            return {"a": Entry(a), "b": Entry(b)}.get(mid)

    import yunshu_gateway.engine as ge

    monkeypatch.setattr(ge, "get_model_manager", lambda: Mgr())
    ra = ChatCompletionRequest(model="a", messages=[{"role": "user", "content": "x"}])
    rb = ChatCompletionRequest(model="b", messages=[{"role": "user", "content": "x"}])
    sd.resolve_sampling(ra)
    sd.resolve_sampling(rb)
    assert (ra.temperature, rb.temperature) == (0.6, 0.9)


def test_real_qwen_file_shape():
    p = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    if not (p / "generation_config.json").exists():
        pytest.skip("27B not on disk")
    got = sd.read_generation_config(p)
    assert got == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_every_generation_entry_resolves():
    root = Path(__file__).resolve().parents[2] / "python/yunshu_gateway/routers"
    for fn, name in (
        ("chat.py", "create_chat_completion"),
        ("chat.py", "_create_chat_completion"),
        ("completions.py", "create_completion"),
        ("responses.py", "create_response"),
        ("anthropic.py", "create_message"),
        ("anthropic.py", "_create_message"),
    ):
        src = (root / fn).read_text()
        m = re.search(rf"async def {name}\(.*?\n(.*?)\n(?:async )?def ", src, re.S)
        assert m and "resolve_sampling(req)" in m.group(1)[:3000], (fn, name)


def test_models_declare_sampling_unset():
    for kind in KINDS:
        req = _build(kind)
        for f in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
            assert getattr(req, f) is None


def test_real_check_script_finds_sampling():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sampdef_real",
        Path(__file__).resolve().parents[2] / "scripts/research/sampdef_real.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    body = {"usage": {"x_yunshu": {"sampling": {"temperature": 1}}}, "id": "x"}
    assert mod.find(body) == {"temperature": 1}
    assert mod.find({"id": 1}) is None


def test_x_yunshu_carries_sampling(model_dir):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway.x_yunshu import RequestInfo

    model_dir(QWEN)
    info = RequestInfo.__new__(RequestInfo)
    info.sampling = None
    tok = current_request_info.set(info)
    try:
        sd.resolve_sampling(_build("chat", temperature=0.5))
    finally:
        current_request_info.reset(tok)
    assert info.sampling["temperature"] == 0.5
    assert info.sampling["top_k"] == 20
    assert info.sampling["source"]["temperature"] == "request"
    assert info.sampling["source"]["top_k"] == "generation_config"
