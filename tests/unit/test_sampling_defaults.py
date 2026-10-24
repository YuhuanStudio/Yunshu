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
    kw.setdefault("max_tokens", 64)  # required by the Messages API
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


# ── per-mode published recommendations ───────────────────────────────────────

from yunshu_engine import sampling_defaults as eng  # noqa: E402

Q38 = "Qwen3.8-27B-oQ4e-mtp"
THINK_TPL = (
    "{% if enable_thinking is undefined or enable_thinking is true %}<think>{% endif %}"
)
NOTHINK_TPL = (
    "{% if enable_thinking is defined and enable_thinking is true %}<think>{% endif %}"
)


@pytest.fixture
def named(tmp_path, monkeypatch):
    def make(name, gc=QWEN, tpl=THINK_TPL):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        if gc is not None:
            (d / "generation_config.json").write_text(json.dumps(gc))
        if tpl is not None:
            (d / "chat_template.jinja").write_text(tpl)
        eng._MODE_CACHE.clear()
        monkeypatch.setattr(sd, "model_path_for", lambda model=None: str(d))
        return d

    return make


def test_thinking_mode_uses_thinking_recommendation(named):
    named(Q38)
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k, req.presence_penalty) == (
        1.0,
        0.95,
        20,
        0.0,
    )


def test_non_thinking_mode_switches_set(named):
    named(Q38)
    for kw in (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ):
        req = _build("chat", **kw)
        sd.resolve_sampling(req)
        assert (req.temperature, req.top_p, req.presence_penalty) == (0.7, 0.80, 1.5), (
            kw
        )


def test_anthropic_thinking_param_picks_mode(named):
    named(Q38)
    off = _build("anthropic", thinking={"type": "disabled"})
    sd.resolve_sampling(off)
    assert off.temperature == 0.7 and off.presence_penalty == 1.5
    on = _build(
        "anthropic", thinking={"type": "enabled", "budget_tokens": 512}, max_tokens=2048
    )
    sd.resolve_sampling(on)
    assert on.temperature == 1.0 and on.presence_penalty == 0.0
    adaptive = _build("anthropic", thinking={"type": "adaptive"})
    sd.resolve_sampling(adaptive)
    assert adaptive.temperature == 1.0


def test_responses_enable_thinking_false(named):
    named(Q38)
    req = _build("responses", enable_thinking=False)
    sd.resolve_sampling(req)
    assert req.temperature == 0.7 and req.top_p == 0.80


def test_constrained_request_is_non_thinking(named):
    named(Q38)
    req = _build("chat", response_format={"type": "json_object"})
    sd.resolve_sampling(req)
    assert req.temperature == 0.7


def test_template_default_off_model_uses_non_thinking_set(named):
    named("Qwen3.5-2B-MLX-bf16", gc=None, tpl=NOTHINK_TPL)
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.presence_penalty) == (1.0, 1.0, 2.0)


def test_explicit_beats_recommendation_and_fields_are_independent(named):
    named(Q38)
    req = _build("chat", temperature=0.2, top_k=0)
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_k, req.top_p) == (0.2, 0, 0.95)


def test_unknown_model_falls_through_to_generation_config(named):
    named("SomeOther-7B", gc={"temperature": 0.6})
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.presence_penalty) == (0.6, 1.0, 0.0)


def test_profile_field_missing_falls_to_generation_config(named, monkeypatch, tmp_path):
    f = tmp_path / "p.json"
    f.write_text(
        json.dumps({"profiles": [{"names": ["m"], "thinking": {"top_p": 0.5}}]})
    )
    monkeypatch.setattr(eng, "PROFILES_PATH", f)
    eng._PROFILES = None
    named("m", gc={"top_k": 7})
    r = eng.effective_defaults(sd.model_path_for(), "thinking")
    assert r["top_p"] == {"value": 0.5, "source": "model_card"}
    assert r["top_k"] == {"value": 7, "source": "generation_config"}
    assert r["temperature"]["source"] == "api_default"
    eng._PROFILES = None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"profiles": "x"}',
        '{"profiles": [1, {"names": []}, {"names": ["m"], "thinking": {"temperature": 99, "top_k": 1.5}}]}',
    ],
)
def test_malformed_profile_file_never_breaks(named, monkeypatch, tmp_path, content):
    f = tmp_path / "p.json"
    f.write_text(content)
    monkeypatch.setattr(eng, "PROFILES_PATH", f)
    eng._PROFILES = None
    named("m", gc=None)
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.top_p, req.top_k) == (1.0, 1.0, 0)
    eng._PROFILES = None


def test_missing_profile_file(named, monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "PROFILES_PATH", tmp_path / "none.json")
    eng._PROFILES = None
    named(Q38)
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.temperature, req.presence_penalty) == (
        1.0,
        0.0,
    )  # generation_config only
    eng._PROFILES = None


def test_api_policy_ignores_model(named, monkeypatch):
    from yunshu_engine import settings

    named(Q38)
    monkeypatch.setenv("YUNSHU_SAMPLING_DEFAULTS", "api")
    settings.reload() if hasattr(settings, "reload") else None
    req = _build("chat")
    sd.resolve_sampling(req)
    assert (req.top_p, req.top_k) == (1.0, 0)


def test_shipped_profiles_are_valid_and_cited():
    doc = json.loads(eng.PROFILES_PATH.read_text())
    assert doc["version"] == 1
    for e in doc["profiles"]:
        assert e["source_url"].startswith("https://huggingface.co/")
        assert e["read"] and e["quote"]
        assert set(e["thinking"]) <= set(eng._RANGES) | {"min_p"}
        for m in ("thinking", "non_thinking"):
            for k, v in e[m].items():
                assert eng._valid(k, v) is not None, (e["names"], m, k)
    assert len(eng.load_profiles()) == len(doc["profiles"])


@pytest.mark.parametrize(
    "name,norm",
    [
        ("Qwen3.8-27B-oQ4e-mtp", "qwen3.8-27b"),
        ("Qwen3.5-9B-MLX-4bit", "qwen3.5-9b"),
        ("Qwen/Qwen3.5-4B", "qwen3.5-4b"),
        ("/x/y/Qwen3.5-0.8B-MLX-bf16/", "qwen3.5-0.8b"),
    ],
)
def test_name_matching(name, norm):
    assert eng.normalize_name(name) == norm
    assert eng.profile_for(name) is not None


def test_x_yunshu_reports_mode_and_sources(named):
    from yunshu_engine.request_tracker import current_request_info
    from yunshu_gateway.x_yunshu import RequestInfo

    named(Q38)
    info = RequestInfo.__new__(RequestInfo)
    info.sampling = None
    tok = current_request_info.set(info)
    try:
        sd.resolve_sampling(_build("chat", enable_thinking=False))
    finally:
        current_request_info.reset(tok)
    s = info.sampling
    assert s["mode"] == "non_thinking" and s["mode_from"] == "request"
    assert s["temperature"] == 0.7 and s["presence_penalty"] == 1.5
    assert s["source"]["temperature"] == "model_card"
    assert s["source"]["frequency_penalty"] == "api_default"


def test_surfaces_card_log_doctor(named):
    d = named(Q38)
    by = eng.defaults_by_mode(d)
    assert by["default_mode"] == "thinking" and by["profile"]["source_url"]
    assert by["non_thinking"]["temperature"] == {"value": 0.7, "source": "model_card"}
    line = eng.log_line(d)
    assert "thinking T=1.0" in line and "non_thinking T=0.7" in line
    from yunshu_cli.doctor import check_sampling

    c = check_sampling(str(d))[0]
    assert c.status == "ok" and "T=0.7" in c.detail
    from yunshu_engine.model_card import _effective_sampling

    assert _effective_sampling(d)["thinking"]["top_k"]["value"] == 20


def test_old_hardcoded_default_gone():
    root = Path(__file__).resolve().parents[2] / "python/yunshu_gateway/routers"
    for fn in ("chat.py", "anthropic.py"):
        assert "default=0.7" not in (root / fn).read_text().split("temperature")[1][:80]


def test_modes_check_script_finds_sampling():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sampdef_modes",
        Path(__file__).resolve().parents[2] / "scripts/research/sampdef_modes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.find({"a": [{"x_yunshu": {"sampling": {"mode": "thinking"}}}]}) == {
        "mode": "thinking"
    }
    prof, mode = mod.profile("/nonexistent/Qwen3.8-27B-oQ4e-mtp")
    assert prof["thinking"]["top_k"] == 20 and mode == "non_thinking"


def test_effort_none_and_absent_anthropic_thinking_use_the_non_thinking_profile(named):
    named(Q38)
    off = _build("anthropic")  # no thinking field = disabled
    sd.resolve_sampling(off)
    assert off.temperature == 0.7 and off.presence_penalty == 1.5
    for kind in ("chat", "responses"):
        req = _build(kind, reasoning_effort="none")
        sd.resolve_sampling(req)
        assert req.temperature == 0.7 and req.presence_penalty == 1.5
