"""Decisions: record encoding, the MLX joint schema head against a torch transcription of the
reference, checkpoint detection (fail closed), the two wire formats and the openai 3.26 SDK."""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from yunshu_engine import decision_engine as de
from yunshu_engine.decision_engine import (
    DecisionEngine,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Option,
    Question,
)


@pytest.fixture()
def cpu():
    """Unit tests never touch the GPU: MLX runs on the CPU device here."""
    import mlx.core as mx

    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def tok(text: str) -> list[int]:
    return [ord(c) for c in text]


def detok(ids) -> str:
    return "".join(chr(i) for i in ids)


def yesno(name=None, text="is it raining?") -> Question:
    return Question(
        "predicate", name, text, (Option("true", True), Option("false", False))
    )


def choice(name, ids, text="pick one") -> Question:
    return Question("choice", name, text, tuple(Option(i, i, f"desc {i}") for i in ids))


def score(name, labels, text="rate it") -> Question:
    return Question(
        "score",
        name,
        text,
        tuple(Option(str(i), lb, f"level {lb}") for i, lb in enumerate(labels)),
    )


# ── encoding ─────────────────────────────────────────────────────────────────────────────


def test_encode_spans_point_at_the_question_and_option_text():
    qs = [
        yesno("wet"),
        choice("mood", ["sad", "glad", "mad"]),
        score("q", ["lo", "hi"]),
    ]
    rec = de.encode_record(tok, "STATE-TEXT", qs)
    text = detok(rec.input_ids)
    assert text.startswith("<|im_start|>system\n")
    assert "STATE:\nSTATE-TEXT\n\nSCHEMA FIELDS:" in text
    assert text.endswith("JOINT SCHEMA DECISIONS:")
    q0 = rec.questions[0]
    assert detok(rec.input_ids[slice(*q0.question_span)]) == "is it raining?"
    assert q0.option_ids == ("true", "false")
    assert detok(rec.input_ids[slice(*q0.option_spans[0])]).startswith(
        '{"description":'
    )
    # choices are sorted by id for the model; request order is restored in readout
    q1 = rec.questions[1]
    assert q1.option_ids == ("glad", "mad", "sad")
    assert '"option_id":"glad"' in detok(rec.input_ids[slice(*q1.option_spans[0])])
    assert [q.question_type for q in rec.questions] == [0, 1, 2]
    # scores keep level order and are numbered
    assert rec.questions[2].option_ids == ("0", "1")


def test_encode_places_media_between_prefix_and_state():
    rec = de.encode_record(tok, "S", [yesno()], media_ids=[9000, 9001])
    ids = list(rec.input_ids)
    assert ids[rec.media_offset : rec.media_offset + 2] == [9000, 9001]
    assert detok(ids[rec.media_offset + 2 :]).startswith("S\n\nSCHEMA")


def test_encode_rejects_overlong_input_instead_of_truncating():
    with pytest.raises(DecisionError, match="limit"):
        de.encode_record(tok, "x" * 400, [yesno()], max_length=300)


def test_question_ids_are_unique_and_default_to_position():
    assert de.question_ids([yesno(), yesno("a")]) == ["q1", "a"]
    with pytest.raises(DecisionError, match="duplicate"):
        de.question_ids([yesno("a"), yesno("a")])
    with pytest.raises(DecisionError, match="duplicate"):
        de.question_ids([yesno(), yesno("q1")])


def test_non_text_state_is_rendered_as_sorted_compact_json():
    rec = de.encode_record(tok, {"b": 1, "a": [1, 2]}, [yesno()])
    assert 'STATE:\n{"a":[1,2],"b":1}\n' in detok(rec.input_ids)


def test_readout_restores_request_order_and_refuses_non_finite():
    q = choice("c", ["z", "a", "m"])
    model_ids = ("a", "m", "z")  # sorted order the model saw
    probs = de.readout(q, [0.0, 0.0, math.log(3)], model_ids)
    assert probs is not None and abs(sum(probs) - 1) < 1e-9
    assert probs[0] > probs[1] and probs[0] == pytest.approx(
        3 / 5
    )  # z first in request order
    assert de.readout(q, [0.0, float("nan"), 1.0], model_ids) is None
    assert de.readout(q, [0.0, float("inf"), 1.0], model_ids) is None


# ── head parity with the torch reference ─────────────────────────────────────────────────

CFG = {
    "hidden_size": 24,
    "width": 16,
    "routing_layers": 2,
    "layers": 2,
    "heads": 2,
    "feedforward": 20,
}


def _torch_reference(cfg):
    """The reference's JointSchemaHead forward, transcribed onto torch.nn layers."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    class Routing(torch.nn.Module):
        def __init__(self, w, h, ff):
            super().__init__()
            self.query_norm = torch.nn.LayerNorm(w)
            self.memory_norm = torch.nn.LayerNorm(w)
            self.attention = torch.nn.MultiheadAttention(w, h, batch_first=True)
            self.feedforward_norm = torch.nn.LayerNorm(w)
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(w, ff),
                torch.nn.GELU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(ff, w),
                torch.nn.Dropout(0.0),
            )

        def forward(self, q, m):
            r, _ = self.attention(
                self.query_norm(q),
                self.memory_norm(m),
                self.memory_norm(m),
                need_weights=False,
            )
            q = q + r
            return q + self.feedforward(self.feedforward_norm(q))

    class Head(torch.nn.Module):
        def __init__(
            self, hidden_size, width, routing_layers, layers, heads, feedforward
        ):
            super().__init__()
            L = lambda: torch.nn.Linear(hidden_size, width, bias=False)  # noqa: E731
            self.hidden_norm = torch.nn.LayerNorm(hidden_size)
            self.memory_projection, self.question_projection = L(), L()
            self.option_question_projection, self.global_projection = L(), L()
            self.option_context_projection, self.option_lexical_projection = L(), L()
            self.type_embedding = torch.nn.Embedding(3, width)
            self.evidence_layers = torch.nn.ModuleList(
                [Routing(width, heads, feedforward) for _ in range(routing_layers)]
            )
            self.option_summary_norm = torch.nn.LayerNorm(width)
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.TransformerDecoderLayer(
                        width,
                        heads,
                        feedforward,
                        0.0,
                        "gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(layers)
                ]
            )
            self.field_norm = torch.nn.LayerNorm(width)
            self.option_norm = torch.nn.LayerNorm(width)
            self.residual_scorer = torch.nn.Sequential(
                torch.nn.Linear(width * 4, width),
                torch.nn.GELU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(width, 1),
            )
            self.prior_logit_scale = torch.nn.Parameter(torch.tensor(0.3))
            self.joint_logit_scale = torch.nn.Parameter(torch.tensor(0.7))
            self.residual_gate = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, hidden, input_ids, record, emb):
            seq = self.hidden_norm(hidden)
            memory = self.memory_projection(seq).unsqueeze(0)
            gv = seq[-1]
            qv = torch.stack(
                [
                    seq[a:b].mean(0)
                    for a, b in (q.question_span for q in record.questions)
                ]
            )
            ctxs, lexs, counts = [], [], []
            for q in record.questions:
                ctxs.append(torch.stack([seq[a:b].mean(0) for a, b in q.option_spans]))
                lexs.append(
                    torch.stack(
                        [emb[input_ids[a:b]].mean(0) for a, b in q.option_spans]
                    )
                )
                counts.append(len(q.option_spans))
            oq = [
                self.option_context_projection(c)
                + self.option_lexical_projection(l)
                + self.option_question_projection(qv[i]).unsqueeze(0)
                for i, (c, l) in enumerate(zip(ctxs, lexs, strict=True))
            ]
            routed = torch.cat(oq, 0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed = layer(routed, memory)
            split = list(torch.split(routed[0], counts, 0))
            base = self.question_projection(qv)
            sums = []
            for f, o in zip(base, split, strict=True):
                w = torch.softmax(torch.matmul(o, f) / math.sqrt(o.shape[-1]), 0)
                sums.append(torch.sum(w.unsqueeze(-1) * o, 0))
            types = torch.tensor([q.question_type for q in record.questions])
            fields = (
                base
                + self.option_summary_norm(torch.stack(sums))
                + self.global_projection(gv).unsqueeze(0)
                + self.type_embedding(types)
            ).unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])
            out = []
            for i, (field, lex, ro) in enumerate(zip(fields, lexs, split, strict=True)):
                anchor = F.normalize(qv[i] + gv, dim=-1)
                prior = self.prior_logit_scale.clamp(
                    max=math.log(100.0)
                ).exp() * torch.matmul(F.normalize(lex, dim=-1), anchor)
                opts = self.option_norm(ro)
                rf = field.unsqueeze(0).expand_as(opts)
                cos = F.cosine_similarity(rf, opts, dim=-1)
                feats = torch.cat([rf, opts, rf * opts, torch.abs(rf - opts)], -1)
                res = self.residual_scorer(feats).squeeze(-1)
                joint = (
                    self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cos + res
                )
                out.append(prior + torch.sigmoid(self.residual_gate) * joint)
            return out

    return Head(**cfg)


def test_mlx_head_matches_the_torch_reference_and_the_key_schema(cpu):
    torch = pytest.importorskip("torch")
    import mlx.core as mx

    torch.manual_seed(0)
    ref = _torch_reference(CFG).eval()
    with torch.no_grad():  # LayerNorm/params at init are trivial; perturb everything
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    sd = {k: v.detach() for k, v in ref.state_dict().items()}
    shapes = de.expected_head_shapes(CFG)
    assert {
        k: tuple(v.shape) for k, v in sd.items()
    } == shapes  # exactly the checkpoint's keys

    qs = [
        yesno("a"),
        choice("b", ["x", "y", "z"]),
        score("c", ["l0", "l1", "l2", "l3"]),
    ]
    rec = de.encode_record(lambda s: [ord(c) % 50 for c in s], "some state", qs)
    n = len(rec.input_ids)
    hidden = torch.randn(n, CFG["hidden_size"])
    emb = torch.randn(64, CFG["hidden_size"])
    with torch.no_grad():
        expected = ref(hidden, torch.tensor(rec.input_ids), rec, emb)

    W = {k: mx.array(v.numpy()) for k, v in sd.items()}
    got = de.head_logits(
        W,
        CFG,
        mx.array(hidden.numpy()),
        rec,
        lambda ids: mx.array(emb.numpy())[mx.array(ids)],
    )
    assert [len(g) for g in got] == [len(e) for e in expected] == [2, 3, 4]
    for g, e in zip(got, expected, strict=True):
        assert g == pytest.approx(e.tolist(), abs=2e-4)


# ── checkpoint detection: a head must never be dropped silently ──────────────────────────


def _checkpoint(
    tmp_path, head_cfg=None, with_weights=False, model_type="qwen3_5", hidden=5120
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": {"hidden_size": hidden},
                "vision_config": {},
            }
        )
    )
    if head_cfg is not None:
        (tmp_path / de.HEAD_CONFIG_FILE).write_text(json.dumps(head_cfg))
    if with_weights:
        (tmp_path / de.HEAD_FILE).write_bytes(b"")
    return tmp_path


def test_head_files_route_the_checkpoint_to_the_decision_type(tmp_path):
    from yunshu_engine.model_manager import ModelType, _detect_model_type

    _checkpoint(tmp_path, dict(CFG, hidden_size=5120))
    assert _detect_model_type(str(tmp_path)) is ModelType.DECISION
    only_weights = tmp_path / "w"
    only_weights.mkdir()
    _checkpoint(only_weights, with_weights=True)
    assert (
        _detect_model_type(str(only_weights)) is ModelType.DECISION
    )  # config missing -> still not an LLM
    plain = tmp_path / "p"
    plain.mkdir()
    _checkpoint(plain)
    assert _detect_model_type(str(plain)) is not ModelType.DECISION


def test_engine_fails_closed_on_unknown_or_mismatched_heads(tmp_path):
    def start(path):
        asyncio.run(DecisionEngine(str(path)).start())

    a = tmp_path / "a"
    a.mkdir()
    _checkpoint(a, {"hidden_size": 5120, "width": 8, "something_new": 1})
    with pytest.raises(ValueError, match="unknown head layout"):
        start(a)
    b = tmp_path / "b"
    b.mkdir()
    _checkpoint(b, dict(CFG, hidden_size=4096), hidden=5120)
    with pytest.raises(ValueError, match="does not match the backbone"):
        start(b)
    c = tmp_path / "c"
    c.mkdir()
    _checkpoint(c, dict(CFG, hidden_size=5120), model_type="llama")
    with pytest.raises(ValueError, match="qwen3_5"):
        start(c)
    d = tmp_path / "d"
    d.mkdir()
    _checkpoint(d, with_weights=True)
    with pytest.raises(ValueError, match=de.HEAD_CONFIG_FILE):
        start(d)


def test_head_weights_with_missing_or_misshapen_tensors_are_rejected(tmp_path, cpu):
    import mlx.core as mx

    shapes = de.expected_head_shapes(CFG)
    good = {k: mx.zeros(s) for k, s in shapes.items()}
    mx.save_safetensors(str(tmp_path / de.HEAD_FILE), good)
    assert set(de.load_head_weights(tmp_path, CFG)) == set(shapes)
    bad = dict(good)
    del bad["field_norm.bias"]
    mx.save_safetensors(str(tmp_path / de.HEAD_FILE), bad)
    with pytest.raises(ValueError, match="missing"):
        de.load_head_weights(tmp_path, CFG)
    bad = dict(good, extra_tensor=mx.zeros((1,)))
    mx.save_safetensors(str(tmp_path / de.HEAD_FILE), bad)
    with pytest.raises(ValueError, match="unexpected"):
        de.load_head_weights(tmp_path, CFG)
    bad = dict(good, **{"field_norm.bias": mx.zeros((3,))})
    mx.save_safetensors(str(tmp_path / de.HEAD_FILE), bad)
    with pytest.raises(ValueError, match="shape"):
        de.load_head_weights(tmp_path, CFG)


# ── wire formats ─────────────────────────────────────────────────────────────────────────


class FakeEngine(DecisionEngine):
    """Deterministic: option k of a question gets weight k+1."""

    def __init__(self, multimodal=True):
        super().__init__("/nonexistent/fake-decider")
        self._loaded = True
        self.supports_multimodal = multimodal
        self.seen: list[DecisionRequest] = []

    async def decide(self, req):
        self.seen.append(req)
        out = []
        for q in req.questions:
            if q.name == "boom":
                out.append(None)
                continue
            w = [float(i + 1) for i in range(len(q.options))]
            if q.kind == "predicate":
                w = [3.0, 1.0]
            out.append([x / sum(w) for x in w])
        return DecisionResult(out, 42)


@pytest.fixture()
def served(monkeypatch):
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import decisions

    engine = FakeEngine()

    async def resolve(model_id):
        return engine if model_id == "clef" else None

    monkeypatch.setattr(decisions, "_resolve_decision_engine", resolve)
    app = create_app()
    return app, engine


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    )


PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="


def test_decisions_wire_answers_in_question_order(served):
    app, engine = served
    body = {
        "model": "clef",
        "input": "It is pouring outside.",
        "questions": [
            {"type": "predicate", "instructions": "Is it raining?", "name": "rain"},
            {
                "type": "choice",
                "instructions": "Mood?",
                "choices": [
                    {"value": "sad"},
                    {"value": True, "description": "yes"},
                    {"value": "mad"},
                ],
            },
            {
                "type": "score",
                "instructions": "How wet?",
                "levels": [
                    {"label": "dry"},
                    {"label": "damp"},
                    {"label": "soaked", "description": "drenched"},
                ],
            },
            {"type": "predicate", "instructions": "x", "name": "boom"},
        ],
    }

    async def go():
        async with _client(app) as c:
            return await c.post("/v1/decisions", json=body)

    r = asyncio.run(go())
    assert r.status_code == 200, r.text
    j = r.json()
    a = j["answers"]
    assert [x["type"] for x in a] == ["predicate", "choice", "score", "refusal"]
    assert a[0] == {"type": "predicate", "name": "rain", "probability": 0.75}
    assert a[1]["choice"] == "mad" and a[1]["name"] is None
    assert [p["value"] for p in a[1]["probabilities"]] == ["sad", True, "mad"]
    assert sum(p["probability"] for p in a[1]["probabilities"]) == pytest.approx(1)
    assert a[2]["score"] == pytest.approx((0 * 1 + 1 * 2 + 2 * 3) / 6)
    assert [(p["label"], p["value"]) for p in a[2]["probabilities"]] == [
        ("dry", 0),
        ("damp", 1),
        ("soaked", 2),
    ]
    assert a[3] == {"type": "refusal", "name": "boom"}
    assert (
        j["model"] == "clef"
        and j["usage"]["input_tokens"] == 42
        and j["usage"]["output_tokens"] == 0
    )
    q = engine.seen[0].questions
    assert (
        q[2].options[2].description == "drenched"
        and q[2].options[0].description == "dry"
    )


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda b: b.update(model="nope"), "not found"),
        (lambda b: b.update(questions=[]), "at least one question"),
        (lambda b: b["questions"][0].update(type="rank"), "questions"),
        (
            lambda b: b["questions"].__setitem__(
                0, {"type": "choice", "instructions": "i", "choices": [{"value": "a"}]}
            ),
            "at least 2",
        ),
        (
            lambda b: b["questions"].__setitem__(
                0,
                {
                    "type": "choice",
                    "instructions": "i",
                    "choices": [{"value": "true"}, {"value": True}],
                },
            ),
            "distinct",
        ),
        (
            lambda b: b["questions"].__setitem__(
                0,
                {
                    "type": "score",
                    "instructions": "i",
                    "levels": [{"label": "a"}, {"label": "a"}],
                },
            ),
            "distinct",
        ),
        (lambda b: b["questions"][0].update(criteria={"a": "b"}), "criteria"),
        (lambda b: b.update(input=[{"role": "assistant", "content": "hi"}]), "input"),
        (
            lambda b: b.update(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "https://x/y.png"}
                        ],
                    }
                ]
            ),
            "inline base64",
        ),
        (lambda b: b.update(input=" "), "empty"),
        (lambda b: b["questions"].append(dict(b["questions"][0])), "duplicate"),
    ],
)
def test_decisions_rejects_bad_requests_with_the_openai_error_shape(
    served, mutate, needle
):
    app, _ = served
    body = {
        "model": "clef",
        "input": "text",
        "questions": [{"type": "predicate", "instructions": "q?", "name": "n"}],
    }
    mutate(body)

    async def go():
        async with _client(app) as c:
            return await c.post("/v1/decisions", json=body)

    r = asyncio.run(go())
    assert r.status_code in (400, 404), r.text
    err = r.json()["error"]
    assert needle in err["message"], err
    assert err["type"]


def test_images_reach_the_engine_and_are_refused_without_a_vision_tower(served):
    app, engine = served
    body = {
        "model": "clef",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": PNG},
                ],
            }
        ],
        "questions": [{"type": "predicate", "instructions": "q"}],
    }

    async def go():
        async with _client(app) as c:
            return await c.post("/v1/decisions", json=body)

    assert asyncio.run(go()).status_code == 200
    assert (
        len(engine.seen[-1].images) == 1 and engine.seen[-1].images[0][:4] == b"\x89PNG"
    )
    engine.supports_multimodal = False
    r = asyncio.run(go())
    assert (
        r.status_code == 400
        and "does not accept images" in r.json()["error"]["message"]
    )


def test_a_non_decision_model_is_a_clear_400(served, monkeypatch):
    from yunshu_gateway.routers import decisions

    app, _ = served

    async def resolve(model_id):
        return object()

    monkeypatch.setattr(decisions, "_resolve_decision_engine", resolve)

    async def go():
        async with _client(app) as c:
            return await c.post(
                "/v1/decisions",
                json={
                    "model": "x",
                    "input": "t",
                    "questions": [{"type": "predicate", "instructions": "q"}],
                },
            )

    r = asyncio.run(go())
    assert (
        r.status_code == 400 and "not a decision model" in r.json()["error"]["message"]
    )


def test_systemone_wire_shares_the_internal_request_and_rounds_like_the_reference(
    served,
):
    app, engine = served
    body = {
        "model": "clef",
        "state": {"ticket": "my card was charged twice"},
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {
                "type": "choice",
                "instructions": "Route",
                "criteria": {"billing": "money", "tech": "bugs"},
            },
            "tone": {
                "type": "score",
                "instructions": "Anger",
                "criteria": ["calm", "annoyed", "furious"],
            },
        },
    }

    async def go(b):
        async with _client(app) as c:
            return await c.post("/v1/systemone", json=b)

    r = asyncio.run(go(body))
    assert r.status_code == 200, r.text
    j = r.json()
    assert list(j["answers"]) == ["urgent", "team", "tone"]
    assert j["answers"]["urgent"] == {"type": "noul", "noul": 0.75}
    assert j["answers"]["team"]["choice"] == "tech"
    assert j["answers"]["team"]["probabilities"] == {"billing": 0.3333, "tech": 0.6667}
    tone = j["answers"]["tone"]
    assert (
        tone["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
        and tone["score"] == 1.3333
    )
    assert j["usage"] == {"input_tokens": 42, "output_tokens": 0}
    seen = engine.seen[-1]
    assert [q.name for q in seen.questions] == [
        "urgent",
        "team",
        "tone",
    ] and seen.state == body["state"]
    bad = asyncio.run(
        go(
            {
                "model": "clef",
                "state": "s",
                "questions": {"a": {"type": "choice", "criteria": {"only": "one"}}},
            }
        )
    )
    assert bad.status_code == 422 and "at least 2" in bad.json()["error"]["message"]


# ── the real openai client ───────────────────────────────────────────────────────────────


def test_openai_sdk_decisions_create_round_trips(served):
    openai = pytest.importorskip("openai")
    if tuple(int(x) for x in openai.__version__.split(".")[:2]) < (3, 26):
        pytest.skip("openai >= 3.26 has client.decisions")
    from openai.types import Decision

    app, _ = served

    async def go():
        http = _client(app)
        client = openai.AsyncOpenAI(
            api_key="k", base_url="http://t/v1", http_client=http
        )
        try:
            return await client.decisions.create(
                model="clef",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "storm"},
                            {"type": "input_image", "image_url": PNG},
                        ],
                    }
                ],
                questions=[
                    {"type": "predicate", "instructions": "Raining?", "name": "rain"},
                    {
                        "type": "choice",
                        "instructions": "Mood",
                        "choices": [{"value": "a"}, {"value": False}],
                    },
                    {
                        "type": "score",
                        "instructions": "Wet",
                        "levels": [
                            {"label": "dry"},
                            {"label": "wet", "description": "soaked"},
                        ],
                    },
                    {"type": "predicate", "instructions": "x", "name": "boom"},
                ],
                safety_identifier="user-1",
            )
        finally:
            await http.aclose()

    d = asyncio.run(go())
    assert isinstance(d, Decision)
    assert [a.type for a in d.answers] == ["predicate", "choice", "score", "refusal"]
    assert d.answers[0].probability == 0.75
    assert d.answers[1].choice is False  # typed: boolean, not the string "false"
    assert d.answers[2].probabilities[1].label == "wet"
    assert (
        d.usage.total_tokens == 42 and d.usage.input_tokens_details.cached_tokens == 0
    )


def test_model_card_of_a_decision_checkpoint(tmp_path):
    from yunshu_engine.model_card import build_model_card

    _checkpoint(tmp_path, dict(CFG, hidden_size=5120))
    card = build_model_card(tmp_path)
    assert card.kind == "decision"
    assert (
        "/v1/decisions" in card.api["endpoints"]
        and "/v1/systemone" in card.api["endpoints"]
    )
    assert "decision" in card.capabilities()
    assert card.input_modalities == ["text", "image"] and card.output_modalities == [
        "decision"
    ]


def test_reordering_choices_gives_the_same_token_sequence():
    a = choice("c", ["billing", "support", "sales"])
    b = Question("choice", "c", a.instructions, tuple(reversed(a.options)))
    ra = de.encode_record(tok, "state", [a, score("s", ["x", "y"])])
    rb = de.encode_record(tok, "state", [b, score("s", ["x", "y"])])
    assert (
        ra.input_ids == rb.input_ids
    )  # so the logits are identical: order cannot matter
    # while adding a question changes the sequence: the head decides all fields jointly
    rc = de.encode_record(tok, "state", [a])
    assert rc.input_ids != ra.input_ids
