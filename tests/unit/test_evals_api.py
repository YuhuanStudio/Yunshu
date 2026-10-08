"""Exercise the real OpenAI SDK against TestClient and a fake normal engine route."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import openai
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from yunshu_gateway import evals_runner, files_store
from yunshu_gateway.evals_grading import similarity
from yunshu_gateway.evals_store import EvalStore, get_store
from yunshu_gateway.routers import evals, files


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_EVALS_DIR", str(tmp_path / "evals"))
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("YUNSHU_CHAT_COMPLETIONS_DIR", str(tmp_path / "chats"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    files_store.reset_store()
    calls = []
    state = {"slow": False, "error": False, "canceled": False}

    @asynccontextmanager
    async def lifespan(app):
        yield
        await evals_runner.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(evals.router, prefix="/v1")
    app.include_router(files.router, prefix="/v1")

    @app.post("/v1/chat/completions")
    async def engine(request: Request):
        body = await request.json()
        calls.append(body)
        if state["slow"]:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                state["canceled"] = True
                raise
        if state["error"]:
            return {"error": "bad model"}
        schema = (
            (body.get("response_format") or {}).get("json_schema", {}).get("schema", {})
        )
        content = (
            '{"label":"good"}'
            if "label" in schema.get("properties", {})
            else '{"score":0.75}'
            if schema
            else "hello"
        )
        return dict(
            choices=[
                dict(
                    message=dict(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=dict(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )

    with TestClient(app) as client:
        sdk = openai.OpenAI(
            base_url="http://testserver/v1", api_key="x", http_client=client
        )
        yield sdk, client, calls, state
    files_store.reset_store()


def definition(sdk, criteria=None):
    return sdk.evals.create(
        name="test",
        data_source_config=dict(
            type="custom",
            item_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            include_sample_schema=True,
        ),
        testing_criteria=criteria
        or [
            dict(
                type="string_check",
                name="exact",
                input="{{sample.output_text}}",
                reference="{{item.answer}}",
                operation="eq",
            )
        ],
    )


def source(n=2, sampling=False):
    ds = dict(
        type="jsonl",
        source=dict(
            type="file_content",
            content=[
                dict(
                    item={"answer": "hello"},
                    sample={"output_text": "hello" if i % 2 == 0 else "no"},
                )
                for i in range(n)
            ],
        ),
    )
    if sampling:
        ds.update(
            type="completions",
            model="local",
            input_messages=dict(
                type="template",
                template=[dict(role="user", content="Say {{item.answer}}")],
            ),
            sampling_params={"max_completion_tokens": 16, "temperature": 0},
        )
    return ds


def finished(sdk, eid, rid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        r = sdk.evals.runs.retrieve(rid, eval_id=eid)
        if r.status not in ("queued", "in_progress"):
            return r
        time.sleep(0.01)
    pytest.fail("run did not finish")


def test_sdk_all_twelve_routes_and_persistence(env):
    sdk, client, calls, _ = env
    e = definition(sdk)
    assert e.data_source_config.schema_["properties"]["item"]["required"] == ["answer"]
    assert sdk.evals.retrieve(e.id).testing_criteria[0].name == "exact"
    assert sdk.evals.update(e.id, metadata={"a": "b"}, name="renamed").name == "renamed"
    definition(sdk)
    first = sdk.evals.list(limit=1, order="asc")
    assert first.has_more
    assert len(sdk.evals.list(limit=1, order="asc", after=first.data[0].id).data) == 1
    run = sdk.evals.runs.create(e.id, data_source=source())
    r = finished(sdk, e.id, run.id)
    assert r.status == "completed"
    assert (r.result_counts.total, r.result_counts.passed, r.result_counts.failed) == (
        2,
        1,
        1,
    )
    assert sdk.evals.runs.list(e.id).data[0].id == r.id
    items = sdk.evals.runs.output_items.list(r.id, eval_id=e.id, limit=1)
    assert items.has_more and items.data[0].sample.output[0].content == "hello"
    second = sdk.evals.runs.output_items.list(
        r.id, eval_id=e.id, after=items.data[0].id
    )
    assert second.data[0].status == "fail"
    got = sdk.evals.runs.output_items.retrieve(
        items.data[0].id, eval_id=e.id, run_id=r.id
    )
    assert got.results[0].score == 1
    assert (
        len(sdk.evals.runs.output_items.list(r.id, eval_id=e.id, status="pass").data)
        == 1
    )
    assert EvalStore(get_store().root).run(e.id, r.id)["_outputs"][0]["id"] == got.id
    assert sdk.evals.runs.cancel(r.id, eval_id=e.id).status == "completed"
    assert sdk.evals.runs.delete(r.id, eval_id=e.id).deleted
    with pytest.raises(openai.NotFoundError):
        sdk.evals.runs.retrieve(r.id, eval_id=e.id)
    assert sdk.evals.delete(e.id).deleted
    assert not calls


def test_sampling_and_local_model_graders(env):
    sdk, _, calls, _ = env
    criteria = [
        dict(
            type="score_model",
            name="score",
            model="local",
            input=[dict(role="user", content="Grade {{sample.output_text}}")],
            range=[0, 1],
            pass_threshold=0.5,
        ),
        dict(
            type="label_model",
            name="label",
            model="local",
            input=[dict(role="user", content="Label {{sample.output_text}}")],
            labels=["good", "bad"],
            passing_labels=["good"],
        ),
    ]
    e = definition(sdk, criteria)
    run = sdk.evals.runs.create(e.id, data_source=source(1, sampling=True))
    r = finished(sdk, e.id, run.id)
    assert r.result_counts.passed == 1
    assert r.per_model_usage[0].invocation_count == 3
    assert r.per_model_usage[0].total_tokens == 15
    assert calls[0]["messages"][0]["content"] == "Say hello"
    assert calls[1]["messages"][0]["content"] == "Grade hello"
    assert calls[2]["response_format"]["json_schema"]["schema"]["properties"]["label"][
        "enum"
    ] == ["good", "bad"]


def test_cancel_inflight_and_delete_cascade(env):
    sdk, _, calls, state = env
    state["slow"] = True
    e = definition(sdk)
    r = sdk.evals.runs.create(e.id, data_source=source(3, sampling=True))
    deadline = time.monotonic() + 3
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls
    assert sdk.evals.runs.cancel(r.id, eval_id=e.id).status == "canceled"
    assert state["canceled"] and len(calls) == 1
    assert sdk.evals.delete(e.id).deleted
    assert not get_store().rows("evalrun")


def test_file_and_stored_completions_sources(env):
    sdk, _, _, _ = env
    e = definition(sdk)
    f = sdk.files.create(
        file=("eval.jsonl", json.dumps(source(1)["source"]["content"][0]).encode()),
        purpose="evals",
    )
    r = sdk.evals.runs.create(
        e.id, data_source=dict(type="jsonl", source=dict(type="file_id", id=f.id))
    )
    assert finished(sdk, e.id, r.id).result_counts.passed == 1
    from pathlib import Path

    from yunshu_engine import settings

    root = Path(settings.get("YUNSHU_CHAT_COMPLETIONS_DIR"))
    root.mkdir(parents=True)
    # Exact atomic stored-chat format from apiplanned fc28350d.
    (root / "000000000010_chatcmpl-test.json").write_text(
        json.dumps(
            {
                "completion": {
                    "id": "chatcmpl-test",
                    "created": 10,
                    "model": "local",
                    "metadata": {"tag": "yes"},
                    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                },
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    )
    e = sdk.evals.create(
        data_source_config={"type": "stored_completions", "metadata": {"tag": "yes"}},
        testing_criteria=[
            dict(
                type="string_check",
                name="stored",
                input="{{sample.output_text}}",
                reference="hello",
                operation="eq",
            )
        ],
    )
    r = sdk.evals.runs.create(
        e.id,
        data_source=dict(
            type="completions",
            source=dict(
                type="stored_completions",
                model="local",
                created_after=10,
                created_before=10,
                limit=1,
            ),
        ),
    )
    assert finished(sdk, e.id, r.id).result_counts.passed == 1


def test_validation_errors_and_run_isolation(env):
    sdk, client, _, _ = env
    e, other = definition(sdk), definition(sdk)
    r = sdk.evals.runs.create(e.id, data_source=source(1))
    with pytest.raises(openai.NotFoundError):
        sdk.evals.runs.retrieve(r.id, eval_id=other.id)
    with pytest.raises(openai.BadRequestError):
        sdk.evals.runs.create(
            e.id,
            data_source=dict(
                type="jsonl", source=dict(type="file_content", content=[{"item": {}}])
            ),
        )
    assert client.post("/v1/evals", content="bad").status_code == 400
    assert client.get("/v1/evals?limit=0").status_code == 400
    assert client.get("/v1/evals?order=bad").status_code == 400
    assert (
        client.post(
            "/v1/evals",
            json={
                "data_source_config": {"type": "custom", "item_schema": {}},
                "testing_criteria": [{"name": "x", "type": "python"}],
            },
        ).status_code
        == 400
    )


def test_errors_are_output_items_and_restart_is_explicit(env):
    sdk, _, _, state = env
    state["error"] = True
    e = definition(sdk)
    r = sdk.evals.runs.create(e.id, data_source=source(1, sampling=True))
    assert finished(sdk, e.id, r.id).result_counts.errored == 1
    item = sdk.evals.runs.output_items.list(r.id, eval_id=e.id).data[0]
    assert item.sample.error.code == "evaluation_error"
    get_store().change(r.id, status="in_progress")
    evals_runner.recover()
    assert sdk.evals.runs.retrieve(r.id, eval_id=e.id).error.code == "interrupted"


@pytest.mark.parametrize(
    "metric",
    [
        "cosine",
        "fuzzy_match",
        "bleu",
        "gleu",
        "rouge_l",
        *[f"rouge_{i}" for i in range(1, 6)],
    ],
)
def test_metrics(metric):
    assert similarity(
        "one two three four five", "one two three four five", metric
    ) == pytest.approx(1)
    assert similarity("aaaaa", "zzzzz", metric) == 0
    assert similarity("", "", metric) == (1 if metric == "fuzzy_match" else 0)
    assert similarity("a", "", metric) == 0


def test_meteor_and_rouge_known_values():
    assert similarity("a b c", "a b c", "meteor") == pytest.approx(1 - 0.5 / 27)
    assert similarity("a b c", "a z c", "rouge_l") == pytest.approx(2 / 3)


def test_real_server_probe_on_cpu_fake_route(env, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts" / "research")
    )
    from evals_verify import parser, validate_result
    from route_checks import Ctx, evals_check

    sdk, client, _, _ = env
    c = Ctx(
        url="http://testserver",
        token="",
        model="local",
        kind="vlm",
        http=client,
        oa=sdk,
    )
    evals_check(c)
    assert c.notes["evals"]["invocations"] == 3
    args = parser().parse_args(
        ["--model", "tiny", "--src", "python", "--out", "result.jsonl"]
    )
    assert args.model == "tiny"
    assert validate_result(dict(complete=True, passed=True, routes=12, invocations=3))
    assert not validate_result(
        dict(complete=True, passed=False, routes=12, invocations=3)
    )
    assert not validate_result(
        dict(complete=False, passed=True, routes=12, invocations=3)
    )


def test_sdk_run_uses_real_chat_router_and_cancel_event(env, monkeypatch):
    from types import SimpleNamespace

    from yunshu_gateway.routers import chat

    sdk, client, _, _ = env
    generated = []
    finished_calls = []

    class FakeEngine:
        is_loaded = True
        _tokenizer = None

        async def generate(self, **kw):
            generated.append(kw)
            try:
                await asyncio.sleep(30)
            finally:
                finished_calls.append(True)
            return SimpleNamespace(
                generated_text="hello",
                prompt_token_count=2,
                completion_token_count=1,
                finish_reason="stop",
            )

    monkeypatch.setattr(chat, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    # Replace fixture's fake HTTP route with the actual normal request handler.
    client.app.router.routes = [
        r
        for r in client.app.router.routes
        if getattr(r, "path", "") != "/v1/chat/completions"
    ]
    client.app.include_router(chat.router, prefix="/v1")
    e = definition(sdk)
    r = sdk.evals.runs.create(e.id, data_source=source(1, sampling=True))
    deadline = time.monotonic() + 3
    while not generated and time.monotonic() < deadline:
        time.sleep(0.01)
    assert generated
    sdk.evals.runs.cancel(r.id, eval_id=e.id)
    assert generated[0]["cancel_event"].is_set()
    assert finished_calls == [True]


def test_file_errors_schema_filters_and_object_prefixes(env):
    sdk, client, _, _ = env
    e = definition(sdk)
    with pytest.raises(openai.NotFoundError):
        sdk.evals.runs.create(
            e.id,
            data_source={
                "type": "jsonl",
                "source": {"type": "file_id", "id": "file_missing"},
            },
        )
    with pytest.raises(openai.BadRequestError):
        sdk.evals.runs.create(
            e.id, data_source={"type": "jsonl", "source": {"type": "file_content"}}
        )
    run = sdk.evals.runs.create(e.id, data_source=source(1))
    finished(sdk, e.id, run.id)
    for method in ("get", "delete"):
        assert getattr(client, method)("/v1/evals/" + run.id).status_code == 404
    assert client.post("/v1/evals/" + run.id, json={"name": "bad"}).status_code == 404
    get_store().change(e.id, updated_at=9999999999)
    other = definition(sdk)
    assert sdk.evals.list(order="desc", order_by="updated_at").data[0].id == e.id
    with pytest.raises(openai.BadRequestError):
        sdk.evals.list(order_by="wrong")
    assert sdk.evals.runs.list(e.id).data[0].id == run.id
    assert sdk.evals.retrieve(other.id).object == "eval"
    with pytest.raises(openai.BadRequestError):
        sdk.evals.runs.create(
            e.id,
            data_source={
                "type": "completions",
                "source": {"type": "stored_completions", "limit": -1},
            },
        )


def test_stored_filters_are_intersection_and_zero_timestamp(env):
    from pathlib import Path

    from yunshu_engine import settings
    from yunshu_gateway.evals_runner import source_rows

    root = Path(settings.get("YUNSHU_CHAT_COMPLETIONS_DIR"))
    root.mkdir(parents=True)
    rec = {
        "completion": {
            "id": "chatcmpl-zero",
            "created": 0,
            "model": "local",
            "metadata": {"tag": "yes"},
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        },
        "messages": [],
    }
    (root / "000000000000_chatcmpl-zero.json").write_text(json.dumps(rec))
    s = {"type": "stored_completions", "created_before": 0}
    assert len(source_rows(s, {"metadata": {"tag": "yes"}})) == 1
    assert not source_rows(
        {**s, "metadata": {"tag": "no"}}, {"metadata": {"tag": "yes"}}
    )
    rec["completion"]["created"] = 1
    (root / "000000000000_chatcmpl-zero.json").write_text(json.dumps(rec))
    assert not source_rows(s, {})


def test_actual_stored_chat_ingestion_after_apiplanned_merge(env, monkeypatch):
    from types import SimpleNamespace

    from yunshu_gateway.routers import chat

    sdk, client, _, _ = env

    class FakeEngine:
        is_loaded = True
        _tokenizer = None

        async def generate(self, **kw):
            return SimpleNamespace(
                generated_text="hello",
                prompt_token_count=2,
                completion_token_count=1,
                finish_reason="stop",
            )

    monkeypatch.setattr(chat, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    client.app.router.routes = [
        r
        for r in client.app.router.routes
        if getattr(r, "path", "") != "/v1/chat/completions"
    ]
    client.app.include_router(chat.router, prefix="/v1")
    response = sdk.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
        store=True,
        metadata={"check": "evals"},
    )
    assert response.choices[0].message.content == "hello"
    e = sdk.evals.create(
        data_source_config={
            "type": "stored_completions",
            "metadata": {"check": "evals"},
        },
        testing_criteria=[
            {
                "type": "string_check",
                "name": "stored_output",
                "input": "{{sample.output_text}}",
                "reference": "hello",
                "operation": "eq",
            },
            {
                "type": "string_check",
                "name": "stored_input",
                "input": "{{item.input_trajectory.0.content}}",
                "reference": "hi",
                "operation": "eq",
            },
        ],
    )
    r = sdk.evals.runs.create(
        e.id,
        data_source={
            "type": "completions",
            "source": {"type": "stored_completions", "model": "local"},
        },
    )
    assert finished(sdk, e.id, r.id).result_counts.passed == 1
    sample = sdk.evals.runs.output_items.list(r.id, eval_id=e.id).data[0].sample
    assert sample.model == "local" and sample.usage.total_tokens == 3


def test_cancel_before_worker_starts_releases_registry(env):
    _, client, _, _ = env

    async def queued():
        rid = "cancel-before-start"
        evals_runner.start(get_store(), rid, client.app, {})
        await evals_runner.cancel(rid)
        assert rid not in evals_runner._tasks

    asyncio.run(queued())


def test_sdk_content_blocks_use_normal_chat_wire_format(env):
    sdk, _, calls, _ = env
    e = definition(
        sdk,
        [
            {
                "type": "label_model",
                "name": "label",
                "model": "local",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Grade {{sample.output_text}}",
                            },
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,eA==",
                                "detail": "low",
                            },
                            "plain",
                        ],
                    }
                ],
                "labels": ["good"],
                "passing_labels": ["good"],
            }
        ],
    )
    ds = source(1, sampling=True)
    ds["input_messages"]["template"][0]["content"] = {
        "type": "output_text",
        "text": "Say {{item.answer}}",
    }
    r = sdk.evals.runs.create(e.id, data_source=ds)
    assert finished(sdk, e.id, r.id).result_counts.passed == 1
    assert calls[0]["messages"][0]["content"] == [{"type": "text", "text": "Say hello"}]
    assert calls[1]["messages"][0]["content"] == [
        {"type": "text", "text": "Grade hello"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,eA==", "detail": "low"},
        },
        {"type": "text", "text": "plain"},
    ]
    sample = sdk.evals.runs.output_items.list(r.id, eval_id=e.id).data[0].sample
    assert isinstance(sample.input[0].content, str)


def test_nonfinite_inline_and_file_json_are_400(env):
    sdk, client, _, _ = env
    e = definition(sdk)
    ds = source(1)
    ds["source"]["content"][0]["sample"]["output_text"] = float("nan")
    raw = json.dumps({"data_source": ds})
    assert (
        client.post(
            f"/v1/evals/{e.id}/runs",
            content=raw,
            headers={"content-type": "application/json"},
        ).status_code
        == 400
    )
    f = sdk.files.create(
        file=("bad.jsonl", json.dumps(ds["source"]["content"][0]).encode()),
        purpose="evals",
    )
    with pytest.raises(openai.BadRequestError):
        sdk.evals.runs.create(
            e.id,
            data_source={"type": "jsonl", "source": {"type": "file_id", "id": f.id}},
        )


def test_concurrent_create_after_parent_delete_has_no_orphan_or_inference(
    env, monkeypatch
):
    import threading

    sdk, _, calls, _ = env
    e = definition(sdk)
    entered, release = threading.Event(), threading.Event()
    original = evals_runner.source_rows

    def delayed(source, config):
        entered.set()
        assert release.wait(5)
        return original(source, config)

    monkeypatch.setattr(evals_runner, "source_rows", delayed)
    outcomes = []

    def create():
        try:
            sdk.evals.runs.create(e.id, data_source=source(1, sampling=True))
            outcomes.append("created")
        except openai.NotFoundError:
            outcomes.append("not_found")

    thread = threading.Thread(target=create)
    thread.start()
    try:
        assert entered.wait(3)
        assert sdk.evals.delete(e.id).deleted
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and outcomes == ["not_found"]
    assert not get_store().rows("evalrun") and not calls


def test_deleted_runs_cannot_be_resurrected_by_progress_or_late_start(env):
    from yunshu_gateway.conversations_store import ConversationError
    from yunshu_gateway.evals_store import new_id

    sdk, client, calls, _ = env
    e = definition(sdk)
    store = get_store()
    rec = {
        "id": new_id("evalrun"),
        "eval_id": e.id,
        "created_at": 0,
        "status": "queued",
    }
    store.create_run(rec)

    async def late():
        evals_runner.start(store, rec["id"], client.app, {})
        task = evals_runner._tasks[rec["id"]]
        assert store.delete_eval(e.id) == [rec["id"]]
        await task  # missing record is normal termination, not an unhandled exception
        assert rec["id"] not in evals_runner._tasks

    asyncio.run(late())
    with pytest.raises(ConversationError):
        store.save_progress(rec)
    assert not store.rows("evalrun") and not calls
    # A crash midway through a multi-file cascade can leave a child file.
    store.save(rec)
    evals_runner.recover()
    assert not store.rows("evalrun")


@pytest.mark.parametrize(
    "a,b,metric,expected",
    [
        ("a a b", "a b b", "cosine", 0.8),
        ("abc", "axc", "fuzzy_match", 2 / 3),
        ("", " ", "fuzzy_match", 0),
        (" ", "  ", "fuzzy_match", 2 / 3),
        ("a a b", "a b b", "rouge_1", 2 / 3),
        ("a a b", "a b b", "rouge_2", 0.5),
        ("a", "a", "rouge_5", 0),
        ("a x b", "a b", "rouge_l", 0.8),
        ("a b", "a", "bleu", 0),
        ("a b", "a", "gleu", 1 / 3),
        ("a x b", "a b", "meteor", 10 / 21),
        ("a b", "a x b", "meteor", 10 / 29),
    ],
)
def test_classic_metric_numeric_fixtures(a, b, metric, expected):
    assert similarity(a, b, metric) == pytest.approx(expected)


def test_bleu_brevity_penalty():
    import math

    assert similarity("a b", "a b c d", "bleu") == pytest.approx(math.exp(-1))


def test_stored_media_parts_survive_sampling(env):
    from pathlib import Path

    from yunshu_engine import settings

    sdk, _, calls, _ = env
    parts = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
    ]
    root = Path(settings.get("YUNSHU_CHAT_COMPLETIONS_DIR"))
    root.mkdir(parents=True)
    (root / "000000000001_chatcmpl-media.json").write_text(
        json.dumps(
            {
                "completion": {
                    "id": "chatcmpl-media",
                    "created": 1,
                    "model": "local",
                    "metadata": {},
                    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                },
                "messages": [{"role": "user", "content": "hi", "content_parts": parts}],
            }
        )
    )
    e = sdk.evals.create(
        data_source_config={"type": "stored_completions"},
        testing_criteria=[
            {
                "type": "string_check",
                "name": "exact",
                "input": "{{sample.output_text}}",
                "reference": "hello",
                "operation": "eq",
            }
        ],
    )
    r = sdk.evals.runs.create(
        e.id,
        data_source={
            "type": "completions",
            "model": "local",
            "input_messages": {
                "type": "item_reference",
                "item_reference": "item.input_trajectory",
            },
            "source": {"type": "stored_completions"},
        },
    )
    assert finished(sdk, e.id, r.id).result_counts.passed == 1
    assert calls[0]["messages"][0]["content"] == parts


def test_sample_input_string_shape_for_null_and_empty_media():
    assert evals_runner.sample_inputs(
        [{"role": "assistant", "content": None}, {"role": "user", "content": []}]
    ) == [{"role": "assistant", "content": ""}, {"role": "user", "content": "[]"}]


def test_cpu_grader_keeps_metadata_responsive_and_stops_on_cancel(env, monkeypatch):
    import threading

    from yunshu_gateway.evals_grading import GradingCancelledError

    sdk, _, calls, _ = env
    entered, exited = threading.Event(), threading.Event()

    def blocked(criterion, context, cancel_event=None):
        assert cancel_event is not None
        entered.set()
        try:
            assert cancel_event.wait(5)
            raise GradingCancelledError("canceled")
        finally:
            exited.set()

    monkeypatch.setattr(evals_runner, "lexical", blocked)
    e = definition(sdk)
    r = sdk.evals.runs.create(e.id, data_source=source(3))
    assert entered.wait(3)
    # These ordinary routes must complete while the CPU worker is still blocked.
    assert sdk.evals.retrieve(e.id).id == e.id
    assert not exited.is_set()
    assert sdk.evals.runs.cancel(r.id, eval_id=e.id).status == "canceled"
    assert exited.wait(3)
    assert sdk.evals.runs.retrieve(r.id, eval_id=e.id).result_counts.total == 0
    assert not calls


@pytest.mark.parametrize(
    "metric", ["rouge_l", "meteor", "bleu", "gleu", "rouge_1", "fuzzy_match"]
)
def test_lexical_loops_observe_cancellation(metric):
    from yunshu_gateway.evals_grading import GradingCancelledError

    class CancelAfterChecks:
        checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks >= 3

    with pytest.raises(GradingCancelledError):
        similarity("a " * 50, "a " * 50, metric, CancelAfterChecks())


def test_fuzzy_cancel_wrapper_preserves_scores():
    from threading import Event

    for a, b in [
        ("aabca", "abaca"),
        ("a " * 30, "a " * 29 + "b"),
        ("", " "),
        ("abc", "axc"),
    ]:
        assert similarity(a, b, "fuzzy_match", Event()) == similarity(
            a, b, "fuzzy_match"
        )


def test_score_grader_respects_explicit_reasoning_effort(env):
    sdk, _, calls, _ = env
    e = definition(
        sdk,
        [
            {
                "type": "score_model",
                "name": "score",
                "model": "local",
                "input": [{"role": "user", "content": "Grade this"}],
                "sampling_params": {"reasoning_effort": "high"},
                "pass_threshold": 0.5,
            }
        ],
    )
    r = sdk.evals.runs.create(e.id, data_source=source(1))
    assert finished(sdk, e.id, r.id).result_counts.passed == 1
    assert calls[0]["reasoning_effort"] == "high"
    assert "enable_thinking" not in calls[0]
