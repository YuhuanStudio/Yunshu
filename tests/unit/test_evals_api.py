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
    assert similarity("", "", metric) == 1
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
