"""Real-server check of Anthropic API version 2026-10-21 with the anthropic 2.x SDK.

Starts `yunshu serve` on a small model (isolated HOME, token auth, port 18990-18999), then drives it
with the SDK: plain message, stream, tool call, count_tokens, a 2026-10-21 document-in-assistant
request (expect 400), the same body on 2023-06-01 (expect 200), a citations-in-user 400, a batch
creation 400, a `role: "system"` message and beta `clear_at`. Exits nonzero on the first failure.

    python scripts/research/anth1021_real.py MODEL_DIR --port 18991
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

TOKEN = "anth1021-secret"
DOC = {
    "type": "document",
    "source": {"type": "text", "media_type": "text/plain", "data": "hello"},
}
TOOL = {
    "name": "get_weather",
    "description": "Get the weather of a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def wait_ready(url, proc, secs=300):
    for _ in range(secs):
        if proc.poll() is not None:
            raise Fail(f"server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise Fail("server not ready")


def run_checks(url, results):
    import anthropic
    import httpx

    cl = anthropic.Anthropic(base_url=url, api_key=TOKEN, max_retries=0, timeout=120)
    expect(anthropic.__version__.startswith("2."), f"SDK {anthropic.__version__}")

    def step(name, fn):
        try:
            results[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": repr(exc)[:300]}
            raise

    def plain():
        m = cl.messages.create(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "Say hi."}],
        )
        expect(m.content and m.usage.output_tokens > 0, "empty message")
        return m.content[0].text[:40]

    def stream():
        with cl.messages.stream(
            model="m", max_tokens=16, messages=[{"role": "user", "content": "Say hi."}]
        ) as s:
            text = "".join(s.text_stream)
            final = s.get_final_message()
        expect(text and final.stop_reason, "stream empty")
        return text[:40]

    def tool():
        m = cl.messages.create(
            model="m",
            max_tokens=128,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
            messages=[{"role": "user", "content": "Weather in Paris?"}],
        )
        uses = [b for b in m.content if b.type == "tool_use"]
        expect(uses and uses[0].name == "get_weather", f"no tool_use: {m.content}")
        return uses[0].input

    def count():
        c = cl.messages.count_tokens(
            model="m", messages=[{"role": "user", "content": "Say hi."}]
        )
        expect(c.input_tokens > 0, "count")
        return c.input_tokens

    msgs_doc = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [DOC]},
    ]

    def doc_400():
        try:
            cl.messages.create(model="m", max_tokens=8, messages=msgs_doc)
        except anthropic.BadRequestError as e:
            expect("document" in str(e), str(e))
            return str(e)[:160]
        raise Fail("expected 400")

    def doc_old_200():
        r = httpx.post(
            url + "/v1/messages",
            json={"model": "m", "max_tokens": 8, "messages": msgs_doc},
            headers={
                "authorization": f"Bearer {TOKEN}",
                "anthropic-version": "2023-06-01",
            },
            timeout=120,
        )
        expect(r.status_code == 200, f"{r.status_code} {r.text[:200]}")
        return r.status_code

    def cite_400():
        body = {
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "t",
                            "citations": [
                                {
                                    "type": "char_location",
                                    "cited_text": "a",
                                    "document_index": 0,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        for path, ver, want in (
            ("/v1/messages", "2026-10-21", 400),
            ("/v1/messages/count_tokens", "2026-10-21", 400),
            ("/v1/messages", "2023-06-01", 200),
        ):
            r = httpx.post(
                url + path,
                json=body,
                headers={"authorization": f"Bearer {TOKEN}", "anthropic-version": ver},
                timeout=120,
            )
            expect(
                r.status_code == want, f"{path} {ver}: {r.status_code} {r.text[:200]}"
            )
        return "ok"

    def batch_400():
        r = httpx.post(
            url + "/v1/messages/batches",
            json={
                "requests": [
                    {
                        "custom_id": "a",
                        "params": {"model": "m", "max_tokens": 8, "messages": msgs_doc},
                    }
                ]
            },
            headers={
                "authorization": f"Bearer {TOKEN}",
                "anthropic-version": "2026-10-21",
            },
            timeout=60,
        )
        expect(r.status_code == 400, f"{r.status_code} {r.text[:200]}")
        return r.json()["error"]["message"][:120]

    def system_role():
        m = cl.messages.create(
            model="m",
            max_tokens=16,
            messages=[
                {"role": "system", "content": "Answer in one word."},
                {"role": "user", "content": "Capital of France?"},
            ],
        )
        expect(m.content, "empty")
        return m.content[0].text[:40]

    def clear_at():
        r = httpx.post(
            url + "/v1/messages",
            json={
                "model": "m",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "system", "content": "x", "clear_at": "next_user_message"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ],
            },
            headers={"authorization": f"Bearer {TOKEN}"},
            timeout=120,
        )
        expect(r.status_code == 200, f"{r.status_code} {r.text[:200]}")
        return r.status_code

    def bad_key():
        bad = anthropic.Anthropic(base_url=url, api_key="wrong", max_retries=0)
        try:
            bad.messages.count_tokens(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        except anthropic.AuthenticationError:
            return "401"
        raise Fail("wrong key accepted")

    for name, fn in (
        ("plain", plain),
        ("stream", stream),
        ("tool", tool),
        ("count_tokens", count),
        ("document_in_assistant_2026_400", doc_400),
        ("document_in_assistant_2023_200", doc_old_200),
        ("citations_in_user", cite_400),
        ("batch_create_400", batch_400),
        ("system_role", system_role),
        ("clear_at", clear_at),
        ("wrong_key_401", bad_key),
    ):
        step(name, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/anth1021-home")
    a = ap.parse_args()
    expect(18990 <= a.port <= 18999, "port outside 18990-18999")
    os.makedirs(a.home, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_TOKEN=TOKEN,
        HF_HUB_OFFLINE="1",
    )
    env.pop("YUNSHU_AUTH_DISABLED", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            a.model,
            "--port",
            str(a.port),
        ],
        cwd=tree,
        env=env,
        stdout=open(os.path.join(a.home, "server.log"), "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    results: dict = {}
    rc = 1
    try:
        wait_ready(url, proc)
        run_checks(url, results)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:300]
    finally:
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
