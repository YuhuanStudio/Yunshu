"""Real-server check of the API gap fixes with the official openai / anthropic SDKs.

Starts `yunshu serve` on a small model (isolated HOME, ports 18990-18999) and drives it:

  host   GET /v1/yunshu/host with no token configured answers 200; with a token it needs the token
  cm     Anthropic Messages `context_management` (clear_tool_uses / clear_thinking): the edits are
         applied and `applied_edits` is reported (non-stream, stream, count_tokens)
  bg     Responses `background` + `stream`: stream, then resume with `starting_after`, then cancel
  st     Responses background + web_search, custom tool beside web_search
  compact  Messages compact_20260112: pause, round trip, continue
  n      chat completions and completions `n>1` with `stream=true`

    python scripts/research/apigaps_real.py MODEL_DIR --item host|cm|bg|n --port 18993

Exits nonzero on the first failed check; the server is killed on every exit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

TOKEN = "apigaps-secret"


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


def status_of(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def check_host(url, results, token):
    code = status_of(url + "/v1/yunshu/host")
    results["host_without_token"] = code
    if token:
        expect(code == 401, f"token server must refuse a tokenless host read: {code}")
        auth = status_of(url + "/v1/yunshu/host", {"Authorization": f"Bearer {TOKEN}"})
        results["host_with_token"] = auth
        expect(auth == 200, f"host with token: {auth}")
    else:
        expect(code == 200, f"host read with no token configured: {code}")


def check_n(url, results, token):
    import openai

    cl = openai.OpenAI(base_url=url + "/v1", api_key=token or "x", max_retries=0)
    seen: dict[int, str] = {}
    finished = set()
    usage = None
    for chunk in cl.chat.completions.create(
        model="m",
        n=3,
        stream=True,
        max_tokens=12,
        temperature=0.9,
        seed=5,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "Name a color."}],
    ):
        if chunk.usage:
            usage = chunk.usage
        for c in chunk.choices:
            seen[c.index] = seen.get(c.index, "") + (c.delta.content or "")
            if c.finish_reason:
                finished.add(c.index)
    results["chat_n3_stream"] = {"choices": seen, "usage": usage and usage.model_dump()}
    expect(sorted(seen) == [0, 1, 2], f"choice indexes {sorted(seen)}")
    expect(finished == {0, 1, 2}, f"finish_reason per choice: {finished}")
    expect(all(seen.values()), f"an empty choice: {seen}")
    expect(usage is not None and usage.completion_tokens > 0, "no usage chunk")
    plain = cl.chat.completions.create(
        model="m",
        n=3,
        max_tokens=12,
        temperature=0.9,
        seed=5,
        messages=[{"role": "user", "content": "Name a color."}],
    )
    expect(len(plain.choices) == 3, "non-stream n=3")
    got: dict[int, str] = {}
    for chunk in cl.completions.create(
        model="m", prompt="The sky is", n=2, stream=True, max_tokens=8, seed=3
    ):
        for c in chunk.choices:
            got[c.index] = got.get(c.index, "") + c.text
    results["completions_n2_stream"] = got
    expect(sorted(got) == [0, 1] and all(got.values()), f"completions: {got}")


def check_bg(url, results, token):
    import openai

    cl = openai.OpenAI(base_url=url + "/v1", api_key=token or "x", max_retries=0)
    events = []
    rid = None
    for ev in cl.responses.create(
        model="m",
        input="Count from 1 to 5.",
        max_output_tokens=40,
        background=True,
        stream=True,
    ):
        events.append(ev)
        if rid is None and getattr(ev, "response", None) is not None:
            rid = ev.response.id
    expect(rid and events[-1].type == "response.completed", "background stream")
    seqs = [e.sequence_number for e in events]
    expect(seqs == sorted(seqs), f"sequence order {seqs}")
    got = cl.responses.retrieve(rid)
    expect(got.status == "completed" and got.output_text, f"retrieve: {got.status}")
    cut = seqs[2]
    again = list(cl.responses.retrieve(rid, stream=True, starting_after=cut))
    expect(
        [e.sequence_number for e in again] == [s for s in seqs if s > cut],
        "starting_after replay",
    )
    results["background_stream"] = {
        "events": len(events),
        "resumed_after": cut,
        "replayed": len(again),
    }
    # Cancel a long one mid-stream.
    stream = cl.responses.create(
        model="m",
        input="Write a very long story about a dragon.",
        max_output_tokens=2000,
        background=True,
        stream=True,
    )
    rid2 = None
    for n, ev in enumerate(stream, 1):
        if rid2 is None and getattr(ev, "response", None) is not None:
            rid2 = ev.response.id
        if n >= 8:
            break
    stream.close()
    cancelled = cl.responses.cancel(rid2)
    expect(cancelled.status == "cancelled", f"cancel: {cancelled.status}")
    for _ in range(40):
        if cl.responses.retrieve(rid2).status == "cancelled":
            break
        time.sleep(0.25)
    expect(
        cl.responses.retrieve(rid2).status == "cancelled", "cancelled stays cancelled"
    )
    results["background_cancel"] = "cancelled"


def check_cm(url, results, token):
    import anthropic

    cl = anthropic.Anthropic(
        base_url=url, api_key=token or "x", max_retries=0, timeout=300
    )
    msgs = [{"role": "user", "content": "Read the files."}]
    for i in range(4):
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{i}",
                        "name": "read",
                        "input": {"path": f"f{i}.txt"},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{i}",
                        "content": f"contents of file {i} " * 40,
                    }
                ],
            }
        )
    edit = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 2},
        "keep": {"type": "tool_uses", "value": 1},
    }
    tool = {
        "name": "read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    kw = dict(
        model="m",
        max_tokens=24,
        tools=[tool],
        messages=msgs,
        betas=["context-management-2025-06-27"],
        context_management={"edits": [edit]},
    )
    plain = cl.beta.messages.create(**{**kw, "context_management": None, "betas": []})
    m = cl.beta.messages.create(**kw)
    applied = m.context_management.applied_edits if m.context_management else None
    expect(applied, f"no applied_edits: {m.context_management}")
    expect(applied[0].cleared_tool_uses == 3, f"cleared {applied[0]}")
    expect(
        m.usage.input_tokens < plain.usage.input_tokens,
        f"input not smaller: {m.usage.input_tokens} vs {plain.usage.input_tokens}",
    )
    with cl.beta.messages.stream(**kw) as s:
        for _ in s:
            pass
        final = s.get_final_message()
    expect(
        final.context_management and final.context_management.applied_edits,
        "stream message_delta context_management",
    )
    cnt = cl.beta.messages.count_tokens(
        model="m",
        tools=[tool],
        messages=msgs,
        betas=["context-management-2025-06-27"],
        context_management={"edits": [edit]},
    )
    expect(
        cnt.context_management
        and cnt.context_management.original_input_tokens > cnt.input_tokens,
        f"count_tokens: {cnt}",
    )
    results["context_management"] = {
        "applied": [a.model_dump() for a in applied],
        "input_tokens_with_edit": m.usage.input_tokens,
        "input_tokens_without": plain.usage.input_tokens,
        "count_tokens": cnt.model_dump(),
    }
    try:
        cl.beta.messages.create(
            **{**kw, "context_management": {"edits": [{"type": "bogus_edit"}]}}
        )
    except anthropic.BadRequestError:
        results["bogus_edit"] = "400"
    else:
        raise Fail("bogus edit accepted")


def check_st(url, results, token):
    """Responses background + web_search, and a custom tool beside web_search (keyless search
    providers: the default `auto` backend; a blocked network is recorded, not hidden)."""
    import openai

    cl = openai.OpenAI(
        base_url=url + "/v1", api_key=token or "x", max_retries=0, timeout=300
    )
    ws = {"type": "web_search"}
    job = cl.responses.create(
        model="m",
        input="Search the web for 'MLX Apple silicon' and answer in one sentence.",
        tools=[ws],
        tool_choice={"type": "web_search"},
        max_output_tokens=200,
        background=True,
    )
    expect(
        job.status in ("queued", "in_progress"), f"background+web_search: {job.status}"
    )
    got = job
    for _ in range(240):
        got = cl.responses.retrieve(job.id)
        if got.status not in ("queued", "in_progress"):
            break
        time.sleep(0.5)
    kinds = [o.type for o in got.output]
    results["background_web_search"] = {"status": got.status, "output": kinds}
    expect(got.status == "completed", f"background web_search ended {got.status}")
    expect("web_search_call" in kinds, f"no web_search_call item: {kinds}")
    streamed = []
    for ev in cl.responses.create(
        model="m",
        input="Search the web for 'MLX' and answer in one sentence.",
        tools=[ws],
        tool_choice={"type": "web_search"},
        max_output_tokens=200,
        background=True,
        stream=True,
    ):
        streamed.append(ev.type)
    expect(streamed[-1] == "response.completed", f"stream ended {streamed[-1:]}")
    results["background_stream_web_search"] = len(streamed)
    both = cl.responses.create(
        model="m",
        input="Create the file a.txt containing hello with the apply_patch tool.",
        tools=[
            ws,
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Edit files with a patch",
            },
        ],
        tool_choice={"type": "custom", "name": "apply_patch"},
        max_output_tokens=200,
    )
    kinds = [o.type for o in both.output]
    results["custom_plus_web_search"] = {"status": both.status, "output": kinds}
    expect(
        both.status in ("completed", "incomplete"), f"custom+web_search: {both.status}"
    )
    expect("custom_tool_call" in kinds, f"no custom_tool_call: {kinds}")


def check_compact(url, results, token):
    import anthropic

    cl = anthropic.Anthropic(
        base_url=url, api_key=token or "x", max_retries=0, timeout=900
    )
    msgs = []
    for i in range(80):
        msgs.append(
            {
                "role": "user",
                "content": f"Note {i}: the code name is project-{i}. "
                + "filler text " * 330,
            }
        )
        msgs.append({"role": "assistant", "content": f"Noted project-{i}."})
    msgs.append(
        {
            "role": "user",
            "content": "Which project number was Note 3 about? Answer briefly.",
        }
    )
    edit = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 50000},
        "pause_after_compaction": True,
    }
    kw = dict(model="m", max_tokens=48, messages=msgs, betas=["compact-2026-01-12"])
    paused = cl.beta.messages.create(**kw, context_management={"edits": [edit]})
    expect(paused.stop_reason == "compaction", f"stop_reason {paused.stop_reason}")
    block = paused.content[0]
    expect(block.type == "compaction" and block.content, f"block {block}")
    results["compaction_pause"] = {
        "summary_chars": len(block.content),
        "iterations": [i.model_dump() for i in (paused.usage.iterations or [])],
    }
    follow = cl.beta.messages.create(
        model="m",
        max_tokens=48,
        betas=["compact-2026-01-12"],
        messages=[
            *msgs,
            {
                "role": "assistant",
                "content": [{"type": "compaction", "content": block.content}],
            },
            {
                "role": "user",
                "content": "Which project number was Note 3 about? Answer briefly.",
            },
        ],
    )
    results["compaction_roundtrip"] = {
        "input_tokens": follow.usage.input_tokens,
        "text": follow.content[0].text[:80] if follow.content else None,
    }
    expect(
        follow.usage.input_tokens < 8000,
        f"round trip kept {follow.usage.input_tokens} tokens",
    )
    both = cl.beta.messages.create(
        **{**kw},
        context_management={"edits": [{**edit, "pause_after_compaction": False}]},
    )
    expect(both.content[0].type == "compaction", "compaction block leads the reply")
    expect(
        any(b.type == "text" for b in both.content),
        "reply follows the compaction block",
    )
    results["compaction_continue"] = [b.type for b in both.content]


def server_log_path(home, item, env=None, now=None):
    """A log path no other run reuses: the gpuq job id when set, else a timestamp and pid."""
    env = os.environ if env is None else env
    tag = env.get("GPUQ_JOB_ID") or env.get("GPUQ_JOB") or ""
    if not tag:
        tag = f"{time.strftime('%m%d-%H%M%S', time.localtime(now))}-{os.getpid()}"
    return os.path.join(home, f"server-{item}-{tag}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument(
        "--item", choices=("host", "cm", "bg", "n", "st", "compact"), required=True
    )
    ap.add_argument("--port", type=int, default=18993)
    ap.add_argument("--token", action="store_true", help="serve with a token")
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/apigaps-home")
    a = ap.parse_args()
    expect(18990 <= a.port <= 18999, "port outside 18990-18999")
    home = os.path.join(a.home, a.item)
    os.makedirs(home, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=home,
        YUNSHU_HOME=home,
        PYTHONPATH=os.path.join(tree, "python"),
        HF_HUB_OFFLINE="1",
        YUNSHU_VLM_APC_DISK="0",
    )
    env.pop("YUNSHU_AUTH_TOKEN", None)
    env.pop("YUNSHU_AUTH_DISABLED", None)
    if a.token:
        env["YUNSHU_AUTH_TOKEN"] = TOKEN
    log_path = server_log_path(home, a.item)
    print(f"server log: {log_path}", flush=True)
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
        stdout=open(log_path, "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    results: dict = {}
    rc = 1
    try:
        wait_ready(url, proc)
        fn = {
            "host": check_host,
            "cm": check_cm,
            "bg": check_bg,
            "n": check_n,
            "st": check_st,
            "compact": check_compact,
        }[a.item]
        fn(url, results, TOKEN if a.token else None)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:400]
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
