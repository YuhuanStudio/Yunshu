"""Real-server check of OpenAI code_interpreter / shell / containers and Anthropic REPL state +
programmatic tool calling, all under srt, with the official SDKs.

Starts `yunshu serve` (isolated HOME, token auth, port 18990-18999, YUNSHU_CODE_EXECUTION=1, srt on
PATH), then drives it. Suite ``small`` (a 3B model): containers API, code_interpreter (plain, stream,
state across responses, file output and download), hosted shell, Anthropic REPL state across calls
and programmatic tool calling. Suite ``big`` (27B): a CSV of SHA-256 hashes through code_interpreter
(verified against hashlib after download), programmatic tool calling with parallel calls, and hosted
shell. Exits nonzero on any failure (fail closed: a step that did not run is a failure).

    python scripts/research/codeexec2_real.py MODEL_DIR --suite small --port 18991
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

TOKEN = "codeexec2-secret"
SRT_BIN = "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
CI = {"type": "code_interpreter", "container": {"type": "auto"}}
SH = {"type": "shell", "environment": {"type": "container_auto"}}
INCLUDE = ["code_interpreter_call.outputs"]
CE = {"type": "code_execution_20260120", "name": "code_execution"}


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def pick_port():
    """A free port in 18990-18999 (other workers use the range too)."""
    import socket

    for port in range(18999, 18989, -1):
        with socket.socket() as sk:
            try:
                sk.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise Fail("no free port in 18990-18999")


def wait_ready(url, proc, secs=900):
    for _ in range(secs):
        if proc.poll() is not None:
            raise Fail(f"server exited rc={proc.returncode}")
        with contextlib.suppress(Exception):
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        time.sleep(1)
    raise Fail("server not ready")


def attempt(fn, tries=3):
    """A small model may phrase a call differently: retry the step, fail closed."""
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Fail as e:
            last = e
            print("attempt failed:", str(e)[:400], file=sys.stderr, flush=True)
    raise last  # type: ignore[misc]


def items(r, kind):
    return [i for i in r.output if i.type == kind]


def logs_of(item):
    return "".join(o.logs for o in (item.outputs or []) if o.type == "logs")


# ── OpenAI ────────────────────────────────────────────────────────────────────
def openai_suite(url, results, big=False, cl=None):
    import openai

    cl = cl or openai.OpenAI(
        base_url=url + "/v1", api_key=TOKEN, max_retries=0, timeout=1800
    )

    def step(name, fn):
        try:
            results[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": repr(exc)[:600]}
            raise

    def create(prompt, tools, choice, **kw):
        return cl.responses.create(
            model="m",
            input=prompt,
            tools=tools,
            tool_choice=choice,
            include=INCLUDE,
            max_output_tokens=kw.pop("max_output_tokens", 900),
            **kw,
        )

    def containers_api():
        c = cl.containers.create(
            name="check", expires_after={"anchor": "last_active_at", "minutes": 5}
        )
        expect(c.id.startswith("cntr_") and c.status == "running", f"{c}")
        expect(c.expires_after and c.expires_after.minutes == 5, "expires_after")
        expect(cl.containers.retrieve(c.id).id == c.id, "retrieve")
        expect(c.id in [x.id for x in cl.containers.list().data], "list")
        f = cl.containers.files.create(c.id, file=("in.txt", b"hello"))
        expect(f.path == "/mnt/data/in.txt" and f.source == "user", f"file {f}")
        expect(
            cl.containers.files.content.retrieve(f.id, container_id=c.id).read()
            == b"hello",
            "content",
        )
        expect(len(cl.containers.files.list(c.id).data) == 1, "files list")
        cl.containers.files.delete(f.id, container_id=c.id)
        expect(len(cl.containers.files.list(c.id).data) == 0, "file delete")
        cl.containers.delete(c.id)
        try:
            cl.containers.retrieve(c.id)
        except openai.NotFoundError:
            return "lifecycle ok"
        raise Fail("deleted container still retrievable")

    def plain():
        def go():
            r = create(
                "Use the tool to run exactly this Python code: print(123*456)",
                [CI],
                {"type": "code_interpreter"},
            )
            ci = items(r, "code_interpreter_call")
            expect(ci, f"no code_interpreter_call: {[i.type for i in r.output]}")
            expect(ci[0].status == "completed", f"status {ci[0].status}")
            expect("56088" in logs_of(ci[0]), f"logs {logs_of(ci[0])!r}")
            expect(ci[0].container_id.startswith("cntr_"), "container id")
            expect(cl.containers.retrieve(ci[0].container_id), "container missing")
            return {"container": ci[0].container_id, "code": ci[0].code[:80]}

        return attempt(go)

    def streaming():
        def go():
            names = []
            final = None
            with cl.responses.stream(
                model="m",
                input="Use the tool to run exactly this Python code: print('streamed-ok')",
                tools=[CI],
                tool_choice={"type": "code_interpreter"},
                include=INCLUDE,
                max_output_tokens=600,
            ) as s:
                for e in s:
                    names.append(e.type)
                final = s.get_final_response()
            for n in (
                "response.code_interpreter_call.in_progress",
                "response.code_interpreter_call_code.done",
                "response.code_interpreter_call.interpreting",
                "response.code_interpreter_call.completed",
            ):
                expect(n in names, f"missing event {n}")
            ci = items(final, "code_interpreter_call")
            expect(ci and "streamed-ok" in logs_of(ci[0]), "stream logs")
            return {"events": len(names)}

        return attempt(go)

    def state_across_responses():
        def go():
            r1 = create(
                "Use the tool to run exactly this Python code: keep_me = 41",
                [CI],
                {"type": "code_interpreter"},
                store=True,
            )
            r2 = create(
                "Use the tool to run exactly this Python code: print(keep_me + 1)",
                [CI],
                {"type": "code_interpreter"},
                previous_response_id=r1.id,
                store=True,
            )
            ci = items(r2, "code_interpreter_call")
            expect(ci, "no call in second response")
            expect(
                ci[0].container_id
                == items(r1, "code_interpreter_call")[0].container_id,
                "container not reused",
            )
            expect("42" in logs_of(ci[0]), f"state lost: {logs_of(ci[0])!r}")
            return {"logs": logs_of(ci[0])}

        return attempt(go)

    def file_output():
        def go():
            r = create(
                "Use the tool to run exactly this Python code: "
                "open('report.csv','w').write('a,b\\n1,2\\n')\nThen tell the user "
                "the file is [report.csv](sandbox:/mnt/data/report.csv).",
                [CI],
                {"type": "code_interpreter"},
            )
            ci = items(r, "code_interpreter_call")
            expect(ci, "no call")
            cid = ci[0].container_id
            files = cl.containers.files.list(cid).data
            expect(
                [f.path for f in files] == ["/mnt/data/report.csv"],
                f"files {[f.path for f in files]}; code={ci[0].code!r} logs={logs_of(ci[0])!r}",
            )
            data = cl.containers.files.content.retrieve(
                files[0].id, container_id=cid
            ).read()
            expect(
                b"1,2" in data, f"content {data!r}"
            )  # a small model may drop the header
            cites = [
                a
                for m in items(r, "message")
                for c in m.content
                for a in (getattr(c, "annotations", None) or [])
                if a.type == "container_file_citation"
            ]
            return {"bytes": len(data), "citations": len(cites)}

        return attempt(go)

    def shell():
        def go():
            r = create(
                "Use the tool to run exactly this command: echo shell-ok && python3 -c 'print(6*7)'",
                [SH],
                {"type": "shell"},
            )
            sc, so = items(r, "shell_call"), items(r, "shell_call_output")
            expect(sc and so, f"items {[i.type for i in r.output]}")
            out = so[0].output[0]
            expect(out.outcome.type == "exit" and out.outcome.exit_code == 0, f"{out}")
            expect("shell-ok" in out.stdout and "42" in out.stdout, f"{out.stdout!r}")
            return {"commands": sc[0].action.commands}

        return attempt(go)

    def sha_csv():
        words = ["alpha", "beta", "gamma", "delta"]

        def go():
            r = create(
                "Write and run a Python program that computes the SHA-256 hex digest of each of these "
                "strings: "
                + ", ".join(words)
                + ". Save a CSV with the header `name,sha256` and one row per string to report.csv in "
                "the current directory, then tell me the first row.",
                [CI],
                "auto",
                max_output_tokens=3000,
            )
            cis = items(r, "code_interpreter_call")
            expect(cis, f"no code_interpreter_call: {[i.type for i in r.output]}")
            cid = cis[0].container_id
            files = cl.containers.files.list(cid).data
            csvs = [f for f in files if f.path.endswith(".csv")]
            expect(csvs, f"no csv: {[f.path for f in files]}")
            rows = (
                cl.containers.files.content.retrieve(csvs[0].id, container_id=cid)
                .read()
                .decode()
                .strip()
                .splitlines()
            )
            expect(rows[0].replace(" ", "") == "name,sha256", f"header {rows[0]!r}")
            got = dict(r.split(",") for r in rows[1:])
            for w in words:
                expect(
                    got.get(w) == hashlib.sha256(w.encode()).hexdigest(), f"hash {w}"
                )
            return {"calls": len(cis), "rows": len(rows) - 1}

        return attempt(go, tries=2)

    step("containers_api", containers_api)
    if not big:
        for name, fn in (
            ("ci_plain", plain),
            ("ci_stream", streaming),
            ("ci_state_across_responses", state_across_responses),
            ("ci_file_output", file_output),
            ("shell_hosted", shell),
        ):
            step(name, fn)
    else:
        step("ci_sha256_csv", sha_csv)
        step("shell_hosted", shell)


# ── Anthropic ─────────────────────────────────────────────────────────────────
def anthropic_suite(url, results, big=False, cl=None):
    import anthropic

    cl = cl or anthropic.Anthropic(
        base_url=url, api_key=TOKEN, max_retries=0, timeout=1800
    )

    def step(name, fn):
        try:
            results[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": repr(exc)[:600]}
            raise

    def py_results(m):
        return [b for b in m.content if b.type == "code_execution_tool_result"]

    def repl_state():
        def go():
            kw = dict(
                model="m",
                max_tokens=600,
                tools=[CE],
                tool_choice={"type": "tool", "name": "code_execution"},
            )
            m1 = cl.beta.messages.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Use the tool to run exactly this Python code: kept = 6",
                    }
                ],
                **kw,
            )
            r1 = py_results(m1)
            expect(
                r1 and r1[0].content.type == "code_execution_result", f"{m1.content}"
            )
            expect(m1.container, "no container")
            m2 = cl.beta.messages.create(
                container=m1.container.id,
                messages=[
                    {
                        "role": "user",
                        "content": "Use the tool to run exactly this Python code: print(kept * 7)",
                    }
                ],
                **kw,
            )
            r2 = py_results(m2)
            expect(r2 and "42" in r2[0].content.stdout, f"state lost: {m2.content}")
            return {"stdout": r2[0].content.stdout}

        return attempt(go)

    def ptc_loop(prompt, tool, answer, tries=3, max_tokens=1200):
        """Drive programmatic tool calling the way a client does: answer tool_use blocks that carry
        a code_execution caller until the code finishes. Returns (final message, calls seen)."""

        def go():
            msgs = [{"role": "user", "content": prompt}]
            seen = []
            container = None
            for _ in range(8):
                kw = {"container": container} if container else {}
                m = cl.beta.messages.create(
                    model="m",
                    max_tokens=max_tokens,
                    tools=[CE, tool],
                    messages=msgs,
                    **kw,
                )
                container = m.container.id if m.container else container
                tus = [b for b in m.content if b.type == "tool_use"]
                if m.stop_reason != "tool_use" or not tus:
                    return m, seen
                for t in tus:
                    expect(
                        t.caller and t.caller.type.startswith("code_execution_"),
                        f"tool_use without code caller: {t.caller}",
                    )
                    seen.append((t.name, dict(t.input)))
                msgs.append(
                    {
                        "role": "assistant",
                        "content": [b.model_dump(exclude_none=True) for b in m.content],
                    }
                )
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": t.id,
                                "content": answer(t.name, dict(t.input)),
                            }
                            for t in tus
                        ],
                    }
                )
            raise Fail("too many pauses")

        return attempt(go, tries=tries)

    def ptc():
        tool = {
            "name": "get_weather",
            "description": "Current weather for a city; returns a short text.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "allowed_callers": ["code_execution_20260120"],
        }

        def go():
            m, seen = ptc_loop(
                "Use code execution to call get_weather for the city Paris from Python code "
                "(`result = await get_weather(city='Paris')`), print the result, and tell me what it says.",
                tool,
                lambda n, a: f"Sunny, 21C in {a.get('city')}",
            )
            expect(seen, "the code never called the tool")
            rs = py_results(m)
            expect(
                rs and "Sunny, 21C" in rs[0].content.stdout,
                f"no result in stdout: {[b.type for b in m.content]}",
            )
            return {"calls": seen, "stdout": rs[0].content.stdout[:100]}

        return go()

    def ptc_big():
        inventory = {"apple": 7, "pear": 11, "plum": 5}
        tool = {
            "name": "get_stock",
            "description": "Units in stock for one product; returns the number as text.",
            "input_schema": {
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
            },
            "allowed_callers": ["code_execution_20260120"],
        }

        def go():
            m, seen = ptc_loop(
                "Use code execution to call get_stock for each of: apple, pear, plum (call them "
                "concurrently with asyncio.gather), add up the stock numbers in Python and print the total.",
                tool,
                lambda n, a: str(inventory.get(a.get("product"), 0)),
                max_tokens=3000,
            )
            expect(len(seen) >= 3, f"calls {seen}")
            rs = py_results(m)
            expect(
                rs and str(sum(inventory.values())) in rs[0].content.stdout,
                f"total {rs and rs[0].content.stdout!r}",
            )
            return {"calls": len(seen), "stdout": rs[0].content.stdout[:80]}

        return attempt(go, tries=2)

    if not big:
        step("anthropic_repl_state", repl_state)
        step("anthropic_ptc", ptc)
    else:
        step("anthropic_ptc_parallel", ptc_big)


SUITES = {"small": False, "big": True}


def run_suites(url, results, big, oai=None, ant=None):
    openai_suite(url, results, big, oai)
    anthropic_suite(url, results, big, ant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--suite", choices=tuple(SUITES), default="small")
    ap.add_argument("--port", type=int, default=0, help="0: first free in 18990-18999")
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/codeexec2-home")
    a = ap.parse_args()
    a.port = a.port or pick_port()
    expect(18990 <= a.port <= 18999, "port outside 18990-18999")
    os.makedirs(a.home, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        PATH=f"{SRT_BIN}:/opt/homebrew/bin:{os.environ['PATH']}",
        YUNSHU_AUTH_TOKEN=TOKEN,
        YUNSHU_CODE_EXECUTION="1",
        YUNSHU_CODE_EXECUTION_TIMEOUT="120",
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
        stdout=open(os.path.join(a.home, f"server-{a.suite}.log"), "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    results: dict = {}
    rc = 1
    try:
        wait_ready(url, proc)
        run_suites(url, results, SUITES[a.suite])
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:600]
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
