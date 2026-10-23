"""Real-server check of the Anthropic code_execution tool under srt, with the anthropic 2.x SDK.

Starts `yunshu serve` (isolated HOME, token auth, port 18990-18999, YUNSHU_CODE_EXECUTION=1, srt on
PATH), then drives it with the SDK. Suite ``basic`` (small model): plain, streaming, container reuse
and file output. Suite ``sha`` (27B): a task that has the model write and run a script producing a
CSV of SHA-256 hashes, verified against hashlib after downloading the file. Exits nonzero on failure.

    python scripts/research/codeexec_real.py MODEL_DIR --suite basic --port 18991
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

TOKEN = "codeexec-secret"
TOOLS = [{"type": "code_execution_20250825", "name": "code_execution"}]
FORCE = {"type": "tool", "name": "code_execution"}
SRT_BIN = "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def wait_ready(url, proc, secs=900):
    for _ in range(secs):
        if proc.poll() is not None:
            raise Fail(f"server exited rc={proc.returncode}")
        with contextlib.suppress(Exception):
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        time.sleep(1)
    raise Fail("server not ready")


def types(msg):
    return [b.type for b in msg.content]


def results_of(msg):
    return [b for b in msg.content if b.type == "bash_code_execution_tool_result"]


def attempt(fn, tries=3):
    """A small model may phrase the command differently: retry the step, fail closed."""
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Fail as e:
            last = e
            print("attempt failed:", str(e)[:300], file=sys.stderr, flush=True)
    raise last  # type: ignore[misc]


def download(cl, fid):
    base = str(cl.base_url).rstrip("/")
    r = cl._client.get(
        f"{base}/v1/files/{fid}/content", headers={"authorization": f"Bearer {TOKEN}"}
    )
    expect(r.status_code == 200, f"download {r.status_code}")
    return r.content


def basic(url, results, cl=None):
    import anthropic

    cl = cl or anthropic.Anthropic(
        base_url=url, api_key=TOKEN, max_retries=0, timeout=600
    )
    expect(anthropic.__version__.startswith("2."), f"SDK {anthropic.__version__}")

    def step(name, fn):
        try:
            results[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "detail": repr(exc)[:400]}
            raise

    def create(prompt, **kw):
        return cl.beta.messages.create(
            model="m",
            max_tokens=700,
            tools=TOOLS,
            tool_choice=FORCE,
            messages=[{"role": "user", "content": prompt}],
            **kw,
        )

    def plain():
        def go():
            m = create(
                "Use the tool to run exactly this bash command: python3 -c 'print(123*456)'"
            )
            expect("server_tool_use" in types(m), f"no tool use: {types(m)}")
            r = results_of(m)
            expect(
                r and r[0].content.type == "bash_code_execution_result",
                "no result block",
            )
            expect(r[0].content.return_code == 0, f"rc {r[0].content.return_code}")
            expect("56088" in r[0].content.stdout, f"stdout {r[0].content.stdout!r}")
            expect(
                m.container and m.container.id.startswith("container_"), "no container"
            )
            return {"types": types(m), "container": m.container.id}

        return attempt(go)

    def streaming():
        def go():
            with cl.beta.messages.stream(
                model="m",
                max_tokens=700,
                tools=TOOLS,
                tool_choice=FORCE,
                messages=[
                    {
                        "role": "user",
                        "content": "Use the tool to run exactly: echo streamed-ok",
                    }
                ],
            ) as s:
                names = [e.type for e in s]
                m = s.get_final_message()
            r = results_of(m)
            expect(r and "streamed-ok" in r[0].content.stdout, f"{types(m)}")
            expect(m.container is not None, "no container in stream")
            expect(
                "input_json" in " ".join(names) or "content_block_delta" in names,
                "no deltas",
            )
            return {"types": types(m)}

        return attempt(go)

    def reuse():
        def go():
            m1 = create("Use the tool to run exactly: echo persisted > note.txt && ls")
            cid = m1.container.id
            expect(
                results_of(m1) and results_of(m1)[0].content.return_code == 0,
                "first run failed",
            )
            m2 = create("Use the tool to run exactly: cat note.txt", container=cid)
            expect(m2.container.id == cid, "container changed")
            r = results_of(m2)
            expect(
                r and "persisted" in r[0].content.stdout,
                f"stdout {r and r[0].content.stdout!r}",
            )
            m3 = create(
                "Use the tool to run exactly: echo x", container="container_" + "0" * 24
            )
            return m3

        try:
            go()
        except anthropic.BadRequestError as e:
            expect("not found" in str(e), str(e)[:200])
            return "unknown container -> 400"
        raise Fail("unknown container accepted")

    def file_output():
        def go():
            m = create(
                'Use the tool to run exactly: printf "a,b\\n1,2\\n" > "$OUTPUT_DIR/out.csv" && ls "$OUTPUT_DIR"'
            )
            r = results_of(m)
            expect(
                r and r[0].content.content,
                f"no file in result: {r and r[0].content.model_dump()}",
            )
            fid = r[0].content.content[0].file_id
            data = download(cl, fid)
            expect(b"1,2" in data, f"file {data!r}")
            return {"file_id": fid, "bytes": len(data)}

        return attempt(go)

    for name, fn in (
        ("plain", plain),
        ("stream", streaming),
        ("container_reuse", reuse),
        ("file_output", file_output),
    ):
        step(name, fn)


def sha(url, results, cl=None):
    import anthropic

    cl = cl or anthropic.Anthropic(
        base_url=url, api_key=TOKEN, max_retries=0, timeout=1800
    )
    words = ["alpha", "beta", "gamma", "delta"]
    prompt = (
        "Write and run a Python script that computes the SHA-256 hex digest of each of these strings: "
        + ", ".join(words)
        + ". Save the result as a CSV with the header `name,sha256` and one row per string to "
        '"$OUTPUT_DIR/hashes.csv" (copy it into $OUTPUT_DIR in the same command), then list $OUTPUT_DIR '
        "and tell me the first row."
    )

    def go():
        m = cl.beta.messages.create(
            model="m",
            max_tokens=4000,
            tools=TOOLS,
            messages=[{"role": "user", "content": prompt}],
        )
        files = [
            x.file_id
            for r in results_of(m)
            if r.content.type == "bash_code_execution_result"
            for x in r.content.content
        ]
        expect(files, f"no output file; blocks={types(m)}")
        rows = download(cl, files[-1]).decode().strip().splitlines()
        expect(rows[0].replace(" ", "") == "name,sha256", f"header {rows[0]!r}")
        got = dict(r.split(",") for r in rows[1:])
        for w in words:
            expect(got.get(w) == hashlib.sha256(w.encode()).hexdigest(), f"hash of {w}")
        return {
            "types": types(m),
            "rows": len(rows) - 1,
            "final_text": m.content[-1].text[:200]
            if m.content[-1].type == "text"
            else "",
        }

    results["sha256_csv"] = {"ok": True, "detail": attempt(go, tries=2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--suite", choices=("basic", "sha"), default="basic")
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/codeexec-home")
    a = ap.parse_args()
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
        (basic if a.suite == "basic" else sha)(url, results)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:400]
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
