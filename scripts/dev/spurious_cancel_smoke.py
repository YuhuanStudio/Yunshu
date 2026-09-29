"""Smoke: no spurious 'cancelled' on /v1/responses (sequential + concurrent), chat, messages."""

import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 18993
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=120).read().decode()


def resp_terminal(text):
    for ln in text.splitlines():
        if ln.startswith("data:") and "response." in ln:
            d = json.loads(ln[5:])
            if d["type"] in ("response.completed", "response.incomplete"):
                return d["type"], (d["response"].get("incomplete_details") or {})
    return None, {}


def one(_i):
    b = {"model": "m", "input": "Say hi.", "max_output_tokens": 60, "stream": True}
    return resp_terminal(post("/v1/responses", b))


msgs = [{"role": "user", "content": "hi"}]
p = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "yunshu_cli",
        "serve",
        "--model",
        sys.argv[1],
        "--port",
        str(PORT),
    ],
    env=dict(os.environ, PYTHONPATH="python"),
    start_new_session=True,
)
bad = 0
try:
    for _ in range(200):
        try:
            if urllib.request.urlopen(BASE + "/health/ready", timeout=3).status == 200:
                break
        except Exception:
            time.sleep(2)
    seq = [one(i) for i in range(20)]
    with cf.ThreadPoolExecutor(8) as ex:
        con = list(ex.map(one, range(8)))
    for name, res in (("seq", seq), ("conc", con)):
        b = [r for r in res if r[1].get("reason") == "cancelled" or r[0] is None]
        print(name, len(res), "spurious:", len(b), {r[0] for r in res}, flush=True)
        bad += len(b)
    c = json.loads(
        post("/v1/chat/completions", {"model": "m", "max_tokens": 30, "messages": msgs})
    )
    print("chat finish:", c["choices"][0]["finish_reason"])
    cs = post(
        "/v1/chat/completions",
        {"model": "m", "max_tokens": 30, "stream": True, "messages": msgs},
    )
    print("chat stream 'cancel' mentions:", cs.count("cancel"))
    m = json.loads(
        post("/v1/messages", {"model": "m", "max_tokens": 30, "messages": msgs})
    )
    print("messages stop:", m.get("stop_reason"))
    ms = post(
        "/v1/messages",
        {"model": "m", "max_tokens": 30, "stream": True, "messages": msgs},
    )
    print("messages stream:", [x for x in ms.splitlines() if "stop_reason" in x][:1])
    print("RESULT", "FAIL" if bad else "PASS")
finally:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
