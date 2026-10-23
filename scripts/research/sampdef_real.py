"""Real-server check: an absent temperature / top_p / top_k uses the model's generation_config.

Builds a model dir of symlinks to MODEL_DIR with a known generation_config (0.35 / 0.8 / 17), serves it
(isolated HOME, token auth, port 18990-18999) and sends chat, completions, Responses and Anthropic Messages
requests with and without sampling parameters. Reads `x_yunshu.sampling` from every answer and the model
card. Exits nonzero on the first failure.

    python scripts/research/sampdef_real.py MODEL_DIR --port 18992
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
import urllib.request

TOKEN = "sampdef-secret"
GC = {"temperature": 0.35, "top_p": 0.8, "top_k": 17}


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def call(url, path, body=None, headers=None):
    h = {"content-type": "application/json", "authorization": f"Bearer {TOKEN}"}
    h.update(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url + path, data, h, method="POST" if body is not None else "GET"
    )
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def find(obj):
    """The first ``x_yunshu.sampling`` object anywhere in a response."""
    if isinstance(obj, dict):
        x = obj.get("x_yunshu")
        if isinstance(x, dict) and "sampling" in x:
            return x["sampling"]
        for v in obj.values():
            r = find(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v)
            if r:
                return r
    return None


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


def check(url, results):
    msg = [{"role": "user", "content": "Say hi."}]
    routes = {
        "chat": ("/v1/chat/completions", {"messages": msg, "max_tokens": 8}),
        "completions": ("/v1/completions", {"prompt": "Hi", "max_tokens": 8}),
        "responses": ("/v1/responses", {"input": "Say hi.", "max_output_tokens": 16}),
        "anthropic": ("/v1/messages", {"messages": msg, "max_tokens": 8}),
    }
    mid = call(url, "/v1/models")["data"][0]["id"]
    for name, (path, body) in routes.items():
        body = {"model": mid, **body}
        hdr = {"anthropic-version": "2023-06-01"} if name == "anthropic" else {}
        s = find(call(url, path, body, hdr))
        expect(s, f"{name}: no x_yunshu.sampling")
        got = (s["temperature"], s["top_p"], s["top_k"])
        expect(got == (0.35, 0.8, 17), f"{name} absent: {s}")
        expect(s["source"]["temperature"] == "generation_config", f"{name}: {s}")
        s2 = find(call(url, path, {**body, "temperature": 0.7, "top_p": 1.0}, hdr))
        got2 = (s2["temperature"], s2["top_p"], s2["top_k"])
        expect(got2 == (0.7, 1.0, 17), f"{name} explicit: {s2}")
        expect(s2["source"]["temperature"] == "request", f"{name}: {s2}")
        s3 = find(call(url, path, {**body, "temperature": 0}, hdr))
        expect(s3["temperature"] == 0, f"{name} greedy: {s3}")
        results[name] = {"absent": s, "explicit": s2}
    card = call(url, f"/v1/models/{mid}")
    sd = (card.get("yunshu") or {}).get("sampling_defaults")
    expect(sd and sd["temperature"]["value"] == 0.35, f"model card: {sd}")
    results["card"] = sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=18992)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/sampdef-home")
    a = ap.parse_args()
    expect(18990 <= a.port <= 18999, "port outside 18990-18999")
    os.makedirs(a.home, exist_ok=True)
    md = os.path.join(a.home, "model")
    os.makedirs(md, exist_ok=True)
    for f in os.listdir(a.model):
        dst = os.path.join(md, f)
        if f != "generation_config.json" and not os.path.lexists(dst):
            os.symlink(os.path.join(os.path.abspath(a.model), f), dst)
    with open(os.path.join(md, "generation_config.json"), "w") as fh:
        json.dump(GC, fh)
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
            md,
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
        check(url, results)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:500]
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
