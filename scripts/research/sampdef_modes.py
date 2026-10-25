"""Real-server check: sampling defaults per mode, through the official OpenAI and Anthropic SDKs.

Starts the Yunshu CLI on MODEL_DIR (isolated HOME, token auth, port 18990-18999), sends requests without
sampling parameters in thinking and non-thinking mode on chat, Responses and Anthropic Messages, reads
``x_yunshu.sampling`` from each answer and compares with the model's published profile. Exits nonzero on
the first mismatch; the process is killed on exit.

    python scripts/research/sampdef_modes.py MODEL_DIR --port 18992
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
FIELDS = ("temperature", "top_p", "top_k", "presence_penalty")


class Fail(AssertionError):  # noqa: N818
    pass


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def find(obj):
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


def wait_ready(url, proc, secs=420):
    for _ in range(secs):
        if proc.poll() is not None:
            raise Fail(f"server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise Fail("server not ready")


def profile(model_dir):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    from yunshu_engine.sampling_defaults import default_mode, profile_for

    return profile_for(model_dir), default_mode(model_dir)


def check(url, model_dir, results):
    import anthropic
    import openai

    prof, dmode = profile(model_dir)
    expect(prof, f"no sampling profile matches {model_dir}")
    oa = openai.OpenAI(base_url=url + "/v1", api_key=TOKEN, timeout=240)
    an = anthropic.Anthropic(base_url=url, api_key=TOKEN, timeout=240)
    mid = oa.models.list().data[0].id
    msg = [{"role": "user", "content": "Say hi."}]
    calls = {}
    for mode, flag in (("thinking", True), ("non_thinking", False)):
        calls[("chat", mode)] = lambda f=flag: oa.chat.completions.create(
            model=mid, messages=msg, max_tokens=24, extra_body={"enable_thinking": f}
        ).model_dump()
        calls[("responses", mode)] = lambda f=flag: oa.responses.create(
            model=mid,
            input="Say hi.",
            max_output_tokens=24,
            extra_body={"enable_thinking": f},
        ).model_dump()
        calls[("anthropic", mode)] = lambda f=flag: an.messages.create(
            model=mid,
            messages=msg,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024}
            if f
            else {"type": "disabled"},
        ).model_dump()
    calls[("chat", "default")] = lambda: oa.chat.completions.create(
        model=mid, messages=msg, max_tokens=24
    ).model_dump()
    calls[("anthropic", "default")] = lambda: an.messages.create(
        model=mid, messages=msg, max_tokens=24
    ).model_dump()
    calls[("chat", "explicit")] = lambda: oa.chat.completions.create(
        model=mid,
        messages=msg,
        max_tokens=24,
        temperature=0.2,
        extra_body={"enable_thinking": False},
    ).model_dump()
    for (route, mode), fn in calls.items():
        s = find(fn())
        expect(s, f"{route}/{mode}: no x_yunshu.sampling")
        if mode == "explicit":
            expect(
                s["temperature"] == 0.2 and s["source"]["temperature"] == "request", s
            )
            expect(
                s["source"]["top_p"] == "model_card" and s["mode"] == "non_thinking", s
            )
        else:
            want_mode = dmode if mode == "default" else mode
            expect(
                s["mode"] == want_mode,
                f"{route}/{mode}: mode {s['mode']} != {want_mode}",
            )
            for f in FIELDS:
                want = prof[want_mode].get(f)
                if want is not None:
                    expect(s[f] == want, f"{route}/{mode}: {f}={s[f]} want {want}")
                    expect(
                        s["source"][f] == "model_card", f"{route}/{mode}: {f} source"
                    )
        results[f"{route}/{mode}"] = s
    card = oa.models.retrieve(mid).model_dump()
    sd = (card.get("yunshu") or {}).get("sampling_defaults")
    expect(
        sd and sd["thinking"]["temperature"]["source"] == "model_card", f"card: {sd}"
    )
    results["card"] = sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=18992)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/sampdef-home2")
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
    log = os.path.join(a.home, f"server-{a.port}.log")
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
        stdout=open(log, "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    results: dict = {}
    rc = 1
    try:
        wait_ready(url, proc)
        check(url, a.model, results)
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        results["_error"] = repr(exc)[:800]
    finally:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    with contextlib.suppress(Exception):
        results["_startup_log"] = [ln for ln in open(log) if "Sampling defaults" in ln][
            :2
        ]
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
