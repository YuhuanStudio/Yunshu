"""Media prompts on the round driver: identity cells against a real VL checkpoint.

Starts one ``yunshu serve`` (the given source tree), then sends greedy requests and records their
texts: an image prompt cold, repeated (exact repeat), beside concurrent text requests, a second image
(the image must matter), and a follow-up turn on the first image (cached prefix). Run it on the
pre-driver tree and on the driver tree and compare ``cells``: every cell must be identical across
the two trees and within a tree (solo == concurrent == repeat). The server is killed on exit.

    python scripts/research/media_identity.py --src <tree>/python --model <ckpt> --out out.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from io import BytesIO
from pathlib import Path


def image_url(kind: int, size: int = 224) -> str:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size // 2), ("blue", "green", "black")[kind % 3])
    boxes = [
        (0, 0, size // 2 - 1, size // 2 - 1),
        (size // 2, 0, size - 1, size // 2 - 1),
        (size // 4, size // 8, 3 * size // 4, 3 * size // 8),
    ]
    ImageDraw.Draw(im).rectangle(boxes[kind % 3], fill="red")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(url: str, model: str, messages: list, max_tokens: int = 48) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def image_msg(kind: int, question: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url(kind)}},
                {"type": "text", "text": question},
            ],
        }
    ]


TEXTS = [
    "Name three prime numbers greater than 50.",
    "Write the word 'harbor' backwards.",
    "What is the capital of Japan? One word.",
]


def cells(url: str, model: str) -> dict:
    out: dict = {}
    q = "Describe where the red rectangle is in this image in one sentence."
    out["image_a_cold"] = chat(url, model, image_msg(0, q))
    out["image_a_repeat"] = chat(url, model, image_msg(0, q))
    out["image_b"] = chat(url, model, image_msg(1, q))
    out["image_b_repeat"] = chat(url, model, image_msg(1, q))
    solo_text = [chat(url, model, [{"role": "user", "content": t}]) for t in TEXTS]
    out["text_solo"] = solo_text
    # concurrent: the image request beside text requests
    res: dict = {}

    def run(key, messages):
        res[key] = chat(url, model, messages)

    threads = [threading.Thread(target=run, args=("image", image_msg(0, q)))]
    threads += [
        threading.Thread(
            target=run, args=(f"text{i}", [{"role": "user", "content": t}])
        )
        for i, t in enumerate(TEXTS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out["image_a_concurrent"] = res["image"]
    out["text_concurrent"] = [res[f"text{i}"] for i in range(len(TEXTS))]
    # follow-up turn on the first image (history + new question: a cached prefix)
    history = image_msg(0, q) + [
        {"role": "assistant", "content": out["image_a_cold"]},
        {"role": "user", "content": "Now name the background colour in one word."},
    ]
    out["image_a_followup"] = chat(url, model, history)
    out["image_a_followup_repeat"] = chat(url, model, history)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the tree's python/ directory")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18996)
    ap.add_argument("--expect-driver-media", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=a.src, YUNSHU_OBSERVE="0", YUNSHU_CONSOLE="0")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = a.out.with_suffix(".server.log")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get(
            "YUNSHU_BIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
        ),
        "serve",
        "-m",
        a.model,
        "--port",
        str(a.port),
    ]
    with open(log, "wb") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    url = f"http://127.0.0.1:{a.port}"
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                sys.exit(f"server exited rc={proc.returncode}; see {log}")
            try:
                urllib.request.urlopen(url + "/health", timeout=2).read()
                break
            except Exception:
                if time.time() > deadline:
                    sys.exit("server did not become ready")
                time.sleep(2)
        model = json.loads(
            urllib.request.urlopen(url + "/v1/models", timeout=10).read()
        )["data"][0]["id"]
        result = {"model": model, "src": a.src, "cells": cells(url, model)}
        engaged = b"Round driver: media prompt" in log.read_bytes()
        result["driver_media_engaged"] = engaged
        a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        c = result["cells"]
        problems = []
        if c["image_a_cold"] != c["image_a_repeat"]:
            problems.append("image repeat differs")
        if c["image_a_cold"] != c["image_a_concurrent"]:
            problems.append("image concurrent differs from solo")
        if c["image_b"] != c["image_b_repeat"]:
            problems.append("image B repeat differs")
        if c["image_a_followup"] != c["image_a_followup_repeat"]:
            problems.append("follow-up repeat differs")
        if c["text_solo"] != c["text_concurrent"]:
            problems.append("text rows beside the image row changed")
        if c["image_a_cold"] == c["image_b"]:
            problems.append("two different images gave the same text (image ignored?)")
        if a.expect_driver_media and not engaged:
            problems.append("driver media path did not engage (log line missing)")
        print(json.dumps({"problems": problems, "engaged": engaged}), flush=True)
        if problems:
            sys.exit("media identity FAILED: " + "; ".join(problems))
    finally:
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
