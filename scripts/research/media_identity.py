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


def image_url(kind: int, size: int = 224, tall: int | None = None) -> str:
    from PIL import Image, ImageDraw

    im = Image.new(
        "RGB", (size, tall or size // 2), ("blue", "green", "black")[kind % 3]
    )
    h = im.size[1]
    boxes = [
        (0, 0, size // 2 - 1, h // 2),
        (size // 2, 0, size - 1, h // 2),
        (size // 4, h // 4, 3 * size // 4, 3 * h // 4),
    ]
    d = ImageDraw.Draw(im)
    d.rectangle(boxes[kind % 3], fill="red")
    for i in range(0, max(size, h), 97):  # texture so a large image is not constant
        d.line((i, 0, 0, i), fill=(255, 255, 0), width=3)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(
    url: str, model: str, messages: list, max_tokens: int = 48, logprobs: bool = False
):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    choice = data["choices"][0]
    if not logprobs:
        return choice["message"]["content"]
    steps = []
    for item in (choice.get("logprobs") or {}).get("content") or []:
        top = sorted(
            (t["logprob"] for t in item.get("top_logprobs") or []), reverse=True
        )
        steps.append(
            {
                "token": item["token"],
                "logprob": item["logprob"],
                "gap": (top[0] - top[1]) if len(top) > 1 else None,
            }
        )
    return {"text": choice["message"]["content"], "steps": steps}


def image_msg(
    kind: int, question: str, size: int = 224, tall: int | None = None
) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url(kind, size, tall)},
                },
                {"type": "text", "text": question},
            ],
        }
    ]


def multi_image_msg(kinds: list[int], question: str) -> list:
    """Several images in one prompt (what a video becomes: its frames as images)."""
    parts = [{"type": "image_url", "image_url": {"url": image_url(k)}} for k in kinds]
    return [{"role": "user", "content": parts + [{"type": "text", "text": question}]}]


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


WIDE_QUESTIONS = [
    "Describe where the red rectangle is in this image in one sentence.",
    "What colour is the background? Answer in one short sentence.",
    "Is the red shape on the left, right or in the middle? Explain briefly.",
    "List the colours you can see in this image.",
]


def cells_wide(url: str, model: str, count: int) -> list:
    """``count`` different small-image prompts with per-token logprobs (greedy): the input of the
    cross-version comparison (media_wide_compare.py)."""
    out = []
    for i in range(count):
        kind = i % 3
        size = (192, 224, 256, 288)[(i // 3) % 4]
        question = WIDE_QUESTIONS[(i // 12 + i) % len(WIDE_QUESTIONS)]
        rec = chat(
            url, model, image_msg(kind, question, size), max_tokens=40, logprobs=True
        )
        out.append({"kind": kind, "size": size, "question": question, **rec})
    return out


def cells_wide_text(url: str, model: str, count: int) -> list:
    """Text-only prompts of the same shape as the wide image prompts (the control: whether the two
    trees agree bit for bit on text where they do not on media)."""
    out = []
    topics = ("a harbor", "the moon", "a violin", "a glacier", "tea", "a lighthouse")
    for i in range(count):
        topic = topics[i % 6]
        question = WIDE_QUESTIONS[(i // 6) % len(WIDE_QUESTIONS)]
        text = (
            f"Consider {topic} number {i}. In one or two sentences: "
            + question.replace("this image", topic)
        )
        rec = chat(
            url,
            model,
            [{"role": "user", "content": text}],
            max_tokens=40,
            logprobs=True,
        )
        out.append({"kind": i % 6, "size": 0, "question": text, **rec})
    return out


def cells_long(url: str, model: str, size: int, tall: int) -> dict:
    """A several-thousand-token image prompt: cold, repeat, beside text rows, follow-up turn; and a
    multi-image prompt (the frames of a video)."""
    out: dict = {}
    q = "Describe where the red rectangle is in this image in one sentence."
    long_msg = image_msg(0, q, size, tall)
    out["long_image_cold"] = chat(url, model, long_msg)
    out["long_image_repeat"] = chat(url, model, long_msg)
    res: dict = {}

    def run(key, messages):
        res[key] = chat(url, model, messages)

    threads = [threading.Thread(target=run, args=("image", long_msg))]
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
    out["long_image_concurrent"] = res["image"]
    out["long_text_concurrent"] = [res[f"text{i}"] for i in range(len(TEXTS))]
    history = long_msg + [
        {"role": "assistant", "content": out["long_image_cold"]},
        {"role": "user", "content": "Now name the background colour in one word."},
    ]
    out["long_image_followup"] = chat(url, model, history)
    out["long_image_followup_repeat"] = chat(url, model, history)
    mq = "How many images are there and what colour is the background of the first?"
    out["multi_image"] = chat(url, model, multi_image_msg([0, 1, 2], mq))
    out["multi_image_repeat"] = chat(url, model, multi_image_msg([0, 1, 2], mq))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the tree's python/ directory")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18996)
    ap.add_argument("--expect-driver-media", action="store_true")
    ap.add_argument(
        "--wide", type=int, default=0, help="N small-image prompts with logprobs"
    )
    ap.add_argument(
        "--wide-text",
        type=int,
        default=0,
        help="N text-only control prompts with logprobs",
    )
    ap.add_argument(
        "--long",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="also run the long-image cells",
    )
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
        if a.long:
            result["cells"].update(cells_long(url, model, *a.long))
        if a.wide:
            result["wide"] = cells_wide(url, model, a.wide)
        if a.wide_text:
            result["wide_text"] = cells_wide_text(url, model, a.wide_text)
        engaged = b"Round driver: media prompt" in log.read_bytes()
        result["driver_media_engaged"] = engaged
        if a.long:
            import re

            counts = [
                int(n)
                for n in re.findall(rb"media prompt (\d+) tokens", log.read_bytes())
            ]
            result["media_prompt_tokens_max"] = max(counts, default=0)
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
        if a.long:
            if c["long_image_cold"] != c["long_image_repeat"]:
                problems.append("long image repeat differs")
            if c["long_image_cold"] != c["long_image_concurrent"]:
                problems.append("long image concurrent differs from solo")
            if c["long_image_followup"] != c["long_image_followup_repeat"]:
                problems.append("long follow-up repeat differs")
            if c["multi_image"] != c["multi_image_repeat"]:
                problems.append("multi-image repeat differs")
            if c["text_solo"] != c["long_text_concurrent"]:
                problems.append("text rows beside the long image row changed")
        if a.long and a.expect_driver_media:
            if result["media_prompt_tokens_max"] < 2000:
                problems.append("long image prompt is not several thousand tokens")
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
