"""Tiny HTTP parity check; run via gpuq on the shared GPU.

Both arms load the same MTP checkpoint and install the same target kernels.
The off arm suppresses drafting at the runner boundary (the public VLM
spec_decode field currently does not control drafting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18998)
    parser.add_argument("--server", choices=("on", "off"))
    args = parser.parse_args()
    if args.server:
        from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

        original = VLMBatchRunner.iter_tokens

        def generate(self, *a, **kw):
            kw["allow_draft"] = args.server == "on"
            stats = kw.setdefault("stats", RunStats())
            tokens = []
            try:
                for token in original(self, *a, **kw):
                    tokens.append(int(token))
                    yield token
            finally:
                print(
                    "m3compat_tokens "
                    + json.dumps({"ids": tokens, "draft": stats.used_draft}),
                    flush=True,
                )

        VLMBatchRunner.iter_tokens = generate
        sys.argv = [
            "yunshu",
            "serve",
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        runpy.run_module("yunshu_cli", run_name="__main__")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    prompts = [
        "Return the integers from 1 to 10 separated by commas.",
        "Write a Python function that returns the sum of a list.",
        "台灣的首都是哪裡？請用一句話回答。",
    ]
    for arm in ("off", "on"):
        log_path = args.out.with_suffix(f".{arm}.log")
        env = dict(
            os.environ,
            YUNSHU_AUTH_DISABLED="1",
            YUNSHU_VLM_DRAFT="mtp",
            YUNSHU_VLM_APC_MEMORY_GB="0",
        )
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--model",
                    args.model,
                    "--out",
                    str(args.out),
                    "--port",
                    str(args.port),
                    "--server",
                    arm,
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                base = f"http://127.0.0.1:{args.port}"
                for _ in range(240):
                    if proc.poll() is not None:
                        raise RuntimeError(f"{arm} server exited; see {log_path}")
                    try:
                        with urllib.request.urlopen(
                            base + "/v1/models", timeout=2
                        ) as response:
                            models = json.load(response)
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError("server startup timeout")
                model = models["data"][0]["id"]
                rows = []
                for prompt in prompts:
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 64,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    req = urllib.request.Request(
                        base + "/v1/chat/completions",
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=180) as response:
                        result = json.load(response)
                    message = result["choices"][0]["message"]
                    rows.append(
                        {
                            "message": message,
                            "tokens": result["usage"]["completion_tokens"],
                            "finish": result["choices"][0]["finish_reason"],
                        }
                    )
                results[arm] = rows
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        log_text = log_path.read_text()
        if "Speculative decoding: mtp" not in log_text:
            raise RuntimeError(f"MTP not engaged; see {log_path}")
        receipts = [
            json.loads(line.removeprefix("m3compat_tokens "))
            for line in log_text.splitlines()
            if line.startswith("m3compat_tokens ")
        ]
        if len(receipts) != len(prompts):
            raise RuntimeError("missing committed-token receipts")
        if any(row["draft"] != (arm == "on") for row in receipts):
            raise RuntimeError("draft arm did not engage the requested mode")
        results[arm + "_token_ids"] = [row["ids"] for row in receipts]
        results[arm + "_mode"] = "mtp; runner allow_draft=" + str(arm == "on")
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    equal = (
        results["on"] == results["off"]
        and results["on_token_ids"] == results["off_token_ids"]
    )
    results.update(
        equal=equal,
        digest=hashlib.sha256(
            json.dumps(
                results["on_token_ids"], sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest(),
        complete=True,
    )
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(
        json.dumps({"equal": equal, "complete": True, "out": str(args.out)}), flush=True
    )
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
