"""Real HTTP prompt-score oracle. Run through gpuq, never directly on a model.

The server hook computes an independent plain full forward on fresh caches before
admitting the scoring row. No driver prefill/readout/scoring helper is reused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def compare(scores, reference, tolerance=0.0):
    assert isinstance(scores, list) and len(scores) == len(reference)
    assert scores[0] is None and reference[0] is None
    errors = []
    for got, want in zip(scores[1:], reference[1:], strict=True):
        assert got["token_id"] == want["token_id"]
        errors.append(abs(got["logprob"] - want["logprob"]))
        assert [x["token_id"] for x in got["top_logprobs"]] == [
            x["token_id"] for x in want["top_logprobs"]
        ]
        errors.extend(
            abs(g["logprob"] - w["logprob"])
            for g, w in zip(got["top_logprobs"], want["top_logprobs"], strict=True)
        )
    maximum = max(errors, default=0.0)
    assert maximum <= tolerance, f"plain forward max error {maximum} > {tolerance}"
    return maximum


def serve(args):
    import mlx.core as mx

    from yunshu_engine.round_driver.driver import RoundDriver
    from yunshu_engine.text_logprobs import top_logprobs_device

    original = RoundDriver.add

    def add(driver, req):
        if req.prompt_logprobs is not None:
            lm = driver.lm
            out = lm(mx.array([req.ids], mx.int32), cache=lm.make_cache())
            logits = (out.logits if hasattr(out, "logits") else out)[0].astype(
                mx.float32
            )
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            ti, tv = top_logprobs_device(lp[:-1], req.prompt_logprobs)
            realized = mx.take_along_axis(
                lp[:-1], mx.array(req.ids[1:])[:, None], axis=-1
            )[:, 0].tolist()
            ids, vals = ti.tolist(), tv.tolist()
            entries = [None] + [
                {
                    "token_id": token,
                    "logprob": value,
                    "top_logprobs": [
                        {"token_id": t, "logprob": v}
                        for t, v in zip(ids[i], vals[i], strict=True)
                    ],
                }
                for i, (token, value) in enumerate(
                    zip(req.ids[1:], realized, strict=True)
                )
            ]
            Path(args.reference).write_text(
                json.dumps(
                    {
                        "scores": entries,
                        "layer": "generic"
                        if driver.generic
                        else "flashnext"
                        if driver.flash
                        else "packed-qwen35",
                    }
                )
            )
        return original(driver, req)

    RoundDriver.add = add
    from yunshu_cli import main

    sys.argv = ["yunshu", "serve", "-m", args.model, "--port", str(args.port)]
    main()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "serve", "dry-run"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference")
    p.add_argument("--port", type=int, default=18998, choices=range(18990, 19000))
    p.add_argument("--tolerance", type=float, default=0.0)
    args = p.parse_args()
    if args.mode == "dry-run":
        assert Path(args.model, "config.json").is_file()
        print("dry-run complete: arguments and checkpoint config exist", flush=True)
        return
    if args.mode == "serve":
        serve(args)
        return
    import httpx

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = out.parent / (out.stem + "-home")
    home.mkdir(exist_ok=True)
    reference = out.with_suffix(".reference.json")
    env = dict(
        os.environ,
        HOME=str(home),
        XDG_CACHE_HOME=str(home / "cache"),
        TMPDIR=str(home),
        HF_HUB_OFFLINE="1",
    )
    log = out.with_suffix(".server.log")
    with log.open("w") as stream:
        proc = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "serve",
                "--model",
                args.model,
                "--out",
                str(out),
                "--reference",
                str(reference),
                "--port",
                str(args.port),
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{args.port}"
            with httpx.Client(base_url=url, timeout=300, trust_env=False) as client:
                deadline = time.monotonic() + 240
                while True:
                    assert proc.poll() is None, f"server exited: {log}"
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    assert time.monotonic() < deadline, f"server start timeout: {log}"
                    time.sleep(0.5)
                models = client.get("/v1/models").json()
                model = models["data"][0]["id"]
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain why the sky is blue in one sentence.",
                        }
                    ],
                    "max_tokens": 3,
                    "temperature": 0,
                    "enable_thinking": False,
                }
                # Warm the APC first: the next requests must still recompute all scores.
                client.post("/v1/chat/completions", json=payload).raise_for_status()
                checks = []
                for _ in range(2):
                    response = client.post(
                        "/v1/chat/completions", json=dict(payload, prompt_logprobs=3)
                    )
                    response.raise_for_status()
                    body = response.json()
                    ref = json.loads(reference.read_text())
                    scores = body["choices"][0]["prompt_logprobs"]
                    error = compare(scores, ref["scores"], args.tolerance)
                    checks.append(
                        {
                            "tokens": len(scores),
                            "max_error": error,
                            "layer": ref["layer"],
                        }
                    )
                out.write_text(
                    json.dumps({"checks": checks, "complete": True, "passed": True})
                    + "\n"
                )
                print(out.read_text(), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
