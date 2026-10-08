"""CPU-tested real-server console route probe. Run only through gpuq/yv."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--src", required=True)
    return p


def validate_latency(data):
    marks = data["milestones_ms"]
    required = (
        "gateway_receive",
        "gateway_admit",
        "model_lease",
        "template_start",
        "template_end",
        "engine_submit",
        "engine_admit",
        "first_decode",
        "sse_first_flush",
    )
    missing = [k for k in required if k not in marks]
    if missing:
        raise AssertionError(f"missing latency milestones: {missing}")
    if any(v < 0 for v in marks.values()):
        raise AssertionError("negative latency milestone")
    durations = data["durations_ms"]
    for key in (
        "model_lease",
        "gateway_admit",
        "template_tokenize",
        "engine_queue",
        "sse_first_flush",
    ):
        if durations[key] is None or durations[key] < 0:
            raise AssertionError(f"missing/negative duration: {key}")
    return durations


def run(a):
    import httpx
    from covaudit_session import Srv
    from route_checks import REGISTRY, Ctx

    out = Path(a.out)
    scratch = out.parent / (out.stem + "-scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    models = scratch / "models"
    models.mkdir(exist_ok=True)
    (models / "consolefeat-seed").symlink_to(
        Path(a.model).resolve(), target_is_directory=True
    )
    token = "consolefeat-probe-token"
    result = {
        "device": "M5",
        "model": a.model,
        "complete": False,
        "checks": {},
        "failures": [],
    }
    server = None
    try:
        os.environ["COVAUDIT_BIN"] = str(Path(sys.executable).parent / "yunshu")
        # Port pool contention is retried; no server outside 18990-18999.
        deadline = time.monotonic() + 180
        while True:
            try:
                server = Srv(
                    a.model,
                    a.src,
                    scratch / "home",
                    scratch / "server.log",
                    ["YUNSHU_AUTH_TOKEN=" + token, "YUNSHU_VLM_APC_DISK=0"],
                    models_dir=str(models),
                    token=token,
                )
                break
            except RuntimeError as exc:
                if "free port" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        server.wait_ready()
        with httpx.Client(
            base_url=server.url,
            headers={"Authorization": "Bearer " + token},
            timeout=180,
        ) as client:
            ctx = Ctx(server.url, token, "consolefeat-model", "multi", http=client)
            ctx.notes["console_model_path"] = a.model
            for name in ("console_registration_cancel", "console_host_latency"):
                REGISTRY[name].fn(ctx)
                result["checks"][name] = "PASS"
            r = client.post(
                "/v1/yunshu/models/register", json={"model": ctx.model, "path": a.model}
            )
            r.raise_for_status()
            body = {
                "model": ctx.model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16,
                "temperature": 0,
                "stream": True,
            }
            r = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"X-Request-Id": "consolefeat-latency"},
            )
            r.raise_for_status()
            if "data: [DONE]" not in r.text:
                raise AssertionError("stream did not complete")
            rows = client.get("/v1/yunshu/requests/recent").json()["data"]
            row = next(
                row for row in rows if row["request_id"] == "consolefeat-latency"
            )
            result["latency_ms"] = validate_latency(row["latency"])
            result["milestones_ms"] = row["latency"]["milestones_ms"]
            result["checks"]["stream_latency"] = "PASS"
            result["notes"] = ctx.notes
            result["server_log_tail"] = server.log_tail(40)
            result["complete"] = True
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if server:
            server.kill()
        out.write_text(json.dumps(result, ensure_ascii=False) + "\n")
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    run(parser().parse_args())
