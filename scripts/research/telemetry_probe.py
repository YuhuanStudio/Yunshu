"""Real Apple host telemetry during HTTP decode. Run exclusively through gpuq."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--draft", choices=("off", "mtp"), default="off")
    return p


def validate(samples: list[dict], response: dict) -> dict:
    ok = [
        s["telemetry"]
        for s in samples
        if s.get("telemetry", {}).get("state") in ("ok", "partial")
    ]
    if not ok:
        raise RuntimeError("no valid IOReport readings")
    peak = max(s.get("watts", {}).get("gpu") or 0 for s in ok)
    frequencies = [
        s["gpu"]["frequency_mhz"]
        for s in ok
        if s.get("gpu", {}).get("frequency_mhz") is not None
    ]
    temps = [
        s["temperature"]["die_max_c"]
        for s in ok
        if s.get("temperature", {}).get("die_max_c") is not None
    ]
    energy = response.get("energy") or {}
    decode = energy.get("decode", {})
    if not (
        0 < peak < 200
        and frequencies
        and all(0 < f < 3000 for f in frequencies)
        and temps
        and all(0 < t < 130 for t in temps)
        and (decode.get("joules") or 0) > 0
    ):
        raise RuntimeError(
            f"implausible or missing telemetry: peak={peak}, MHz={frequencies}, temps={temps}, energy={energy}"
        )
    return {
        "gpu_peak_watts": peak,
        "gpu_mhz_min": min(frequencies),
        "gpu_mhz_max": max(frequencies),
        "die_max_c": max(temps),
        "energy": energy,
        "samples": len(ok),
    }


def main(argv=None):
    a = parser().parse_args(argv)
    if a.tokens < 1:
        raise ValueError("tokens must be positive")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        a.out.write_text(json.dumps({"complete": "dry-run", "model": a.model}) + "\n")
        return
    import tfbench

    server = tfbench.Srv(
        "yunshu",
        {
            "YUNSHU_TELEMETRY": "on",
            "YUNSHU_AUTH_TOKEN": "k",
            "YUNSHU_VLM_DRAFT": a.draft,
        },
        "telemetry-probe",
        a.model,
    )
    samples = []
    try:
        # Ensure the sampler has a baseline before the measured request.
        time.sleep(2)
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(
                tfbench.send,
                server.url,
                {
                    "model": server.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Computing evolved through mechanical calculators, programmable machines and integrated circuits. "
                            * 128
                            + "Write a long, detailed explanation of the history of computing.",
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": a.tokens,
                    "seed": 1234,
                },
            )
            finished_at = None
            while True:
                req = urllib.request.Request(
                    server.url + "/v1/yunshu/host",
                    headers={"Authorization": "Bearer k"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    sample = json.load(response)
                samples.append(sample)
                print(json.dumps({"sample": sample}), flush=True)
                if pending.done():
                    if finished_at is None:
                        finished_at = time.perf_counter()
                    if time.perf_counter() - finished_at >= 1.1:
                        break
                time.sleep(0.1 if finished_at is not None else 1.0)
            result = pending.result()
        summary = validate(samples, result)
        record = {
            "complete": True,
            "device": "M5",
            "engaged": tfbench.engaged_spec_mode("yunshu", server.log.read_text()),
            "response": {k: v for k, v in result.items() if k not in ("_text", "head")},
            "summary": summary,
            "host_samples": samples,
        }
        a.out.write_text(json.dumps(record) + "\n")
        print(json.dumps({"complete": True, "summary": summary}), flush=True)
    finally:
        server.kill()


if __name__ == "__main__":
    main()
