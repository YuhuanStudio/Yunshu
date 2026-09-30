"""Run a real, isolated SearXNG (venv under /Volumes/P5Plus/yunshu-test-envs/searxng) and check that
Yunshu's SearXNG provider parses its JSON. Loopback only (port 18997); the instance is killed on exit.

    python searxng_probe.py "python programming language"
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Volumes/P5Plus/yunshu-test-envs/searxng")
PORT = 18997
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "python programming language"
    cfg = ROOT / "settings.yml"
    cfg.write_text(
        "use_default_settings: true\n"
        "server:\n  secret_key: yunshu-test-only\n  bind_address: 127.0.0.1\n"
        f"  port: {PORT}\n  limiter: false\n  public_instance: false\n"
        "search:\n  formats: [html, json]\n"
    )
    env = {
        **os.environ,
        "SEARXNG_SETTINGS_PATH": str(cfg),
        "PYTHONPATH": str(ROOT / "src"),
    }
    log = (ROOT / "searxng.log").open("wb")
    proc = subprocess.Popen(
        [str(ROOT / "venv/bin/python"), "-m", "searx.webapp"],
        cwd=ROOT / "src",
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    print("searxng exited early; see", ROOT / "searxng.log")
                    return 2
                time.sleep(1)
        print(f"searxng up after {time.time() - t0:.0f}s")
        raw = urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/search?q={q.replace(' ', '+')}&format=json",
            timeout=60,
        ).read()
        data = json.loads(raw)
        print(
            "raw result keys:", sorted(data.keys()), "n =", len(data.get("results", []))
        )
        if data.get("results"):
            print("first raw result keys:", sorted(data["results"][0].keys()))

        from yunshu_gateway.server_tools import search

        async def go():
            import httpx

            async with httpx.AsyncClient(timeout=60) as c:
                return await search.SearXNG(f"http://127.0.0.1:{PORT}").search(
                    q, limit=5, client=c
                )

        rows = asyncio.run(go())
        print("provider parsed", len(rows), "results")
        for r in rows[:5]:
            print(
                " -",
                r.title[:70],
                "|",
                r.url[:80],
                "|",
                r.snippet[:70].replace("\n", " "),
                "|",
                r.page_age,
            )
        return 0 if rows else 1
    finally:
        os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
