#!/usr/bin/env python3
"""Start one engine server, run the capability matrix against it, kill the server (kill -9 on exit).

Runs INSIDE a gpuq job:  gpuq submit --label capmatrix-27b -- python scripts/research/capmatrix_serve.py \
    --engine yunshu --model /path --out DIR
Engines other than `yunshu` use the snapshot registry (bench_engines.build_launch, the 27B checkpoint).
Exits nonzero if the server never comes up or the verdict is not all_pass (fail closed).
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dev"))
import capmatrix  # noqa: E402


def ready(url: str, proc, limit: float) -> str:
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=3) as r:
                return json.load(r)["data"][0]["id"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("server not ready")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument(
        "--model",
        help="checkpoint path (yunshu only; others use the snapshot registry)",
    )
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", default="vision,thinking")
    ap.add_argument("--ready-s", type=float, default=900)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    home = a.out / f"home-{a.engine}"
    home.mkdir(parents=True, exist_ok=True)
    log = a.out / f"server-{a.engine}.log"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "OPENAI_", "YUNSHU_"))
    }
    if a.engine == "yunshu":
        binp = os.environ.get(
            "TFB_YUNSHU_BIN",
            "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu",
        )
        cmd = [binp, "serve", "-m", a.model, "--port", str(a.port)]
        env.update(
            HF_HUB_OFFLINE="1",
            NO_PROXY="127.0.0.1",
            PYTHONPATH=str(HERE.parents[1] / "python"),
        )
    else:
        import bench_engines as be

        la = be.build_launch(a.engine, a.port, home, os.environ)
        for link, target in la.links.items():
            Path(link).parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(FileExistsError):
                os.symlink(target, link)
        for path, text in la.files.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        cmd, env = la.cmd, la.env
    url = f"http://127.0.0.1:{a.port}"
    with open(log, "wb") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    try:
        model = ready(url, proc, a.ready_s)
        print(f"engine={a.engine} model={model} ready", flush=True)
        v = capmatrix.run(
            capmatrix.load_matrix(),
            url,
            model,
            a.engine,
            set(filter(None, a.features.split(","))),
        )
        v["tag"] = a.out.name
        (a.out / f"verdict-{a.engine}.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False) + "\n"
        )
        for i, r in v["rows"].items():
            print(
                f"{r['status']:5} {i}"
                + ("" if r["status"] in ("pass", "na") else f"  {r['detail']}"),
                flush=True,
            )
        print(f"counts={v['counts']} complete={v['complete']}", flush=True)
        return 0 if v["complete"] else 1
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait(30)


if __name__ == "__main__":
    sys.exit(main())
