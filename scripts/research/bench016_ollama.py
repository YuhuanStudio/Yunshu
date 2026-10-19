"""Own-port Ollama bootstrap for tfbench. Invoke only in gpuq; never uses user HOME."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def environment(home: Path, port: int, ctx: int) -> dict:
    if not 18990 <= port <= 18999:
        raise ValueError("Ollama benchmark port outside shared pool")
    return {
        **os.environ,
        "HOME": str(home),
        "TMPDIR": str(home),
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": str(home / "models"),
        "OLLAMA_CONTEXT_LENGTH": str(ctx),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_FLASH_ATTENTION": "1",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--ctx", required=True, type=int)
    a = ap.parse_args()
    env = environment(a.home, a.port, a.ctx)
    model_file = a.home / "Modelfile"
    model_file.write_text(
        f"FROM {a.model}\nPARAMETER temperature 0\nPARAMETER num_ctx {a.ctx}\n"
    )
    server = subprocess.Popen([a.binary, "serve"], env=env)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError("Ollama server exited before readiness")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{a.port}/api/version", timeout=1
                ).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Ollama startup timeout")
        subprocess.run(
            [a.binary, "create", "bench016", "-f", str(model_file)],
            env=env,
            timeout=300,
            check=True,
        )
        return server.wait()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
