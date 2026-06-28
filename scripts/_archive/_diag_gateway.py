"""Discriminate the gateway N=32 drop: is it synchronous per-request CPU on the
event loop (tokenization/validation/response-build) that scales with PROMPT SIZE,
or fixed per-request overhead?

Launch the yunshu engine-loop gateway, hit it at N=16/32 with a TINY prompt
(cheap tokenization) and the BIG bench prompt (~1300 tok, expensive tokenization).
If TINY scales N16→N32 but BIG drops, the loop-blocking synchronous tokenizer
work is the cause.

Run: PYTHONPATH=. uv run python scripts/_diag_gateway.py ./models/Qwen2.5-3B-Instruct-bf16
"""
from __future__ import annotations
import asyncio, os, signal, socket, subprocess, sys, time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen2.5-3B-Instruct-bf16"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIG = "Topic: " + ("Explain how photosynthesis works in detail. " * 40)
TINY = "Hi."
MT = 64


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


async def main():
    import httpx
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"),
               YUNSHU_MODEL=MODEL, YUNSHU_AUTH_DISABLED="true", YUNSHU_ENGINE_LOOP="1")
    log = open("/tmp/diag_gw.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "yunshu_gateway.main:app",
                             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                            env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        async with httpx.AsyncClient() as c:
            # wait ready
            t0 = time.time(); mid = None
            while time.time() - t0 < 180:
                try:
                    r = await c.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                    if r.status_code == 200:
                        mid = (r.json().get("data") or [{}])[0].get("id"); break
                except Exception:
                    pass
                await asyncio.sleep(2)
            if not mid:
                print("server not ready"); return

            async def req(content):
                t = time.perf_counter()
                r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json={"model": mid, "messages": [{"role": "user", "content": content}],
                                       "max_tokens": MT, "temperature": 0.0}, timeout=600)
                return (r.json().get("usage") or {}).get("completion_tokens", 0)

            await req("warmup")
            for label, prompt in (("TINY prompt", TINY), ("BIG prompt(~1300tok)", BIG)):
                row = {}
                for N in (16, 32):
                    w0 = time.perf_counter()
                    cts = await asyncio.gather(*[req(f"[{i}] " + prompt) for i in range(N)])
                    wall = time.perf_counter() - w0
                    row[N] = round(sum(cts) / wall, 1)
                trend = "SCALES" if row[32] >= row[16] else "DROPS"
                print(f"  {label:22} N=16 {row[16]:>7}   N=32 {row[32]:>7}   → {trend}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=10)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        log.close()


if __name__ == "__main__":
    asyncio.run(main())
