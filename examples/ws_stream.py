"""Stream two chats concurrently over ONE WebSocket, cancelling one mid-way.

python examples/ws_stream.py [ws://127.0.0.1:8000/v1/stream] [model]
"""

import asyncio
import sys

sys.path.insert(0, "python")
from yunshu_client import YunshuStream  # noqa: E402


async def run(conn, name, prompt, model, stop_after=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    n = 0
    async for m in conn.chat(body, id=name, with_done=True):
        if m["type"] == "done":
            print(f"\n[{name}] {m['reason']} {m['stats']}")
            return
        delta = (m["data"].get("choices") or [{}])[0].get("delta", {})
        print(f"[{name}] {delta.get('content') or ''}", end="", flush=True)
        n += 1
        if stop_after and n == stop_after:
            await conn.cancel(name)


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/v1/stream"
    model = sys.argv[2] if len(sys.argv) > 2 else "default"
    async with YunshuStream(url) as conn:
        print("session:", conn.session["limits"])
        await asyncio.gather(
            run(conn, "story", "Tell me a short story.", model),
            run(conn, "poem", "Write a long poem.", model, stop_after=8),
        )


asyncio.run(main())
