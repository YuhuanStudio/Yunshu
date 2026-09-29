"""OpenAI Realtime conformance checks, driven by the official `openai` SDK.

    python scripts/dev/realtime_conformance.py --url http://127.0.0.1:18990 [--model M] \
        [--audio] [--uds PATH]

Runs the GA client (`client.realtime.connect`) and the beta client
(`client.beta.realtime.connect`) against a server and prints one PASS/FAIL line
per check. Used by tests/unit/test_realtime_conformance.py (fake engine) and by
the real-model smoke test. Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import struct
import sys

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return bool(ok)


def in_order(types: list[str], wanted: list[str]) -> bool:
    it = iter(types)
    return all(any(t == w for t in it) for w in wanted)


async def collect_until(conn, stop_types: set[str], timeout: float = 120.0) -> list:
    events = []

    async def go():
        async for ev in conn:
            events.append(ev)
            if ev.type in stop_types:
                return

    await asyncio.wait_for(go(), timeout)
    return events


def tone_pcm16(seconds: float = 0.6, rate: int = 24000) -> bytes:
    n = int(seconds * rate)
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n)
    )


async def ga_text(client, model: str) -> None:
    async with client.realtime.connect(model=model) as conn:
        ev = await conn.recv()
        check("ga: first event is session.created", ev.type == "session.created")
        s = ev.session
        check("ga: session.type == realtime", getattr(s, "type", None) == "realtime")
        check("ga: session.model set from ?model=", getattr(s, "model", None) == model)
        audio = getattr(s, "audio", None)
        check(
            "ga: session.audio.input.format audio/pcm",
            audio is not None and audio.input.format.type == "audio/pcm",
        )

        await conn.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": "Answer in one short sentence.",
                "audio": {"input": {"turn_detection": None}},
            }
        )
        ups = await collect_until(conn, {"session.updated", "error"})
        upd = ups[-1]
        check("ga: session.updated", upd.type == "session.updated", upd.type)
        if upd.type == "session.updated":
            check(
                "ga: update applied (instructions, output_modalities, VAD off)",
                upd.session.instructions == "Answer in one short sentence."
                and list(upd.session.output_modalities) == ["text"]
                and upd.session.audio.input.turn_detection is None,
            )

        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello."}],
            }
        )
        evs = await collect_until(conn, {"conversation.item.done", "error"})
        types = [e.type for e in evs]
        check(
            "ga: conversation.item.added then .done",
            in_order(types, ["conversation.item.added", "conversation.item.done"]),
            str(types),
        )

        await conn.response.create()
        evs = await collect_until(conn, {"response.done", "error"})
        types = [e.type for e in evs]
        check(
            "ga: response lifecycle order",
            in_order(
                types,
                [
                    "response.created",
                    "response.output_item.added",
                    "response.content_part.added",
                    "response.output_text.delta",
                    "response.output_text.done",
                    "response.content_part.done",
                    "response.output_item.done",
                    "response.done",
                ],
            ),
            str(sorted(set(types))),
        )
        check(
            "ga: no beta-only event names leak",
            not any(
                t
                in (
                    "response.text.delta",
                    "conversation.item.created",
                    "conversation.created",
                )
                for t in types
            ),
        )
        done = evs[-1]
        check(
            "ga: response.done status completed",
            getattr(done.response, "status", "") == "completed",
            str(getattr(done.response, "status", "")),
        )
        check(
            "ga: response.done has usage + output items",
            done.response.usage is not None and len(done.response.output) >= 1,
        )
        text = "".join(e.delta for e in evs if e.type == "response.output_text.delta")
        check("ga: streamed text non-empty", bool(text.strip()), repr(text[:60]))
        out = done.response.output[0]
        check(
            "ga: output item content is output_text",
            out.content
            and out.content[0].type == "output_text"
            and out.content[0].text == text,
        )

        # unknown event => error event, socket stays usable
        await conn.send({"type": "no.such.event"})
        evs = await collect_until(conn, {"error"}, timeout=10)
        check(
            "ga: unknown event -> error event",
            evs[-1].type == "error" and evs[-1].error.type == "invalid_request_error",
        )

        # cancel mid-response
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Write a 3000 word essay about the history of the ocean.",
                    }
                ],
            }
        )
        await collect_until(conn, {"conversation.item.done", "error"})
        await conn.response.create()
        got = 0

        async def cancel_soon():
            nonlocal got
            async for ev in conn:
                if ev.type == "response.output_text.delta":
                    got += 1
                    if got == 3:
                        await conn.response.cancel()
                if ev.type == "response.done":
                    return ev

        done = await asyncio.wait_for(cancel_soon(), 60)
        check(
            "ga: response.cancel -> response.done(cancelled)",
            done.response.status in ("cancelled", "incomplete"),
            f"{done.response.status} after {got} deltas",
        )

        # tool round-trip declared via session tools
        await conn.session.update(
            session={
                "type": "realtime",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        await collect_until(conn, {"session.updated", "error"})


async def beta_text(client, model: str) -> None:
    async with client.beta.realtime.connect(model=model) as conn:
        ev = await conn.recv()
        check("beta: first event session.created", ev.type == "session.created")
        ev2 = await conn.recv()
        check(
            "beta: conversation.created follows",
            ev2.type == "conversation.created",
            ev2.type,
        )
        await conn.session.update(
            session={"modalities": ["text"], "turn_detection": None}
        )
        await collect_until(conn, {"session.updated", "error"})
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hi."}],
            }
        )
        await collect_until(conn, {"conversation.item.created", "error"})
        await conn.response.create()
        evs = await collect_until(conn, {"response.done", "error"})
        types = [e.type for e in evs]
        check(
            "beta: response.text.delta lifecycle",
            in_order(
                types,
                [
                    "response.created",
                    "response.output_item.added",
                    "response.text.delta",
                    "response.text.done",
                    "response.done",
                ],
            ),
            str(sorted(set(types))),
        )


async def ga_audio(client, model: str) -> None:
    """input_audio_buffer with manual commit: audio in, transcript + reply out."""
    async with client.realtime.connect(model=model) as conn:
        await conn.recv()
        await conn.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["text"],
                "audio": {"input": {"turn_detection": None}},
            }
        )
        await collect_until(conn, {"session.updated", "error"})
        pcm = tone_pcm16()
        step = 4800
        for i in range(0, len(pcm), step):
            await conn.input_audio_buffer.append(
                audio=base64.b64encode(pcm[i : i + step]).decode()
            )
        await conn.input_audio_buffer.commit()
        evs = await collect_until(conn, {"input_audio_buffer.committed", "error"}, 120)
        types = [e.type for e in evs]
        check(
            "ga-audio: input_audio_buffer.committed",
            "input_audio_buffer.committed" in types,
            str(types),
        )
        check(
            "ga-audio: no error on audio commit",
            "error" not in types,
            str([getattr(e, "error", None) for e in evs if e.type == "error"]),
        )
        await conn.response.create()
        evs = await collect_until(conn, {"response.done", "error"}, 180)
        types = [e.type for e in evs]
        check(
            "ga-audio: response.create after committed audio -> response.done",
            types[-1] == "response.done" and "error" not in types,
            str(sorted(set(types))),
        )
        text = "".join(e.delta for e in evs if e.type == "response.output_text.delta")
        check("ga-audio: reply text non-empty", bool(text.strip()), repr(text[:60]))


async def main(args) -> int:
    from openai import AsyncOpenAI

    kw = {}
    if args.uds:
        import httpx

        kw["http_client"] = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=args.uds)
        )
    client = AsyncOpenAI(
        base_url=args.url.rstrip("/") + "/v1",
        api_key=args.api_key,
        websocket_base_url=args.url.replace("http", "ws", 1).rstrip("/") + "/v1",
        **kw,
    )
    await ga_text(client, args.model)
    await beta_text(client, args.model)
    if args.audio:
        await ga_audio(client, args.model)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key", default="x")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--uds", default=None)
    sys.exit(asyncio.run(main(ap.parse_args())))
