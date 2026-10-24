"""The console sends nothing to any engine while the pause file says so."""

import asyncio
import json

import httpx

from yunshu_console.discovery import Discovery
from yunshu_console.pause import Pause
from yunshu_console.poller import Poller


def put(path, **kw):
    path.write_text(json.dumps({"pause": True, "until": None, "reason": "r", **kw}))


def test_pause_states(tmp_path):
    f = tmp_path / "p.json"
    now = [1000.0]
    p = Pause(f, clock=lambda: now[0])
    assert p.state() is None  # no file
    put(f, until=1500.0, reason="ab-speed")
    assert p.state() == {"reason": "ab-speed", "until": 1500.0}
    now[0] = 1500.0
    assert p.state() is None  # past until
    f.write_text("{not json")
    assert p.state() is None
    f.write_text('{"pause": false}')
    assert p.state() is None
    put(f)  # until null: stale after 2 h of mtime

    mt = f.stat().st_mtime
    now[0] = mt + 60
    assert p.paused
    now[0] = mt + 2 * 3600 + 1
    assert not p.paused
    assert Pause(None).state() is None


def test_poller_and_discovery_send_nothing_while_paused(tmp_path):
    f = tmp_path / "p.json"
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(500)

    pause = Pause(f)
    poller = Poller(
        "http://127.0.0.1:18990",
        None,
        interval_s=0.25,
        pause=pause,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    disc = Discovery(
        ["http://127.0.0.1:18990"],
        interval_s=0.5,
        pause=pause,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def go(seconds):
        tasks = [asyncio.create_task(poller.run()), asyncio.create_task(disc.run())]
        await asyncio.sleep(seconds)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    put(f)
    asyncio.run(go(0.8))
    assert seen == []
    f.unlink()  # resumes by itself
    asyncio.run(go(0.8))
    assert seen
