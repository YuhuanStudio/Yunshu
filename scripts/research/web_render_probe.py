"""CPU-only Chromium fixture probe: inline JS, guarded same-origin fetch, blocked egress.

No listening server or real origin traffic. Set PLAYWRIGHT_BROWSERS_PATH and
TMPDIR to external build storage; Chromium is launched with --disable-gpu.
"""

import argparse
import asyncio
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def valid(rows):
    return bool(
        rows
        and rows[-1].get("complete") is True
        and rows[-1].get("pass") is True
        and any(row.get("check") for row in rows)
        and all(row.get("pass") is True for row in rows if row.get("check"))
    )


async def run(args):
    from yunshu_gateway.server_tools.webfetch import FetchResult
    from yunshu_gateway.tavily import render

    rows, calls = [], []
    if args.dry_run:
        rows.append({"check": "dry_run", "pass": True})
    else:

        async def fetch(url, **kwargs):
            calls.append(url)
            if url == "https://fixture.example/":
                text = '<html><body><main id="content">Loading</main><script src="/bundle.js"></script></body></html>'
                media = "text/html"
            elif url == "https://fixture.example/bundle.js":
                # The bundle exceeds the prose cap and must be replayed whole.
                text = (
                    "/*"
                    + "x" * 50000
                    + "*/"
                    + """
                    fetch('/data').then(r => r.text()).then(t => {
                        document.getElementById('content').innerHTML = '<h1>Fixture</h1><p>' + t + '</p>';
                    });
                    fetch('http://127.0.0.1:8000/private').catch(() => {});
                    new WebSocket('wss://external.example/socket');
                    navigator.serviceWorker.register('/worker.js').catch(() => {});
                """
                )
                media = "application/javascript"
            elif url == "https://fixture.example/data":
                text = (
                    "AUTHORED_RENDER_42. This article came from a guarded same-origin fetch. "
                    * 10
                )
                media = "text/plain"
            else:
                raise AssertionError(f"Unexpected origin request: {url}")
            assert kwargs.get("preserve_body") is True
            return FetchResult(url, "", text, media, "")

        original = render.page
        render.page = fetch
        try:
            result = await render.render(
                "https://fixture.example/", automated=False, timeout=15
            )
        finally:
            render.page = original
        rows.extend(
            [
                {"check": "rendered_js", "pass": "AUTHORED_RENDER_42" in result.text},
                {
                    "check": "egress_blocked",
                    "pass": set(calls)
                    == {
                        "https://fixture.example/",
                        "https://fixture.example/bundle.js",
                        "https://fixture.example/data",
                    },
                    "fetch_calls": calls,
                },
                {"check": "metadata", "pass": result.metadata.get("rendered") is True},
            ]
        )
    rows.append(
        {"complete": True, "pass": all(row["pass"] for row in rows), "device": "CPU"}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return 0 if valid(rows) else 1


def main(argv=None):
    return asyncio.run(run(parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
