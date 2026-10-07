"""Real model Tavily gate; all SERP/page content is a loopback fixture.

Run through yv/gpuq. The baseline exercises the existing web-search API, the
candidate exercises new Tavily routes. A CPU dry-run validates parser/output.
"""

import argparse
import json
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate(rows):
    checks = [row for row in rows if "check" in row]
    return bool(
        rows
        and rows[-1].get("complete") is True
        and checks
        and all(row.get("pass") is True for row in checks)
    ), "Missing or failed Tavily checks"


def load_dependencies():
    # Registry first, avoiding route_checks_tools's historic import cycle.
    import route_checks  # noqa: F401
    from covaudit_session import Srv, free_port
    from m3sweep_jobs import routes_make_ctx
    from route_checks_tavily import tavily_routes
    from route_checks_tools import FakeBackend, _web_search_provider

    return (
        Srv,
        free_port,
        routes_make_ctx,
        FakeBackend,
        _web_search_provider,
        tavily_routes,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(row):
        rows.append(row)
        args.out.write_text("".join(json.dumps(value) + "\n" for value in rows))
        print(json.dumps(row), flush=True)

    if args.dry_run:
        record({"check": "dry_run", "pass": True})
        record({"complete": True, "dry_run": True})
        return 0
    (
        Srv,
        free_port,
        routes_make_ctx,
        FakeBackend,
        _web_search_provider,
        tavily_routes,
    ) = load_dependencies()
    root = args.out.parent / (args.out.stem + "-scratch")
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    os.environ.setdefault("HF_HOME", "/Volumes/P5Plus/hf-cache")
    os.environ.setdefault("TMPDIR", str(root / "tmp"))
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    server = None
    fake = FakeBackend(free_port(), page_results=True)
    try:
        server = Srv(
            args.model,
            args.src,
            root,
            args.out.with_suffix(".server.log"),
            [
                f"YUNSHU_SEARXNG_URL={fake.url}",
                "YUNSHU_WEB_SEARCH_PROVIDER=searxng",
                "YUNSHU_WEB_FETCH_ALLOW_PRIVATE=1",
                "YUNSHU_WEB_RESEARCH=0",
                "YUNSHU_VLM_APC_DISK=0",
                *(
                    [f"YUNSHU_WEB_SEARCH_HEALTH_FILE={root / 'health.json'}"]
                    if not args.baseline
                    else []
                ),
            ],
        )
        server.wait_ready()
        ctx = routes_make_ctx(server, None, "multi", model_id=Path(args.model).name)
        ctx.fake = fake
        numbers = _web_search_provider(ctx) if args.baseline else tavily_routes(ctx)
        record(
            {
                "check": "existing_web_search" if args.baseline else "tavily_routes",
                "pass": True,
                "device": "M5",
                "numbers": numbers,
                "baseline": args.baseline,
            }
        )
        record({"complete": True, "pass": True, "device": "M5"})
    finally:
        if server:
            server.stop()
        fake.close()
    return 0 if validate(rows)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
