"""One 27B server for CLI census + SDK probes; run only inside gpuq.

--dry-run validates dependencies without importing MLX or starting any process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source", type=Path)
    p.add_argument("--source-ref")
    p.add_argument("--census-from", type=Path)
    p.add_argument(
        "--scenarios",
        default="cc_edit,cx_edit,oc_edit,oc_bash,cc_websearch,cx_websearch,cc_mcp,cx_mcp,cx_ws,cc_context,cc_image,sdk",
    )
    return p


def summarize(results, expected=()):
    return {
        "complete": True,
        "pass": bool(results)
        and set(expected) <= set(results)
        and all(r.get("ok") and r.get("rc", 0) == 0 for r in results.values()),
        "results": results,
    }


def protocol_probes(url, model, out):
    """Capture wire evidence for display semantics and thinking block termination."""
    import httpx
    import replay

    rows = []
    with httpx.Client(base_url=url, timeout=180) as client:
        for display in ("summarized", "omitted", "updates"):
            for stream in (False, True):
                body = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Compute 17 times 23. Think briefly and answer with the number.",
                        }
                    ],
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 64,
                        "display": display,
                    },
                    "max_tokens": 256,
                    "temperature": 0,
                    "stream": stream,
                }
                response = client.post("/v1/messages", json=body)
                row = {
                    "display": display,
                    "stream": stream,
                    "status": response.status_code,
                    "wire": response.text,
                }
                problems = []
                if response.status_code != 200:
                    problems.append("HTTP status")
                elif stream:
                    events = [
                        v
                        for _, v in replay.sse_events(response.text)
                        if isinstance(v, dict)
                    ]
                    open_thinking = set()
                    signed = set()
                    for event in events:
                        if event.get("content_block", {}).get("type") == "thinking":
                            open_thinking.add(event["index"])
                        delta = event.get("delta", {})
                        if delta.get("type") == "signature_delta":
                            signed.add(event["index"])
                        if (
                            display in ("omitted", "updates")
                            and delta.get("type") == "thinking_delta"
                            and delta.get("thinking")
                        ):
                            problems.append("hidden reasoning exposed")
                        if (
                            event["type"] == "content_block_stop"
                            and event["index"] in open_thinking
                            and event["index"] not in signed
                        ):
                            problems.append("missing signature_delta")
                    row["thinking_blocks"] = len(open_thinking)
                    if not open_thinking:
                        problems.append("no thinking exercised")
                    if not events or events[-1].get("type") != "message_stop":
                        problems.append("no successful terminal")
                else:
                    content = response.json().get("content", [])
                    thinking = [b for b in content if b.get("type") == "thinking"]
                    row["thinking_blocks"] = len(thinking)
                    if not thinking:
                        problems.append("no thinking exercised")
                    if display in ("omitted", "updates") and any(
                        b.get("thinking") for b in thinking
                    ):
                        problems.append("hidden reasoning exposed")
                row.update(problems=sorted(set(problems)), ok=not problems)
                rows.append(row)
                print("protocol", display, stream, row["problems"], flush=True)
                (out / "protocol.json").write_text(json.dumps(rows, indent=2))
    return {"ok": bool(rows) and all(r["ok"] for r in rows)}


def main(argv=None):
    a = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    if not (a.model / "config.json").is_file():
        raise SystemExit("missing model config")
    if a.dry_run:
        print(
            json.dumps(
                {
                    "complete": True,
                    "source": str(root),
                    "scenarios": a.scenarios.split(","),
                }
            )
        )
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        AGENTIC_YUNSHU_SRC=str(a.source or root / "python"),
        CENSUS_OUT=str(a.out / "census"),
        AGENTIC_BUILD=str(a.out / "agentic"),
    )
    import census
    import servers

    census.BUILD = a.out / "scratch"
    original_census = census.Census
    census.Census = lambda log, script, port=0: original_census(
        log, script, port=port or servers.free_ports(1)[0]
    )
    if a.census_from:
        shutil.copytree(a.census_from, a.out / "census")
    else:
        for name in census.S:
            census.run_session(name, timeout=90)
    import census_diff

    old = Path(
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/agentapi/main-census-1007/census"
    )
    previous, current = (
        census_diff.signature(old),
        census_diff.signature(a.out / "census"),
    )
    (a.out / "delta-1007.json").write_text(
        json.dumps(
            {
                "new": census_diff.diff(previous, current),
                "not_observed": census_diff.diff(current, previous),
            },
            indent=2,
        )
    )
    import e2e_agents as e
    import fake_searxng

    # Isolate every artifact, including agent scratch and server settings, from other workers.
    e.census.BUILD = a.out / "scratch"
    servers.SERVER_HOME = a.out / "server-home"
    sx = fake_searxng.make_server(servers.free_ports(1)[0])
    threading.Thread(target=sx.serve_forever, daemon=True).start()
    sx_url = f"http://127.0.0.1:{sx.server_address[1]}"
    os.environ.update(
        YUNSHU_SEARXNG_URL=sx_url,
        YUNSHU_WEB_FETCH_ALLOW_PRIVATE="1",
        YUNSHU_VLM_APC_DISK="0",
        YUNSHU_VLM_APC_DISK_DIR=f"/Volumes/Micron/yunshu-apc/{a.out.name}",
        YUNSHU_VLM_DRAFT="/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2",
    )
    # Every helper listener stays inside the reserved port pool as well.
    original_make = e.fake_searxng.make_server
    e.fake_searxng.make_server = lambda port=0: original_make(
        port or servers.free_ports(1)[0]
    )
    original_proxy = e.CaptureProxy
    e.CaptureProxy = lambda upstream, log: original_proxy(
        upstream, log, port=servers.free_ports(1)[0]
    )
    srv, _ = servers.start_server(
        1,
        lambda port: servers.Server("yunshu", str(a.model), port, a.out / "server.log"),
    )
    try:
        protocol = protocol_probes(srv.url, srv.model_id, a.out)
        sys.argv = [
            "e2e_agents",
            "--url",
            srv.url,
            "--out",
            str(a.out / "agents"),
            "--scenarios",
            a.scenarios,
            "--timeout",
            "180",
            "--sdk-skip",
            "anthropic_web_fetch_public",
        ]
        agent_error = None
        try:
            e.main()
        except SystemExit:
            pass
        except Exception as exc:
            agent_error = repr(exc)
        results_path = a.out / "agents/results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        results["census"] = {
            "ok": all(
                (a.out / "census" / name / "meta.json").exists()
                and json.loads((a.out / "census" / name / "meta.json").read_text()).get(
                    "rc"
                )
                == 0
                for name in census.S
            )
        }
        if agent_error:
            results["agent_harness"] = {"ok": False, "error": agent_error}
        results["thinking_protocol"] = protocol
        import replay

        sys.argv = [
            "replay",
            "--url",
            srv.url,
            "--census",
            str(a.out / "census"),
            "--out",
            str(a.out / "replay"),
            "--per-session",
            "2",
            "--max-tokens",
            "64",
        ]
        results["replay"] = {"ok": True}
        try:
            replay.main()
        except SystemExit as exc:
            results["replay"] = {"ok": exc.code in (None, 0)}
        summary = summarize(
            results, [*a.scenarios.split(","), "census", "replay", "thinking_protocol"]
        )
        summary.update(
            commit=a.source_ref
            or subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            device="m5",
            engaged=srv.engaged_spec_mode(),
        )
        (a.out / "complete.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return 0 if summary["pass"] else 1
    finally:
        srv.kill()
        sx.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
