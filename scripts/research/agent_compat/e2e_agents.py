"""End-to-end: real coding agents against a real Yunshu server, with the server-side tools live.

Starts (all on loopback, ports 18990-18999):
  - a fake SearXNG (fixed corpus with a checkable fact) and a tiny MCP server,
  - Yunshu serving --model, configured with YUNSHU_SEARXNG_URL and web_fetch on loopback allowed,
  - a full-capture proxy between each agent and Yunshu.
Then runs each scenario with the isolated agent CLIs (scripts/research/agentic launcher: scrubbed env,
isolated HOME / CLAUDE_CONFIG_DIR / CODEX_HOME, sandbox-exec with loopback-only network), and checks the
agent's final answer for the fact only the tools can supply. Kills its own server on exit.

    python e2e_agents.py --model /path/to/model --scenarios cc_websearch,cx_websearch,cx_mcp,cc_mcp
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("AGENTIC_YUNSHU_SRC", str(HERE.parents[2] / "python"))

import census  # noqa: E402
import fake_searxng  # noqa: E402
import servers  # noqa: E402
import tiny_mcp  # noqa: E402
from capture_proxy import CaptureProxy  # noqa: E402

FACT = "Cloud Book 42"

TUIS = {
    "tui_cc": ("claude", ["/status", "/model", "/context", "/cost"]),
    "tui_cx": ("codex", ["/status", "/model"]),
}

SCENARIOS = {
    "cc_websearch": dict(
        agent="claude",
        cfg="cc_allow_web",
        prompt=(
            "Use the WebSearch tool to find the reference release code name of Yunshu, "
            "then answer with just that code name."
        ),
        expect=[FACT],
    ),
    "cc_mcp": dict(
        agent="claude",
        cfg="cc_mcp",
        prompt="Use the tiny MCP add tool to add 19 and 23 and answer with only the number.",
        expect=["42"],
    ),
    "cx_websearch": dict(
        agent="codex",
        cfg="cx_web_live",
        prompt=(
            "Search the web for the reference release code name of Yunshu, then answer with just that code name."
        ),
        expect=[FACT],
    ),
    "cx_mcp": dict(
        agent="codex",
        cfg="cx_mcp_http",
        prompt="Use the tiny MCP add tool to add 19 and 23 and answer with only the number.",
        expect=["42"],
    ),
    "cx_ws": dict(
        agent="codex",
        cfg="cx_ws",
        prompt="Reply with the single word: pong",
        expect=["pong"],
        proxy=False,
    ),
    "cc_context": dict(
        agent="claude", cfg=None, prompt="/context", expect=["Context Usage"]
    ),
    "cc_edit": dict(
        agent="claude",
        cfg=None,
        prompt="Create a file named hello.txt containing exactly the text hi-there, then run ls to confirm it exists.",
        check_files={"hello.txt": "hi-there"},
    ),
    "cx_edit": dict(
        agent="codex",
        cfg=None,
        prompt="Create a file named hello.txt containing exactly the text hi-there, then run ls to confirm it exists.",
        check_files={"hello.txt": "hi-there"},
    ),
    "oc_edit": dict(
        agent="opencode",
        cfg=None,
        prompt="Create a file named hello.txt containing exactly the text hi-there, then run ls to confirm it exists.",
        check_files={"hello.txt": "hi-there"},
    ),
    "cc_image": dict(
        agent="claude",
        cfg=None,
        setup="red_png",
        prompt="Use the Read tool to look at the image pic.png, then tell me its dominant color in one word.",
        expect=["red"],
    ),
    "oc_bash": dict(
        agent="opencode",
        cfg=None,
        prompt="Run the shell command `echo yunshu-ok` with the bash tool and tell me its output.",
        expect=["yunshu-ok"],
    ),
}


def apply_launch_config(agent: str, launch, info, run: Path, model_id: str):
    """Configure the agent the way `yunshu launch` would, from the served model's real card."""
    sys.path.insert(0, str(HERE.parents[2] / "python"))
    from yunshu_cli.integrations.agent_config import (
        claude_code_env,
        codex_catalog,
        codex_provider_toml,
        opencode_provider,
    )

    if agent == "claude":
        env = claude_code_env(info, "http://unused", "unused")
        for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(k)
        launch.env.update(env)
    elif agent == "codex":
        home = launch.home / ".codex"
        cat = home / "yunshu-models.json"
        cat.write_text(json.dumps(codex_catalog(info)))
        cfg = home / "config.toml"
        import re

        text = cfg.read_text()
        for key in ("model_context_window", "model_auto_compact_token_limit"):
            text = re.sub(rf"^{key} = .*\n", "", text, flags=re.M)
        block = codex_provider_toml(info, "http://unused", str(cat)).split(
            "\n[model_providers"
        )[0]
        keep = [
            ln
            for ln in block.splitlines()
            if ln.split(" = ")[0]
            in (
                "model_catalog_json",
                "model_context_window",
                "model_auto_compact_token_limit",
                "model_reasoning_summary",
            )
        ]
        cfg.write_text("\n".join(keep) + "\n" + text)
    elif agent == "opencode":
        cfgp = launch.home / ".config" / "opencode" / "opencode.json"
        cfg = json.loads(cfgp.read_text())
        prov = opencode_provider(info, "http://unused", "k")
        cfg["provider"]["yunshu"]["models"] = {model_id: prov["models"][info.id]}
        cfgp.write_text(json.dumps(cfg, indent=1))


def red_png(work: Path):
    """A 64x64 solid red PNG (pure stdlib)."""
    import struct
    import zlib

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))
    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    (work / "pic.png").write_bytes(png)


SETUPS = {"red_png": red_png}


def cx_mcp_http(home, launch, ctx):
    p = home / ".codex" / "config.toml"
    p.write_text(
        p.read_text()
        + f'\n[mcp_servers.tiny]\nurl = "http://127.0.0.1:{ctx["mcp_port"]}/mcp"\n'
    )


census.HOOKS["cx_mcp_http"] = cx_mcp_http
_orig_mcp = census.HOOKS["cc_mcp"]


def cc_mcp_http(home, launch, ctx):
    cfg = ctx["run"] / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tiny": {
                        "type": "http",
                        "url": f"http://127.0.0.1:{ctx['mcp_port']}/mcp",
                    }
                }
            }
        )
    )
    s = home / ".claude" / "settings.json"
    d = json.loads(s.read_text())
    d["permissions"]["allow"].append("mcp__tiny")
    s.write_text(json.dumps(d))
    launch.cmd += ["--mcp-config", str(cfg)]


census.HOOKS["cc_mcp"] = cc_mcp_http


def final_text(agent: str, stdout: str) -> str:
    """The agent's final answer from its machine-readable output."""
    if agent == "claude":
        for line in reversed(stdout.splitlines()):
            with contextlib.suppress(ValueError):
                d = json.loads(line)
                if d.get("type") == "result":
                    return str(d.get("result", ""))
    if agent == "codex":
        msgs = []
        for line in stdout.splitlines():
            with contextlib.suppress(ValueError):
                d = json.loads(line)
                it = d.get("item") or {}
                if (
                    d.get("type") == "item.completed"
                    and it.get("type") == "agent_message"
                ):
                    msgs.append(it.get("text", ""))
        return "\n".join(msgs[-1:])
    if agent == "opencode":
        parts = []
        for line in stdout.splitlines():
            with contextlib.suppress(ValueError):
                d = json.loads(line)
                if d.get("type") == "text":
                    parts.append((d.get("part") or {}).get("text", ""))
        return "".join(parts)
    return stdout[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument(
        "--sdk-skip", default="", help="comma list of e2e_sdk checks to skip"
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", nargs="*", default=[], help="extra `yunshu serve` args")
    ap.add_argument(
        "--launch-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="configure each agent the way `yunshu launch` does (real window, catalog, limits)",
    )
    a = ap.parse_args()
    out = Path(
        a.out or census.OUT_ROOT.parent / f"{time.strftime('%Y-%m-%d')}-agent-e2e"
    )
    out.mkdir(parents=True, exist_ok=True)

    sx = fake_searxng.make_server(0)
    threading.Thread(target=sx.serve_forever, daemon=True).start()
    sx_url = f"http://127.0.0.1:{sx.server_address[1]}"
    mcp_port = servers.free_ports(3)[2]
    threading.Thread(target=tiny_mcp.serve_http, args=(mcp_port,), daemon=True).start()
    os.environ.update(YUNSHU_SEARXNG_URL=sx_url, YUNSHU_WEB_FETCH_ALLOW_PRIVATE="1")
    port = servers.free_ports(1)[0]
    srv = servers.Server("yunshu", a.model, port, out / "server.log", extra=a.extra)
    results = {}
    try:
        srv.start()
        model_id = srv.model_id
        print("server ready", srv.url, model_id, f"{srv.ready_s:.0f}s", flush=True)
        import httpx

        sys.path.insert(0, str(HERE.parents[2] / "python"))
        from yunshu_cli.integrations.agent_config import ModelInfo

        card_item = next(
            m
            for m in httpx.get(srv.url + "/v1/models").json()["data"]
            if m["id"] == model_id
        )
        card_info = ModelInfo.from_models_item(card_item)
        (out / "model_card.json").write_text(json.dumps(card_item, indent=1))
        # what `yunshu launch` would hand each agent for THIS model's real card (no agent needed)
        for tool in ("claude", "codex", "opencode"):
            lp = subprocess.run(
                [str(servers.YUNSHU_BIN), "launch", tool, "--dry-run", "-u", srv.url],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HOME": str(servers.SERVER_HOME),
                    "PYTHONPATH": os.environ["AGENTIC_YUNSHU_SRC"],
                },
            )
            (out / f"launch_{tool}.txt").write_text(lp.stdout + lp.stderr)
        for name in a.scenarios.split(","):
            if name.startswith("sdk"):
                sdk_py = os.environ.get(
                    "SDK_PYTHON",
                    "/Volumes/P5Plus/yunshu-test-cache/api-ext/sdkvenv/bin/python",
                )
                skip = a.sdk_skip
                cmd = [
                    sdk_py,
                    str(HERE / "e2e_sdk.py"),
                    "--url",
                    srv.url,
                    "--model",
                    model_id,
                    "--search-fact",
                    FACT,
                    "--page-url",
                    f"{sx_url}/page/yunshu-search",
                    "--mcp-url",
                    f"http://127.0.0.1:{mcp_port}/mcp",
                    "--skip",
                    skip,
                ]
                t0 = time.time()
                p = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=a.timeout * 3
                )
                (out / "sdk.txt").write_text(p.stdout + "\n" + p.stderr)
                results[name] = dict(
                    ok=p.returncode == 0, secs=round(time.time() - t0, 1)
                )
                print(p.stdout, p.stderr[-1500:], flush=True)
                continue
            if name in TUIS:
                import census_tui

                agent, inputs = TUIS[name]
                cfg_ = None
                census_tui.run_tui(
                    agent,
                    inputs,
                    f"e2e_{name}",
                    srv.url,
                    model_id,
                    6.0,
                    cfg_,
                    configure=(
                        (
                            lambda ln, ag=agent: apply_launch_config(
                                ag, ln, card_info, ln.cwd, model_id
                            )
                        )
                        if a.launch_config
                        else None
                    ),
                )
                dst = out / name
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(
                    census.OUT_ROOT / f"e2e_{name}",
                    dst,
                    ignore=shutil.ignore_patterns("home", "sandbox.sb"),
                )
                results[name] = dict(ok=True, note="see tui.txt")
                continue
            sc = SCENARIOS[name]
            run = census.BUILD / "runs" / f"e2e_{name}"
            shutil.rmtree(run, ignore_errors=True)
            run.mkdir(parents=True)
            work = census.BUILD / "work" / f"e2e_{name}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=work)
            if sc.get("setup"):
                SETUPS[sc["setup"]](work)
            proxy = (
                CaptureProxy(srv.url, run / "requests.jsonl")
                if sc.get("proxy", True)
                else None
            )
            base = proxy.url if proxy else srv.url
            launch = census.agents.prepare(
                sc["agent"], run, work, base, model_id, sc["prompt"]
            )
            if a.launch_config:
                apply_launch_config(sc["agent"], launch, card_info, run, model_id)
            hook = sc.get("cfg")
            if hook:
                census.HOOKS[hook](
                    launch.home, launch, dict(run=run, work=work, mcp_port=mcp_port)
                )
            t0 = time.time()
            proc = subprocess.Popen(
                launch.cmd,
                cwd=launch.cwd,
                env=launch.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                so, se = proc.communicate(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, 9)
                so, se = proc.communicate()
                se = "TIMEOUT\n" + (se or "")
            if proxy:
                proxy.stop()
            text = final_text(sc["agent"], so)
            ok = all(x.lower() in text.lower() for x in sc.get("expect", []))
            for fname, want in (sc.get("check_files") or {}).items():
                f = work / fname
                ok = ok and f.is_file() and want in f.read_text(errors="replace")
                text += f" [file {fname}: {'ok' if f.is_file() and want in f.read_text(errors='replace') else 'MISSING/WRONG'}]"
            results[name] = dict(
                ok=ok,
                secs=round(time.time() - t0, 1),
                rc=proc.returncode,
                answer=text[:400],
            )
            print(
                f"{name}: {'PASS' if ok else 'FAIL'} {results[name]['secs']}s rc={proc.returncode} :: {text[:160]!r}",
                flush=True,
            )
            dst = out / name
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(
                run, dst, ignore=shutil.ignore_patterns("home", "sandbox.sb")
            )
            (dst / "stdout.txt").write_text(so)
            (dst / "stderr.txt").write_text(se)
        (out / "results.json").write_text(json.dumps(results, indent=1))
    finally:
        srv.kill()
        sx.shutdown()


if __name__ == "__main__":
    main()
