"""Agent-compat census: run real coding agents against a scripted recording server.

Reuses the isolated CLI installs and the sandboxed launcher of scripts/research/agentic (its
`agents.prepare`: scrubbed env, isolated HOME / CLAUDE_CONFIG_DIR / CODEX_HOME, sandbox-exec with
loopback-only network). Nothing here touches the user's real ~/.claude, ~/.codex or ~/.config.

    python census.py list
    python census.py run cc_plain cx_shell ...   # or: run all
    python census.py summarize <run_dir>         # endpoint / header / tool-type table

Raw output goes to docs/research/runs/<date>-agent-census/<session>/ (gitignored).
"""

from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTIC = Path(os.environ.get("AGENTIC_SRC") or "/nonexistent")
if not AGENTIC.is_dir():
    for cand in (
        HERE.parent / "agentic",
        Path(
            "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.claude/worktrees/agent-ab474b259e82063f7/scripts/research/agentic"
        ),
        Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/research/agentic"),
    ):
        if (cand / "agents.py").is_file():
            AGENTIC = cand
            break
sys.path.insert(0, str(AGENTIC))
sys.path.insert(0, str(HERE))

import agents  # noqa: E402  (from scripts/research/agentic)
from census_server import Census  # noqa: E402

OUT_ROOT = Path(
    os.environ.get(
        "CENSUS_OUT",
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/"
        + time.strftime("%Y-%m-%d")
        + "-agent-census",
    )
)
BUILD = Path("/Volumes/P5Plus/yunshu-build/agent-compat")
MCP = str(HERE / "tiny_mcp.py")
PY = sys.executable
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# ---- sessions --------------------------------------------------------------------------
# agent, prompt, script (mock model steps), extra config hooks (name of a function below), timeout
S = {
    "cc_plain": dict(agent="claude", prompt="Say hello.", script=[{"text": "Hello!"}]),
    "cc_bash_edit": dict(
        agent="claude",
        prompt="Create hello.txt containing hi, then run ls.",
        script=[
            {
                "tool": "Write",
                "input": {"file_path": "$WORK/hello.txt", "content": "hi\n"},
            },
            {"tool": "Bash", "input": {"command": "ls", "description": "list"}},
            {
                "tool": "Edit",
                "input": {
                    "file_path": "$WORK/hello.txt",
                    "old_string": "hi",
                    "new_string": "hello",
                },
            },
            {"text": "Done."},
        ],
    ),
    "cc_websearch": dict(
        agent="claude",
        prompt="Search the web for yunshu.",
        script=[
            {"tool": "WebSearch", "input": {"query": "yunshu local inference"}},
            {"text": "Done."},
        ],
        cfg="cc_allow_web",
    ),
    "cc_webfetch": dict(
        agent="claude",
        prompt="Fetch the page.",
        script=[
            {
                "tool": "WebFetch",
                "input": {"url": "http://127.0.0.1:$PORT/page", "prompt": "summarize"},
            },
            {"text": "Done."},
        ],
        cfg="cc_allow_web",
    ),
    "cc_task": dict(
        agent="claude",
        prompt="Use a sub-agent to list files.",
        script=[
            {
                "tool": ["Task", "Agent"],
                "input": {
                    "description": "list",
                    "prompt": "list files",
                    "subagent_type": "general-purpose",
                },
            },
            {"tool": "Bash", "input": {"command": "ls", "description": "list"}},
            {"text": "sub done"},
            {"text": "Done."},
        ],
    ),
    "cc_mcp": dict(
        agent="claude",
        prompt="Use the tiny MCP echo tool.",
        script=[
            {"tool": "mcp__tiny__echo", "input": {"text": "ping"}},
            {"text": "Done."},
        ],
        cfg="cc_mcp",
    ),
    "cc_image": dict(
        agent="claude",
        prompt="Look at the image pic.png.",
        script=[
            {"tool": "Read", "input": {"file_path": "$WORK/pic.png"}},
            {"text": "A pixel."},
        ],
    ),
    "cc_thinking": dict(
        agent="claude",
        prompt="Think hard: 17*23?",
        script=[{"text": "391"}],
        cfg="cc_thinking",
    ),
    "cc_slash_context": dict(agent="claude", prompt="/context", script=[{"text": "x"}]),
    "cc_slash_cost": dict(agent="claude", prompt="/cost", script=[{"text": "x"}]),
    "cc_slash_compact": dict(
        agent="claude",
        prompt="/compact",
        script=[{"text": "<summary>s</summary>"}],
        cfg="cc_two_turn",
    ),
    "cc_discovery": dict(
        agent="claude", prompt="Say hi.", script=[{"text": "hi"}], cfg="cc_discovery"
    ),
    "cc_discovery2": dict(
        agent="claude",
        prompt="Say hi.",
        script=[{"text": "hi"}],
        cfg="cc_discovery",
        model="qwen3.8-27b",
        models_payload={
            "data": [
                {
                    "type": "model",
                    "id": "qwen3.8-27b",
                    "display_name": "Qwen3.8 27B",
                    "created_at": "2026-01-01T00:00:00Z",
                    "max_input_tokens": 131072,
                    "max_tokens": 65536,
                    "capabilities": {
                        "image_input": {"supported": True},
                        "thinking": {
                            "supported": True,
                            "types": {
                                "enabled": {"supported": True},
                                "adaptive": {"supported": True},
                            },
                        },
                        "effort": {
                            "supported": True,
                            "low": {"supported": True},
                            "medium": {"supported": True},
                            "high": {"supported": True},
                        },
                    },
                }
            ],
            "has_more": False,
            "first_id": "qwen3.8-27b",
            "last_id": "qwen3.8-27b",
        },
    ),
    "cc_statusline": dict(
        agent="claude", prompt="Say hi.", script=[{"text": "hi"}], cfg="cc_statusline"
    ),
    "cx_plain": dict(agent="codex", prompt="Say hello.", script=[{"text": "Hello!"}]),
    "cx_shell": dict(
        agent="codex",
        prompt="Run ls.",
        script=[
            {
                "tool": ["shell", "exec_command", "shell_command"],
                "input": {"command": ["ls"], "cmd": "ls"},
            },
            {"text": "Done."},
        ],
    ),
    "cx_websearch": dict(
        agent="codex",
        prompt="Search the web for yunshu.",
        script=[{"text": "x"}],
        cfg="cx_web_live",
    ),
    "cx_mcp": dict(
        agent="codex",
        prompt="Use the tiny echo tool.",
        script=[{"text": "x"}],
        cfg="cx_mcp",
    ),
    "cx_reasoning": dict(
        agent="codex", prompt="Think.", script=[{"text": "x"}], cfg="cx_reasoning"
    ),
    "cx_compact": dict(
        agent="codex",
        prompt="Run ls then say done.",
        script=[
            {"tool": ["exec_command"], "input": {"cmd": "ls"}},
            {"text": "SUMMARY: ran ls."},
            {"text": "Done."},
            {"text": "Done."},
        ],
        cfg="cx_compact",
    ),
    "cx_catalog": dict(
        agent="codex", prompt="Say hello.", script=[{"text": "x"}], cfg="cx_catalog"
    ),
    "cx_ws": dict(
        agent="codex", prompt="Say hello.", script=[{"text": "x"}], cfg="cx_ws"
    ),
    "oc_plain": dict(
        agent="opencode", prompt="Say hello.", script=[{"text": "Hello!"}]
    ),
    "oc_bash": dict(
        agent="opencode",
        prompt="Run ls.",
        script=[
            {"tool": "bash", "input": {"command": "ls", "description": "list"}},
            {"text": "Done."},
        ],
    ),
}


# ---- per-session config tweaks (applied after agents.prepare wrote the base config) -----
def cc_allow_web(home, launch, ctx):
    p = home / ".claude" / "settings.json"
    s = json.loads(p.read_text())
    s["permissions"]["deny"] = []
    s["permissions"]["allow"] += ["WebSearch", "WebFetch"]
    p.write_text(json.dumps(s))


def cc_mcp(home, launch, ctx):
    cfg = ctx["run"] / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"tiny": {"command": PY, "args": [MCP]}}}))
    s = home / ".claude" / "settings.json"
    d = json.loads(s.read_text())
    d["permissions"]["allow"].append("mcp__tiny")
    s.write_text(json.dumps(d))
    launch.cmd += ["--mcp-config", str(cfg)]


def cc_thinking(home, launch, ctx):
    launch.env["MAX_THINKING_TOKENS"] = "8000"
    s = home / ".claude" / "settings.json"
    d = json.loads(s.read_text())
    d["alwaysThinkingEnabled"] = True
    s.write_text(json.dumps(d))


def cc_discovery(home, launch, ctx):
    launch.env.pop("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", None)
    launch.env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    launch.env["ANTHROPIC_AUTH_TOKEN"] = "sk-yunshu-bench-dummy"


def cc_two_turn(home, launch, ctx):
    pass


def cc_statusline(home, launch, ctx):
    s = home / ".claude" / "settings.json"
    d = json.loads(s.read_text())
    out = ctx["run"] / "statusline.jsonl"
    d["statusLine"] = {"type": "command", "command": f"cat >> {out}; echo x"}
    s.write_text(json.dumps(d))


def _toml_append(home, text):
    p = home / ".codex" / "config.toml"
    p.write_text(p.read_text() + "\n" + text)


def cx_web_live(home, launch, ctx):
    p = home / ".codex" / "config.toml"
    p.write_text(
        p.read_text().replace('web_search = "disabled"', 'web_search = "live"')
    )


def cx_mcp(home, launch, ctx):
    _toml_append(home, f'[mcp_servers.tiny]\ncommand = "{PY}"\nargs = ["{MCP}"]\n')


def cx_reasoning(home, launch, ctx):
    p = home / ".codex" / "config.toml"
    t = p.read_text().replace(
        'web_search = "disabled"',
        'web_search = "disabled"\nmodel_reasoning_effort = "high"\n'
        'model_reasoning_summary = "detailed"\nmodel_supports_reasoning_summaries = true',
    )
    p.write_text(t)


def cx_compact(home, launch, ctx):
    p = home / ".codex" / "config.toml"
    import re

    t = re.sub(
        r"model_auto_compact_token_limit = \d+",
        "model_auto_compact_token_limit = 50",
        p.read_text(),
    )
    p.write_text(t)


CATALOG = {
    "models": [
        {
            "slug": "census-model",
            "base_instructions": "You are a coding agent.",
            "display_name": "Census Model",
            "description": "local model",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "fast"},
                {"effort": "medium", "description": "balanced"},
                {"effort": "high", "description": "deep"},
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 1,
            "supports_reasoning_summaries": True,
            "support_verbosity": False,
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "supports_parallel_tool_calls": True,
            "context_window": 131072,
            "input_modalities": ["text", "image"],
            "experimental_supported_tools": [],
        }
    ]
}


def cx_catalog(home, launch, ctx):
    cat = home / ".codex" / "catalog.json"
    cat.write_text(json.dumps(CATALOG))
    p = home / ".codex" / "config.toml"
    p.write_text(f'model_catalog_json = "{cat}"\n' + p.read_text())


def cx_ws(home, launch, ctx):
    p = home / ".codex" / "config.toml"
    p.write_text(
        p.read_text().replace(
            'wire_api = "responses"',
            'wire_api = "responses"\nsupports_websockets = true',
        )
    )


HOOKS = {
    f.__name__: f
    for f in (
        cc_discovery,
        cc_allow_web,
        cc_mcp,
        cc_thinking,
        cc_two_turn,
        cc_statusline,
        cx_web_live,
        cx_mcp,
        cx_reasoning,
        cx_compact,
        cx_ws,
        cx_catalog,
    )
}


def fill(obj, work: Path, port: int):
    if isinstance(obj, str):
        return obj.replace("$WORK", str(work)).replace("$PORT", str(port))
    if isinstance(obj, list):
        return [fill(x, work, port) for x in obj]
    if isinstance(obj, dict):
        return {k: fill(v, work, port) for k, v in obj.items()}
    return obj


def run_session(name: str, timeout: int = 180):
    spec = S[name]
    run = BUILD / "runs" / name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    work = BUILD / "work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "pic.png").write_bytes(PNG)
    (work / "README.md").write_text("# demo\n")
    subprocess.run(["git", "init", "-q"], cwd=work)
    srv = Census(run / "requests.jsonl", [], port=0)
    srv.script = fill(spec["script"], work, srv.port)
    srv.models_payload = spec.get("models_payload")
    if spec.get("model"):
        srv.model = spec["model"]
    try:
        launch = agents.prepare(
            spec["agent"],
            run,
            work,
            srv.url,
            spec.get("model", "census-model"),
            spec["prompt"],
        )
        # prepare() wrapped the command in sandbox-exec; hooks edit configs and may append args
        ctx = dict(run=run, work=work)
        hook = spec.get("cfg")
        if hook:
            HOOKS[hook](launch.home, launch, ctx)
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
            so, se = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            so, se = proc.communicate()
            rc, se = -9, "TIMEOUT\n" + (se or "")
        (run / "stdout.txt").write_text(so)
        (run / "stderr.txt").write_text(se)
        (run / "meta.json").write_text(
            json.dumps(
                dict(
                    name=name,
                    rc=rc,
                    secs=round(time.time() - t0, 1),
                    cmd=agents.describe(launch.cmd)[:400],
                )
            )
        )
        print(f"{name}: rc={rc} {time.time() - t0:.0f}s")
    finally:
        srv.stop()
        dst = OUT_ROOT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run, dst, ignore=shutil.ignore_patterns("home", "sandbox.sb"))
        homedst = dst / "home"
        for sub in (".claude", ".codex", ".config"):
            src = run / "home" / sub
            if src.exists():
                shutil.copytree(
                    src,
                    homedst / sub,
                    ignore=shutil.ignore_patterns("cache", "node_modules", "*.sqlite*"),
                    dirs_exist_ok=True,
                )


def load(run: Path):
    f = run / "requests.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def tool_ids(body):
    out = []
    for t in (body or {}).get("tools") or []:
        out.append(t.get("type") or "function") if not t.get("name") and not t.get(
            "function"
        ) else out.append(
            (t.get("type") or "custom")
            + ":"
            + (t.get("name") or (t.get("function") or {}).get("name", "?"))
        )
    return out


def summarize(root: Path):
    rows = collections.defaultdict(lambda: collections.Counter())
    hdr = collections.defaultdict(lambda: collections.defaultdict(set))
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for r in load(d):
            key = f"{r['method']} {r['path'].split('?')[0]} -> {r['status']}"
            rows[d.name][key] += 1
            for h in (
                "anthropic-beta",
                "anthropic-version",
                "openai-beta",
                "user-agent",
                "x-stainless-package-version",
                "originator",
                "session_id",
                "x-client-request-id",
                "x-app",
                "accept",
                "content-type",
            ):
                if h in r["headers"] or h.title() in r["headers"]:
                    v = r["headers"].get(h) or r["headers"].get(h.title())
                    hdr[d.name][h].add(v[:160])
    for name, c in rows.items():
        print(f"\n== {name}")
        for k, n in c.most_common():
            print(f"  {n:3d}  {k}")
        for h, vs in hdr[name].items():
            print(f"  hdr {h}: {sorted(vs)[:4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "summarize"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(S))
    elif a.cmd == "run":
        for n in list(S) if a.names in ([], ["all"]) else a.names:
            run_session(n, a.timeout)
    else:
        summarize(Path(a.names[0]) if a.names else OUT_ROOT)


if __name__ == "__main__":
    main()
