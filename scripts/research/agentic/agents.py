"""Headless coding-agent launchers: opencode, Claude Code, Codex CLI.

Each run gets a fresh, fully isolated environment under ``BUILD`` (HOME, XDG_*, TMPDIR,
CLAUDE_CONFIG_DIR, CODEX_HOME, git/npm config), a scrubbed environment (nothing inherited
except PATH basics), and a dummy API key. The whole agent process tree is additionally wrapped
in a macOS ``sandbox-exec`` profile that

* denies all network except loopback (so a stray telemetry / update / registry call fails
  loudly instead of leaving the machine),
* denies reading the user's real agent configs, keychain, shell rc and git/npm config,
* denies writing anywhere except the run's own directories.

The pinned agent CLIs live in ``AGENTIC_CLIS`` (see ``install_agents.sh``); the user's own
installs are never used.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

CLIS = Path(
    os.environ.get("AGENTIC_CLIS", "/Volumes/P5Plus/yunshu-test-envs/agentic-clis")
)
BUILD = Path(os.environ.get("AGENTIC_BUILD", "/Volumes/P5Plus/yunshu-build/agentic"))
BIN = CLIS / "node_modules" / ".bin"
PYENV = CLIS / "pyenv"
CODEX_BIN = (
    CLIS
    / "node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
)
API_KEY = "sk-yunshu-bench-dummy"
REAL_HOME = Path(os.path.expanduser("~"))

# Hidden-from-agent parts of the real home (read + write denied by the sandbox).
DENY_HOME = [
    ".claude",
    ".claude.json",
    ".codex",
    ".config",
    ".ssh",
    ".gnupg",
    ".aws",
    ".npmrc",
    ".gitconfig",
    ".zshrc",
    ".zprofile",
    ".zshenv",
    ".bashrc",
    ".bash_profile",
    ".netrc",
    ".local/share/opencode",
    ".cache",
    "Library/Keychains",
    "Library/Application Support",
    "Library/Preferences",
    "Library/Caches",
    "Documents",
    "Desktop",
    "Downloads",
]


@dataclass
class Launch:
    cmd: list[str]
    env: dict[str, str]
    cwd: Path
    home: Path
    notes: dict = field(default_factory=dict)


def base_env(home: Path, port_hint: str = "") -> dict[str, str]:
    """Scrubbed environment: only what is set here reaches the agent."""
    tmp = home / "tmp"
    for d in (
        home,
        tmp,
        home / ".config",
        home / ".local" / "share",
        home / ".cache",
        home / ".local" / "state",
    ):
        d.mkdir(parents=True, exist_ok=True)
    gitcfg = home / "gitconfig"
    gitcfg.write_text(
        "[user]\n\tname = bench\n\temail = bench@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n[safe]\n\tdirectory = *\n"
    )
    return {
        "PATH": f"{PYENV}/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "TMPDIR": str(tmp) + "/",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "dumb",
        "CI": "1",
        "NO_COLOR": "1",
        "SHELL": "/bin/zsh",
        "USER": os.environ.get("USER", "bench"),
        "GIT_CONFIG_GLOBAL": str(gitcfg),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "npm_config_userconfig": str(home / ".npmrc"),
        "npm_config_globalconfig": str(home / ".npmglobalrc"),
        "npm_config_cache": str(home / ".npm"),
        "npm_config_update_notifier": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "DO_NOT_TRACK": "1",
    }


def sandbox_profile(writable: list[Path]) -> str:
    lines = ["(version 1)", "(allow default)", "(deny network*)"]
    lines += [
        '(allow network-outbound (remote ip "localhost:*"))',
        '(allow network-inbound (local ip "localhost:*"))',
        '(allow network-bind (local ip "localhost:*"))',
    ]
    for p in DENY_HOME:
        lines.append(f'(deny file-read* file-write* (subpath "{REAL_HOME / p}"))')
    lines.append("(deny file-write*)")
    for p in writable:
        lines.append(f'(allow file-write* (subpath "{p}"))')
    lines += [
        '(allow file-write* (literal "/dev/null") (literal "/dev/tty") (regex #"^/dev/(fd|pts)/"))',
        '(allow file-write* (subpath "/private/var/folders"))',
    ]
    return "\n".join(lines) + "\n"


def wrap_sandbox(cmd: list[str], profile_path: Path, writable: list[Path]) -> list[str]:
    profile_path.write_text(sandbox_profile(writable))
    return ["/usr/bin/sandbox-exec", "-f", str(profile_path), *cmd]


def agent_version(agent: str) -> str:
    import subprocess

    exe = {"opencode": "opencode", "claude": "claude", "codex": "codex"}[agent]
    try:
        return subprocess.run(
            [str(CODEX_BIN if agent == "codex" else BIN / exe), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "HOME": str(BUILD / "install-home")},
        ).stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


def prepare(
    agent: str,
    run_dir: Path,
    workdir: Path,
    base_url: str,
    model: str,
    prompt: str,
    context_len: int = 131072,
    max_output: int = 32768,
) -> Launch:
    """Build the command + environment for one headless run.

    ``base_url`` is the recording proxy root (no ``/v1``); the agent must reach only it.
    """
    home = run_dir / "home"
    env = base_env(home)
    fn = {"opencode": _opencode, "claude": _claude, "codex": _codex}[agent]
    launch = fn(env, home, workdir, base_url, model, prompt, context_len, max_output)
    writable = [run_dir, workdir]
    launch.cmd = wrap_sandbox(launch.cmd, run_dir / "sandbox.sb", writable)
    return launch


def _opencode(env, home, workdir, base_url, model, prompt, ctx, max_out):
    cfgdir = home / ".config" / "opencode"
    cfgdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "yunshu": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Local engine",
                "options": {"baseURL": base_url + "/v1", "apiKey": API_KEY},
                "models": {
                    model: {
                        "name": model,
                        "tool_call": True,
                        "reasoning": True,
                        "limit": {"context": ctx, "output": max_out},
                    }
                },
            }
        },
        "model": f"yunshu/{model}",
        "small_model": f"yunshu/{model}",
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    }
    (cfgdir / "opencode.json").write_text(json.dumps(cfg, indent=1))
    env.update(
        OPENCODE_DISABLE_AUTOUPDATE="true",
        OPENCODE_DISABLE_MODELS_FETCH="true",
        OPENCODE_DISABLE_LSP_DOWNLOAD="true",
        OPENCODE_DISABLE_SHARE="true",
        OPENCODE_DISABLE_DEFAULT_PLUGINS="true",
        OPENCODE_DISABLE_CLAUDE_CODE="true",
        OPENCODE_DISABLE_EXTERNAL_SKILLS="true",
        OPENCODE_DISABLE_PRUNE="true",
        OPENCODE_CONFIG=str(cfgdir / "opencode.json"),
    )
    cmd = [
        str(BIN / "opencode"),
        "run",
        "--pure",
        "--format",
        "json",
        "--model",
        f"yunshu/{model}",
        "--dir",
        str(workdir),
        prompt,
    ]
    return Launch(cmd, env, workdir, home)


def _claude(env, home, workdir, base_url, model, prompt, ctx, max_out):
    cdir = home / ".claude"
    cdir.mkdir(parents=True, exist_ok=True)
    settings = {
        "permissions": {
            "allow": [
                "Bash",
                "Edit",
                "Write",
                "MultiEdit",
                "Read",
                "Glob",
                "Grep",
                "NotebookEdit",
                "TodoWrite",
            ],
            "deny": ["WebFetch", "WebSearch"],
            "defaultMode": "acceptEdits",
        },
        "includeCoAuthoredBy": False,
        "autoUpdatesChannel": "stable",
        "env": {"DISABLE_AUTOUPDATER": "1"},
    }
    (cdir / "settings.json").write_text(json.dumps(settings, indent=1))
    # Pre-seed the "onboarding done / trust this folder" state so -p never blocks or prompts.
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "numStartups": 5,
                "theme": "dark",
                "customApiKeyResponses": {"approved": [API_KEY[-20:]], "rejected": []},
                "projects": {
                    str(workdir): {
                        "hasTrustDialogAccepted": True,
                        "hasCompletedProjectOnboarding": True,
                        "allowedTools": [],
                    }
                },
            }
        )
    )
    # with CLAUDE_CONFIG_DIR set, the global state lives inside it (the interactive TUI reads this one)
    (cdir / ".claude.json").write_text((home / ".claude.json").read_text())
    env.update(
        CLAUDE_CONFIG_DIR=str(cdir),
        CLAUDE_CODE_TMPDIR=str(home / "tmp"),  # else it writes /tmp/claude-<uid>/
        ANTHROPIC_BASE_URL=base_url,
        ANTHROPIC_API_KEY=API_KEY,
        ANTHROPIC_MODEL=model,
        ANTHROPIC_SMALL_FAST_MODEL=model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=model,
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1",
        DISABLE_ERROR_REPORTING="1",
        DISABLE_AUTOUPDATER="1",
        DISABLE_BUG_COMMAND="1",
        DISABLE_COST_WARNINGS="1",
        CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
        CLAUDE_CODE_DISABLE_TERMINAL_TITLE="1",
        CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY="1",
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(max_out),
        CLAUDE_CODE_USE_BEDROCK="0",
        CLAUDE_CODE_USE_VERTEX="0",
        BASH_DEFAULT_TIMEOUT_MS="600000",
    )
    cmd = [
        str(BIN / "claude"),
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--setting-sources",
        "user",
        "--strict-mcp-config",
        "--add-dir",
        str(workdir),
    ]
    return Launch(cmd, env, workdir, home)


def _codex(env, home, workdir, base_url, model, prompt, ctx, max_out):
    chome = home / ".codex"
    chome.mkdir(parents=True, exist_ok=True)
    toml = f"""model = "{model}"
model_provider = "yunshu"
approval_policy = "never"
sandbox_mode = "danger-full-access"
check_for_update_on_startup = false
model_context_window = {ctx}
model_auto_compact_token_limit = {int(ctx * 0.8)}
web_search = "disabled"
disable_paste_burst = true

[analytics]
enabled = false

[feedback]
enabled = false

[model_providers.yunshu]
name = "Local engine"
base_url = "{base_url}/v1"
env_key = "YUNSHU_API_KEY"
wire_api = "responses"
request_max_retries = 1
stream_max_retries = 1
"""
    (chome / "config.toml").write_text(toml)
    env.update(CODEX_HOME=str(chome), YUNSHU_API_KEY=API_KEY)
    cmd = [
        str(CODEX_BIN),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "-C",
        str(workdir),
        "--dangerously-bypass-approvals-and-sandbox",  # we ARE the sandbox: sandbox-exec, see module doc
        prompt,
    ]
    return Launch(cmd, env, workdir, home)


def describe(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
