"""Yunshu CLI — service: run the server in the background at login (launchd).

``yunshu service install`` writes a per-user launchd agent
(``~/Library/LaunchAgents/com.yuhuanstudio.yunshu.plist``) that runs
``yunshu serve`` with the options given at install time, restarts it if it
crashes, and logs to ``~/Library/Logs/Yunshu/yunshu.log``. Nothing runs as
root and nothing outside those two paths is touched.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from yunshu_engine import paths, settings

from ._output import emit, fail, is_json

console = Console()
service_app = typer.Typer(
    help="Run Yunshu in the background at login (macOS launchd agent).",
    no_args_is_help=True,
)

# Passed through to the service when set in the installing shell, so the
# service finds the same Hugging Face cache and endpoint.
_PASSTHROUGH_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT", "HF_HUB_OFFLINE")
_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _target() -> str:
    return f"{_domain()}/{paths.SERVICE_LABEL}"


def log_file() -> Path:
    return paths.log_dir() / "yunshu.log"


def serve_args(
    model: str | None,
    models_dir: str | None,
    host: str,
    port: int,
    config: str | None,
    set_: list[str],
) -> list[str]:
    args = [sys.executable, "-m", "yunshu_cli", "serve", "--host", host]
    args += ["--port", str(port)]
    if model:
        p = Path(model).expanduser()
        args += ["--model", str(p.resolve()) if p.exists() else model]
    if models_dir:
        args += ["--models-dir", str(Path(models_dir).expanduser().resolve())]
    if config:
        args += ["--config", str(Path(config).expanduser().resolve())]
    for item in set_:
        args += ["--set", item]
    return args


def build_plist(program: list[str]) -> dict:
    env = {"PATH": _PATH}
    env.update({k: os.environ[k] for k in _PASSTHROUGH_ENV if os.environ.get(k)})
    log = str(log_file())
    return {
        "Label": paths.SERVICE_LABEL,
        "ProgramArguments": program,
        "RunAtLoad": True,
        # Restart after a crash, not after a clean stop.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": env,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _loaded() -> bool:
    return _launchctl("print", _target()).returncode == 0


def parse_print(text: str) -> dict:
    """pid / state / last exit code from ``launchctl print`` output."""
    out: dict = {}
    for key, pattern in (
        ("pid", r"^\s*pid = (\d+)"),
        ("state", r"^\s*state = (\S+)"),
        ("last_exit_code", r"^\s*last exit code = (.+)$"),
    ):
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            out[key] = int(m.group(1)) if key == "pid" else m.group(1).strip()
    return out


def _installed_address(plist_path: Path) -> tuple[str, int] | None:
    try:
        args = plistlib.loads(plist_path.read_bytes())["ProgramArguments"]
        host = args[args.index("--host") + 1]
        port = int(args[args.index("--port") + 1])
    except (OSError, KeyError, ValueError, IndexError, plistlib.InvalidFileException):
        return None
    return ("127.0.0.1" if host == "0.0.0.0" else host), port


@service_app.command("install")
def install(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model path or Hugging Face repo id to serve."
    ),
    models_dir: str | None = typer.Option(
        None, "--models-dir", "-d", help="Serve every model in this directory."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    config: str | None = typer.Option(
        None, "--config", "-c", help="TOML settings file the service reads."
    ),
    set_: list[str] = typer.Option(
        [], "--set", help="Setting override KEY=VALUE (repeatable)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the launchd agent and change nothing."
    ),
    no_start: bool = typer.Option(
        False, "--no-start", help="Write the agent but do not start it now."
    ),
    force: bool = typer.Option(False, "--force", help="Replace an installed agent."),
):
    """Install the launchd agent: start Yunshu at login and restart it on crash."""
    if sys.platform != "darwin":
        fail("`yunshu service` uses launchd and runs on macOS only.", code=2)
    if not model and not models_dir and not settings.get("YUNSHU_MODEL"):
        fail(
            "Say what to serve: --model <path or repo id> or --models-dir <dir> "
            "(or put YUNSHU_MODEL in the --config file).",
            code=2,
        )
    if model:
        from .doctor import check_model

        bad = [c for c in check_model(model, {}) if c.status == "fail"]
        if bad:
            fail(f"{bad[0].detail}. {bad[0].fix}", code=2)
    if config and not Path(config).expanduser().is_file():
        fail(f"Config file not found: {config}", code=2)

    plist = build_plist(serve_args(model, models_dir, host, port, config, set_))
    dest = paths.launch_agent_plist()
    if dry_run:
        text = plistlib.dumps(plist).decode()
        emit({"path": str(dest), "plist": plist}, human=lambda: print(text))
        return
    if dest.exists() and not force:
        fail(
            f"Already installed at {dest}. Use --force to replace it, or "
            "`yunshu service uninstall` first.",
            code=1,
        )
    if dest.exists() and _loaded():
        _launchctl("bootout", _target())
    dest.parent.mkdir(parents=True, exist_ok=True)
    paths.log_dir().mkdir(parents=True, exist_ok=True)
    dest.write_bytes(plistlib.dumps(plist))
    started = False
    if not no_start:
        r = _launchctl("bootstrap", _domain(), str(dest))
        if r.returncode != 0:
            fail(f"Wrote {dest} but launchctl could not start it: {r.stderr.strip()}")
        started = True
    emit(
        {"path": str(dest), "log": str(log_file()), "started": started},
        human=lambda: console.print(
            f"[green]✓ Installed[/] {dest}\n"
            f"  {'Started; it' if started else 'It'} runs at login on "
            f"http://{host}:{port} and restarts after a crash.\n"
            f"  Logs: {log_file()}  (yunshu service logs -f)"
        ),
    )


@service_app.command("uninstall")
def uninstall():
    """Stop the service and remove the launchd agent (models and logs are kept)."""
    dest = paths.launch_agent_plist()
    was_loaded = _loaded()
    if was_loaded:
        _launchctl("bootout", _target())
    removed = dest.exists()
    if removed:
        dest.unlink()
    emit(
        {"removed": removed, "stopped": was_loaded},
        human=lambda: console.print(
            f"[green]✓ Removed[/] {dest}" if removed else "Not installed."
        ),
    )


@service_app.command("start")
def start():
    """Start the installed service now."""
    dest = paths.launch_agent_plist()
    if not dest.exists():
        fail("Not installed. Run `yunshu service install --model <model>` first.")
    r = (
        _launchctl("kickstart", _target())
        if _loaded()
        else _launchctl("bootstrap", _domain(), str(dest))
    )
    if r.returncode != 0:
        fail(f"launchctl: {r.stderr.strip() or r.stdout.strip()}")
    emit({"started": True}, human=lambda: console.print("[green]✓ Started[/]"))


@service_app.command("stop")
def stop():
    """Stop the service until the next login (or `yunshu service start`)."""
    if _loaded():
        _launchctl("bootout", _target())
    emit({"stopped": True}, human=lambda: console.print("[green]✓ Stopped[/]"))


@service_app.command("restart")
def restart():
    """Restart the running service (e.g. after changing its config file)."""
    if not _loaded():
        fail("The service is not running. Use `yunshu service start`.")
    r = _launchctl("kickstart", "-k", _target())
    if r.returncode != 0:
        fail(f"launchctl: {r.stderr.strip()}")
    emit({"restarted": True}, human=lambda: console.print("[green]✓ Restarted[/]"))


@service_app.command("status")
def status():
    """Installed? running? healthy? — and where the logs are."""
    dest = paths.launch_agent_plist()
    info: dict = {
        "installed": dest.exists(),
        "plist": str(dest),
        "log": str(log_file()),
    }
    r = _launchctl("print", _target())
    info["loaded"] = r.returncode == 0
    if info["loaded"]:
        info.update(parse_print(r.stdout))
    addr = _installed_address(dest) if dest.exists() else None
    if addr:
        info["url"] = f"http://{addr[0]}:{addr[1]}"
        info["healthy"] = _healthy(info["url"])
    if is_json():
        emit(info)
        return
    if not info["installed"]:
        console.print("Not installed. `yunshu service install --model <model>`")
        return
    state = (
        "running" if info.get("pid") else ("loaded" if info["loaded"] else "stopped")
    )
    console.print(
        f"Service: [bold]{state}[/]"
        + (f" (pid {info['pid']})" if info.get("pid") else "")
    )
    if "last_exit_code" in info:
        console.print(f"Last exit: {info['last_exit_code']}")
    if addr:
        mark = "[green]healthy[/]" if info["healthy"] else "[yellow]not answering[/]"
        console.print(f"Server: {info['url']} {mark}")
    console.print(f"Agent: {dest}\nLogs: {log_file()}")


def _healthy(url: str) -> bool:
    try:
        import httpx

        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


@service_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Lines to show."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep printing."),
):
    """Show the service log."""
    path = log_file()
    if not path.exists():
        fail(f"No log yet at {path}.")
    cmd = ["tail", "-n", str(lines)] + (["-F"] if follow else []) + [str(path)]
    if follow:
        os.execvp("tail", cmd)
    print(subprocess.run(cmd, capture_output=True, text=True).stdout, end="")
