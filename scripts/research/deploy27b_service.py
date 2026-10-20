"""gpuq-only isolated launchd crash/restart and wheel-upgrade acceptance check.

Refuses to touch a pre-existing service. The test plist and HOME live on P5Plus;
launchd's label is the product label because the console restart API targets it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic.servers import free_ports

LABEL = "com.yuhuanstudio.yunshu"


def service_pid(text):
    match = re.search(r"^\s*pid = (\d+)$", text, re.M)
    return int(match[1]) if match else None


def require_owned_job(state, program):
    """An inactive submitted job still needs bootout after a startup failure."""
    if str(program) not in state:
        raise RuntimeError("launchd job ownership could not be verified")
    return service_pid(state)


def require_owned(state, program):
    """Never signal a launchd job unless its executable is our isolated interpreter."""
    if str(program) not in state or service_pid(state) is None:
        raise RuntimeError("launchd job ownership could not be verified")
    return service_pid(state)


def submit_args(job):
    """Run the isolated plist's command when noowners prevents plist bootstrap.

    launchctl submit creates a keepalive job in the caller's launchd domain. Its
    clean-exit policy differs from the product plist, so record the transport and
    use this only for the crash/API-restart/upgrade checks, never plist validation.
    No environment is inherited from the worker or the user's actual HOME.
    """
    argv = job["ProgramArguments"]
    code = (
        "import os; "
        f"os.chdir({job['WorkingDirectory']!r}); "
        f"out=os.open({job['StandardOutPath']!r}, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600); "
        "os.dup2(out,1); os.close(out); "
        f"err=os.open({job['StandardErrorPath']!r}, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600); "
        "os.dup2(err,2); os.close(err); "
        f"os.execve({argv[0]!r}, {argv!r}, {job['EnvironmentVariables']!r})"
    )
    # The isolated Python interpreter is after nice's two arguments.
    python_bin = argv[3] if argv[0] == "/usr/bin/nice" else argv[0]
    return [
        "submit",
        "-l",
        job["Label"],
        "--",
        python_bin,
        "-c",
        code,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", type=Path, required=True, dest="python_bin")
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument(
        "--launch-method", choices=("bootstrap", "submit"), default="bootstrap"
    )
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output = a.output.absolute()
    root = a.output.parent / a.output.stem
    if not str(root).startswith("/Volumes/P5Plus/") or a.output.exists():
        ap.error("unique P5Plus output required")
    python_bin, wheel, model = (p.absolute() for p in (a.python_bin, a.wheel, a.model))
    target = f"gui/{os.getuid()}/{LABEL}"

    def ctl(*args):
        return subprocess.run(
            ["launchctl", *args], capture_output=True, text=True, timeout=20
        )

    if ctl("print", target).returncode == 0:
        raise RuntimeError(
            "existing service belongs to someone else; refusing to replace it"
        )
    home = root / "home"
    home.mkdir(parents=True)
    tmp = root / "tmp"
    tmp.mkdir()
    port = free_ports(1)[0]
    url = f"http://127.0.0.1:{port}"
    admin = secrets.token_urlsafe(32)
    data_dir = home / ".yunshu"
    models = data_dir / "models" / "deploy27b"
    models.parent.mkdir(parents=True)
    models.symlink_to(model, target_is_directory=True)
    config = data_dir / "config.toml"
    config.write_text(f'auth_token = "{admin}"\nvlm_apc_memory_gb = 1.0\n')
    config.chmod(0o600)
    # launchd refuses to bootstrap a plist from a noowners volume such as P5Plus
    # ("Bootstrap failed: 5: Input/output error"), so only this few-KB file lives on
    # the internal disk, in a private temp dir removed at exit, and so does the
    # server log (launchd exits 78 EX_CONFIG when StandardOutPath is on that volume;
    # it is copied to P5Plus at exit). HOME, models and the venv stay on P5Plus.
    plist_dir = Path(tempfile.mkdtemp(prefix="deploy27b-plist-"))
    plist = plist_dir / f"{LABEL}.plist"
    env = {
        "HOME": str(home),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(tmp),
        "HF_HOME": str(root / "hf"),
        "HF_HUB_OFFLINE": "1",
    }
    program = [
        "/usr/bin/nice",
        "-n",
        "15",
        str(python_bin),
        "-m",
        "yunshu_cli",
        "serve",
        "--no-console",
        "--model",
        str(models),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": LABEL,
                "ProgramArguments": program,
                "RunAtLoad": True,
                "KeepAlive": {"SuccessfulExit": False},
                "ThrottleInterval": 2,
                "WorkingDirectory": str(home),
                "EnvironmentVariables": env,
                "StandardOutPath": str(plist_dir / "server.log"),
                "StandardErrorPath": str(plist_dir / "server.log"),
            }
        )
    )
    plist.chmod(0o600)

    def boot_job():
        if a.launch_method == "submit":
            return ctl(*submit_args(plistlib.loads(plist.read_bytes())))
        return ctl("bootstrap", f"gui/{os.getuid()}", str(plist))

    events = []
    owned = False

    def api(path, method="GET", body=None, token=admin):
        request = urllib.request.Request(
            url + path,
            method=method,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def ready(previous=None):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            pid = service_pid(ctl("print", target).stdout)
            try:
                if pid and pid != previous and api("/health/ready").get("ready"):
                    return pid
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        raise RuntimeError("service did not return ready with a new pid")

    def chat(token=admin):
        reply = api(
            "/v1/chat/completions",
            "POST",
            {
                "model": api("/v1/models")["data"][0]["id"],
                "messages": [{"role": "user", "content": "Reply READY only."}],
                "max_tokens": 32,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            token,
        )
        if not reply.get("choices") or not reply["choices"][0]["message"].get(
            "content"
        ):
            raise RuntimeError("empty first response")

    result = {
        "complete": False,
        "passed": False,
        "events": events,
        "port": port,
        "launch_method": a.launch_method,
        "plist_bootstrap_verified": False,
    }
    try:
        boot = boot_job()
        if boot.returncode:
            raise RuntimeError("bootstrap failed: " + boot.stderr.strip())
        owned = True
        result["plist_bootstrap_verified"] = a.launch_method == "bootstrap"
        pid = ready()
        chat()
        key = api(
            "/v1/yunshu/keys",
            "POST",
            {"name": "deploy27b-upgrade", "scopes": ["infer"]},
        )
        key_secret = key.pop("secret")
        before_keys = hashlib.sha256((data_dir / "keys.json").read_bytes()).hexdigest()
        before_config = config.read_bytes()
        require_owned(ctl("print", target).stdout, python_bin)
        os.kill(pid, signal.SIGKILL)
        restarted = ready(pid)
        chat()
        events.append(
            {
                "check": "crash_restart_first_request",
                "passed": True,
                "old_pid": pid,
                "new_pid": restarted,
            }
        )
        require_owned(ctl("print", target).stdout, python_bin)
        if ctl("bootout", target).returncode:
            raise RuntimeError("could not stop owned old-wheel service")
        owned = False
        install_env = {
            **os.environ,
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "TMPDIR": str(tmp),
        }
        with (root / "upgrade.log").open("w") as log:
            upgrade = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python_bin),
                    "--no-deps",
                    "--reinstall-package",
                    "yunshu",
                    str(wheel),
                ],
                env=install_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if upgrade.returncode:
            raise RuntimeError("upgrade install failed")
        if ctl("print", target).returncode == 0:
            raise RuntimeError("another service appeared during upgrade")
        if boot_job().returncode:
            raise RuntimeError("upgraded bootstrap failed")
        owned = True
        pid = ready(restarted)
        chat(key_secret)
        if (
            hashlib.sha256((data_dir / "keys.json").read_bytes()).hexdigest()
            != before_keys
        ):
            raise RuntimeError("API keys changed during upgrade")
        if config.read_bytes() != before_config or models.resolve() != model.resolve():
            raise RuntimeError("settings/model link changed during upgrade")
        events.append(
            {"check": "upgrade_preserves_models_settings_keys", "passed": True}
        )
        change = api(
            "/v1/yunshu/config", "PATCH", {"settings": {"YUNSHU_KEEP_ALIVE_TIMEOUT": 7}}
        )
        if not change.get("restart_required") or not change.get("restart", {}).get(
            "available"
        ):
            raise RuntimeError("console did not offer required restart")
        api("/v1/yunshu/service/restart", "POST", {"confirm": True})
        new_pid = ready(pid)
        chat(key_secret)
        effective = api("/v1/yunshu/config")["settings"]
        if not any(
            r["name"] == "YUNSHU_KEEP_ALIVE_TIMEOUT" and r["value"] == 7
            for r in effective
        ):
            raise RuntimeError("restart setting did not take effect")
        events.append(
            {
                "check": "console_restart_first_request",
                "passed": True,
                "old_pid": pid,
                "new_pid": new_pid,
            }
        )
        result.update(complete=True, passed=True)
    except Exception as exc:  # noqa: BLE001
        result.update(complete=True, error=str(exc))
    finally:
        if owned:
            state = ctl("print", target)
            require_owned_job(state.stdout, python_bin)
            ctl("bootout", target)
        if (plist_dir / "server.log").exists():
            shutil.copyfile(plist_dir / "server.log", root / "server.log")
        shutil.rmtree(plist_dir, ignore_errors=True)
        a.output.write_text(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
