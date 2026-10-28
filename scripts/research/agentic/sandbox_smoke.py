"""CPU smoke for the srt sandbox the agentic harness generates (no GPU, no engine).

Runs, under the exact settings ``agents.wrap_srt`` writes: every agent CLI ``--version``, a zsh
heredoc write, a loopback fetch (allowed), a public-internet fetch (blocked), a write outside the
run dirs (blocked) and a read of a denied real-HOME path (blocked). Exits nonzero on any miss.
"""

from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agents  # noqa: E402

PORT = 0  # picked from 18990-18999 in main()


def run(cmd: list[str], env: dict, cwd: Path, settings: Path, writable, timeout=60):
    full = agents.wrap_srt(cmd, settings, writable, PORT)
    p = subprocess.run(
        full, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    run_dir = agents.BUILD / "sandbox-smoke"
    shutil.rmtree(run_dir, ignore_errors=True)
    work = run_dir / "work"
    work.mkdir(parents=True)
    env = agents.base_env(run_dir / "home")
    writable = [run_dir, work]
    settings = run_dir / "srt-settings.json"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    global PORT
    srv = None
    for PORT in range(18999, 18989, -1):
        try:
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
            break
        except OSError:
            continue
    if srv is None:
        print("no free port in 18990-18999", file=sys.stderr)
        return 2
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    results: dict[str, dict] = {}

    def check(name, cmd, ok):
        rc, out = run(cmd, env, work, settings, writable)
        good = ok(rc, out)
        results[name] = {"pass": good, "rc": rc, "out": out[:300]}
        print(
            f"[{'PASS' if good else 'FAIL'}] {name}: rc={rc} {out[:120]!r}", flush=True
        )

    try:
        for a in ("opencode", "claude", "codex"):
            exe = agents.CODEX_BIN if a == "codex" else agents.BIN / a
            check(
                f"{a} --version", [str(exe), "--version"], lambda rc, o: rc == 0 and o
            )
        z = ["/bin/zsh", "-c"]
        check(
            "zsh heredoc write",
            [*z, f"cat > {work}/f <<EOF\nhello\nEOF\ncat {work}/f"],
            lambda rc, o: rc == 0 and o.endswith("hello"),
        )
        check(
            "loopback allowed",
            [*z, f"curl -s -m 5 http://127.0.0.1:{PORT}/"],
            lambda rc, o: rc == 0 and o.endswith("ok"),
        )
        check(
            "internet blocked",
            [*z, "curl -s -m 8 -o /dev/null -w '%{http_code}' https://example.com"],
            lambda rc, o: o.endswith("000") or rc != 0,
        )
        check(
            "write outside blocked",
            [*z, f"echo x > {agents.REAL_HOME}/yunshu_sandbox_probe"],
            lambda rc, o: (
                rc != 0 and not (agents.REAL_HOME / "yunshu_sandbox_probe").exists()
            ),
        )
        check(
            "denied read blocked",
            [*z, f"ls {agents.REAL_HOME}/Documents"],
            lambda rc, o: rc != 0 and "not permitted" in o.lower(),
        )
    finally:
        srv.shutdown()
    (run_dir / "smoke.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r["pass"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
