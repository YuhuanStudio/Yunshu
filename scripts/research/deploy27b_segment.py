"""gpuq-only deployment segment: reuse realistic soak and the agentic server lifecycle.

Run a tiny pilot first, then one --minutes 60 job. Each segment has its
own HOME and outputs; the controller resumes only completed, clean segments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic import servers
from soak_realistic import Client, idle_verdict, memory_ledger
from soak_realistic import main as soak_main


def parse_vm(swapusage: str, vm_stat: str, pressure: str) -> dict:
    row: dict = {}
    for key in ("total", "used"):
        part = swapusage.split(f"{key} =")[1].split()[0]
        scale = {"M": 1.0, "G": 1024.0}[part[-1]]
        row[f"swap_{key}_mb"] = float(part[:-1]) * scale
    for line in vm_stat.splitlines():
        for key in ("Pageouts", "Swapouts", "Swapins", "Pageins"):
            if line.startswith(f"{key}:"):
                row[key.lower()] = int(line.split(":")[1].strip().rstrip("."))
    row["free_pct"] = next(
        (
            int(x.split(":")[1].strip().rstrip("%"))
            for x in pressure.splitlines()
            if "free percentage" in x
        ),
        None,
    )
    return row


def vm_sample() -> dict:
    """Host-wide swap and paging counters (no process content)."""

    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout

    row: dict = {"t": time.time()}
    try:
        row.update(
            parse_vm(
                run("/usr/sbin/sysctl", "-n", "vm.swapusage"),
                run("/usr/bin/vm_stat"),
                run("/usr/bin/memory_pressure", "-Q"),
            )
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = type(exc).__name__
    return row


class PressureSampler(threading.Thread):
    def __init__(self, path: Path, every: float = 30.0):
        super().__init__(daemon=True)
        self.path, self.every, self._stop_evt = path, every, threading.Event()

    def run(self):
        with self.path.open("a") as f:
            while True:
                f.write(json.dumps(vm_sample()) + "\n")
                f.flush()
                if self._stop_evt.wait(self.every):
                    return

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--resident-load-gib", type=int, default=0)
    ap.add_argument("--apc-gb", type=float, default=8)
    ap.add_argument("--max-memory-gb", type=float, default=None)
    ap.add_argument("--prefill-buffer-gb", type=float, default=None)
    ap.add_argument(
        "--pressure-log",
        type=Path,
        default=None,
        help="sample host swap / pageouts / memory pressure every 30 s into this JSONL",
    )
    a = ap.parse_args()
    if not 0 <= a.resident_load_gib <= 24:
        ap.error("resident load must be 0..24 GiB")
    if a.output.exists():
        ap.error("output exists; unique paths required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    root = a.output.parent / a.output.stem
    servers.SERVER_HOME = root / "home"
    os.environ["AGENTIC_YUNSHU_SRC"] = str(
        Path(__file__).resolve().parents[2] / "python"
    )
    os.environ["HF_HOME"] = str(root / "hf")
    os.environ["TMPDIR"] = str(root / "tmp")
    is_27b = "27B" in Path(a.model).name
    if is_27b:
        os.environ["AGENTIC_YUNSHU_DRAFTER"] = (
            "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
        )
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    server, _ = servers.start_server(
        1,
        lambda port: servers.Server(
            "yunshu",
            a.model,
            port,
            root / "server.log",
            [
                "--no-console",
                "--set",
                "YUNSHU_AUTH_DISABLED=true",
                "--set",
                f"YUNSHU_VLM_APC_MEMORY_GB={a.apc_gb:g}",
                "--set",
                "YUNSHU_CACHE_RESERVE_GB=100",
                *(
                    ["--set", f"YUNSHU_MAX_MEMORY_GB={a.max_memory_gb:g}"]
                    if a.max_memory_gb is not None
                    else []
                ),
                *(
                    ["--set", f"YUNSHU_PREFILL_BUFFER_CACHE_GB={a.prefill_buffer_gb:g}"]
                    if a.prefill_buffer_gb is not None
                    else []
                ),
            ],
        ),
    )
    try:
        assert server.proc is not None
        mode = server.engaged_spec_mode() if is_27b else None
        if is_27b and mode != "dflash":
            raise RuntimeError(f"27B expected DFlash, engaged {mode}")
        # Establish a resident, exercised baseline: lazy weight residency and one-time
        # initialization are not a leak caused by the subsequent workload.
        warm = Client(server.url, server.model_id, 180).chat(
            [{"role": "user", "content": "Reply READY only."}], 32
        )
        if warm.get("error") or not warm.get("done") or not warm.get("content"):
            raise RuntimeError("baseline warm request failed")
        time.sleep(40)

        def diagnose(phase):
            if not a.diagnostics:
                return True
            # Only allocation metadata and vmmap's region summary, never content.
            path = root / f"{phase}-census.json"
            ok = True
            try:
                with urllib.request.urlopen(
                    server.url + "/debug/memory-census?min_mib=8", timeout=120
                ) as response:
                    sample = json.load(response)
                ok = isinstance(sample.get("active_mib"), (int, float))
                path.write_text(json.dumps(sample) + "\n")
            except Exception as exc:  # noqa: BLE001
                ok = False
                path.write_text(json.dumps({"error": type(exc).__name__}) + "\n")
            with (root / f"{phase}-vmmap.txt").open("w") as log:
                try:
                    proc = subprocess.run(
                        ["/usr/bin/vmmap", "-summary", str(server.proc.pid)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                    ok = ok and proc.returncode == 0
                    print(f"{phase} vmmap rc={proc.returncode}", flush=True)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    ok = False
                    log.write(type(exc).__name__ + "\n")
            return ok

        if not diagnose("before"):
            raise RuntimeError("baseline allocation diagnostics incomplete")
        if a.pilot:
            before = memory_ledger(server.url)
            load = []
            if a.resident_load_gib:
                swap = before.get("host", {}).get("swap_used_bytes")
                if swap is None or swap > 64 * 2**20:
                    a.output.write_text(
                        json.dumps(
                            dict(
                                complete=True,
                                passed=False,
                                problems=[
                                    "pre-existing host swap exceeds 64 MiB or is unknown; no extra load allocated"
                                ],
                                before=before,
                                engaged=mode,
                            )
                        )
                        + "\n"
                    )
                    return 1
                available = before.get("free_bytes")
                if available is None or available < (a.resident_load_gib + 16) * 2**30:
                    raise RuntimeError(
                        "insufficient host headroom for controlled resident load"
                    )
                # Incompressible 16 MiB chunks represent a bounded desktop memory load.
                # This is a controlled surrogate, not a claim about actual application RAM.
                load = [os.urandom(16 * 2**20) for _ in range(a.resident_load_gib * 64)]
                print(f"holding controlled other load {len(load) / 64} GiB", flush=True)
                time.sleep(60)
            loaded = memory_ledger(server.url)
            cli = Client(server.url, server.model_id, 180)
            result = cli.chat(
                [{"role": "user", "content": "Reply with the word READY."}], 32
            )
            time.sleep(40)
            after = memory_ledger(server.url)
            problems = idle_verdict(before, after)
            if (
                a.resident_load_gib
                and after.get("host", {}).get("swap_used_bytes", 2**30) > 64 * 2**20
            ):
                problems.append("resident load caused swap above 64 MiB")
            if result.get("error") or not result.get("done") or not result["content"]:
                problems.append("pilot response failed")
            row = dict(
                complete=True,
                passed=not problems,
                problems=problems,
                before=before,
                engaged=mode,
                loaded=loaded,
                controlled_other_load_gib=a.resident_load_gib,
                after=after,
            )
            a.output.write_text(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            return 1 if problems else 0
        load = []
        sampler = None
        if a.resident_load_gib:
            # Bounded incompressible neighbour load held for the whole soak.
            load = [os.urandom(16 * 2**20) for _ in range(a.resident_load_gib * 64)]
            print(f"holding controlled other load {len(load) / 64} GiB", flush=True)
        if a.pressure_log:
            sampler = PressureSampler(a.pressure_log)
            sampler.start()
        sys.argv = [
            "soak_realistic.py",
            "--url",
            server.url,
            "--model",
            server.model_id,
            "--pid",
            str(server.proc.pid),
            "--minutes",
            str(a.minutes),
            "--long-doc-lines",
            "1600",
            "--seed",
            str(a.seed),
            "--output",
            str(a.output),
            "--note",
            f"deploy27b 32K+ agentic segment draft={mode}",
        ]
        rc = soak_main()
        if sampler:
            sampler.stop()
        del load
        diagnostics_ok = diagnose("after")
        return rc if diagnostics_ok else 1
    finally:
        server.kill()
        # Only this segment's cache; preserve JSONL/log evidence, never delete model links.
        cache = servers.SERVER_HOME / ".yunshu" / "cache"
        if cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache)


if __name__ == "__main__":
    sys.exit(main())
