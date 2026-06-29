"""Yunshu Distributed Launcher — starts multi-process ring mesh.

Generates MLX_HOSTFILE and launches N processes with MLX_RANK env vars.
Supports:
  - Local multi-process (multiple ports on localhost)
  - Multi-node (different IPs via --hostfile or manual config)

Usage:
    # Local 2-process test
    PYTHONPATH=. uv run python scripts/launch_mesh.py -n 2 python worker.py

    # Local 4-process test
    PYTHONPATH=. uv run python scripts/launch_mesh.py -n 4 python worker.py

    # Multi-node (2 IPs, 1 process each)
    PYTHONPATH=. uv run python scripts/launch_mesh.py --hosts 192.168.1.1,192.168.1.2 python worker.py

    # Custom base port
    PYTHONPATH=. uv run python scripts/launch_mesh.py -n 2 --port 30000 python worker.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading


def build_hostfile(num_procs: int, hosts: list[str] | None, base_port: int) -> list[list[str]]:
    """Build JSON hostfile data for mx.distributed ring backend.

    Each entry is a list of "ip:port" strings for one node.
    For local mode, each "node" is localhost with a unique port.
    """
    if hosts:
        if len(hosts) != num_procs:
            print(f"Error: {num_procs} procs but {len(hosts)} hosts", file=sys.stderr)
            sys.exit(1)
        return [[f"{h}:{base_port + i}"] for i, h in enumerate(hosts)]
    else:
        return [[f"127.0.0.1:{base_port + i}"] for i in range(num_procs)]


def main():
    parser = argparse.ArgumentParser(description="Yunshu distributed launcher")
    parser.add_argument("-n", "--num-procs", type=int, default=2, help="Number of processes")
    parser.add_argument("--hosts", type=str, help="Comma-separated host IPs (one per process)")
    parser.add_argument("--port", type=int, default=29700, help="Base port number")
    parser.add_argument("--verbose", action="store_true", help="Enable MLX_RING_VERBOSE")
    parser.add_argument("--timeout", type=int, default=60, help="Per-process timeout in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args()

    # fix: the docstring shows `launch_mesh.py -n 2 -- python worker.py`
    # form, but the original check `args.command[0] == "--"` rejected that
    # exact usage. Strip the `--` separator first, then check emptiness.
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("Usage: launch_mesh.py [-n N] [--hosts IP1,IP2] [--port PORT] [--] <command>", file=sys.stderr)
        sys.exit(1)

    hostfile_data = build_hostfile(args.num_procs, args.hosts and args.hosts.split(","), args.port)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(hostfile_data, f)
        hostfile = f.name

    print(f"Yunshu Mesh Launcher: {args.num_procs} processes")
    print(f"Hostfile: {json.dumps(hostfile_data)}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)

    procs = []
    results = {}

    def run_proc(rank, p):
        try:
            stdout, stderr = p.communicate(timeout=args.timeout)
            results[rank] = (p.returncode, stdout.decode(), stderr.decode())
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            results[rank] = (-1, (stdout or b"").decode(), (stderr or b"").decode() + " TIMEOUT")

    for rank in range(args.num_procs):
        env = os.environ.copy()
        env['MLX_HOSTFILE'] = hostfile
        env['MLX_RANK'] = str(rank)
        if args.verbose:
            env['MLX_RING_VERBOSE'] = '1'

        p = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)
        print(f"  Rank {rank}: pid={p.pid}")

    threads = []
    for rank, p in enumerate(procs):
        t = threading.Thread(target=run_proc, args=(rank, p))
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=args.timeout + 10)

    max_rc = 0
    for rank in range(args.num_procs):
        if rank in results:
            rc, stdout, stderr = results[rank]
            print(f"\n--- Rank {rank} (exit={rc}) ---")
            if stdout.strip():
                print(stdout)
            if "TIMEOUT" in (stderr or ""):
                print("  TIMEOUT")
                max_rc = max(max_rc, 1)
            elif stderr.strip():
                print(f"  STDERR: {stderr[-500:]}")
            max_rc = max(max_rc, rc)
        else:
            print(f"\n--- Rank {rank}: STILL RUNNING ---")
            max_rc = 1

    os.unlink(hostfile)

    print(f"\n{'=' * 60}")
    if max_rc == 0:
        print("All processes exited successfully")
    else:
        print(f"Failed with max exit code {max_rc}")
    sys.exit(max_rc)


if __name__ == "__main__":
    main()
