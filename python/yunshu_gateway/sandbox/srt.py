"""Locate srt and build its settings for one container (pure functions, unit-testable without srt)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MIN_NODE = (22, 12)

# Allowed names may not resolve to these (srt already blocks loopback / link-local / own addresses).
PRIVATE_RANGES = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
    "fc00::/7",
]

# Roots denied for reading; the container and the interpreter prefix are carved back in.
DENY_READ_ROOTS = ["/Users", "/Volumes", "/home"]


@dataclass(frozen=True)
class SrtInfo:
    node: str
    cli: str
    version: str = ""
    problem: str = ""

    def argv(self, settings: Path, inner: list[str]) -> list[str]:
        return [
            self.node,
            self.cli,
            "--settings",
            str(settings),
            "-c",
            shlex.join(inner),
        ]


_NODE_VERSIONS: dict[tuple[str, float], tuple[int, ...] | None] = {}


def _node_version(node: str) -> tuple[int, ...] | None:
    try:
        key = (node, os.stat(node).st_mtime)
    except OSError:
        return None
    if key not in _NODE_VERSIONS:
        _NODE_VERSIONS[key] = _probe_node(node)
    return _NODE_VERSIONS[key]


def _probe_node(node: str) -> tuple[int, ...] | None:
    try:
        out = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_srt() -> SrtInfo | None:
    """The srt CLI on PATH (or under ~/.yunshu/srt), with a Node new enough to run it. None when absent;
    an ``SrtInfo`` with ``problem`` set when srt exists but cannot run."""
    srt = shutil.which("srt")
    if srt is None:
        cand = Path.home() / ".yunshu" / "srt" / "node_modules" / ".bin" / "srt"
        srt = str(cand) if cand.exists() else None
    if srt is None:
        return None
    cli = str(Path(srt).resolve())
    node = shutil.which("node")
    if node is None:
        cand = Path(srt).parent / "node"
        node = str(cand) if cand.exists() else None
    if node is None:
        return SrtInfo("", cli, problem="Node.js (>= 22.12) was not found on PATH")
    v = _node_version(node)
    if v is None or v < MIN_NODE:
        return SrtInfo(
            node,
            cli,
            problem=f"Node {'.'.join(map(str, v)) if v else '?'} is too old (srt needs >= 22.12)",
        )
    version = ""
    pkg = Path(cli).parent.parent / "package.json"
    with contextlib.suppress(OSError, ValueError):
        version = json.loads(pkg.read_text()).get("version", "")
    return SrtInfo(node, cli, version=version)


def interpreter_prefixes() -> list[str]:
    """Directories the sandboxed Python needs to read: the base (non-venv) interpreter prefix."""
    return sorted({sys.base_prefix, sys.base_exec_prefix})


def parse_allowlist(value: str | None) -> list[str]:
    return [h.strip() for h in (value or "").split(",") if h.strip()]


def host_allowed(host: str, patterns: list[str]) -> bool:
    """Is ``host`` (a name or ``*.name`` pattern, optional ``:port``) covered by one of ``patterns``?"""
    h = host.strip().lower()
    if not h:
        return False
    for p in patterns:
        p = p.strip().lower()
        if p == h:
            return True
        if p.startswith("*.") and not h.startswith("*."):
            if h.split(":")[0].endswith(p[1:].split(":")[0]) and (
                ":" not in p or h.endswith(":" + p.rsplit(":", 1)[1])
            ):
                return True
        if h.startswith("*.") and p == h:
            return True
    return False


def build_settings(
    container_dir: Path,
    *,
    network_allow: list[str] | None = None,
    read_extra: list[str] | None = None,
) -> dict:
    """srt settings for one container.

    * writes: only the container directory (work/, home/, tmp/ and per-call out/ live in it);
    * reads: everything outside the user areas stays readable (system libraries), ``/Users``,
      ``/Volumes`` and ``/home`` are denied, then only this container and the Python prefix are
      allowed back, so the user's home (keys, ``~/.yunshu``), other volumes and other containers
      cannot be read;
    * network: none unless hosts are listed; loopback is never reachable (``allowLocalBinding``
      false), and allowed names cannot resolve into private ranges.
    """
    allow_read = [str(container_dir), *interpreter_prefixes(), *(read_extra or [])]
    return {
        "network": {
            "allowedDomains": list(network_allow or []),
            "deniedDomains": [],
            "deniedResolvedAddresses": list(PRIVATE_RANGES),
            "allowLocalBinding": False,
        },
        "filesystem": {
            # the shared root too, whatever the user set it to: other containers stay unreadable
            "denyRead": [*DENY_READ_ROOTS, str(container_dir.parent)],
            "allowRead": allow_read,
            "allowWrite": [str(container_dir)],
            "denyWrite": [],
        },
    }
