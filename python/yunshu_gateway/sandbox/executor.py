"""Run tool calls for one container: bash under srt, the text editor in-process (path-confined)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from yunshu_engine import settings

from .containers import Container, ContainerManager, get_manager
from .srt import SrtInfo, build_settings, find_srt, parse_allowlist
from .supervisor import run_supervised

logger = logging.getLogger("yunshu.sandbox")

MAX_FILES_PER_CALL = 20
VIEW_LIMIT = 200_000  # characters handed back by one text-editor view
MAX_CONCURRENT = 2
INSTALL_HINT = (
    "Code execution needs Anthropic's sandbox-runtime: `npm install -g "
    "@anthropic-ai/sandbox-runtime` (Node >= 22.12), then restart Yunshu."
)


class SandboxUnavailableError(RuntimeError):
    """srt is missing or cannot run; the request must fail (never run unsandboxed)."""


@dataclass
class ExecResult:
    ok: bool = True
    error_code: str | None = None  # Anthropic error_code vocabulary
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    files: list[dict] = field(default_factory=list)  # [{"file_id", "filename"}]
    data: dict = field(default_factory=dict)  # text-editor result fields
    text: str = ""  # what the model reads
    elapsed: float = 0.0
    notes: list[str] = field(default_factory=list)


def _clip(b: bytes, truncated: bool) -> str:
    s = b.decode("utf-8", errors="replace")
    return s + ("\n[output truncated]" if truncated else "")


def model_text(r: ExecResult) -> str:
    if not r.ok:
        return f"Error: {r.error_code}" + (f" ({r.text})" if r.text else "")
    parts = [f"exit code: {r.return_code}"]
    if r.stdout:
        parts.append("stdout:\n" + r.stdout)
    if r.stderr:
        parts.append("stderr:\n" + r.stderr)
    if r.files:
        parts.append("files saved: " + ", ".join(f["filename"] for f in r.files))
    return "\n".join(parts)


def _srt_tmp() -> str:
    d = Path("/tmp") / f"yunshu-srt-{os.getuid()}"
    d.mkdir(mode=0o700, exist_ok=True)
    return str(d)


class CodeExecutor:
    def __init__(self, manager: ContainerManager | None = None):
        self._manager = manager
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem: asyncio.Semaphore | None = None

    @property
    def manager(self) -> ContainerManager:
        return self._manager or get_manager()

    def srt(self) -> SrtInfo:
        info = find_srt()
        if info is None:
            raise SandboxUnavailableError(
                "The sandbox runtime (srt) is not installed. " + INSTALL_HINT
            )
        if info.problem:
            raise SandboxUnavailableError(
                f"srt cannot run: {info.problem}. " + INSTALL_HINT
            )
        return info

    def _lock(self, cid: str) -> asyncio.Lock:
        return self._locks.setdefault(cid, asyncio.Lock())

    def _env(self, c: Container, out_dir: Path) -> dict[str, str]:
        py_bin = os.path.join(sys.base_prefix, "bin")
        return {
            "PATH": f"{py_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(c.home),
            # srt itself puts unix sockets under its TMPDIR (a path length limit applies), so the
            # process env carries a short one and `env` hands the command its container tmp
            "TMPDIR": _srt_tmp(),
            "OUTPUT_DIR": str(out_dir),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "TERM": "dumb",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
        }

    def stage_file(self, c: Container, file_id: str) -> str:
        """Copy a Files API file into the container's work directory; returns the file name."""
        from ..files_store import get_store

        store = get_store()
        meta = store.get_meta(file_id)
        name = os.path.basename(meta.get("filename") or file_id) or file_id
        (c.work / name).write_bytes(store.read(file_id))
        return name

    async def bash(
        self, c: Container, command: str, timeout: float | None = None
    ) -> ExecResult:
        t0 = time.monotonic()
        if not isinstance(command, str) or not command.strip():
            return ExecResult(False, "invalid_tool_input", text="command is required")
        info = self.srt()
        limit = float(settings.get("YUNSHU_CODE_EXECUTION_TIMEOUT"))
        if timeout:
            limit = max(1.0, min(limit, float(timeout)))
        mem = int(settings.get("YUNSHU_CODE_EXECUTION_MEMORY_MB")) * 1024 * 1024
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        async with self._lock(c.id):
            out_dir = c.out / uuid.uuid4().hex[:12]
            out_dir.mkdir(mode=0o700)
            settings_path = c.path.parent / ".meta" / f"{c.id}.srt.json"
            settings_path.write_text(
                json.dumps(
                    build_settings(
                        c.path,
                        network_allow=parse_allowlist(
                            settings.get("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW")
                        ),
                    )
                )
            )
            argv = info.argv(
                settings_path,
                [
                    "/usr/bin/env",
                    f"TMPDIR={c.tmp}/",
                    f"TMPPREFIX={c.tmp / 'zsh'}",
                    "/bin/bash",
                    "-c",
                    command,
                ],
            )
            try:
                async with self._sem:
                    r = await run_supervised(
                        argv,
                        env=self._env(c, out_dir),
                        cwd=str(c.work),
                        timeout=limit,
                        mem_bytes=mem,
                    )
                files = self._capture(out_dir)
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)
        res = ExecResult(
            stdout=_clip(r.stdout, r.stdout_truncated),
            stderr=_clip(r.stderr, r.stderr_truncated),
            return_code=r.return_code,
            files=files,
            elapsed=time.monotonic() - t0,
        )
        violations = res.stderr.count("Operation not permitted") + res.stderr.count(
            "sandbox_violations"
        )
        if r.timed_out:
            res.ok, res.error_code = False, "execution_time_exceeded"
            res.text = f"exceeded {limit:g} s"
        elif r.oom:
            res.ok, res.error_code = False, "unavailable"
            res.text = f"killed: memory above {mem // 2**20} MB"
        elif r.output_overflow:
            res.ok, res.error_code = False, "output_file_too_large"
            res.text = "output exceeded 1 MiB"
        # metadata only: never the command, its output or file contents
        logger.info(
            "code_exec container=%s rc=%s ms=%d out_bytes=%d files=%d timed_out=%s oom=%s "
            "overflow=%s violations=%d",
            c.id,
            r.return_code,
            res.elapsed * 1000,
            len(r.stdout) + len(r.stderr),
            len(files),
            r.timed_out,
            r.oom,
            r.output_overflow,
            violations,
        )
        res.text = res.text or model_text(res)
        return res

    def _capture(self, out_dir: Path) -> list[dict]:
        """Top-level regular files left in $OUTPUT_DIR become Files API entries."""
        from ..files_store import FileStoreError, get_store

        found: list[dict] = []
        for p in sorted(out_dir.iterdir())[:MAX_FILES_PER_CALL]:
            if p.is_symlink() or not p.is_file():
                continue
            try:
                meta = get_store().put(
                    p.read_bytes(),
                    p.name,
                    purpose="code_execution_output",
                    downloadable=True,
                )
            except FileStoreError as e:
                logger.warning("code_exec output %s not stored: %s", p.name, e)
                continue
            found.append({"file_id": meta["id"], "filename": p.name})
        return found

    # ── text editor (runs in the gateway, confined to the container) ─────────
    def _resolve(self, c: Container, path: str) -> Path | None:
        if not isinstance(path, str) or not path or "\0" in path:
            return None
        base = c.work.resolve()
        p = Path(path)
        if p.is_absolute():
            with contextlib.suppress(ValueError):
                p.resolve().relative_to(c.path.resolve())
                return p.resolve()
            p = Path(path.lstrip("/"))
        full = (base / p).resolve()
        try:
            full.relative_to(c.path.resolve())
        except ValueError:
            return None
        return full

    async def editor(self, c: Container, args: dict) -> ExecResult:
        t0 = time.monotonic()
        cmd = args.get("command") if isinstance(args, dict) else None
        p = self._resolve(c, args.get("path") if isinstance(args, dict) else None)
        if p is None or cmd not in ("view", "create", "str_replace"):
            return ExecResult(False, "invalid_tool_input", text="bad command or path")
        async with self._lock(c.id):
            r = await asyncio.to_thread(self._edit, p, cmd, args)
        r.elapsed = time.monotonic() - t0
        logger.info("code_exec_editor container=%s cmd=%s ok=%s", c.id, cmd, r.ok)
        return r

    def _edit(self, p: Path, cmd: str, a: dict) -> ExecResult:
        if cmd == "create":
            text = a.get("file_text")
            if not isinstance(text, str):
                return ExecResult(
                    False, "invalid_tool_input", text="file_text required"
                )
            existed = p.exists()
            if p.is_dir():
                return ExecResult(
                    False, "invalid_tool_input", text="path is a directory"
                )
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return ExecResult(
                data={
                    "type": "text_editor_code_execution_create_result",
                    "is_file_update": existed,
                },
                text=f"{'Updated' if existed else 'Created'} {a['path']}",
            )
        if not p.exists():
            return ExecResult(False, "file_not_found", text=str(a.get("path")))
        if cmd == "view":
            if p.is_dir():
                names = sorted(os.listdir(p))
                body = "\n".join(names)
                return ExecResult(
                    data={
                        "type": "text_editor_code_execution_view_result",
                        "file_type": "text",
                        "content": body,
                        "num_lines": len(names),
                        "start_line": 1,
                        "total_lines": len(names),
                    },
                    text=body,
                )
            raw = p.read_bytes()
            ext = p.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"):
                kind = "pdf" if ext == ".pdf" else "image"
                return ExecResult(
                    data={
                        "type": "text_editor_code_execution_view_result",
                        "file_type": kind,
                        "content": base64.b64encode(raw).decode(),
                    },
                    text=f"[{kind} file, {len(raw)} bytes]",
                )
            lines = raw.decode("utf-8", errors="replace").splitlines()
            start, end = 1, len(lines)
            vr = a.get("view_range")
            if (
                isinstance(vr, list)
                and len(vr) == 2
                and all(isinstance(x, int) for x in vr)
            ):
                start = max(1, vr[0])
                end = len(lines) if vr[1] == -1 else min(len(lines), vr[1])
            body = "\n".join(lines[start - 1 : end])[:VIEW_LIMIT]
            return ExecResult(
                data={
                    "type": "text_editor_code_execution_view_result",
                    "file_type": "text",
                    "content": body,
                    "num_lines": max(0, end - start + 1),
                    "start_line": start,
                    "total_lines": len(lines),
                },
                text=body,
            )
        # str_replace
        old, new = a.get("old_str"), a.get("new_str", "")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return ExecResult(False, "invalid_tool_input", text="old_str required")
        text = p.read_text(errors="replace")
        n = text.count(old)
        if n != 1:
            return ExecResult(
                False,
                "invalid_tool_input",
                text="old_str matched " + ("nothing" if n == 0 else f"{n} places"),
            )
        start_line = text[: text.index(old)].count("\n") + 1
        p.write_text(text.replace(old, new, 1))
        old_l, new_l = old.split("\n"), new.split("\n")
        return ExecResult(
            data={
                "type": "text_editor_code_execution_str_replace_result",
                "old_start": start_line,
                "old_lines": len(old_l),
                "new_start": start_line,
                "new_lines": len(new_l),
                "lines": [f"-{x}" for x in old_l] + [f"+{x}" for x in new_l],
            },
            text=f"Edited {a['path']}",
        )


_executor: CodeExecutor | None = None


def get_executor() -> CodeExecutor:
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor
