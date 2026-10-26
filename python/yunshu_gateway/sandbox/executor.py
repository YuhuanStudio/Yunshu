"""Run tool calls for one container: bash under srt, the text editor in-process (path-confined)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import shutil
import stat as stat_mod
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from yunshu_engine import settings

from . import files as cfiles
from .containers import MEMORY_LIMITS, Container, ContainerManager, get_manager
from .repl import ReplError, ReplOutcome, get_pool
from .srt import SrtInfo, build_settings, find_srt, host_allowed, parse_allowlist
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
    calls: list[dict] = field(
        default_factory=list
    )  # python: tool calls the cell is waiting on
    changed: list = field(
        default_factory=list
    )  # python: files created or modified (files.Entry)
    state_lost: bool = (
        False  # python: the process was replaced, earlier variables are gone
    )


def _clip(b: bytes, truncated: bool) -> str:
    s = b.decode("utf-8", errors="replace")
    return s + ("\n[output truncated]" if truncated else "")


def model_text(r: ExecResult) -> str:
    if not r.ok:
        return f"Error: {r.error_code}" + (f" ({r.text})" if r.text else "")
    parts = [f"exit code: {r.return_code}"]
    if r.state_lost:
        parts.append(
            "note: the Python process was restarted; variables from earlier calls are gone"
        )
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

    def memory_bytes(self, c: Container) -> int:
        mb = int(settings.get("YUNSHU_CODE_EXECUTION_MEMORY_MB"))
        if c.memory_limit in MEMORY_LIMITS:
            mb = min(mb, MEMORY_LIMITS[c.memory_limit])
        return mb * 1024 * 1024

    def network_allow(self, c: Container) -> list[str]:
        """What the sandbox may reach: a container can only narrow the server's list, never widen it."""
        server = parse_allowlist(settings.get("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW"))
        if c.network_allow is None:
            return server
        return [h for h in c.network_allow if host_allowed(h, server)]

    def srt_argv(
        self, c: Container, inner: list[str], info: SrtInfo | None = None
    ) -> list[str]:
        info = info or self.srt()
        settings_path = c.path.parent / ".meta" / f"{c.id}.srt.json"
        settings_path.write_text(
            json.dumps(build_settings(c.path, network_allow=self.network_allow(c)))
        )
        return info.argv(settings_path, inner)

    def env(self, c: Container, out_dir: Path) -> dict[str, str]:
        return self._env(c, out_dir)

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
        from ..files_store import FileStoreError, get_store

        store = get_store()
        meta = store.get_meta(file_id)
        name = os.path.basename(meta.get("filename") or file_id) or file_id
        rel = cfiles.clean_relpath(name) or file_id
        try:
            cfiles.write_file(c, rel, store.read(file_id))
        except cfiles.FileOpError as e:
            raise FileStoreError(400, f"cannot stage {name}: {e.message}") from e
        self.manager.add_user_file(c.id, rel)
        return rel

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
        mem = self.memory_bytes(c)
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        async with self._lock(c.id):
            out_dir = c.out / uuid.uuid4().hex[:12]
            out_dir.mkdir(mode=0o700)
            argv = self.srt_argv(
                c,
                [
                    "/usr/bin/env",
                    f"TMPDIR={c.tmp}/",
                    f"TMPPREFIX={c.tmp / 'zsh'}",
                    "/bin/bash",
                    "-c",
                    command,
                ],
                info,
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

    # ── python REPL (persistent process per container) ───────────────────────
    def _repl_tools(self, tools: list[dict] | None) -> list[dict]:
        out = []
        for t in tools or []:
            props = (t.get("schema") or {}).get("properties") or {}
            req = (t.get("schema") or {}).get("required") or []
            order = [k for k in req if k in props] + [k for k in props if k not in req]
            out.append({"name": t["name"], "params": order})
        return out

    async def python(
        self,
        c: Container,
        code: str,
        *,
        tools: list[dict] | None = None,
        timeout: float | None = None,
        fresh: bool = False,
        meta: dict | None = None,
    ) -> ExecResult:
        """Run a cell in the container's Python process. ``tools`` ([{name, schema}]) are awaitable
        in the cell; a cell that calls one returns with ``calls`` set and waits for
        :meth:`python_resume`. ``fresh`` starts a clean process first (stateless call)."""
        if not isinstance(code, str) or not code.strip():
            return ExecResult(False, "invalid_tool_input", text="code is required")
        self.srt()
        limit = float(settings.get("YUNSHU_CODE_EXECUTION_TIMEOUT"))
        if timeout:
            limit = max(1.0, min(limit, float(timeout)))
        sess = get_pool().get(self, c)
        if fresh:
            await sess.close()
            sess.proc = None
            sess._lost = False
        before = await asyncio.to_thread(cfiles.snapshot, c)
        m = dict(meta or {})
        m["before"] = before
        m["limit"] = limit
        try:
            out = await sess.run(
                code,
                self._repl_tools(tools),
                timeout=limit,
                mem_bytes=self.memory_bytes(c),
                meta=m,
            )
        except ReplError as e:
            return ExecResult(False, "unavailable", text=e.message)
        return await self._python_result(c, sess, out)

    async def python_resume(
        self, c: Container, results: dict[str, tuple[object, bool]]
    ) -> ExecResult:
        sess = get_pool().peek(c.id)
        if sess is None:
            return ExecResult(
                False, "unavailable", text="the code execution is no longer running"
            )
        try:
            out = await sess.resume(results)
        except ReplError as e:
            return ExecResult(False, "unavailable", text=e.message)
        return await self._python_result(c, sess, out)

    def python_pending(self, c: Container) -> dict[str, dict] | None:
        sess = get_pool().peek(c.id)
        return dict(sess.pending) if sess is not None and sess.paused else None

    async def _python_result(self, c: Container, sess, out: ReplOutcome) -> ExecResult:
        res = ExecResult(
            stdout=out.stdout,
            stderr=out.stderr,
            return_code=out.return_code,
            elapsed=out.elapsed,
            state_lost=out.state_lost,
        )
        limit = sess.meta.get("limit")
        if out.status == "tool_calls":
            res.calls = out.calls
            res.text = "waiting for tool results"
            return res
        if out.status == "timeout":
            res.ok, res.error_code = False, "execution_time_exceeded"
            res.text = f"exceeded {limit:g} s; the Python process was restarted"
        elif out.status == "oom":
            res.ok, res.error_code = False, "unavailable"
            res.text = (
                f"killed: memory above {self.memory_bytes(c) // 2**20} MB; "
                "the Python process was restarted"
            )
        elif out.status == "overflow":
            res.ok, res.error_code = False, "output_file_too_large"
            res.text = "output exceeded 1 MiB; the Python process was restarted"
        elif out.status == "too_many_calls":
            res.ok, res.error_code = False, "too_many_requests"
            res.text = (
                "the code made too many tool calls (or one call was too large); "
                "the Python process was restarted"
            )
        elif out.status == "crashed":
            res.ok, res.error_code = False, "unavailable"
            res.text = "the Python process exited" + (
                f": {out.stderr.strip()[-300:]}" if out.stderr.strip() else ""
            )
        else:
            before = sess.meta.get("before") or {}
            res.changed = await asyncio.to_thread(cfiles.changed_since, c, before)
        logger.info(
            "code_exec_python container=%s status=%s rc=%s ms=%d out_bytes=%d changed=%d",
            c.id,
            out.status,
            out.return_code,
            out.elapsed * 1000,
            len(out.stdout) + len(out.stderr),
            len(res.changed),
        )
        res.text = res.text or model_text(res)
        return res

    def store_outputs(self, c: Container, entries: list) -> list[dict]:
        """Copy files a cell created into the Files API (Anthropic returns them as file ids)."""
        from ..files_store import FileStoreError, get_store

        found: list[dict] = []
        for e in entries[:MAX_FILES_PER_CALL]:
            try:
                data = cfiles.read_file(c, e.rel, max_bytes=64 * 1024 * 1024)
                meta = get_store().put(
                    data,
                    os.path.basename(e.rel),
                    purpose="code_execution_output",
                    downloadable=True,
                )
            except (cfiles.FileOpError, FileStoreError) as err:
                logger.warning("code_exec output not stored: %s", err)
                continue
            found.append({"file_id": meta["id"], "filename": os.path.basename(e.rel)})
        return found

    # ── text editor (runs in the gateway, confined to the container) ─────────
    def _resolve(self, c: Container, path: str | None) -> tuple[Path, list[str]] | None:
        """(container root, components below it), normalised lexically. The editor opens these
        with ``files.open_confined`` (descriptor walk, O_NOFOLLOW), never through a resolved
        path, so a link the sandbox swaps in between check and use cannot lead out."""
        if not isinstance(path, str) or not path or "\0" in path:
            return None
        roots = [c.path, c.path.resolve()]
        rel = None
        if path.startswith("/"):
            for r in roots:
                with contextlib.suppress(ValueError):
                    rel = os.path.normpath(Path(path).relative_to(r))
                    break
            if rel is None:
                rel = os.path.normpath("work/" + path.lstrip("/"))
        else:
            rel = os.path.normpath("work/" + path)
        if rel == ".":
            return roots[1], []
        parts = rel.split("/")
        if parts[0] == ".." or any(len(x.encode()) > cfiles.MAX_NAME for x in parts):
            return None
        return roots[1], parts

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

    @staticmethod
    def _read_fd(fd: int) -> bytes:
        chunks = []
        while b := os.read(fd, 1 << 20):
            chunks.append(b)
        return b"".join(chunks)

    @staticmethod
    def _replace_fd(fd: int, data: bytes) -> ExecResult | None:
        """Overwrite the regular file behind ``fd`` (None on success)."""
        if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
            return ExecResult(False, "invalid_tool_input", text="not a regular file")
        os.ftruncate(fd, 0)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        return None

    def _edit(self, p: tuple[Path, list[str]], cmd: str, a: dict) -> ExecResult:
        root, parts = p
        name = parts[-1] if parts else ""
        if cmd == "create":
            text = a.get("file_text")
            if not isinstance(text, str):
                return ExecResult(
                    False, "invalid_tool_input", text="file_text required"
                )
            if not parts:
                return ExecResult(
                    False, "invalid_tool_input", text="path is a directory"
                )
            existed = True
            try:
                try:
                    fd = cfiles.open_confined(root, parts, os.O_WRONLY)
                except FileNotFoundError:
                    existed = False
                    fd = cfiles.open_confined(
                        root, parts, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mkdirs=True
                    )
            except IsADirectoryError:
                return ExecResult(
                    False, "invalid_tool_input", text="path is a directory"
                )
            except OSError:
                return ExecResult(
                    False, "invalid_tool_input", text="cannot write to that path"
                )
            try:
                bad = self._replace_fd(fd, text.encode())
            finally:
                os.close(fd)
            if bad:
                return bad
            return ExecResult(
                data={
                    "type": "text_editor_code_execution_create_result",
                    "is_file_update": existed,
                },
                text=f"{'Updated' if existed else 'Created'} {a['path']}",
            )
        try:
            fd = cfiles.open_confined(root, parts, os.O_RDONLY)
        except FileNotFoundError:
            return ExecResult(False, "file_not_found", text=str(a.get("path")))
        except OSError:
            return ExecResult(False, "invalid_tool_input", text="cannot read that path")
        try:
            mode = os.fstat(fd).st_mode
            if stat_mod.S_ISDIR(mode):
                entries = sorted(os.listdir(fd))
                raw = b""
            elif stat_mod.S_ISREG(mode):
                entries = None
                raw = self._read_fd(fd)
            else:  # FIFO, device, socket: never read (the open was O_NONBLOCK, a read would block)
                return ExecResult(
                    False, "invalid_tool_input", text="not a regular file"
                )
        finally:
            os.close(fd)
        if cmd == "view":
            if entries is not None:
                names = entries
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
            ext = os.path.splitext(name)[1].lower()
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
        if entries is not None:
            return ExecResult(False, "invalid_tool_input", text="path is a directory")
        text = raw.decode("utf-8", errors="replace")
        n = text.count(old)
        if n != 1:
            return ExecResult(
                False,
                "invalid_tool_input",
                text="old_str matched " + ("nothing" if n == 0 else f"{n} places"),
            )
        start_line = text[: text.index(old)].count("\n") + 1
        try:
            wfd = cfiles.open_confined(root, parts, os.O_WRONLY)
        except OSError:
            return ExecResult(
                False, "invalid_tool_input", text="cannot write to that path"
            )
        try:
            bad = self._replace_fd(wfd, text.replace(old, new, 1).encode())
        finally:
            os.close(wfd)
        if bad:
            return bad
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
