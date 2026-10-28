"""Files inside a container's working directory, as the gateway (not the sandbox) touches them.

Sandboxed code controls everything under ``work/``, symlinks included, so the gateway never opens a
path it built from that tree with a plain ``open``: every component is opened relative to a directory
descriptor with ``O_NOFOLLOW`` (a symlink anywhere on the way is an error, not a detour to the
host's files), the final object must be a regular file, and names are validated before they are used.
OpenAI's ``/mnt/data`` is the working directory: ``/mnt/data/a/b.csv`` is ``work/a/b.csv``.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat as stat_mod
from dataclasses import dataclass
from pathlib import Path

from .containers import Container

MNT = "/mnt/data"
MAX_DEPTH = 8
MAX_NAME = 200
MAX_LIST = 2000
MAX_FILE_BYTES = 256 * 1024 * 1024


class FileOpError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # "not_found" | "invalid" | "too_large" | "exists"
        self.message = message


def clean_relpath(name: str | None) -> str | None:
    """A path relative to the working directory, or None when it is not acceptable."""
    if not isinstance(name, str) or not name or "\0" in name or "\\" in name:
        return None
    if name == MNT or name.startswith(MNT + "/"):
        name = name[len(MNT) :]
    name = name.lstrip("/")
    parts = name.split("/")
    if not parts or len(parts) > MAX_DEPTH:
        return None
    for p in parts:
        if p in ("", ".", "..") or len(p.encode()) > MAX_NAME:
            return None
    return "/".join(parts)


def file_id(container_id: str, rel: str) -> str:
    """Stable id of a file: the same path in the same container always has the same id."""
    h = hashlib.sha256(f"{container_id}\0{rel}".encode()).hexdigest()
    return "cfile_" + h[:24]


def _open_work(c: Container) -> int:
    try:
        return os.open(str(c.work), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise FileOpError("not_found", "The container has no working directory.") from e


def _walk(c: Container, parts: list[str], create: bool) -> int:
    """Descriptor of the directory holding ``parts[-1]``; every step refuses symlinks."""
    fd = _open_work(c)
    try:
        for comp in parts[:-1]:
            try:
                nxt = os.open(
                    comp,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except FileNotFoundError:
                if not create:
                    raise FileOpError("not_found", "No such file.") from None
                os.mkdir(comp, 0o700, dir_fd=fd)
                nxt = os.open(
                    comp,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except OSError as e:
                raise FileOpError("invalid", "Path crosses a link or a file.") from e
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


class LinkEscapeError(PermissionError):
    """A link met on the way leads out of the container root (or loops)."""


MAX_LINK_HOPS = 16


def _link_target_parts(
    target: str, root: Path, extra_roots: tuple[str, ...]
) -> tuple[bool, list[str]]:
    """``(absolute, components)`` of a link target; an absolute target counts only when it names
    a place inside the container (its real path, or OpenAI's ``/mnt/data`` = ``work/``)."""
    if not target.startswith("/"):
        return False, [p for p in target.split("/") if p not in ("", ".")]
    for prefix, base in ((MNT, ["work"]),) + tuple((r, []) for r in extra_roots):
        if target == prefix or target.startswith(prefix.rstrip("/") + "/"):
            rest = target[len(prefix) :].split("/")
            return True, base + [p for p in rest if p not in ("", ".")]
    raise LinkEscapeError(errno.EACCES, "link leaves the container")


def open_confined(
    root: Path,
    parts: list[str],
    flags: int,
    *,
    mode: int = 0o600,
    mkdirs: bool = False,
    extra_roots: tuple[str, ...] = (),
) -> int:
    """Open ``root/parts`` with every component relative to a directory descriptor and
    ``O_NOFOLLOW``: a link swapped in at any point is an error (ELOOP / ENOTDIR), never a detour
    out of ``root``. ``O_NONBLOCK`` keeps a FIFO from hanging the open; the caller still checks
    ``fstat`` for the kind of object it got. Raises OSError (FileNotFoundError when missing).

    A symlink the sandboxed code made is followed by hand, one hop at a time, as long as its
    target stays inside ``root`` (relative targets, ``..`` never above ``root``, absolute targets
    under ``root``, ``extra_roots`` or ``/mnt/data``); anything that leaves the container, or more
    than ``MAX_LINK_HOPS`` links, raises :class:`LinkEscapeError` (an ``OSError``)."""
    stack = [os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)]
    pending = list(parts) or ["."]
    hops = 0
    try:
        while pending:
            comp = pending.pop(0)
            last = not pending
            if comp == "." or comp == "":
                if last:
                    return os.open(
                        ".",
                        flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                        mode,
                        dir_fd=stack[-1],
                    )
                continue
            if comp == "..":
                if len(stack) == 1:
                    raise LinkEscapeError(errno.EACCES, "link leaves the container")
                os.close(stack.pop())
                if last:
                    return os.open(
                        ".",
                        flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                        mode,
                        dir_fd=stack[-1],
                    )
                continue
            cur = stack[-1]
            try:
                is_link = stat_mod.S_ISLNK(os.lstat(comp, dir_fd=cur).st_mode)
            except FileNotFoundError:
                is_link = False
                if not last and mkdirs:
                    os.mkdir(comp, 0o700, dir_fd=cur)
            if is_link:
                hops += 1
                if hops > MAX_LINK_HOPS:
                    raise LinkEscapeError(errno.ELOOP, "too many links")
                absolute, tparts = _link_target_parts(
                    os.readlink(comp, dir_fd=cur), root, extra_roots
                )
                if absolute:
                    while len(stack) > 1:
                        os.close(stack.pop())
                pending = tparts + pending
                if not pending:
                    pending = ["."]
                continue
            if last:
                return os.open(
                    comp, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=cur
                )
            stack.append(
                os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
            )
        raise FileNotFoundError(errno.ENOENT, "empty path")
    finally:
        for fd in stack:
            os.close(fd)


def write_file(
    c: Container, rel: str, data: bytes, *, max_bytes: int = MAX_FILE_BYTES
) -> int:
    if len(data) > max_bytes:
        raise FileOpError("too_large", f"File exceeds {max_bytes} bytes.")
    parts = rel.split("/")
    dfd = _walk(c, parts, create=True)
    try:
        try:
            fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
                dir_fd=dfd,
            )
        except OSError as e:
            raise FileOpError("invalid", "Cannot write to that path.") from e
        try:
            if not stat_mod.S_ISREG(os.fstat(fd).st_mode):
                raise FileOpError("invalid", "Not a regular file.")
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
        finally:
            os.close(fd)
    finally:
        os.close(dfd)
    return len(data)


def read_file(c: Container, rel: str, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    parts = rel.split("/")
    dfd = _walk(c, parts, create=False)
    try:
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except FileNotFoundError:
            raise FileOpError("not_found", "No such file.") from None
        except OSError as e:
            raise FileOpError("invalid", "Cannot read that path.") from e
        try:
            st = os.fstat(fd)
            if not stat_mod.S_ISREG(st.st_mode):
                raise FileOpError("not_found", "No such file.")
            if st.st_size > max_bytes:
                raise FileOpError("too_large", f"File exceeds {max_bytes} bytes.")
            chunks = []
            left = max_bytes + 1
            while left > 0:
                b = os.read(fd, min(1 << 20, left))
                if not b:
                    break
                chunks.append(b)
                left -= len(b)
            data = b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)
    if len(data) > max_bytes:
        raise FileOpError("too_large", f"File exceeds {max_bytes} bytes.")
    return data


def delete_file(c: Container, rel: str) -> None:
    parts = rel.split("/")
    dfd = _walk(c, parts, create=False)
    try:
        try:
            st = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
            if stat_mod.S_ISDIR(st.st_mode):
                raise FileOpError("not_found", "No such file.")
            os.unlink(parts[-1], dir_fd=dfd)
        except FileNotFoundError:
            raise FileOpError("not_found", "No such file.") from None
    finally:
        os.close(dfd)


@dataclass(frozen=True)
class Entry:
    rel: str
    size: int
    mtime_ns: int


def scan(c: Container, limit: int = MAX_LIST) -> list[Entry]:
    """Regular files under work/ (symlinks and special files are not files of the container).
    Walks descriptors with O_NOFOLLOW: a link in place of work/ or a directory lists nothing."""
    out: list[Entry] = []
    try:
        root = _open_work(c)
    except FileOpError:
        return out
    stack: list[tuple[str, int, int]] = [("", root, 0)]
    try:
        while stack and len(out) < limit:
            base, fd, depth = stack.pop()
            try:
                with os.scandir(fd) as it:
                    entries = list(it)
                for e in entries:
                    rel = f"{base}/{e.name}" if base else e.name
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat_mod.S_ISREG(st.st_mode):
                        out.append(Entry(rel, st.st_size, st.st_mtime_ns))
                        if len(out) >= limit:
                            break
                    elif stat_mod.S_ISDIR(st.st_mode) and depth + 1 < MAX_DEPTH:
                        try:
                            sub = os.open(
                                e.name,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd,
                            )
                        except OSError:
                            continue
                        stack.append((rel, sub, depth + 1))
            except OSError:
                pass
            finally:
                os.close(fd)
    finally:
        for _b, fd, _d in stack:
            os.close(fd)
    return sorted(out, key=lambda x: (x.mtime_ns, x.rel))


def snapshot(c: Container) -> dict[str, tuple[int, int]]:
    return {e.rel: (e.size, e.mtime_ns) for e in scan(c)}


def changed_since(c: Container, before: dict[str, tuple[int, int]]) -> list[Entry]:
    """Files created or modified since ``before`` was taken."""
    return [e for e in scan(c) if before.get(e.rel) != (e.size, e.mtime_ns)]


def record(c: Container, e: Entry) -> dict:
    """The OpenAI ``container.file`` object."""
    return {
        "id": file_id(c.id, e.rel),
        "object": "container.file",
        "bytes": e.size,
        "container_id": c.id,
        "created_at": int(e.mtime_ns // 1_000_000_000),
        "path": f"{MNT}/{e.rel}",
        "source": "user" if e.rel in c.user_files else "assistant",
    }


def find(c: Container, fid: str) -> Entry | None:
    for e in scan(c):
        if file_id(c.id, e.rel) == fid:
            return e
    return None


def mime_for(rel: str) -> str:
    import mimetypes

    return mimetypes.guess_type(rel)[0] or "application/octet-stream"
