"""The persistent Python process of a container (runs INSIDE the sandbox, standard library only).

The gateway copies this file into the container and starts it under srt. It speaks JSON lines on the
descriptors it was started with (a private duplicate; fd 1 and 2 are re-pointed at capture files so
user code, subprocesses included, cannot write into the protocol by accident):

  gateway -> worker   {"op": "exec", "id", "code", "tools": [{"name", "params": [..]}]}
                      {"op": "tool_result", "call_id", "content", "is_error"}
  worker  -> gateway  {"op": "ready"}
                      {"op": "tool_call", "call_id", "name", "input"}
                      {"op": "done", "id", "return_code", "stdout", "stderr"}

Globals survive between ``exec`` messages (a REPL). Top-level ``await`` works; the last expression's
value is echoed like a notebook cell. Tools the request declared are async functions: awaiting one
hands the call to the gateway and resumes with the client's result.
"""

import ast
import asyncio
import builtins
import inspect
import io
import json
import os
import sys
import threading
import traceback

CAP = 64 * 1024  # bytes handed back per stream
OVERFLOW = 1024 * 1024  # bytes written by one cell before the process kills itself
MNT = "/mnt/data"


def main() -> None:
    out_path, err_path = sys.argv[1], sys.argv[2]
    proto_in = os.fdopen(os.dup(0), "r", encoding="utf-8", errors="replace")
    proto_fd = os.dup(1)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    ofd = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o600)
    efd = os.open(err_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o600)
    os.dup2(ofd, 1)
    os.dup2(efd, 2)
    sys.stdout = io.TextIOWrapper(
        os.fdopen(1, "wb", closefd=False),
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        os.fdopen(2, "wb", closefd=False),
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
    wlock = threading.Lock()

    def send(obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode()
        with wlock:
            view = memoryview(data)
            while view:
                view = view[os.write(proto_fd, view) :]

    _mnt_shim()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    inbox: asyncio.Queue[dict] = asyncio.Queue()
    waiting: dict[str, asyncio.Future] = {}

    def reader() -> None:
        for line in proto_in:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            loop.call_soon_threadsafe(dispatch, msg)
        os._exit(0)  # the gateway went away

    def dispatch(msg: dict) -> None:
        if msg.get("op") == "tool_result":
            fut = waiting.pop(str(msg.get("call_id")), None)
            if fut is not None and not fut.done():
                fut.set_result(msg)
        else:
            inbox.put_nowait(msg)

    def watch_output() -> None:
        import time

        while True:
            time.sleep(0.05)
            try:
                if os.fstat(1).st_size + os.fstat(2).st_size > OVERFLOW:
                    send({"op": "overflow"})
                    os._exit(75)
            except OSError:
                return

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=watch_output, daemon=True).start()
    g: dict = {"__name__": "__main__", "__builtins__": builtins}
    counter = [0]

    def make_tool(name: str, params: list[str]):
        async def tool(*args, **kwargs):
            if len(args) > len(params):
                raise TypeError(
                    f"{name}() takes at most {len(params)} positional arguments"
                )
            payload = dict(zip(params, args, strict=False))
            for k, v in kwargs.items():
                if k in payload:
                    raise TypeError(f"{name}() got multiple values for argument '{k}'")
                payload[k] = v
            counter[0] += 1
            call_id = f"call_{os.getpid()}_{counter[0]}"
            fut = loop.create_future()
            waiting[call_id] = fut
            send(
                {"op": "tool_call", "call_id": call_id, "name": name, "input": payload}
            )
            res = await fut
            content = res.get("content")
            if res.get("is_error"):
                raise RuntimeError(
                    content if isinstance(content, str) else json.dumps(content)
                )
            return content

        tool.__name__ = name
        return tool

    async def run_cell(code: str) -> int:
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            body = compile(tree, "<cell>", "exec", flags=flags)
            tail = compile(last, "<cell>", "eval", flags=flags) if last else None
        except SyntaxError:
            traceback.print_exc(limit=0)
            return 1
        try:
            r = eval(body, g)  # noqa: S307 - the point of this process
            if inspect.iscoroutine(r):
                await r
            if tail is not None:
                v = eval(tail, g)  # noqa: S307
                if inspect.iscoroutine(v):
                    v = await v
                if v is not None:
                    g["_"] = v
                    print(repr(v))
            return 0
        except SystemExit as e:
            c = e.code
            return c if isinstance(c, int) else (0 if c is None else 1)
        except BaseException:  # noqa: BLE001
            et, ev, tb = sys.exc_info()
            frames = traceback.extract_tb(tb)
            cell = [f for f in frames if f.filename == "<cell>"]
            sys.stderr.write("Traceback (most recent call last):\n")
            sys.stderr.write("".join(traceback.format_list(cell or frames)))
            sys.stderr.write("".join(traceback.format_exception_only(et, ev)))
            return 1

    def take(fd: int) -> str:
        size = os.fstat(fd).st_size
        data = os.pread(fd, min(size, CAP), 0)
        os.ftruncate(fd, 0)
        text = data.decode("utf-8", errors="replace")
        return text + ("\n[output truncated]" if size > CAP else "")

    async def serve() -> None:
        send({"op": "ready"})
        while True:
            msg = await inbox.get()
            if msg.get("op") != "exec":
                continue
            for t in msg.get("tools") or []:
                g[t["name"]] = make_tool(t["name"], list(t.get("params") or []))
            rc = await run_cell(str(msg.get("code") or ""))
            sys.stdout.flush()
            sys.stderr.flush()
            if os.fstat(1).st_size + os.fstat(2).st_size > OVERFLOW:
                send({"op": "overflow"})
                os._exit(75)
            send(
                {
                    "op": "done",
                    "id": msg.get("id"),
                    "return_code": rc,
                    "stdout": take(1),
                    "stderr": take(2),
                }
            )

    loop.run_until_complete(serve())


def _mnt_shim() -> None:
    """OpenAI documents files under /mnt/data; here that is the working directory."""
    cwd = os.getcwd()

    def fix(p):
        try:
            s = os.fspath(p)
        except TypeError:
            return p
        if isinstance(s, bytes):
            m = MNT.encode()
            if s == m or s.startswith(m + b"/"):
                return os.fsencode(cwd) + s[len(m) :]
            return p
        if s == MNT or s.startswith(MNT + "/"):
            return cwd + s[len(MNT) :]
        return p

    def wrap(fn, nargs=1):
        def inner(*a, **k):
            return fn(*[fix(x) for x in a[:nargs]], *a[nargs:], **k)

        inner.__name__ = getattr(fn, "__name__", "wrapped")
        return inner

    orig_open = builtins.open
    builtins.open = io.open = wrap(orig_open)  # type: ignore[assignment]
    for name in (
        "listdir",
        "scandir",
        "stat",
        "lstat",
        "remove",
        "unlink",
        "mkdir",
        "makedirs",
        "rmdir",
        "chdir",
        "utime",
    ):
        if hasattr(os, name):
            setattr(os, name, wrap(getattr(os, name)))
    orig_rename = os.rename

    def rename(a, b, *x, **k):
        return orig_rename(fix(a), fix(b), *x, **k)

    os.rename = rename  # type: ignore[assignment]


if __name__ == "__main__":
    main()
