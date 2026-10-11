"""Submit a flashnext80 harness job to gpuq only after its exact argv parses (no 0-second argparse failures).

    flashnext80_submit.py --label L [--priority 0 --timeout 18 --stall 10 --mem-gb 95 --out F ...] \
        -- /path/to/python scripts/research/flashnext80_footprint.py --tree ... --env K=V ...

The harness script (the first ``*.py`` after the interpreter) is imported and its ``build_parser()`` must accept
the remaining arguments, otherwise nothing is submitted and the exit code is 2.  Scripts without a
``build_parser`` are submitted unchecked.  Every argument is passed through as a separate word (no shell).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

GPUQ = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq"


def split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        raise SystemExit(
            "usage: flashnext80_submit.py [gpuq options] -- python script.py args..."
        )
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def find_script(command: list[str]) -> tuple[int, Path] | None:
    for index, word in enumerate(command):
        if word.endswith(".py") and Path(word).is_file():
            return index, Path(word)
    return None


def check_command(command: list[str]) -> str | None:
    """None when the harness accepts the command's arguments, else the parser's error text."""
    found = find_script(command)
    if found is None:
        return None
    index, path = found
    spec = importlib.util.spec_from_file_location(f"_check_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    build = getattr(module, "build_parser", None)
    if build is None:
        return None
    parser = build()
    errors: list[str] = []
    parser.error = lambda message: (
        errors.append(message) or (_ for _ in ()).throw(ValueError(message))
    )  # type: ignore[method-assign]
    try:
        parser.parse_args(command[index + 1 :])
    except ValueError:
        return errors[0] if errors else "invalid arguments"
    except SystemExit as exc:  # --help or a parser that exits itself
        return None if exc.code in (0, None) else f"parser exited {exc.code}"
    return None


def main(argv: list[str] | None = None) -> int:
    options, command = split_command(list(sys.argv[1:] if argv is None else argv))
    problem = check_command(command)
    if problem:
        print(
            f"flashnext80_submit: refusing to submit, the harness rejects its arguments: {problem}",
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ)
    env.setdefault("GPUQ_DIR", "/Volumes/P5Plus/yunshu-gpuq")
    env.setdefault("GPUQ_OWNER", "sonnet-flashnext80")
    return subprocess.call([GPUQ, "submit", *options, "--", *command], env=env)


if __name__ == "__main__":
    sys.exit(main())
