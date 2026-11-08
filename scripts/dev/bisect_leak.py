"""Find the smallest set of test files that, run before a target file in one process, make it fail (delta debugging).

usage: bisect_leak.py TARGET FILE...   (run from the repo root; prints the minimal set)
"""

import subprocess
import sys

MARK = "not serial and not fullcmp_e2e"


def fails(files: list[str], target: str) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
         "-m", MARK, *files, target],
        capture_output=True, text=True,
    )
    return f"FAILED {target}" in r.stdout


def ddmin(files: list[str], target: str) -> list[str]:
    n = 2
    while len(files) >= 2:
        size = max(1, len(files) // n)
        chunks = [files[i : i + size] for i in range(0, len(files), size)]
        for i, c in enumerate(chunks):
            comp = [f for j, ch in enumerate(chunks) if j != i for f in ch]
            if fails(c, target):
                files, n = c, 2
                print("narrow ->", len(files), flush=True)
                break
            if fails(comp, target):
                files, n = comp, max(n - 1, 2)
                print("narrow ->", len(files), flush=True)
                break
        else:
            if n >= len(files):
                break
            n = min(len(files), n * 2)
    return files


if __name__ == "__main__":
    target, files = sys.argv[1], sys.argv[2:]
    assert fails(files, target), "does not reproduce"
    print("MINIMAL:", ddmin(files, target))
