"""Freeze the measured source at GPU job start so ongoing edits cannot mix builds."""

import hashlib
import os
import shutil
import time
from pathlib import Path


def freeze(out, harness):
    source = Path(os.environ["TFB_YUNSHU_SRC"])
    target = Path(out).parent / ("source-" + str(time.time_ns()))
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    digest = hashlib.sha256()
    for path in sorted(target.rglob("*.py")):
        digest.update(str(path.relative_to(target)).encode())
        digest.update(path.read_bytes())
    harness.YUNSHU_SRC = str(target)
    return {"source": str(target), "sha256": digest.hexdigest()}
