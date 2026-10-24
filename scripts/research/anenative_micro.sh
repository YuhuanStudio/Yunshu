#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-anenative
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$TMPDIR"
exec /Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_micro.py "$@"
