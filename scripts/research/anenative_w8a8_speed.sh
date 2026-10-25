#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-anenative2
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
mkdir -p "$TMPDIR"
exec /Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_w8a8_speed.py "$@"
