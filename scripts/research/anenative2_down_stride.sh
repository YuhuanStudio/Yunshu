#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/anenative2_down_stride.py
