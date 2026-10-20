#!/bin/sh
# gpuq entry: tool-guided spec on == off and decode speed on this tree (probe_tools_identity.py).
set -e
cd "$(dirname "$0")/../.."
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/probe_tools_identity.py \
  --src "$PWD/python" --out "$1"
