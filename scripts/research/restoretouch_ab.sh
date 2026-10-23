#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
TREE=${1:?tree}
ARM=${2:?arm}
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=$TREE/python TFB_OUT=$ROOT/docs/research/restoretouch/ab TFB_PORT_LAST=18996
export PYTHONPATH=$TREE/python
trap 'rm -rf "/Volumes/Micron/yunshu-apc/warmprobe-$ARM"' EXIT
/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/restore_touch_probe.py" --plain --memory --tree "$TREE" --arm "$ARM" --reps 1 --env YUNSHU_AUTH_TOKEN=k
