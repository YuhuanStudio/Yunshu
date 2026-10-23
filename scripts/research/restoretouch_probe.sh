#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=$ROOT/python TFB_OUT=$ROOT/docs/research/restoretouch/probe TFB_PORT_LAST=18996
export PYTHONPATH=$ROOT/python
/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/restore_touch_probe.py" --tree "$ROOT" --arm restoretouch2-probe --reps 1
rm -rf /Volumes/Micron/yunshu-apc/warmprobe-restoretouch2-probe
