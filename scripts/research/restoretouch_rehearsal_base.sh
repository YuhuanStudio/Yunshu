#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=/Volumes/P5Plus/yunshu-build/wt-tailfast-base/python TFB_PORT_LAST=18996
export TFB_OUT=$ROOT/docs/research/restoretouch/rehearsal-base/tfb
export PYTHONPATH=$TFB_YUNSHU_SRC
/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/splitinv_rehearsal.py" --ctx 32768 --kind code --reply-tokens 256 --out "$ROOT/docs/research/restoretouch/rehearsal-base/code.jsonl"
