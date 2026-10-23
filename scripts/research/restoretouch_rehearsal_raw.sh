#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=$ROOT/python TFB_PORT_LAST=18996
export TFB_OUT=$ROOT/docs/research/restoretouch/rehearsal-raw/tfb
export PYTHONPATH=$TFB_YUNSHU_SRC
for kind in code prose; do
/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/splitinv_rehearsal.py" --ctx 32768 --kind "$kind" --reply-tokens 256 512 2048 --out "$ROOT/docs/research/restoretouch/rehearsal-raw/$kind.jsonl"
done
printf '{"complete":true,"logprobs_checked":false}\n'
