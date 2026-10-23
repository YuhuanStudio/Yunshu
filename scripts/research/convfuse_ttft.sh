#!/bin/bash
# Cold TTFT, one variant/rep per gpuq job: convfuse_ttft.sh <name> <tree> <rep>
set -euo pipefail
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
TFB=/Volumes/P5Plus/yunshu-build/wt-kernelstudy/scripts/research/tfbench.py
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/convfuse
name=$1; src=$2; rep=$3
mkdir -p $OUT
env TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 \
  TFB_YUNSHU_SRC=$src/python TFB_OUT=$OUT/tfb.$name.$rep TFB_PORT_LAST=18996 \
  $PY $TFB --engine yunshu --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
  --out $OUT/ttft.$name.$rep.jsonl --tag=-convfuse-$name-$rep \
  --env YUNSHU_VLM_APC_DISK=0 \
  --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 \
  --part decode --rep $rep --only-ctx 8192 --only-ctx 32768 --only-ctx 131072 --only-kind prose \
  --decode-tokens 32 --long-ask --phases cold
