#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-turn2b
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/turn2b-profile-1022a
mkdir -p "$OUT"
env TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 \
 TFB_YUNSHU_SRC=/Volumes/P5Plus/yunshu-build/wt-codex-turn2b/python TFB_OUT=$OUT/tfb TFB_PORT_LAST=18996 \
 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/turn2prefill_profile.py" \
 --engine yunshu --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
 --out "$OUT/profile.jsonl" --tag=-turn2b-profile-1022a \
 --env YUNSHU_VLM_APC_DISK=0 \
 --env YUNSHU_VLM_APC_DISK_DIR=/Volumes/Micron/yunshu-apc/turn2b-profile-1022a \
 --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 \
 --part decode --rep 0 --only-ctx 32768 --only-kind code --decode-tokens 512 --long-ask --phases cold,warm,turn2
printf '{"complete":true}\n' > "$OUT/complete.json"
