#!/bin/bash
# One gpuq job owns both sustained ANE work and GPU server. No foreign process is killed.
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-anenative
LABEL=${1:?unique-label}
OUT=/Volumes/P5Plus/yunshu-build/codex/anenative-concurrent-$LABEL
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
ANE=/Volumes/P5Plus/venvs/anesplit-mlx/bin/python
cd "$ROOT"
mkdir -p "$OUT/tmp"
export PYTHONPATH=$ROOT/python TMPDIR=$OUT/tmp
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=$ROOT/python TFB_PORT_LAST=18996
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pid=
cleanup() { if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; fi; }
trap cleanup EXIT
run() {
 local arm=$1 backend=$2
 if [ "$backend" != off ]; then
   "$ANE" scripts/research/anenative_load.py --cache "$OUT/cache" --backend "$backend" \
     --ready "$OUT/ready.json" --out "$OUT/load-$arm.json" > "$OUT/load-$arm.log" 2>&1 &
   pid=$!
   for ((i=0;i<240;i++)); do
     [ -f "$OUT/ready.json" ] && break
     kill -0 "$pid" 2>/dev/null || { cat "$OUT/load-$arm.log"; return 1; }
     sleep 1
   done
   [ -f "$OUT/ready.json" ] || return 1
 fi
 export TFB_OUT=$OUT/tfb-$arm
 "$PY" scripts/research/tfbench.py --engine yunshu \
   --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --out "$OUT/$arm.jsonl" --tag=-anenative-$arm \
   --env YUNSHU_VLM_APC_DISK=0 --env YUNSHU_VLM_APC_DISK_DIR=/Volumes/Micron/yunshu-apc/anenative-$LABEL-$arm \
   --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 \
   --part decode --rep 0 --only-ctx 32768 --only-kind prose --only-kind code --decode-tokens 512 --long-ask --phases cold
 rm -rf "/Volumes/Micron/yunshu-apc/anenative-$LABEL-$arm"
 if [ -n "$pid" ]; then
   kill -TERM "$pid"
   wait "$pid"
   pid=
   "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["complete"]' "$OUT/load-$arm.json"
 fi
}
run base off
run cand native
run cand-r1 native
run base-r1 off
run coreml coreml
run coreml-r1 coreml
run base-r2 off
run cand-r2 native
run coreml-r2 coreml
run base-r3 off
printf '{"complete":true}\n' > "$OUT/complete.json"
