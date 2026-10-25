#!/bin/bash
# Prove harness and sustained load before loading the 27B server.
set -euo pipefail
cd "${ANENATIVE_ROOT:-/Volumes/P5Plus/yunshu-build/wt-anenative2}"
CHANNELS=${1:-2176}
ROWS=${2:-1024}
MODE=${3:-fp16}
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC=$PWD/python TFB_PORT_LAST=18996
export TFB_OUT=/Volumes/P5Plus/yunshu-build/codex/anenative-conc-dry
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
ANE=/Volumes/P5Plus/venvs/anesplit-mlx/bin/python
mkdir -p "$TMPDIR"
nice -n 15 "$PY" -m pytest tests/unit/test_ane_native.py tests/unit/test_ane_native_bench.py -q -p no:cacheprovider
"$PY" scripts/research/anenative_tfbench.py --engine yunshu --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --part decode --only-ctx 32768 --only-kind prose --only-kind code --decode-tokens 512 --long-ask --phases cold --out /Volumes/P5Plus/yunshu-build/codex/anenative-conc-dry.jsonl --dry-run
for backend in native coreml; do
 "$ANE" scripts/research/anenative_load.py --backend "$backend" --channels "$CHANNELS" --mode "$MODE" --rows "$ROWS" --duration 2 \
   --cache /Volumes/P5Plus/yunshu-build/codex/anenative-conc-pilot-cache \
   --ready /Volumes/P5Plus/yunshu-build/codex/anenative-conc-pilot.ready \
   --out /Volumes/P5Plus/yunshu-build/codex/anenative-conc-pilot-$backend.json
 "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["complete"] and r["calls"]>0 and r["rate_samples"]; assert r["whole_mlp"] == (r["mlp_only_model64_tps_upper_bound"] is not None); print({"backend":r["backend"],"calls":r["calls"],"complete":True})' /Volumes/P5Plus/yunshu-build/codex/anenative-conc-pilot-$backend.json
 done
printf '{"complete":true}\n'
