#!/bin/bash
# One concurrency cell: serve the 27B (draft arm), run conc_probe scenarios, kill the server.
# usage: conc_cell.sh TAG ARM(off|default) CTX_CHARS TOKENS SCENARIOS [LOGLEVEL]
set -euo pipefail
TAG=$1; ARM=$2; CTX=$3; TOK=$4; ONLY=$5; LVL=${6:-INFO}
WT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=/Volumes/P5Plus/yunshu-build/concfix/out; mkdir -p "$OUT"
H=/Volumes/P5Plus/yunshu-build/concfix/home-$TAG; mkdir -p "$H"
PORT=${PORT:-18995}
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/concfix
EXTRA=()
[ "$ARM" = off ] && EXTRA=(YUNSHU_VLM_DRAFT=off)
env ${ENVX:-} HOME=$H YUNSHU_HOME=$H/.yunshu YUNSHU_VLM_APC_DISK=0 YUNSHU_CONSOLE=0 YUNSHU_OBSERVE=1 YUNSHU_LOG_LEVEL=$LVL \
  ${EXTRA[@]+"${EXTRA[@]}"} PYTHONPATH=$WT/python \
  "$PY" -m yunshu_cli serve -m ${MODEL_DIR:-/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp} --port $PORT > "$OUT/server-$TAG.log" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null; rm -rf "${H:?}/.yunshu/cache"' EXIT
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null
MODEL=$(curl -s http://127.0.0.1:$PORT/v1/models | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
"$PY" "$WT/scripts/research/conc_probe.py" --url http://127.0.0.1:$PORT --model "$MODEL" \
  --out "$OUT/$TAG.jsonl" --tokens $TOK --ctx-chars $CTX --join ${JOIN:-15} --only "$ONLY"
echo "complete: true"
