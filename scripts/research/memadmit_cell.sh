#!/bin/bash
# One memory-admission cell: serve a checkpoint, run conc_probe scenarios, sample host memory
# (swap, pageouts, swapouts, pressure) every 2 s, kill the server.
# usage: memadmit_cell.sh TAG CTX_CHARS TOKENS SCENARIOS   (env: MODEL_DIR, ENVX, PORT)
set -euo pipefail
TAG=$1; CTX=$2; TOK=$3; ONLY=$4
WT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=/Volumes/P5Plus/yunshu-build/memadmit/out; mkdir -p "$OUT"
H=/Volumes/P5Plus/yunshu-build/memadmit/home-$TAG; mkdir -p "$H"
PORT=${PORT:-18996}
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/memadmit
mkdir -p "$TMPDIR"
vmv() { vm_stat | awk -v k="$1" '$0 ~ k {gsub(/\./,"",$NF); print $NF}'; }
env ${ENVX:-} HOME=$H YUNSHU_HOME=$H/.yunshu YUNSHU_VLM_APC_DISK=0 YUNSHU_CONSOLE=0 YUNSHU_OBSERVE=1 YUNSHU_LOG_LEVEL=INFO \
  YUNSHU_AUTH_DISABLED=true PYTHONPATH=$WT/python \
  "$PY" -m yunshu_cli serve -m ${MODEL_DIR:-/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp} --port $PORT > "$OUT/server-$TAG.log" 2>&1 &
SRV=$!
( while kill -0 $SRV 2>/dev/null; do echo "$(date +%s) rss_kb=$(ps -o rss= -p $SRV | tr -d ' ') swap_used=$(sysctl -n vm.swapusage | tr -s ' ' | cut -d' ' -f7) pageouts=$(vmv Pageouts) swapouts=$(vmv Swapouts) pressure_free=$(memory_pressure 2>/dev/null | grep -o 'free percentage: [0-9]*%' | tr -d '\n')"; sleep 2; done > "$OUT/mem-$TAG.log" 2>&1 ) &
SAMP=$!
trap 'kill -9 $SRV $SAMP 2>/dev/null; rm -rf "${H:?}/.yunshu/cache"' EXIT
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SRV 2>/dev/null || { echo "server exited"; tail -20 "$OUT/server-$TAG.log"; exit 1; }; sleep 2; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null
MODEL=$(curl -s http://127.0.0.1:$PORT/v1/models | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
# CTX may be a space-separated list: the probe runs once per context size (peak memory is
# cumulative, so ascending sizes give the per-size peak).
for C in $CTX; do
  "$PY" "$WT/scripts/research/conc_probe.py" --url http://127.0.0.1:$PORT --model "$MODEL" \
    --out "$OUT/$TAG.jsonl" --tokens $TOK --ctx-chars $C --only "$ONLY"
done
if [ -n "${REFUSE_CTX:-}" ]; then
  "$PY" "$WT/scripts/research/memadmit_refuse.py" --url http://127.0.0.1:$PORT --model "$MODEL" \
    --ctx-chars $REFUSE_CTX --out "$OUT/refuse-$TAG.json"
fi
echo '{"complete": true}' >> "$OUT/$TAG.jsonl"
curl -s http://127.0.0.1:$PORT/metrics > "$OUT/metrics-$TAG.txt" || true
curl -s http://127.0.0.1:$PORT/v1/yunshu/status > "$OUT/status-$TAG.json" || true
grep -E "Memory (admission|plan|pressure)|APC RAM budget" "$OUT/server-$TAG.log" | tail -30 || true
echo "complete: true"
