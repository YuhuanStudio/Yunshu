#!/bin/bash
# gpuq job: conc_probe on this tree. conc_job.sh TAG OUTDIR [--only a,b] [extra conc_probe args]; set CONC_DEBUG=1 for slice logs
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$1; OUT=$2; shift 2
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
H=/Volumes/P5Plus/yunshu-build/agentload-home-conc; mkdir -p "$H" "$OUT"
PORT=18994
export HOME=$H YUNSHU_HOME=$H/.yunshu YUNSHU_VLM_APC_DISK=0 YUNSHU_CONSOLE=0 YUNSHU_DEBUG_ROUTES=1 YUNSHU_AUTH_DISABLED=truePYTHONPATH=$T/python \
  YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2
[ "${CONC_DEBUG:-0}" = 1 ] && export YUNSHU_LOG_LEVEL=DEBUG
LAUNCH=("$PY" -m yunshu_cli)
[ -n "${TAIL_SHARE:-}" ] && LAUNCH=("$PY" "$T/scripts/research/serve_with_share.py" "$TAIL_SHARE")
"${LAUNCH[@]}" serve -m /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --port $PORT > "$OUT/server-$TAG.log" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "${H:?}/.yunshu/cache"' EXIT
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null
MODEL=$(curl -s http://127.0.0.1:$PORT/v1/models | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
"$PY" "$T/scripts/research/conc_probe.py" --url http://127.0.0.1:$PORT --model "$MODEL" --out "$OUT/conc-$TAG.jsonl" --tokens 1500 "$@"
echo "complete: true"
