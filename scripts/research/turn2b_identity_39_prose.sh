#!/bin/bash
set -euo pipefail
test "$(cat /Volumes/P5Plus/yunshu-gpuq/logs/1022-030118-00-turn2b-tests-smoke-1022a.rc)" = 0 || exit 3
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-turn2b
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/turn2b-identity-1022c
mkdir -p "$OUT"
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb" TFB_PORT_LAST=18996
for kind in prose; do
 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/apc_restore_identity.py" \
 --ctx 32768 --kind "$kind" --turn2 --turn2-repeat 39 --compare-spec-off --tokens 512 --trace-tail --out "$OUT/$kind.jsonl" \
 --env YUNSHU_VLM_APC_DISK=0 \
 --env YUNSHU_VLM_APC_DISK_DIR=/Volumes/Micron/yunshu-apc/turn2b-identity-1022c \
 --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2
 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python - "$OUT/$kind.jsonl" "$OUT/tfb/out" <<'PYV'
import json,sys
from pathlib import Path
r=json.loads(open(sys.argv[1]).readline())
suffix=r['partial']['pt']-r['partial']['cached']
assert 516 <= suffix <= 642, ("wrong short-tail path",suffix)
assert r['generated_tokens']>=500, r['generated_tokens']
for mode in ('A','B','off'):
    lines=(Path(sys.argv[2])/('server-apcid-'+mode+'-32768.log')).read_text().splitlines()
    rows=[json.loads(x.split('TURN2TAIL ')[1])['rows'] for x in lines if 'TURN2TAIL ' in x]
    assert rows and all(512<n<640 for n in rows), (mode,rows)
    print('M64 identity engaged '+mode+' rows='+str(rows),flush=True)
print('M64 identity coverage suffix='+str(suffix),flush=True)
PYV
done
printf '{"complete":true}\n' > "$OUT/complete.json"
