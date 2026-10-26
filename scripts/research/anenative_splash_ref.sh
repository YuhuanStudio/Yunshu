#!/bin/bash
# Splash 1.4.0 cold TTFT with its ANE split on (splash-latest) or off (splash-noane), via gaptable.
set -euo pipefail
ENGINE=${1:?engine}; CTX=${2:?ctx}; REP=${3:?rep}
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python /Volumes/P5Plus/yunshu-build/wt-final-0152c4bb/scripts/research/gaptable.py run \
  --manifest /Volumes/P5Plus/yunshu-build/lead/gaptable-0152c4bb/manifest.json \
  --out /Volumes/P5Plus/yunshu-build/codex/anenative-splash-ref --src /Volumes/P5Plus/yunshu-build/wt-final-0152c4bb \
  --engine "$ENGINE" --rep "$REP" --part decode --ctx "$CTX" --kind prose
