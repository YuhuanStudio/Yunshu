#!/bin/bash
# Two independent shapes, one queue lock; run only after paused-worker smoke passes.
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-anenative2
TAG=${1:?unique-tag}
PIN=/Volumes/P5Plus/yunshu-build/codex/anenative-phase1-fb121a2f
scripts/research/anenative_warm_interference.sh --tag "$TAG-part" --server-src "$PIN" --out "/Volumes/P5Plus/yunshu-build/codex/anenative-warm-$TAG-part" --channels 2176 --rows 1024
scripts/research/anenative_warm_interference.sh --tag "$TAG-full128" --server-src "$PIN" --out "/Volumes/P5Plus/yunshu-build/codex/anenative-warm-$TAG-full128" --channels 17408 --rows 128
printf '{"complete":true}\n'
