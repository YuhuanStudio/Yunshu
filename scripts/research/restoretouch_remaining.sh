#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
"$ROOT/scripts/research/restoretouch_ab.sh" "$ROOT" restoretouch2-ab-cand
"$ROOT/scripts/research/restoretouch_ab.sh" "$ROOT" restoretouch2-ab-cand-r1
"$ROOT/scripts/research/restoretouch_ab.sh" /Volumes/P5Plus/yunshu-build/wt-tailfast-base restoretouch2-ab-base-r1
printf '{"complete":true,"all_arms":true}\n'
