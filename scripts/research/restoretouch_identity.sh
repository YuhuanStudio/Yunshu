#!/bin/bash
set -euo pipefail
export ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-restoretouch TAG=restoretouch2-
/Volumes/P5Plus/yunshu-build/lead/warmttft/id_any.sh "${1:?32768 or rehearsal}"
printf '{"complete":true}\n'
