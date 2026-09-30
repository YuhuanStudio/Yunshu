#!/bin/bash
# Queue the APC audit GPU batch against a frozen source snapshot.   queue_batch.sh SNAPSHOT_PYTHON_DIR
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SNAP=$1
for job in smoke multi-new multi-ssd tiers-ssd tiers-ram tiers-cold text-default text-ssd text-warm; do
  bash "$HERE/queue_jobs.sh" "$job" "$SNAP"
done
AGENTS=codex SRC=$SNAP bash "$HERE/submit_capture.sh" new1
