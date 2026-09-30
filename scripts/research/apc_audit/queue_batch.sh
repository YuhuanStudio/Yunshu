#!/bin/bash
# Queue the APC audit GPU batch against a frozen source snapshot.   queue_batch.sh SNAPSHOT_PYTHON_DIR
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SNAP=$1
for job in ${JOBS:-smoke multi-new multi-ssd tiers-ssd tiers-ram tiers-cold text-default text-ssd text-warm}; do
  bash "$HERE/queue_jobs.sh" "$job" "$SNAP"
done
if [ -n "$CAPTURE" ]; then
  SRC=$SNAP bash "$HERE/submit_capture.sh" "$CAPTURE"
fi
