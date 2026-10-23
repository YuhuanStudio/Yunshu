#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-restoretouch
nice -n 15 /Volumes/P5Plus/yunshu-build/lead/precheck.sh "$PWD" tests/unit/test_buffer_cache.py tests/unit/test_idle_memory.py tests/unit/test_memory_ledger_counters.py tests/unit/test_restore_touch_probe.py tests/unit/test_apc_manager.py
printf '{"complete":true}\n'
