#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-agentcensus
mkdir -p "$PWD/docs/research/agentcensus/tmp"
export TMPDIR="$PWD/docs/research/agentcensus/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/python"
export TFB_YUNSHU_SRC="$PWD/python" TFB_PORT_LAST=18996
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_agentcensus_1022.py -q -p no:cacheprovider
mkdir -p /Volumes/P5Plus/yunshu-build/codex/agentcensus-base-1022a
set +e
PYTHONPATH="$PWD/docs/research/agentcensus/base-python" nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_anthropic_thinking_display.py -q -p no:cacheprovider > /Volumes/P5Plus/yunshu-build/codex/agentcensus-base-1022a/display-before.log 2>&1
before_rc=$?
set -e
if [ "$before_rc" -eq 0 ]; then echo 'Expected baseline regression tests to fail'; exit 2; fi
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_anthropic_thinking_display.py tests/unit/test_coding_agent_api_fixes.py tests/unit/test_agent_compat_stream_check.py -q -p no:cacheprovider > /Volumes/P5Plus/yunshu-build/codex/agentcensus-base-1022a/display-after.log 2>&1
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/agent_compat/census_1022.py --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --source "$PWD/docs/research/agentcensus/base-python" --source-ref 8036c1674dfcd1c448ba6a375bacde0790458602 --out /Volumes/P5Plus/yunshu-build/codex/agentcensus-base-1022a
