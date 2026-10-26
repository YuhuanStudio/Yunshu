#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-sampledspec
export PYTHONPATH="$PWD/python"
export TMPDIR="$PWD/docs/research/sampledspec/tmp"
export HOME="$PWD/docs/research/sampledspec/home"
mkdir -p "$TMPDIR" "$HOME"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest \
 tests/unit/test_sampled_spec_tree.py tests/unit/test_dflash_fast.py \
 tests/unit/test_dflash_copy.py tests/unit/test_dflash_tree.py \
 tests/unit/test_dflash_tree_fallback.py tests/unit/test_dflash_top_k_ids.py \
 tests/unit/test_mtp_tree_logprobs.py tests/unit/test_tool_guide_invariant.py \
 tests/unit/test_constrained_spec.py tests/unit/test_apc_identity_probe.py tests/unit/test_restore_touch_probe.py tests/unit/test_sampledspec_probe.py -q -p no:cacheprovider
printf '{"complete":true}\n' > docs/research/sampledspec/unit.complete.json
