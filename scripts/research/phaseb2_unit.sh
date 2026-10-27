#!/bin/bash
# Full unit suite for the phaseb2 worktree (run under gpuq).
cd /Volumes/P5Plus/yunshu-build/wt-phaseb2 || exit 1
PYTHONPATH=python exec nice -n 15 .venv/bin/python -m pytest -q tests/unit -p no:cacheprovider --maxfail=30 -rf
