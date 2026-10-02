"""Extracted engine modules support either import order and existing package paths."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["yunshu_engine", "python.yunshu_engine"])
@pytest.mark.parametrize(
    "module",
    [
        "engine_policy",
        "engine_sampling",
        "engine_cache",
        "engine_text",
        "engine_templates",
        "engine_embeddings",
        "engine_diagnostics",
        "engine_fast",
        "engine_stream",
        "engine_speculative",
        "engine_ngram",
        "engine_mtp",
    ],
)
def test_extracted_module_imports_before_facade(package, module):
    # Each subprocess starts without a loaded facade, so pytest's collection-time
    # imports cannot hide a circular dependency in a particular extracted module.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {package}.{module}\n"
            f"from {package}.batched_engine import BatchedEngine, GenerationOutput\n",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
