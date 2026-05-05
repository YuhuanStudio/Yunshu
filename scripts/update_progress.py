#!/usr/bin/env python3
"""Progress update script for Yunshu.

Counts tests, modules, lines of code, scans for TODO/FIXME comments,
and updates PROGRESS_GUIDE.md phase percentages based on what's
actually implemented.

Usage:
    cd yunshu
    uv run python scripts/update_progress.py
    uv run python scripts/update_progress.py --dry-run
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent

# ─── Scanning ───────────────────────────────────────────────────

def count_python_loc(directory: Path) -> dict[str, int]:
    """Count Python lines of code across a directory tree."""
    stats = {"files": 0, "lines": 0, "classes": 0, "functions": 0}
    if not directory.exists():
        return stats
    for py_file in directory.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        stats["files"] += 1
        try:
            source = py_file.read_text(encoding="utf-8")
            stats["lines"] += len(source.splitlines())
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    stats["classes"] += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stats["functions"] += 1
        except Exception:
            pass
    return stats


def count_tests(test_dir: Path) -> dict[str, int]:
    """Count test files and extract test counts from pytest output."""
    stats = {"files": 0, "passed": 0, "failed": 0, "total": 0}
    if not test_dir.exists():
        return stats
    for py_file in test_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if py_file.stem.startswith("test_") or py_file.stem.endswith("_test"):
            stats["files"] += 1
    # Try to get actual test counts
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--co", "-q"],
            cwd=_ROOT,
            capture_output=True, text=True,
            timeout=60,
        )
        match = re.search(r"(\d+)\s+test", result.stdout)
        if match:
            stats["total"] = int(match.group(1))
    except Exception:
        pass
    return stats


def scan_todos(directory: Path) -> list[dict[str, str]]:
    """Scan for TODO/FIXME comments."""
    findings = []
    pattern = re.compile(r"#\s*(TODO|FIXME|HACK)\b", re.IGNORECASE)
    if not directory.exists():
        return findings
    for py_file in directory.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                findings.append({
                    "file": str(py_file.relative_to(directory)),
                    "line": str(i),
                    "text": line.strip(),
                })
    return findings


# ─── Phase Completeness Detection ──────────────────────────────

def detect_phase_completions(python_dir: Path) -> dict[str, int]:
    """Detect phase completion percentages by checking for implemented features.

    Returns a dict of phase_name -> percentage (0-100).
    """
    phases: dict[str, int] = {}

    # Phase 0: Platform validation
    phase0_items = {
        "roofline": python_dir / "yunshu_engine" / "roofline.py",
        "benchmark": python_dir / "yunshu_engine" / "benchmark.py",
        "hardware_detect": python_dir / "yunshu_engine" / "hardware.py",
        "metal_kernels": python_dir / "yunshu_engine" / "metal_kernels.py",
    }
    phase0_done = sum(1 for p in phase0_items.values() if p.exists())
    phases["Phase 0"] = min(100, int((phase0_done / len(phase0_items)) * 60) + 10)  # base 10 for CI

    # Phase 1: MVP single-node
    phase1_items = {
        "engine": python_dir / "yunshu_engine" / "engine.py",
        "batched_engine": python_dir / "yunshu_engine" / "batched_engine.py",
        "scheduler": python_dir / "yunshu_engine" / "engine_core.py",
        "gateway_main": python_dir / "yunshu_gateway" / "main.py",
        "chat_router": python_dir / "yunshu_gateway" / "routers" / "chat.py",
        "completions_router": python_dir / "yunshu_gateway" / "routers" / "completions.py",
        "models_router": python_dir / "yunshu_gateway" / "routers" / "models.py",
        "embeddings_router": python_dir / "yunshu_gateway" / "routers" / "embeddings.py",
        "json_schema": python_dir / "yunshu_engine" / "json_schema.py",
        "tool_calling": python_dir / "yunshu_engine" / "tool_call_streamer.py",
        "model_manager": python_dir / "yunshu_engine" / "model_manager.py",
        "thinking_budget": python_dir / "yunshu_engine" / "thinking_budget.py",
        "prefill_progress": python_dir / "yunshu_engine" / "prefill_progress.py",
        "model_discovery": python_dir / "yunshu_engine" / "model_discovery.py",
        "server_metrics": python_dir / "yunshu_engine" / "server_metrics.py",
        "memory_enforcer": python_dir / "yunshu_engine" / "process_memory_enforcer.py",
    }
    phase1_done = sum(1 for p in phase1_items.values() if p.exists())
    phases["Phase 1"] = int((phase1_done / len(phase1_items)) * 100)

    # Phase 2: Distributed (mostly skipped per project direction)
    phase2_items = {
        "mesh": python_dir / "yunshu_mesh" / "__init__.py",
        "control": python_dir / "yunshu_control" / "__init__.py",
        "api_mesh_router": python_dir / "yunshu_api" / "routers" / "mesh.py",
    }
    phase2_done = sum(1 for p in phase2_items.values() if p.exists())
    phases["Phase 2"] = int((phase2_done / len(phase2_items)) * 15)  # Cap at 15% since multi-node not tested

    # Phase 3: Multi-modal + Anthropic + MCP
    phase3_items = {
        "vlm_engine": python_dir / "yunshu_engine" / "vlm_engine.py",
        "audio_engine": python_dir / "yunshu_engine" / "audio_engine.py",
        "image_engine": python_dir / "yunshu_engine" / "image_engine.py",
        "anthropic_router": python_dir / "yunshu_gateway" / "routers" / "anthropic.py",
        "audio_router": python_dir / "yunshu_gateway" / "routers" / "audio.py",
        "images_router": python_dir / "yunshu_gateway" / "routers" / "images.py",
        "mcp_router": python_dir / "yunshu_gateway" / "routers" / "mcp.py",
        "realtime_router": python_dir / "yunshu_gateway" / "routers" / "realtime.py",
    }
    phase3_done = sum(1 for p in phase3_items.values() if p.exists())
    phases["Phase 3"] = int((phase3_done / len(phase3_items)) * 100)

    # Phase 4: Spec decode + Realtime
    phase4_items = {
        "speculative_decoder": python_dir / "yunshu_engine" / "speculative_decoder.py",
        "realtime": python_dir / "yunshu_gateway" / "routers" / "realtime.py",
    }
    phase4_done = sum(1 for p in phase4_items.values() if p.exists())
    phases["Phase 4"] = int((phase4_done / len(phase4_items)) * 100)

    # Phase 5: v1.0 release
    phase5_items = {
        "sdk": python_dir / "yunshu_sdk" / "__init__.py",
        "cli": python_dir / "yunshu_cli" / "__init__.py",
        "release_prep": _ROOT / "scripts" / "release_prep.py",
        "dev_setup": _ROOT / "scripts" / "dev_setup.sh",
        "bench_paper": _ROOT / "scripts" / "benchmark_paper.py",
        "progress_update": _ROOT / "scripts" / "update_progress.py",
        "docs_index": _ROOT / "docs" / "index.html",
        "docs_api": _ROOT / "docs" / "api_reference.html",
        "docs_bench": _ROOT / "docs" / "benchmark.html",
        "bench_results": _ROOT / "docs" / "benchmark_results.md",
    }
    phase5_done = sum(1 for p in phase5_items.values() if p.exists())
    phases["Phase 5"] = int((phase5_done / len(phase5_items)) * 100)

    return phases


def make_progress_bar(pct: int) -> str:
    """Create a text progress bar."""
    filled = int(pct / 100 * 16)
    empty = 16 - filled
    bar = chr(9608) * filled + chr(9617) * empty  # █ and ░
    return f"{bar} {pct:3d}%"


# ─── Update PROGRESS_GUIDE.md ──────────────────────────────────

def update_progress_guide(
    python_stats: dict,
    test_stats: dict,
    todos: list,
    phases: dict[str, int],
    dry_run: bool,
) -> None:
    """Update the phase percentages in PROGRESS_GUIDE.md."""
    guide_path = _ROOT / "PROGRESS_GUIDE.md"
    if not guide_path.exists():
        print(f"  PROGRESS_GUIDE.md not found at {guide_path}")
        return

    content = guide_path.read_text(encoding="utf-8")

    # Find the phase completion section and update percentages
    # Pattern: "Phase N (description):  ████████████░░░░  60%   (comment)"
    phase_patterns = {
        "Phase 0": r"(Phase 0\s*\([^)]*\):\s*)[█░]+\s*\d+%",
        "Phase 1": r"(Phase 1\s*\([^)]*\):\s*)[█░]+\s*\d+%",
        "Phase 2": r"(Phase 2\s*\([^)]*\):\s*)[█░]+\s*\d+%",
        "Phase 3": r"(Phase 3\s*\([^)]*\):\s*)[█░]+\s*\d+%",
        "Phase 4": r"(Phase 4\s*\([^)]*\):\s*)[█░]+\s*\d+%",
        "Phase 5": r"(Phase 5\s*\([^)]*\):\s*)[█░]+\s*\d+%",
    }

    for phase_name, pattern in phase_patterns.items():
        pct = phases.get(phase_name, 0)
        bar = make_progress_bar(pct)
        replacement = rf"\g<1>{bar}"
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            print(f"  {phase_name}: updated to {pct}%")

    # Update the test count line
    test_total = test_stats.get("total", 0)
    content = re.sub(
        r"測試：\s*\d+\s*tests",
        f"測試：     {test_total} tests" if test_total > 0 else None,
        content,
    )

    # Update code count line
    loc = python_stats.get("lines", 0)
    files = python_stats.get("files", 0)
    old_code_line = re.search(r"代碼量：.*", content)
    if old_code_line:
        content = content.replace(
            old_code_line.group(),
            f"代碼量：   {files}+ Python files, {loc}+ lines",
        )
        print(f"  Code stats: {files} files, {loc} lines")

    if dry_run:
        print("\n  [DRY RUN] No changes written.")
    else:
        guide_path.write_text(content, encoding="utf-8")
        print(f"\n  Written to: {guide_path}")


# ─── Main ───────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Update Yunshu progress")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    args = parser.parse_args()

    print("Yunshu Progress Update\n")

    python_dir = _ROOT / "python"
    test_dir = _ROOT / "tests"

    # 1. Count Python LOC
    print("  [1/4] Counting Python code...")
    python_stats = count_python_loc(python_dir)
    print(f"         {python_stats['files']} files, {python_stats['lines']} lines, "
          f"{python_stats['classes']} classes, {python_stats['functions']} functions")

    # 2. Count tests
    print("  [2/4] Counting tests...")
    test_stats = count_tests(test_dir)
    print(f"         {test_stats['files']} test files, {test_stats['total']} tests detected")

    # 3. Scan TODOs
    print("  [3/4] Scanning for TODO/FIXME...")
    todos = scan_todos(python_dir)
    print(f"         {len(todos)} TODO/FIXME/HACK comments found")
    if todos:
        # Group by type
        by_type: dict[str, int] = defaultdict(int)
        for t in todos:
            m = re.match(r"#\s*(TODO|FIXME|HACK)", t["text"], re.IGNORECASE)
            if m:
                by_type[m.group(1).upper()] += 1
        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
        print(f"         ({type_str})")
        # Show first 5
        for t in todos[:5]:
            print(f"           {t['file']}:{t['line']} — {t['text'][:80]}")
        if len(todos) > 5:
            print(f"           ... and {len(todos) - 5} more")

    # 4. Detect phase completions & update
    print("  [4/4] Detecting phase completions...")
    phases = detect_phase_completions(python_dir)
    for phase, pct in phases.items():
        bar = make_progress_bar(pct)
        print(f"         {phase}: {bar}")

    # Update PROGRESS_GUIDE.md
    print("\n  Updating PROGRESS_GUIDE.md...")
    update_progress_guide(python_stats, test_stats, todos, phases, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
