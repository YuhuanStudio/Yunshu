"""Lead condition (2026-11-03): the full-comparison driver never touches the GPU itself.

fullcmp_run.py and every scripts/research module it imports must
  1. import no mlx / mlx_lm / mlx_vlm (checked in a fresh interpreter, sys.modules after import), and
  2. spawn only the allowed targets: gpuq submit / wait / status / cancel / run (the swap gate CPU job),
     the quiet gate's read-only probes (ps, sysctl), the audit, reps and preflight scripts, and one
     read-only git query in bench_engines. Every engine launch is a `gpuq submit` argv (Job.submit_argv).

The AST scan is static: a spawn it cannot classify fails the test. No GPU, no gpuq call, no server.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"

# Modules the driver imports (top level or lazily) from scripts/research.
MODULES = [
    "fullcmp_run",
    "bench_engines",
    "fullcmp_quiet",
    "fullcmp_profile",
    "fullcmp_audit_manifest",
]

FORBIDDEN_IMPORTS = ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio", "yunshu_engine", "yunshu_gateway")

GPUQ_VERBS = {"submit", "wait", "status", "cancel", "run"}
HARNESS_SCRIPTS = {"AUDIT", "REPS", "PREFLIGHT", "QUIET"}
SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output"}
OS_SPAWNS = {"system", "popen", "fork", "forkpty"}

# Spawns whose argv is a variable: (module, enclosing function) -> reason. Each is verified
# separately (run_capture callers, Job.submit_argv) or is a fixed read-only probe.
VARIABLE_ARGV_ALLOWED = {
    ("fullcmp_run", "run_capture"): "gpuq status: callers pass only [GPUQ, 'status'] (checked below)",
    ("fullcmp_run", "swap_gate"): "gpuq run: CPU swap-stability preflight job, no engine",
    ("fullcmp_run", "submit_all"): "Job.submit_argv(): gpuq submit (checked below)",
    ("bench_engines", "tree_version"): "read-only git rev-parse on a pinned tree",
    ("fullcmp_audit_manifest", "tree_head"): "read-only git rev-parse of the pinned worktree (fail closed)",
    ("fullcmp_run", "pre_submit_checks"): "CPU pytest of the explicit quick file list in the snapshot (checked by test_pre_submit_collects_explicit_files_only)",
}
LITERAL_NON_GPUQ_ALLOWED = {
    ("fullcmp_run", "_git"): "read-only git status / rev-parse / archive of the live worktree (harness snapshot)",
    ("fullcmp_run", "_file_blobs"): "read-only git hash-object of the snapshot files (integrity proof)",
    ("fullcmp_audit_manifest", "tree_head"): "read-only git rev-parse of the pinned worktree (fail closed)",
    ("fullcmp_quiet", "run_ps"): "quiet gate: read-only ps",
    ("fullcmp_quiet", "read_loadavg"): "quiet gate: read-only sysctl",
    ("bench_engines", "tree_version"): "read-only git rev-parse on a pinned tree",
}


def _spawn_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        if f.value.id == "subprocess" and f.attr in SUBPROCESS_CALLS:
            return f"subprocess.{f.attr}"
        if f.value.id == "os" and (
            f.attr.startswith("exec") or f.attr.startswith("spawn") or f.attr in OS_SPAWNS
        ):
            return f"os.{f.attr}"
    if isinstance(f, ast.Name) and f.id in SUBPROCESS_CALLS | {"Popen"}:
        return f.id
    return None


class _Scan(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.func: list[str] = []
        self.spawns: list[tuple[str, str, ast.Call]] = []  # (function, spawn name, call)

    def visit_FunctionDef(self, node):  # noqa: N802
        self.func.append(node.name)
        self.generic_visit(node)
        self.func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node):  # noqa: N802
        name = _spawn_name(node)
        if name:
            self.spawns.append((self.func[-1] if self.func else "<module>", name, node))
        self.generic_visit(node)


def _parse(module: str) -> ast.Module:
    return ast.parse((RESEARCH / f"{module}.py").read_text(), filename=f"{module}.py")


def _const(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _classify(module: str, func: str, call: ast.Call) -> str | None:
    """None when the spawn is allowed; otherwise the reason it is not."""
    argv = call.args[0] if call.args else None
    if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
        head = argv.elts[0]
        if isinstance(head, ast.Name) and head.id == "GPUQ":
            verb = _const(argv.elts[1]) if len(argv.elts) > 1 else None
            if verb in GPUQ_VERBS:
                return None
            return f"gpuq with verb {verb!r} not in {sorted(GPUQ_VERBS)}"
        if isinstance(head, ast.Name) and head.id == "HARNESS_PY":
            second = argv.elts[1] if len(argv.elts) > 1 else None
            if isinstance(second, ast.Call) and isinstance(second.func, ast.Name) and second.func.id == "str":
                second = second.args[0] if second.args else None
            if isinstance(second, ast.Name) and second.id in HARNESS_SCRIPTS:
                return None
            return "harness python on a script outside audit/reps/preflight/quiet"
        if _const(head) is not None:
            if (module, func) in LITERAL_NON_GPUQ_ALLOWED:
                return None
            return f"literal spawn {_const(head)!r} not allowed here"
        if (module, func) in VARIABLE_ARGV_ALLOWED:
            return None
        return "argv head is neither GPUQ, HARNESS_PY nor a literal"
    if (module, func) in VARIABLE_ARGV_ALLOWED:
        return None
    return "variable argv outside the allowlist"


def _all_spawns():
    for module in MODULES:
        scan = _Scan(module)
        scan.visit(_parse(module))
        for func, name, call in scan.spawns:
            yield module, func, name, call


def test_every_spawn_is_an_allowed_target():
    violations = []
    seen = set()
    for module, func, name, call in _all_spawns():
        seen.add((module, func))
        reason = _classify(module, func, call)
        if reason:
            violations.append(f"{module}.{func}:{call.lineno} {name}: {reason}")
    assert not violations, "disallowed spawn(s):\n" + "\n".join(violations)
    # The allowlists must match real code (a stale entry hides a removed check).
    for key in VARIABLE_ARGV_ALLOWED:
        if key[0] in MODULES:
            assert key in seen, f"stale allowlist entry {key}"


def test_run_capture_callers_only_query_gpuq_status():
    tree = _parse("fullcmp_run")
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_capture"
    ]
    assert calls, "run_capture has no callers; the allowlist entry is stale"
    for call in calls:
        arg = call.args[0]
        assert isinstance(arg, ast.List), f"run_capture:{call.lineno} takes a non-literal argv"
        assert isinstance(arg.elts[0], ast.Name) and arg.elts[0].id == "GPUQ"
        assert _const(arg.elts[1]) == "status", f"run_capture:{call.lineno} is not gpuq status"


def test_job_submit_argv_is_gpuq_submit():
    tree = _parse("fullcmp_run")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "submit_argv"]
    assert len(fns) == 1, "Job.submit_argv not found exactly once"
    ret = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Return)]
    assert ret and isinstance(ret[0].value, ast.List), "submit_argv must return a literal list"
    elts = ret[0].value.elts
    assert isinstance(elts[0], ast.Name) and elts[0].id == "GPUQ"
    assert _const(elts[1]) == "submit", "every engine launch must be a `gpuq submit` argv"


def test_driver_imports_no_mlx_in_a_fresh_interpreter(tmp_path):
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        sys.path.insert(0, {str(RESEARCH)!r})
        for name in {MODULES!r}:
            importlib.import_module(name)
        print(json.dumps(sorted(sys.modules)))
        """
    )
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    loaded = json.loads(res.stdout.strip().splitlines()[-1])
    bad = [m for m in loaded if m.split(".")[0] in FORBIDDEN_IMPORTS]
    assert not bad, f"driver pulled in GPU/engine modules: {bad}"
