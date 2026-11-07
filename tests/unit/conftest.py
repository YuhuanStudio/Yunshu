"""Shared fixtures for unit tests."""

import os as _os
import sys as _sys
import tempfile as _tempfile

# The dev tools (yv, agentbench, the agentic runner) default to this machine's build volume and main
# checkout venv. Unit tests must never write there, and CI has neither: point them at a throwaway root
# and the running interpreter before any test module imports them.
_DEV_ROOT = _tempfile.mkdtemp(prefix="yunshu-unit-devroot-")
_os.environ["YV_PY"] = _sys.executable
_os.environ["YV_ROOT"] = _os.path.join(_DEV_ROOT, "verify")
_os.environ["AGENTIC_BUILD"] = _os.path.join(_DEV_ROOT, "agentic")
_os.environ["AGENTBENCH_ROOT"] = _os.path.join(_DEV_ROOT, "agentbench")
# Stores, keys, caches and config default to ~/.yunshu (Path.home()). A unit test must never write
# into the developer's real home (scripted chat completions piled up in ~/.yunshu/chat_completions),
# so every test runs with HOME pointed at the throwaway root, removed when the session ends.
_os.makedirs(_os.path.join(_DEV_ROOT, "home"), exist_ok=True)
_os.environ["HOME"] = _os.path.join(_DEV_ROOT, "home")
import atexit as _atexit
import shutil as _shutil

_atexit.register(_shutil.rmtree, _DEV_ROOT, True)

import os
from unittest.mock import MagicMock

import pytest

# helper modules next to the tests (fullcmp_pin_io) import by bare name, whichever file runs first
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _disable_auth(request):
    """Disable auth for all unit tests by default.

    Auth-related tests (auth, rbac, middleware) manage their own auth env
    and are exempted from the global disable.
    """
    modname = request.module.__name__ if request.module else ""

    is_auth_test = any(kw in modname.lower() for kw in ("auth", "rbac", "middleware"))
    if is_auth_test:
        yield
        return

    old = os.environ.get("YUNSHU_AUTH_DISABLED")
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    yield
    if old is None:
        os.environ.pop("YUNSHU_AUTH_DISABLED", None)
    else:
        os.environ["YUNSHU_AUTH_DISABLED"] = old


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path, monkeypatch):
    """Tests never read the developer's ~/.yunshu/config.toml."""
    from yunshu_engine import settings

    monkeypatch.setattr(
        settings, "user_config_path", lambda: tmp_path / "user-config.toml"
    )


@pytest.fixture(autouse=True)
def _no_m3_idle_probe(monkeypatch):
    """gpuq daemon-loop tests never ssh to the laptop for its idle time."""
    monkeypatch.setenv("GPUQ_M3_IDLE_S", "0")


@pytest.fixture(autouse=True)
def _mcp_connector_loopback(monkeypatch):
    """The fake MCP servers in the tests listen on 127.0.0.1; the product default blocks that."""
    monkeypatch.setenv("YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE", "1")


@pytest.fixture(autouse=True, scope="session")
def _fast_drain():
    """Zero drain timeout for all tests — avoids 30s wait on app teardown."""
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"
    yield
    os.environ.pop("YUNSHU_DRAIN_TIMEOUT", None)


@pytest.fixture
def mock_engine():
    """A minimal mock Engine that simulates a loaded+running state."""
    from yunshu_engine.types import EngineConfig

    engine = MagicMock()
    engine._model_name = "test-model"
    engine._model = MagicMock()
    engine._running = True
    engine._loaded = True
    engine._model_path = "/models/test-model"
    engine.config = EngineConfig()
    yield engine


@pytest.fixture
def mock_batched_engine():
    """A minimal mock BatchedEngine that simulates a loaded+running state."""
    engine = MagicMock()
    engine._model_name = "test-model"
    engine._running = True
    engine._loaded = True
    yield engine


@pytest.fixture
def set_engine(mock_engine):
    """Set a mock engine as the global engine, yield it, then clear.

    Uses the engine module's set_engine/get_engine pattern.
    """
    from yunshu_gateway.engine import set_engine as _set_engine

    _set_engine(mock_engine)
    yield mock_engine
    _set_engine(None)


@pytest.fixture
def client(set_engine):
    """FastAPI TestClient with mock engine and auth disabled."""
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _isolate_native_tool_context():
    """Direct router helpers must not leave request context for the next unit test."""
    from yunshu_engine.batched_engine import _REQUEST_TOOL_USE, _REQUEST_TOOLS

    tools = _REQUEST_TOOLS.set(None)
    choice = _REQUEST_TOOL_USE.set(None)
    yield
    _REQUEST_TOOLS.reset(tools)
    _REQUEST_TOOL_USE.reset(choice)


@pytest.fixture(autouse=True)
def _restore_environment():
    """A test that exports to os.environ (serve --uds exports YUNSHU_UDS) must not change the next one's
    world: the random-order pass found test_cli_bind_preflight failing behind such a leak."""
    saved = dict(os.environ)
    yield
    if dict(os.environ) != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _restore_gateway_engine_globals():
    """The gateway keeps its engine and model manager in module globals (init_model_manager, set_engine).
    A test that leaves a manager behind turned /health/ready into 503 for a later test that sets only an
    engine (random-order seed 4244: test_readiness_probe_with_model)."""
    import sys

    mod = sys.modules.get("yunshu_gateway.engine")
    names = ("_engine", "_display_model_id", "_model_manager")
    saved = {n: getattr(mod, n) for n in names if hasattr(mod, n)} if mod else {}
    yield
    mod = sys.modules.get("yunshu_gateway.engine")
    if mod is not None:
        for n, v in saved.items():
            setattr(mod, n, v)
        if (
            not saved
        ):  # the module was first imported by this test: back to its import-time state
            for n in names:
                if hasattr(mod, n):
                    setattr(mod, n, None)


@pytest.fixture(autouse=True)
def _restore_mlx_default_device():
    """A test that moves MLX to the CPU must not leave it there: a later test's Metal kernels would then run
    on the CPU (test_lane_linear failed only in suite order this way). A stub mlx.core some tests swap in is
    left alone."""

    def real_mx():
        mx = _sys.modules.get("mlx.core")
        if hasattr(mx, "default_device") and hasattr(mx, "set_default_device"):
            return mx
        return None

    mx = real_mx()
    previous = mx.default_device() if mx is not None else None
    yield
    mx = real_mx()
    if mx is None:
        return
    if previous is None:
        # mlx was first imported during this test; the process default before it was the GPU when there is one
        previous = mx.gpu if mx.metal.is_available() else mx.cpu
    if mx.default_device() != previous:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _yunshu_loggers_inherit_levels():
    """Each test starts with the yunshu_* package loggers at NOTSET, so ``caplog.at_level(DEBUG)`` reaches them: an earlier
    import or test that configured them (the gateway sets yunshu_engine to INFO) made caplog miss DEBUG lines in
    suite order only (test_shortdecode_probe, test_vlm_next_turn). Their levels are restored afterwards."""
    import logging

    loggers = [
        lg
        for name, lg in logging.root.manager.loggerDict.items()
        # package loggers only: "yunshu.audit" is INFO by design from import time and must keep it
        if name.startswith("yunshu_") and isinstance(lg, logging.Logger)
    ]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.NOTSET)
    yield
    for lg, level in saved:
        lg.setLevel(level)


# fullcmp / tfbench harness markers (by file, so the harness test files stay untouched):
# fullcmp_e2e = the end-to-end chains (11-30 min each: planner -> wrapper -> real tfbench against the
# stub -> audit); fullcmp_quick = every other fullcmp / tfbench unit file, the fast regression subset
# (`pytest -m fullcmp_quick`, under 2 minutes).
_FULLCMP_QUICK_FILES = ("test_fullcmp_", "test_tfbench_fullcmp")


# Tests that validate the REAL full-comparison run (census.json, pin files, manifest, installed
# binaries and venvs under the build volume). They cannot run anywhere but the benchmark machine
# (a CI runner or a laptop has none of it), so they skip when the run dir is absent. The e2e chains
# need it too (every file named test_fullcmp_e2e*). Listed in docs/guides/FULLCOMPARE.md section 7:
# on the benchmark machine none of these may be skipped (the harness preflight counts the skips).
_BENCH_CENSUS = (
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/census.json"
)
_NEEDS_BENCH_MACHINE = frozenset(
    {
        "test_fullcmp_audit_manifest.py::test_build_manifest_cells_pins_and_phases",
        "test_fullcmp_audit_manifest.py::test_duplicate_cell_stops_the_build",
        "test_fullcmp_audit_manifest.py::test_rep_decision_per_cell_reaches_the_audit_rule",
        "test_fullcmp_batch_smoke_reps.py::test_smoke_batch_is_a_valid_batch_choice_and_dry_runs",
        "test_fullcmp_batch_smoke_reps.py::test_unknown_engine_id_is_refused_not_silently_empty",
        "test_fullcmp_engine_tables.py::test_check_all_returns_every_miss_at_once",
        "test_fullcmp_engine_tables.py::test_every_manifest_engine_has_a_complete_pin",
        "test_fullcmp_engine_tables.py::test_every_manifest_engine_resolves_every_table",
        "test_fullcmp_engine_tables.py::test_runner_refuses_on_a_miss_before_any_job",
        "test_fullcmp_engine_tables.py::test_unresolved_is_a_note_not_a_miss",
        "test_fullcmp_pin_per_engine.py::test_cell_env_carries_the_engines_own_tree",
        "test_fullcmp_pin_per_engine.py::test_dev0_cell_still_validates_against_dev0",
        "test_fullcmp_pin_per_engine.py::test_dev0_tree_is_refused_for_the_pb2_engine",
        "test_fullcmp_pin_per_engine.py::test_engine_pin_routes_the_env_per_engine",
        "test_fullcmp_pin_per_engine.py::test_pb2_cell_validates_against_the_pb2_tree_and_venv",
        "test_fullcmp_pins.py::test_every_census_engine_has_a_pin",
        "test_fullcmp_pins.py::test_install_paths_carry_the_pinned_version",
        "test_fullcmp_pins.py::test_missing_binary_leaves_the_digest_empty",
        "test_fullcmp_pins.py::test_pinned_binary_is_the_launched_file_not_the_archive",
        "test_fullcmp_pins.py::test_unestablished_engaged_modes_fail_closed",
        "test_fullcmp_pins.py::test_unsupported_engines_have_no_digest_and_say_why",
        "test_fullcmp_pins.py::test_yunshu_pin_is_dflash_with_the_engaged_line_and_uv_lock",
        "test_fullcmp_profile_rep.py::test_audit_manifest_reps_follow_the_rows_not_a_constant",
        "test_fullcmp_run.py::test_a_cell_flagged_twice_fails_the_batch",
        "test_fullcmp_run.py::test_a_quiet_flagged_cell_is_superseded_and_rerun_in_its_slot",
        "test_fullcmp_run.py::test_allow_foreign_overrides_refusal",
        "test_fullcmp_run.py::test_audit_manifest_carries_census_pins_for_every_engine",
        "test_fullcmp_run.py::test_audit_manifest_names_the_profile_required_engines",
        "test_fullcmp_run.py::test_cells_filter_selects_only_matching_cells_and_refuses_an_empty_match",
        "test_fullcmp_run.py::test_dry_run_touches_no_gpu_or_queue",
        "test_fullcmp_run.py::test_full_run_queues_rep3_for_flagged_cells_and_fails_on_bad_job",
        "test_fullcmp_run.py::test_plan_batch_profile_only_has_only_profile_jobs",
        "test_fullcmp_run.py::test_profile_only_dry_run_prints_only_profile_jobs",
        "test_fullcmp_run.py::test_profile_only_needs_profile_rep_and_engines",
        "test_fullcmp_run.py::test_profile_only_run_skips_the_audit_reps_and_requeues",
        "test_fullcmp_run_quiet.py::test_busy_host_refuses_the_batch_before_anything_is_submitted",
        "test_fullcmp_snapshot.py::test_driver_main_skips_pre_submit_inside_a_test_run",
        "test_fullcmp_submit_argv.py::test_driver_audit_manifest_has_a_pin_for_every_batch_engine",
        "test_fullcmp_submit_argv.py::test_every_b1_submit_argv_passes_gpuq_preflight",
        "test_fullcmp_submit_argv.py::test_every_cell_carries_the_run_id_and_its_manifest_group",
        "test_fullcmp_submit_argv.py::test_gate_is_a_step_inside_the_wrapper_not_the_argv",
        "test_fullcmp_submit_argv.py::test_self_check_is_clean_for_every_b1_job",
        "test_fullcmp_submit_argv.py::test_submit_argv_never_names_the_main_checkout",
        "test_fullcmp_yunshu_pb2.py::test_both_yunshu_pins_record_their_package_versions_and_differences_are_footnoted",
        "test_fullcmp_yunshu_pb2.py::test_package_versions_are_read_from_each_pinned_venv_and_match",
        "test_fullcmp_yunshu_pb2.py::test_pb2_pin_resolves_with_the_census_and_is_its_own_tree",
        "test_fullcmp_yunshu_pb2.py::test_preview_has_its_own_pin_file_and_the_dev0_tree_is_refused_for_it",
    }
)


def _bench_machine_present() -> bool:
    try:
        return os.path.isfile(_BENCH_CENSUS)
    except OSError:  # PermissionError under the CI sandbox counts as absent
        return False


def pytest_collection_modifyitems(config, items):
    machine = _bench_machine_present()
    for item in items:
        name = os.path.basename(str(item.fspath))
        if not name.startswith(_FULLCMP_QUICK_FILES):
            continue
        is_e2e = name.startswith("test_fullcmp_e2e")
        if is_e2e:
            item.add_marker(pytest.mark.fullcmp_e2e)
        else:
            item.add_marker(pytest.mark.fullcmp_quick)
        base = f"{name}::{item.originalname}"
        if not machine and (is_e2e or base in _NEEDS_BENCH_MACHINE):
            item.add_marker(
                pytest.mark.skip(
                    reason="needs the benchmark machine's fullcmp-1103 run dir "
                    "(census, pins, manifest); see docs/guides/FULLCOMPARE.md section 7"
                )
            )


def _loaded_driver_modules():
    """Every loaded copy of fullcmp_run.py under any name (a test may hold a copy that is not in
    sys.modules any more). gc.get_objects() also yields dead weakref proxies, whose isinstance or
    attribute access raises ReferenceError: those are skipped."""
    import gc as _gc
    import types as _types

    found = []
    for o in _gc.get_objects():
        try:
            if isinstance(o, _types.ModuleType) and str(
                getattr(o, "__file__", "")
            ).endswith("fullcmp_run.py"):
                found.append(o)
        except ReferenceError:
            continue
    return found


@pytest.fixture(autouse=True)
def _driver_harness_snapshot_stub(request, monkeypatch):
    """The fullcmp driver snapshots its harness at HEAD and runs the pre-submit subset before a
    batch (fullcmp_run.main). Unit tests call main() from a worktree or from a snapshot (no .git):
    they stub only those two steps, so the snapshot's own tests run without git. test_fullcmp_snapshot
    exercises the real snapshot code and is not stubbed."""
    if request.node.fspath.basename == "test_fullcmp_snapshot.py":
        return
    for mod in _loaded_driver_modules():
        if not hasattr(mod, "snapshot_harness") or not hasattr(
            mod, "pre_submit_checks"
        ):
            continue
        root = mod.REPO_ROOT
        monkeypatch.setattr(
            mod, "snapshot_harness", lambda work, repo=None, _r=root: _r
        )
        monkeypatch.setattr(mod, "use_harness", lambda root: None)
        monkeypatch.setattr(mod, "pre_submit_checks", lambda root, repo=None: "")


# Unit tests never run the driver's pre-submit (it runs a pytest subset: a recursion if a test
# that calls the driver ran it again). Inherited by the driver's subprocesses too.
_sys.modules.setdefault("_fullcmp_presubmit_guard", None)
_os.environ.setdefault("TFB_PRESUBMIT_SKIP", "1")


@pytest.fixture(autouse=True)
def _fullcmp_driver_sandbox(request, monkeypatch, tmp_path_factory):
    """A test that loads the fullcmp driver never reaches the real GPU queue or the real result roots
    (2026-11-04: a unit test submitted five real jobs). Every loaded driver copy gets tmp paths and a
    fake gpuq that only logs its argv; GPUQ_DIR points at tmp, so the driver's own refusal also holds."""
    drivers = [o for o in _loaded_driver_modules() if hasattr(o, "GPUQ")]
    if not drivers:
        return
    sandbox = tmp_path_factory.mktemp("driver-sandbox")  # not the test's tmp_path
    for d in ("gpuq", "wrap", "raw", "home", "tfout", "audit", "quiet", "preflight"):
        (sandbox / d).mkdir(parents=True, exist_ok=True)
    fake = sandbox / "fake_gpuq.sh"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {sandbox / "submits.log"}\n'
        'case "$1" in submit) echo fake-job-$$ ;; esac\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("GPUQ_DIR", str(sandbox / "gpuq"))
    for mod in drivers:
        monkeypatch.setattr(mod, "GPUQ", str(fake))
        monkeypatch.setattr(mod, "GPUQ_DIR_DEFAULT", str(sandbox / "gpuq"))
        monkeypatch.setattr(mod, "WRAP_ROOT", sandbox / "wrap")
        monkeypatch.setattr(mod, "RAW_ROOT", sandbox / "raw")
        monkeypatch.setattr(mod, "HOME_ROOT", sandbox / "home")
        monkeypatch.setattr(mod, "TFOUT_ROOT", sandbox / "tfout")
        monkeypatch.setattr(mod, "AUDIT_DIR", sandbox / "audit")
        monkeypatch.setattr(mod, "QUIET_DIR", sandbox / "quiet")
        monkeypatch.setattr(mod, "PREFLIGHT_DIR", sandbox / "preflight")


_REAL_QUEUE = "/Volumes/P5Plus/yunshu-gpuq"


@pytest.fixture(autouse=True)
def _real_paths_sandbox(tmp_path_factory, monkeypatch):
    """Every unit test: the GPU queue dir, telemetry dir, quiet dir and the Yunshu result output go to
    tmp, and a call to gpuq while GPUQ_DIR is the real queue is refused (a test must not submit real
    jobs or write real raw files). Tests that point GPUQ_DIR at their own tmp dir are unaffected."""
    # its own tmp dir, not the test's tmp_path: tests that list or size their tmp_path must not see it
    box = tmp_path_factory.mktemp("real-paths-sandbox")
    for d in ("gpuq", "telemetry", "quiet", "out", "home"):
        (box / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GPUQ_DIR", str(box / "gpuq"))
    monkeypatch.setenv("TFB_TELEMETRY_DIR", str(box / "telemetry"))
    monkeypatch.setenv("TFB_OUT", str(box / "out"))
    monkeypatch.setenv("QUIET_DIR", str(box / "quiet"))
    import subprocess as _sp

    real_run, real_popen = _sp.run, _sp.Popen

    def _argv0(args):
        a = args[0] if isinstance(args, (list, tuple)) and args else args
        return str(a) if a is not None else ""

    def _guard(args):
        argv = list(args) if isinstance(args, (list, tuple)) else [str(args)]
        if argv and argv[0].endswith("gpuq") and "fake" not in argv[0]:
            gdir = os.environ.get("GPUQ_DIR", "")
            if os.path.realpath(gdir) == os.path.realpath(_REAL_QUEUE):
                raise RuntimeError(
                    f"sandbox: a test tried gpuq {argv[1:3]} on the real queue"
                )

    def run(args, *a, **k):
        _guard(args)
        return real_run(args, *a, **k)

    class GuardedPopen(real_popen):
        def __init__(self, args, *a, **k):
            _guard(args)
            super().__init__(args, *a, **k)

    monkeypatch.setattr(_sp, "run", run)
    monkeypatch.setattr(_sp, "Popen", GuardedPopen)


@pytest.fixture(scope="module", autouse=True)
def _fullcmp_e2e_harness_stand_in(request):
    """The fullcmp e2e chains run tfbench from the live worktree (no snapshot), so the cell header has
    no harness_commit and the audit's harness_identity guard fails. The stand-in is this worktree's
    HEAD and tree: cell_env reads fullcmp_run.HARNESS_COMMIT / HARNESS_TREE at call time. Every loaded
    copy of the driver is set (a test may load it under another name). Real runs get the values from
    use_harness(snapshot); the guard itself is unchanged (test_fullcmp_audit covers a real header
    without harness_commit)."""
    name = request.node.fspath.basename
    if not name.startswith("test_fullcmp_e2e"):
        yield
        return
    import subprocess as _sp

    root = str(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    head = _sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tree = _sp.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    mp = pytest.MonkeyPatch()
    for mod in _loaded_driver_modules():
        if hasattr(mod, "HARNESS_COMMIT"):
            mp.setattr(mod, "HARNESS_COMMIT", head)
            mp.setattr(mod, "HARNESS_TREE", tree)
    yield
    mp.undo()
