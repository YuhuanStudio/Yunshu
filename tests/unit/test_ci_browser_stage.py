"""Keep the browser gate runnable on a Linux host without MLX or a project venv."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_linux_browser_job_installs_console_python():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    build = workflow.split("  build:\n", 1)[1].split("  test-unit:\n", 1)[0]
    install = build.index("uv venv .console-test-venv")
    browser = build.index("pnpm test:browser")
    assert install < browser
    assert "YUNSHU_TEST_PYTHON=$PWD/.console-test-venv/bin/python" in build
    for dependency in ("fastapi==", "httpx==", "uvicorn==", "websockets=="):
        assert dependency in build[install:browser]
    assert "uv sync" not in build  # MLX is macOS-only; the console is CPU-only.


def test_local_browser_stage_is_opt_in_and_uses_checkout_python():
    script = (ROOT / "scripts/dev/ci-local").read_text()
    stage = script.split(
        'if [ "$STAGE" != gpu ] && [ "${YUNSHU_CI_BROWSER:-0}" = 1 ]; then', 1
    )[1].split("\nfi", 1)[0]
    assert "playwright install --with-deps chromium webkit" in stage
    assert 'YUNSHU_TEST_PYTHON="$WORK/.venv/bin/python" pnpm test:browser' in stage


def test_local_gate_runs_full_suite_with_vm_capability_plugin():
    script = (ROOT / "scripts/dev/ci-local").read_text()
    assert "-m pytest tests/unit" in script
    assert "-p scripts.dev.ci_paravirtual" in script
    assert 'cat "$WORK/pytest-paravirtual.log"' in script
    assert "paravirtual pytest rc $prc" in script


def test_vm_fixture_forces_capability_but_keeps_explicit_path_overrides(monkeypatch):
    from importlib.util import module_from_spec, spec_from_file_location
    from types import SimpleNamespace

    from yunshu_engine.utils import hardware

    spec = spec_from_file_location(
        "ci_paravirtual", ROOT / "scripts/dev/ci_paravirtual.py"
    )
    plugin = module_from_spec(spec)
    spec.loader.exec_module(plugin)
    apply = plugin.force_paravirtual_metal.__wrapped__
    monkeypatch.setattr(hardware, "is_paravirtual_metal", lambda: False)
    detector = hardware.is_paravirtual_metal
    apply(
        monkeypatch,
        SimpleNamespace(
            node=SimpleNamespace(
                name="test_identification_requires_both_vm_identifiers[physical]"
            )
        ),
    )
    assert hardware.is_paravirtual_metal is detector
    apply(monkeypatch, SimpleNamespace(node=SimpleNamespace(name="test_runner")))
    assert hardware.is_paravirtual_metal() is True
    hardware.is_paravirtual_metal.cache_clear()
    monkeypatch.setattr(hardware, "is_paravirtual_metal", lambda: False)
    assert hardware.is_paravirtual_metal() is False


def test_cpu_stage_runs_outside_gpuq_and_the_gpu_stage_runs_only_the_unit_suite():
    """2026-11-10: the CPU-only browser suite (~50 min) held the GPU lane inside the gpuq job
    while 25 jobs waited. YUNSHU_CI_STAGE=cpu runs lint/console/browser plus the build and the
    wheel smoke (they need the console bundle the frontend build makes) outside gpuq and skips
    the unit suite; the gpu stage runs only the unit suite (a first split put the wheel check
    in the gpu half without the frontend build: 'console_static/index.html' missing)."""
    script = (ROOT / "scripts/dev/ci-local").read_text()
    guard = script.index('if [ "$STAGE" != cpu ] && [ -z "${GPUQ_RC:-}" ]')
    assert guard < script.index("git clone")
    suite = script.index('if [ "$STAGE" != cpu ]; then\nPROFILE=')
    assert (
        script.index("pnpm test:browser") < suite < script.index("-m pytest tests/unit")
    )
    gpu_exit = script.index('echo "ci-local: GPU stage PASS $SHA"')
    assert script.index("-m pytest tests/unit") < gpu_exit < script.index("uv build -q")
    assert 'if [ "$STAGE" != gpu ]; then\n(cd frontend && pnpm install' in script
    assert '[ "$STAGE" = all ] || WORK="$WORK-$STAGE"' in script
