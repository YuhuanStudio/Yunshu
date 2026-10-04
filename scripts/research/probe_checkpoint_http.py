"""HTTP A/B for synchronous vs first-token-deferred checkpoint admission.

The baseline launcher selects the upstream coordinator method before starting
Yunshu. This is a research-only selection; serving settings remain unchanged.
Every measured stream is parsed by tfbench's fail-closed reader.
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfbench as t  # noqa: E402


def assert_no_full_unit_suites(processes=None):
    """Report unqueued full unit suites that could contaminate this A/B.

    Suites niced to >= 10 (the agents' policy) are ignored. Others only warn:
    gpuq's per-job CPU-contention record flags the run, so it is not aborted.
    Returns the offending pids.
    """
    if processes is None:
        import psutil

        processes = psutil.process_iter(["pid", "cmdline", "nice"])
    active = []
    for process in processes:
        args = process.info.get("cmdline") or []
        # Inspect argv tokens, never a shell/agent prompt that mentions pytest.
        pytest = any(Path(arg).name == "pytest" for arg in args[:4])
        whole = any(arg.rstrip("/").endswith("tests/unit") for arg in args)
        nice = process.info.get("nice")
        if pytest and whole and not (nice is not None and nice >= 10):
            active.append(process.info["pid"])
    if active:
        print(
            f"warning: full unit suites running {active}; gpuq contention record flags this run",
            file=sys.stderr,
        )
    return active


def restore_experiment_launcher(mode):
    """Process-local research patches; no serving flag or APC arithmetic change."""
    patch = ""
    if mode in (
        "reserved",
        "async",
        "barrier",
        "asyncbarrier",
        "paired32",
        "spans",
        "cow",
        "cowasync",
    ):
        # Keep the reference arm upstream even after COW ships. Candidates
        # install the exact production helper through their explicit wrapper.
        patch += (
            "from yunshu_engine.kernels import cache_restore\n"
            "cache_restore.install = lambda: None\n"
        )
    if mode in (
        "async",
        "barrier",
        "asyncbarrier",
        "paired32",
        "spans",
        "cow",
        "cowasync",
    ):
        patch += (
            "import sys\n"
            + f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        )
    if mode in ("cow", "cowasync"):
        patch += "from cow_restore import install as _install_cow\n_cow_counts, _cow_uninstall = _install_cow()\n"
    if mode == "notokprefix":
        # Reference arm: production tokenizer prefix reuse disabled.
        patch += (
            "from yunshu_engine.tokenizer_prefix import TokenizerPrefixCache as _T\n"
            "_T.encode = lambda self, tok, text, add_special_tokens=True: "
            "_T._full_encode(tok, text, add_special_tokens)\n"
        )
    if mode == "fence":
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from tokenizer_fence_cache import install as _fence\n"
            "_fence_counts, _fence_uninstall = _fence()\n"
        )
    if mode == "bucket512":
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from yunshu_engine.kernels import cache_restore\ncache_restore.install()\n"
            "from capacity_bucket import install as _bucket\n"
            "_bucket_counts, _bucket_uninstall = _bucket()\n"
        )
    if mode == "interleave":
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from span_forward import install as _interleave\n"
            "_interleave_counts, _interleave_uninstall = _interleave(preserve_descriptors=True)\n"
        )
    if mode == "spans":
        patch += "from span_forward import install as _install_spans\n_span_counts, _span_uninstall = _install_spans()\n"
    if mode == "paired32":
        patch += "from lane_paired32 import install as _install_paired32\n_paired32_uninstall = _install_paired32()\n"
    if mode in ("barrier", "asyncbarrier"):
        patch += "from lane_final_barrier import install as _install_barrier\n_barrier_uninstall = _install_barrier()\n"
    if mode in ("async", "asyncbarrier", "cowasync"):
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from async_restore import install\n"
            "_async_counts, _async_uninstall = install()\n"
        )
    if mode in ("direct", "keep"):
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from probe_prefill_restore import direct_clone\n"
            "from mlx_vlm import apc_adapters\n"
            "_clone = apc_adapters.clone_cache_entry\n"
            "def _direct(c, **kwargs):\n"
            "    return direct_clone(c, clone=_clone, **kwargs)\n"
            "apc_adapters.clone_cache_entry = _direct\n"
        )
    if mode in ("keep", "pool", "nativepool", "combo"):
        patch += (
            "import mlx.core as mx\n"
            "from yunshu_engine.kernels import buffer_cache\n"
            "_install = buffer_cache.install\n"
            "def _bounded_pool(limit_gib):\n"
            "    result = _install(limit_gib)\n"
            "    mx.set_cache_limit(buffer_cache._STATE['limit'])\n"
            "    return result\n"
            "buffer_cache.install = _bounded_pool\n"
            "buffer_cache.clear_if_over = lambda: None\n"
        )
    if mode in ("wide", "combo"):
        patch += (
            "from yunshu_engine.kernels import lane_linear\n"
            "from yunshu_engine.kernels.tensorfold import lane_qmm\n"
            "lane_linear.PIECE = 512\n"
            "lane_qmm.MAX_ROWS = 512\n"
        )
    if mode in ("native", "nativepool", "combo"):
        patch += (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from native_model_cache import wrap\n"
            "from mlx_vlm.models.qwen3_5.language import Qwen3_5Model\n"
            "Qwen3_5Model.__call__ = wrap(Qwen3_5Model.__call__)\n"
        )
    return patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=(
            "main",
            "interleave",
            "bucket512",
            "fence",
            "notokprefix",
            "async",
            "cow",
            "cowasync",
            "spans",
            "barrier",
            "paired32",
            "asyncbarrier",
            "sync",
            "deferred",
            "reserved",
            "legacy",
            "tf",
            "direct",
            "keep",
            "wide",
            "pool",
            "native",
            "nativepool",
            "combo",
        ),
        required=True,
    )
    ap.add_argument("--draft", choices=("mtp", "off"), default="mtp")
    ap.add_argument(
        "--model", help="Optional small checkpoint for harness smoke checks"
    )
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert_no_full_unit_suites()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    launcher = a.out.parent / f"checkpoint-launcher-{a.mode}-{a.rep}.py"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "from mlx_vlm.apc_coordinator import APCCoordinator\n"
        "from yunshu_engine.apc_manager import _Coordinator\n"
        + (
            "_Coordinator.merge_rows = APCCoordinator.merge_rows\n"
            if a.mode in ("sync", "deferred", "legacy")
            else ""
        )
        + (
            "_Coordinator.flush_deferred_checkpoints = None\n"
            if a.mode in ("sync", "legacy")
            else ""
        )
        + (
            "from yunshu_engine.kernels import lane_linear\n"
            "lane_linear.STOCK_MIN_ROWS = 128\n"
            if a.mode == "legacy"
            else ""
        )
        + restore_experiment_launcher(a.mode)
        + "from yunshu_cli import main\nmain()\n"
    )
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    env = {"YUNSHU_VLM_DRAFT": a.draft, "YUNSHU_VLM_APC_DISK": "0"}
    if a.mode == "legacy":
        env["YUNSHU_PREFILL_BUFFER_CACHE_GB"] = "0"
    server = None
    # Srv constructors can raise after Popen. Remember our own child before any
    # ready check so the finally block always cleans this probe's server.
    import subprocess

    children = []
    original_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    t.subprocess.Popen = tracked_popen
    try:
        engine = "tf-new" if a.mode == "tf" else "yunshu"
        server = t.Srv(
            engine, env, f"checkpoint-{a.mode}-{a.ctx}-{a.rep}", model=a.model
        )
        t.send(server.url, t.req(server.model, "Say hi.", 24, seed=1234))
        with a.out.open("a") as out:
            for kind in ("prose", "code"):
                text = t.load_prompt(f"{kind}-{a.ctx}")
                messages = [{"role": "user", "content": text}]
                reply = ""
                reference = a.out.parent / f"turn2-reply-{kind}-{a.ctx}.json"
                for phase in ("cold", "warm", "turn2"):
                    assert_no_full_unit_suites()
                    body = t.req(server.model, text, 256, seed=1234)
                    body["messages"] = list(messages)
                    if phase == "turn2":
                        # Every engine / arm receives the same second-turn text,
                        # even when their first-turn generation differs.
                        reply = json.loads(reference.read_text())["reply"]
                        body["messages"] += [
                            {"role": "assistant", "content": reply},
                            {
                                "role": "user",
                                "content": "Continue with the next part, at the same length.",
                            },
                        ]
                    result = t.send(server.url, body)
                    assert_no_full_unit_suites()
                    if phase == "cold":
                        reply = result["_text"]
                        if not reference.exists():
                            reference.write_text(json.dumps(dict(reply=reply)))
                    result.pop("_text")
                    record = dict(
                        mode=a.mode,
                        ctx=a.ctx,
                        rep=a.rep,
                        kind=kind,
                        phase=phase,
                        request_sha256=hashlib.sha256(
                            json.dumps(
                                {k: v for k, v in body.items() if k != "model"},
                                sort_keys=True,
                            ).encode()
                        ).hexdigest(),
                        load_1m=os.getloadavg()[0],
                        contended=t.was_contended(),
                        **result,
                    )
                    line = json.dumps(record)
                    out.write(line + "\n")
                    out.flush()
                    print(line, flush=True)
            log = server.log.read_text()
            proof = (
                "drafter /Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
                if a.mode == "tf"
                else f"draft={a.draft}"
            )
            if proof not in log:
                raise RuntimeError(f"engaged mode absent from {server.log}: {proof}")
            experiment_proof = {
                "interleave": "Original-descriptor projection interleave engaged:",
                "bucket512": "Native restore capacity bucket engaged: 512 tokens",
                "fence": "Exact tokenizer fence reuse engaged",
            }.get(a.mode)
            if (
                experiment_proof
                and "27B" in (a.model or t.M)
                and experiment_proof not in log
            ):
                raise RuntimeError(
                    f"candidate not engaged: {experiment_proof}; see {server.log}"
                )
            out.write(
                json.dumps(
                    dict(
                        mode=a.mode,
                        ctx=a.ctx,
                        rep=a.rep,
                        phase="complete",
                        success=True,
                        engaged_mode=server.engaged_spec_mode,
                        experiment_engaged=bool(
                            experiment_proof and experiment_proof in log
                        ),
                        server_log=str(server.log),
                        contended=t.was_contended(),
                    )
                )
                + "\n"
            )
    finally:
        t.subprocess.Popen = original_popen
        if server is not None:
            server.kill()
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
        with contextlib.suppress(OSError):
            launcher.unlink()


if __name__ == "__main__":
    main()
