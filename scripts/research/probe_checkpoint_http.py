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
    """An unqueued unit suite can run Metal tests and contaminate this A/B."""
    if processes is None:
        import psutil

        processes = psutil.process_iter(["pid", "cmdline"])
    active = []
    for process in processes:
        args = process.info.get("cmdline") or []
        # Inspect argv tokens, never a shell/agent prompt that mentions pytest.
        pytest = any(Path(arg).name == "pytest" for arg in args[:4])
        whole = any(arg.rstrip("/").endswith("tests/unit") for arg in args)
        if pytest and whole:
            active.append(process.info["pid"])
    if active:
        raise RuntimeError(f"GPU timing contaminated by full unit suites: {active}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("sync", "deferred", "reserved", "legacy", "tf"),
        required=True,
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
        + "from yunshu_cli import main\nmain()\n"
    )
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    env = {"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_VLM_APC_DISK": "0"}
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
        server = t.Srv(engine, env, f"checkpoint-{a.mode}-{a.ctx}-{a.rep}")
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
                else "Speculative decoding: mtp"
            )
            if proof not in log:
                raise RuntimeError(f"engaged mode absent from {server.log}: {proof}")
            out.write(
                json.dumps(
                    dict(
                        mode=a.mode,
                        ctx=a.ctx,
                        rep=a.rep,
                        phase="complete",
                        success=True,
                        engaged_mode="dflash2" if a.mode == "tf" else "mtp",
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
