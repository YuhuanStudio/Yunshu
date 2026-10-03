"""Cold-only HTTP prefill A/B; optional warm request checks reuse output.

Run only through gpuq. The research launcher changes layout retention, never
APC/restore. Unique --out names also isolate server homes and logs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def dispatch_receipt(log):
    """A live dispatch receipt survives server shutdown without atexit hooks."""
    for line in log.splitlines():
        if "NAX_DISPATCH_ENGAGED " in line:
            return json.loads(line.split("NAX_DISPATCH_ENGAGED ", 1)[1])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm",
        choices=(
            "base",
            "planes",
            "stock",
            "tile128",
            "lane64",
            "lane128",
            "narrow",
            "combo",
        ),
        required=True,
    )
    p.add_argument("--ctx", type=int, choices=(128, 8192, 32768), required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--warm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import tfbench as t

    if a.model:
        t.M = str(a.model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    t.OUT = a.out.parent / a.out.stem
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    launcher = a.out.with_suffix(".launcher.py")
    patch = ""
    custom = a.arm in ("tile128", "lane64", "lane128", "narrow", "combo")
    if a.arm in ("planes", "stock"):
        patch = f"""
from yunshu_engine.kernels import lane_linear
original = lane_linear.LaneLinear.stock
retained = {{}}
def stock(self):
    key = id(self)
    if key not in retained:
        w, s, b = original(self)
        retained[key] = (w,s,b) if {a.arm!r} == 'stock' else (s,b)
    values = retained[key]
    return values if len(values) == 3 else (original(self)[0], *values)
lane_linear.LaneLinear.stock = stock
"""
    elif custom:
        patch = f"""
import sys, atexit, json
sys.path.insert(0, {str(Path(__file__).parent)!r})
import nax_prefill_dispatch as dispatch
dispatch.install({a.arm!r})
from yunshu_engine.kernels import lane_linear
kernel_id = lane_linear.prefill_kernel_id
lane_linear.prefill_kernel_id = lambda: kernel_id() + '+research-{a.arm}'
atexit.register(lambda: print('NAX_DISPATCH_RECEIPT ' + json.dumps(dict(arm={a.arm!r}, calls=sum(dispatch.calls.values()))), flush=True))
"""
    launcher.write_text(
        f"#!{sys.executable}\n" + patch + "\nfrom yunshu_cli import main\nmain()\n"
    )
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    with a.out.open("a") as out:
        out.write(
            json.dumps(
                dict(
                    phase="start",
                    arm=a.arm,
                    launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
                    source=str(Path(__file__).resolve().parents[2]),
                )
            )
            + "\n"
        )
    children = []
    original_popen = subprocess.Popen

    def tracked(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    t.subprocess.Popen = tracked
    server = None
    try:
        server = t.Srv(
            "yunshu",
            {"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_VLM_APC_DISK": "0"},
            a.out.stem,
        )
        t.send(server.url, t.req(server.model, "Say hi.", 24, seed=1234))
        with a.out.open("a") as out:
            for kind in ("prose", "code"):
                text = (
                    "The quick brown fox jumps over the lazy dog. " * 12
                    if a.ctx == 128
                    else t.load_prompt(f"{kind}-{a.ctx}")
                )
                body = t.req(server.model, text, 64 if a.ctx == 128 else 256, seed=1234)
                reference = None
                for phase in ("cold", "warm") if a.warm else ("cold",):
                    result = t.send(server.url, body)
                    result.pop("_text")
                    if reference is None:
                        reference = result["sha"]
                    elif reference != result["sha"]:
                        raise RuntimeError("cold/warm output differs")
                    row = dict(
                        arm=a.arm,
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
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps(row), flush=True)
            log = server.log.read_text()
            if not a.model and "Speculative decoding: mtp" not in log:
                raise RuntimeError("MTP engagement missing")
            if not a.model and "stock-qmm-gt512" not in log:
                raise RuntimeError("stock prefill engagement missing")
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
    receipt = None
    if custom:
        receipt = dispatch_receipt(server.log.read_text())
        if not receipt and not a.model:
            raise RuntimeError("dispatch receipt missing")
        if not a.model and (receipt["arm"] != a.arm or not receipt["calls"]):
            raise RuntimeError("candidate prefill dispatch did not engage")
    row = dict(
        phase="complete",
        success=True,
        arm=a.arm,
        server_log=str(server.log),
        engaged_mode="mtp" if not a.model else "tiny",
        dispatch_receipt=receipt,
        contended=t.was_contended(),
    )
    with a.out.open("a") as out:
        out.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
