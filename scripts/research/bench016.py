"""Fresh-release snapshot controller and guarded tfbench entrypoint (CPU until cell)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_engines as be  # noqa: E402
import bench_snapshot as bs  # noqa: E402
import refsync_jobs as rs  # noqa: E402

ROOT = HERE.parents[1]
GPUQ = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq"
QUEUE = Path("/Volumes/P5Plus/yunshu-gpuq")
ACTIVE = {"pending", "running", "paused", "wait-slot", "held"}


def admission(swap: str, processes: str) -> list[str]:
    """Conservative host gate: never reclaim somebody else's resident engine."""
    errors = []
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", swap)
    if not m:
        errors.append("cannot parse swap usage")
    # Historical host swap belongs to other processes. No owned engine exists
    # before this fresh process launch; reject every resident engine below.
    # gpuq independently rejects >64 MiB new swap during the cell.
    for line in processes.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        cmd = fields[2]
        if int(fields[1]) > 1024 * 1024 and re.search(
            r"(llama-server|tensorfold.*serve|splash.*serve|omlx.*serve|yunshu.*serve|strata.*--|ollama.*runner)",
            cmd,
        ):
            errors.append("foreign resident engine: " + fields[0])
    return errors


def verdict(job, rows: list[dict], receipt: dict) -> list[str]:
    errors = bs.validate_rows(job, rows)
    if receipt.get("rc") != 0:
        errors.append("gpuq rc not zero")
    if receipt.get("contended") is not False:
        errors.append("gpuq contention missing or true")
    if job.stage != "pilot":
        phases = {r.get("phase"): r for r in rows if r.get("part") == "decode"}
        for phase, n in [("cold", 2048), ("warm", 2048), ("turn2", 256)]:
            if phases.get(phase, {}).get("ct") != n:
                errors.append(f"{phase}: missing/short output")
        if phases.get("cold", {}).get("sha") != phases.get("warm", {}).get("sha"):
            errors.append("cold/warm text mismatch")
    return errors


def configure(manifest: dict) -> None:
    archive = Path(manifest["yunshu_src"]).parent.parent / "yunshu-main.tar"
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != manifest["source_archive_sha256"]
    ):
        raise ValueError("pinned source archive digest changed")
    with tarfile.open(archive) as source:
        for entry in source.getmembers():
            if entry.isfile():
                expected = hashlib.sha256(source.extractfile(entry).read()).digest()
                actual = hashlib.sha256(
                    (Path(manifest["yunshu_src"]).parent / entry.name).read_bytes()
                ).digest()
                if expected != actual:
                    raise ValueError("pinned source file changed: " + entry.name)
    import bench016_extra

    bench016_extra.install(be, manifest)
    for key, value in manifest["binaries"].items():
        if not hasattr(be, key) or not Path(value).is_file():
            raise ValueError(f"invalid binary {key}: {value}")
        setattr(be, key, value)
    for engine, info in manifest["engines"].items():
        be.ENGINES[engine] = dataclasses.replace(
            be.ENGINES[engine], product=info["product"]
        )
    # Existing version helpers describe the old snapshots. Bind this run to its audited artifacts.
    original_tree_version = be.tree_version

    def tree_version(src):
        if str(src) == manifest["yunshu_src"]:
            return manifest["yunshu_version"], manifest["yunshu_sha"]
        return original_tree_version(src)

    be.tree_version = tree_version
    old_version = be.engine_version

    def version(engine, src=""):
        info = manifest["engines"].get(engine) or manifest.get("extra_engines", {}).get(
            engine
        )
        return (
            (info["version"], info.get("git_sha")) if info else old_version(engine, src)
        )

    be.engine_version = version
    original_meta = be.meta_row

    def meta(engine, **kwargs):
        row = original_meta(engine, **kwargs)
        info = manifest["engines"].get(engine) or manifest.get("extra_engines", {}).get(
            engine
        )
        if info:
            row["artifact_sha256"] = info["sha256"]
        return row

    be.meta_row = meta


def cell(manifest: dict, argv: list[str]) -> int:
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    vm = subprocess.check_output(["vm_stat"], text=True)
    processes = subprocess.check_output(["ps", "-axo", "pid=,rss=,command="], text=True)
    report = {
        "swap": swap,
        "vm_stat": vm,
        "owned_engine_processes_at_start": [],
        "errors": admission(swap, processes),
    }
    print(json.dumps({"admission": report}), flush=True)
    if report["errors"]:
        return 2
    configure(manifest)
    import tfbench

    sys.argv = ["tfbench.py", *argv]
    return tfbench.main() or 0


def controller(manifest: dict, path: Path) -> int:
    build = path.parent
    configure(manifest)
    rs.configure()
    bs.BUILD = rs.BUILD = build
    bs.LABEL_PREFIX = manifest["label"]
    bs.PRIORITY = 1
    for engine in manifest["order"]:
        bs._STARTUP_S.setdefault(engine, 420 if engine.startswith("fn-") else 90)
    out = build / "runs"
    out.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "GPUQ_DIR": str(QUEUE),
        "GPUQ_OWNER": "bench016",
        "TFB_WORK": "/Volumes/P5Plus/yunshu-build/snapshot014/corpus",
        "TFB_EXACT_PROMPTS": "1",
        "TFB_EXPECT_YUNSHU_SHA": manifest["yunshu_sha"],
        "TFB_YUNSHU_SRC": manifest["yunshu_src"],
        "TMPDIR": str(build),
        "UV_CACHE_DIR": "/Volumes/P5Plus/yunshu-test-envs/bench016/cache",
    }
    state = (
        json.loads((build / "state.json").read_text())
        if (build / "state.json").exists()
        else {}
    )
    jobs = rs.pilots(manifest["order"], out)
    # Rotate which engine starts each repetition; engines remain interleaved per depth.
    for rep in range(3):
        order = manifest["order"][rep:] + manifest["order"][:rep]
        for depth in (32768, 65536, 131072):
            for engine in order:
                jobs.append(
                    rs.make_job(
                        engine, f"L{depth}", depth, ("code",), rep, out, "decode"
                    )
                )
    blocked = set()
    failed = set()
    for job in jobs:
        if job.engine in blocked:
            continue
        for attempt in range(1, 4):
            name = f"{job.name}-try{attempt}"
            output = out / job.engine / f"{name}.jsonl"
            output.parent.mkdir(exist_ok=True, parents=True)
            job.out = output
            job.env = {**env, "TFB_OUT": str(build / "work" / name)}
            job.argv = bs.tfbench_argv(job, output)
            job.argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(path),
                "cell",
                *job.argv[2:],
            ]
            job.argv += [
                "--env",
                "YUNSHU_CONSOLE=0",
                "--env",
                f"YUNSHU_VLM_APC_DISK_DIR={build}/apc",
            ]
            old = state.get(name, {})
            jid = old.get("id")
            if not jid:
                cmd = bs.submit_args(job, attempt)
                # Unique attempt labels across repetitions and failed admission reruns.
                label_index = cmd.index("--label") + 1
                cmd[label_index] += f"-try{attempt}"
                sub = subprocess.run(cmd, env=job.env, capture_output=True, text=True)
                if sub.returncode:
                    raise RuntimeError(sub.stderr)
                jid = sub.stdout.split()[-1]
                state[name] = {"id": jid, "out": str(output)}
                (build / "state.json").write_text(json.dumps(state, indent=2))
                print("SUBMITTED", name, jid, flush=True)
            while True:
                subprocess.run(
                    [GPUQ, "wait", "--max-seconds", "55", jid], env=env, check=False
                )
                receipt = json.loads((QUEUE / "jobs" / f"{jid}.json").read_text())
                if receipt["state"] not in ACTIVE:
                    break
            subprocess.run(
                [GPUQ, "digest", "--label-prefix", manifest["label"], "--since", "48h"],
                env=env,
                check=False,
            )
            log = QUEUE / "logs" / f"{jid}.log"
            if log.exists():
                print(
                    "\n".join(log.read_text(errors="replace").splitlines()[-20:]),
                    flush=True,
                )
            errors = verdict(job, bs.read_rows(output), receipt)
            state[name].update(errors=errors, state=receipt["state"])
            (build / "state.json").write_text(json.dumps(state, indent=2))
            print("VERDICT", name, errors or "PASS", flush=True)
            if not errors:
                failed.discard(job.name)
                break
            failed.add(job.name)
            if receipt.get("contended"):
                continue
            blocked.add(job.engine)
            break
    (build / "controller-complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "pass": not blocked and not failed,
                "blocked": sorted(blocked),
                "failed": sorted(failed),
            }
        )
    )
    return int(bool(blocked or failed))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("mode", choices=["cell", "controller"])
    args, rest = ap.parse_known_args()
    manifest = json.loads(args.manifest.read_text())
    return (
        cell(manifest, rest)
        if args.mode == "cell"
        else controller(manifest, args.manifest.resolve())
    )


if __name__ == "__main__":
    raise SystemExit(main())
