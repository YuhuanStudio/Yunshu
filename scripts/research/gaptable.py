#!/usr/bin/env python3
"""Pinned, interleaved cross-engine cells. Submit each cell through gpuq; never run all inline.

Full plan: gaptable.py plan --manifest FILE --out DIR --src FINAL_MAIN_TREE
Smoke: gaptable.py run --manifest FILE --out DIR --src TREE --engine NAME --smoke
"""

import argparse
import dataclasses
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ENGINES = (
    "yunshu-main",
    "yunshu-tf4",
    "tf-native",
    "splash-latest",
    "splash-noane",
    "llamacpp-mtp",
    "llamacpp-dflash",
    "omlx-latest",
    "mlx-serve",
    "kapitsa",
)


def cells(smoke=False):
    result = [
        ("decode", c, k)
        for c in ((1024, 32768) if smoke else (1024, 32768, 131072))
        for k in (("code",) if smoke else ("code", "prose"))
    ]
    return result if smoke else result + [("conc", 1024, "mixed")]


def schedule(engines, reps=3):
    # Rotate order between reps; one cell per job allows actual cross-engine interleaving.
    return [
        (r, e, part, ctx, kind)
        for part, ctx, kind in cells()
        for r in range(reps)
        for e in engines[r % len(engines) :] + engines[: r % len(engines)]
    ]


def configure(be, manifest):
    import bench016_extra

    for key, value in manifest["binaries"].items():
        setattr(be, key, value)
    bench016_extra.install(be, manifest)
    for engine, info in manifest["engines"].items():
        be.STATIC_VERSIONS[engine] = (info["version"], info.get("git_sha"))
        if engine in be.ENGINES:
            be.ENGINES[engine] = dataclasses.replace(
                be.ENGINES[engine], product=info["product"]
            )
    be.ENGINES["splash-noane"] = dataclasses.replace(
        be.ENGINES["splash-latest"], id="splash-noane"
    )
    be.STATIC_VERSIONS["splash-noane"] = be.STATIC_VERSIONS["splash-latest"]
    launch, detect, paths = be.build_launch, be.detect_mode, be.required_paths

    def build(engine, *args, **kwargs):
        result = launch(
            "splash-latest" if engine == "splash-noane" else engine, *args, **kwargs
        )
        if engine == "splash-noane":
            result.cmd.append("--disable-ane")
            result.flags["ane"] = "disabled explicitly"
        return result

    be.build_launch = build
    be.detect_mode = lambda e, *a: detect(
        "splash-latest" if e == "splash-noane" else e, *a
    )
    be.required_paths = lambda e: paths("splash-latest" if e == "splash-noane" else e)


def validate(path, part, ctx):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not any(r.get("part") == "part_done" for r in rows):
        raise RuntimeError(f"incomplete result: {path}")
    data = [r for r in rows if r.get("part") == part]
    if not data:
        raise RuntimeError("no measured rows")
    for row in data:
        if part == "decode":
            if row.get("cached") is None:
                raise RuntimeError("server omitted cached token accounting")
            if (
                ctx == 32768
                and row["phase"] in ("warm", "turn2")
                and row["cached"] <= 0
            ):
                raise RuntimeError(f"invalid warm cell: cached={row['cached']}")
        elif any(v is None for v in row["cached"]):
            raise RuntimeError("server omitted concurrent cache accounting")
        elif len(row["ttfts"]) != 8 or len(row["per_req_dec"]) != 8:
            raise RuntimeError("missing concurrent per-request metrics")
    return data


def run(a):
    os.environ.update(
        TFB_YUNSHU_SRC=str(Path(a.src).resolve() / "python"),
        TFB_OUT=str(Path(a.out).resolve() / "scratch"),
        TFB_PORT_LAST="18996",
    )
    import bench_engines as be

    configure(be, json.loads(Path(a.manifest).read_text()))
    import tfbench as tb

    Path(a.out).mkdir(parents=True, exist_ok=True)
    scratch = Path(a.out).resolve() / "scratch"
    os.environ["TMPDIR"] = str(scratch / "tmp")
    for part, ctx, kind in cells(True) if a.smoke else [(a.part, a.ctx, a.kind)]:
        dest = Path(a.out) / f"{a.engine}-{part}-{ctx}-{kind}-r{a.rep}.jsonl"
        if dest.exists():
            raise FileExistsError(dest)  # never silently append a second run
        args = [
            "tfbench",
            "--engine",
            a.engine,
            "--part",
            part,
            "--rep",
            str(a.rep),
            "--out",
            str(dest),
            "--tag",
            dest.stem,
            "--ctx-tokens",
            str(ctx + 4096),
            "--idle-s",
            "0",
            "--parallel",
            "8" if part == "conc" else "1",
            "--env",
            "YUNSHU_CONSOLE=0",
            "--env",
            "YUNSHU_VLM_APC_DISK=0",
            "--env",
            "YUNSHU_VLM_DRAFT=" + tb.D,
        ]
        if part == "decode":
            args += [
                "--only-ctx",
                str(ctx),
                "--only-kind",
                kind,
                "--decode-tokens",
                "512",
                "--long-ask",
                "--turn2-tokens",
                "512",
                "--phases",
                "cold,warm,turn2" if ctx == 32768 else "cold",
            ]
        else:
            args += [
                "--conc-ns",
                "8",
                "--conc-ctx",
                "1024",
                "--conc-trials",
                "1",
                "--conc-tokens",
                "256",
            ]
        if a.dry_run:
            args += ["--dry-run"]
        sys.argv = args
        (scratch / "tmp").mkdir(parents=True, exist_ok=True)
        try:
            tb.main()
            if not a.dry_run:
                evidence = []
                for log in (scratch / "out").glob("*.log"):
                    evidence.extend(
                        line
                        for line in log.read_text(errors="replace").splitlines()
                        if re.search(
                            r"prompt cache: .*GiB|streams of .* tokens|DFlash drafter (loaded|attached)|Neural Engine FFN split|(?i:kflash)",
                            line,
                        )
                    )
                with dest.open("a") as f:
                    f.write(
                        json.dumps(
                            dict(
                                part="launch_evidence", engine=a.engine, lines=evidence
                            )
                        )
                        + "\n"
                    )
                validate(dest, part, ctx)
        except Exception as exc:
            logs = []
            for log in (Path(a.out) / "scratch" / "out").glob("*.log"):
                logs.append(
                    dict(file=log.name, tail=log.read_text(errors="replace")[-16000:])
                )
            with dest.open("a") as f:
                f.write(
                    json.dumps(dict(part="failure", error=str(exc), logs=logs)) + "\n"
                )
            raise
        finally:
            # Homes may contain symlinks to checkpoints; rmtree does not follow them.
            scratch = Path(a.out) / "scratch"
            if scratch.exists():
                shutil.rmtree(scratch)
    print(json.dumps({"complete": True, "engine": a.engine, "rep": a.rep}))


def summary(out):
    """Summarize completed independent reps; never collapse them to the last row."""
    groups = {}
    for path in Path(out).glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if any(r.get("part") == "failure" for r in rows):
            continue
        if not any(r.get("part") == "part_done" and r.get("complete") for r in rows):
            continue
        for r in rows:
            if r.get("part") == "decode":
                values = {"TTFT s": r["ttft_s"]}
                if r["phase"] == "cold":
                    values["decode tok/s"] = r["dec_tps"]
                cell = f"{r['ctx']} {r['kind']} {r['phase']}"
            elif r.get("part") == "conc":
                values = {
                    "aggregate tok/s": r["agg_tps"],
                    "per-request decode tok/s": statistics.median(r["per_req_dec"]),
                    "per-request TTFT s": statistics.median(r["ttfts"]),
                }
                cell = "8-concurrent 1K"
            else:
                continue
            for field, value in values.items():
                key = (r["engine"], cell, field)
                rep = r["snapshot_rep"]
                if rep in groups.setdefault(key, {}):
                    raise ValueError(f"duplicate rep {rep}: {key}")
                groups[key][rep] = value
    lines = [
        "| engine | cell | metric | median | min–max | reps | status |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for (engine, cell, field), reps in sorted(groups.items()):
        values = list(reps.values())
        status = "3+ reps" if len(reps) >= 3 else "direction only"
        lines.append(
            f"| {engine} | {cell} | {field} | {statistics.median(values):.3f} | "
            f"{min(values):.3f}–{max(values):.3f} | {len(reps)} | {status} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "plan", "summary"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--src", required=True, help="final main tree; never a stale pinned tree"
    )
    p.add_argument("--engine", choices=ENGINES)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--part", choices=("decode", "conc"), default="decode")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--kind", choices=("code", "prose", "mixed"), default="code")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.action == "summary":
        print(summary(a.out))
        return
    if a.action == "run":
        if not a.engine:
            p.error("run requires --engine")
        return run(a)
    # Emit a reviewable plan; lead submits only after final-main merges.
    sha = subprocess.check_output(
        ["git", "-C", a.src, "rev-parse", "HEAD"], text=True
    ).strip()
    jobs = []
    for rep, engine, part, ctx, kind in schedule(list(ENGINES)):
        jobs.append(
            dict(
                rep=rep,
                engine=engine,
                part=part,
                ctx=ctx,
                kind=kind,
                argv=[
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run",
                    "--manifest",
                    a.manifest,
                    "--out",
                    a.out,
                    "--src",
                    a.src,
                    "--engine",
                    engine,
                    "--rep",
                    str(rep),
                    "--part",
                    part,
                    "--ctx",
                    str(ctx),
                    "--kind",
                    kind,
                ],
            )
        )
    print(json.dumps(dict(yunshu_sha=sha, repetitions=3, jobs=jobs), indent=2))


if __name__ == "__main__":
    main()
