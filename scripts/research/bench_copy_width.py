"""Quiet gpuq copy-width sweep; each arm owns a server and a result file.

Run through gpuq with --expect-complete. Failed arms do not hide later arms.
The explicit MTP override prevents an isolated HOME from choosing another draft.
"""

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import tfbench as bench
from spec_bench_snapshot import freeze, refuse_contended


def main():
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--cost-curve", type=Path)
    ap.add_argument("--compare-policy", action="store_true")
    ap.add_argument(
        "--cost-setting",
        action="store_true",
        help="compare the built-in request-local cost setting with fixed widths",
    )
    a = ap.parse_args()
    if a.cost_setting and a.cost_curve is not None:
        ap.error("--cost-setting and --cost-curve are separate experiments")
    if a.compare_policy and a.cost_curve is None:
        ap.error("--compare-policy needs --cost-curve")
    if refuse_contended(a.output):
        return 0  # gpuq's latched contention makes wait return 3
    a.output.parent.mkdir(parents=True, exist_ok=True)
    source, fingerprint = freeze(a.output)
    bench.YUNSHU_SRC = str(source)
    bench.OUT = a.output.parent / "servers"
    policy = {"enabled": False}
    if a.cost_curve is not None:
        popen = bench.subprocess.Popen

        def policy_server(cmd, **kwargs):
            if policy["enabled"] and cmd[0] == bench.YUNSHU_BIN:
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("copy_cost_server.py")),
                    "--cost-curve",
                    str(a.cost_curve),
                    "--source-dir",
                    str(source),
                    *cmd[1:],
                ]
            return popen(cmd, **kwargs)

        bench.subprocess.Popen = policy_server
    failed = False
    arms = [
        (rows, enabled)
        for rows in a.rows
        for enabled in (
            [False, True]
            if a.compare_policy or a.cost_setting
            else [a.cost_curve is not None]
        )
    ]
    with a.output.open("x") as summary:
        bench.emit(summary, part="snapshot", **fingerprint)
        for rep in range(a.reps):
            # Rotate/reverse: neither thermal drift nor caches always favor an arm.
            order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
            if rep % 2:
                order.reverse()
            for rows, policy_on in order:
                policy["enabled"] = policy_on
                suffix = "-cost" if policy_on else ""
                result = a.output.with_name(
                    f"{a.output.stem}-w{rows}-r{rep}{suffix}.jsonl"
                )
                srv = None
                rc = 1
                record = dict(
                    rows=rows,
                    rep=rep,
                    output=str(result),
                    cost_curve=str(a.cost_curve)
                    if policy_on and a.cost_curve
                    else None,
                    cost_policy="setting"
                    if policy_on and a.cost_setting
                    else "curve"
                    if policy_on
                    else "fixed",
                )
                try:
                    with result.open("x") as out:
                        if bench.was_contended():
                            raise RuntimeError("CPU contended; arm skipped")
                        env = {
                            "YUNSHU_SPEC_COPY_ROWS": str(rows),
                            "YUNSHU_VLM_DRAFT": "mtp",
                        }
                        if a.cost_setting:
                            env["YUNSHU_SPEC_COPY_COST"] = "1" if policy_on else "0"
                        srv = bench.Srv("yunshu", env, result.stem)
                        bench.emit(
                            out,
                            part="session",
                            env=env,
                            load=os.getloadavg(),
                            **fingerprint,
                        )
                        for _ in range(2):
                            bench.send(srv.url, bench.req(srv.model, "Say hi.", 24))
                        args = argparse.Namespace(
                            only_ctx=a.contexts, only_kind=None, engine="yunshu"
                        )
                        bench.part_decode(srv, out, args)
                        if a.agent:
                            bench.part_agent(srv, out, args)
                        if bench.was_contended():
                            raise RuntimeError(
                                "CPU contended during arm; results untrustworthy"
                            )
                        mode = [
                            line.strip()
                            for line in srv.log.read_text().splitlines()
                            if "Speculative decoding:" in line
                        ]
                        if not mode or not any("mtp" in line.lower() for line in mode):
                            raise RuntimeError(f"MTP mode not proved: {mode}")
                        if (
                            policy_on
                            and a.cost_setting
                            and "'copy_cost': True" not in srv.log.read_text()
                        ):
                            raise RuntimeError(
                                "built-in copy cost policy did not engage"
                            )
                        if (
                            policy_on
                            and a.cost_curve is not None
                            and "Copy cost policy:" not in srv.log.read_text()
                        ):
                            raise RuntimeError("copy cost policy did not engage")
                        bench.emit(out, complete=True, mode=mode, load=os.getloadavg())
                    rc = 0
                except Exception:
                    record["error"] = traceback.format_exc()
                    failed = True
                finally:
                    if srv is not None:
                        srv.kill()
                bench.emit(summary, **record, rc=rc)
                print(json.dumps(dict(record, rc=rc)), flush=True)
        bench.emit(summary, complete=True, success=not failed)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
