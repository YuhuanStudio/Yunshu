"""One isolated cross-engine agentbench task; execute only through gpuq."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import tfbench

sys.path.insert(0, str(Path(__file__).parent / "agentic"))
import run_agentic  # noqa: E402
from proxy import RecordingProxy, api_kind  # noqa: E402


def greedy_body(body):
    request = json.loads(body)
    request["temperature"] = 0
    request["top_p"] = 1
    return json.dumps(request).encode()


class GreedyProxy(RecordingProxy):
    def _read_request_body(self, handler):
        body = super()._read_request_body(handler)
        if body and handler.command == "POST" and api_kind(handler.path):
            body = greedy_body(body)
            if "Content-Length" in handler.headers:
                handler.headers.replace_header("Content-Length", str(len(body)))
            else:
                handler.headers["Content-Length"] = str(len(body))
        return body


def validate(rows, task):
    runs = [row for row in rows if row.get("type") == "run"]
    if len(runs) != 1 or runs[0].get("task") != task:
        return False, "missing or unexpected agent task"
    if runs[0].get("requests", 0) < 1:
        return False, "agent made no generation request (infrastructure failure)"
    if not rows or rows[-1].get("complete") is not True:
        return False, "agent task lacks terminal complete record"
    for request in runs[0].get("request_log", []):
        sampling = request.get("sampling")
        if sampling and (
            sampling.get("temperature") != 0 or sampling.get("top_p") != 1
        ):
            return False, "non-greedy agent generation"
    return True, ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", required=True, choices=list(tfbench.bench_engines.ENGINES)
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {"engine": args.engine, "task": args.task, "complete": "dry-run"}
            )
        )
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    server = tfbench.Srv(
        args.engine, {}, f"agent-{args.engine}-{args.task}", ctx_tokens=140000
    )
    try:
        run_agentic.RecordingProxy = GreedyProxy
        run_agentic.TRIM = (*run_agentic.TRIM, "sampling")
        params = SimpleNamespace(
            output=str(args.out),
            engine_label=args.engine,
            agent="opencode",
            tasks=args.task,
            repeat=1,
            shard_i=0,
            shard_n=1,
            serve=None,
            engine_url=server.url,
            server_pid=server.proc.pid,
            model=server.model,
            proxy_port=tfbench.free_port(),
            checkpoint=tfbench.M,
            server_arg=[],
            context=131072,
            max_output=32768,
            timeout_min=20,
            budget_min=25,
            save_bodies=False,
            keep_work=False,
        )
        run_agentic.cmd_run(params)
        with args.out.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "complete": True,
                        "engine": args.engine,
                        "release_sha": tfbench.bench_engines.engine_version(
                            args.engine, tfbench.YUNSHU_SRC
                        )[1],
                        "engaged_mode": server.engaged_spec_mode,
                    }
                )
                + "\n"
            )
        rows = [json.loads(line) for line in args.out.read_text().splitlines()]
        ok, reason = validate(rows, args.task)
        if not ok:
            raise RuntimeError(reason)
    finally:
        server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
