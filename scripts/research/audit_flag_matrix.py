#!/usr/bin/env python3
"""Consolidated p-1 flag measurements with pinned source and fail-closed receipts.

--dry-run builds/validates CPU inputs and every subprocess argv. --smoke runs
short tiny-checkpoint correctness only. Timing run requires a successful smoke
receipt, emits every arm's rc, and continues after arm failures. Use gpuq only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
CAPTURES = MAIN / "docs/research/runs/2026-09-30-agent-census"
CORPORA = MAIN / "reference/omlx/omlx/admin/bench_corpora"
MODEL = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
TINY = Path("/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16")
DRAFT = Path("/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2")
TEXT = Path("/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-bf16")
TEXT_DRAFT = Path("/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit")
AREAS = ("tree", "row_exact", "tools", "driver", "external")


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def source_sha():
    manifest = [
        (str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((ROOT / "python").rglob("*.py"))
    ]
    return sha(manifest)


def captures():
    result = []
    for client, session in (
        ("claude", "cc_bash_edit"),
        ("codex", "cx_shell"),
        ("opencode", "oc_bash"),
    ):
        path = CAPTURES / session / "requests.jsonl"
        record = next(
            json.loads(line)
            for line in path.read_text().splitlines()
            if (json.loads(line).get("body") or {}).get("tools")
        )
        body = record["body"]
        # No headers, credentials or client state are retained in results.
        body.pop("previous_response_id", None)
        if client == "claude":
            body["thinking"] = {"type": "disabled"}
            body["max_tokens"] = 512
            body.pop("output_config", None)
        elif client == "codex":
            body["max_output_tokens"] = 512
            body["store"] = False
            body["reasoning"] = {"effort": "low"}
        else:
            body["max_tokens"] = 512
        body["stream"] = False
        result.append((client, record["path"].split("?")[0], body))
    return result


def request(url, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path,
        None if body is None else json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer k",
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def free_port():
    for port in range(18990, 19000):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("no free audit port")


def launcher(port):
    # Instrument only this research server, never production engine files.
    import uvicorn

    from yunshu_engine import dflash_tree, mtp_tree
    from yunshu_engine.kernels import omlx
    from yunshu_engine.vlm_engine import VLMEngine

    if os.environ.get("AUDIT_TINY_SMOKE") == "1":
        # The bf16 tiny checkpoint has no packed LaneLinear projections.
        # Exercise its narrow supported tree; the 27B job retains 15 nodes.
        mtp_tree.NODES = 7
    count = [0]
    for module in (dflash_tree, mtp_tree):
        original = module.search_tree

        def search(*a, _original=original, **kw):
            count[0] += 1
            return _original(*a, **kw)

        module.search_tree = search
    events, guide = VLMEngine._runner_events, VLMEngine._tool_guide

    def trace(self, ids, **kw):
        tokens, complete = [], False
        first = last = None
        prompt_digest = sha(ids.tolist())
        start = count[0]
        try:
            for event in events(self, ids, **kw):
                if event[1] is not None:
                    tokens.append(int(event[1]))
                    last = time.perf_counter()
                    first = first or last
                complete = complete or event[3] is not None
                yield event
        finally:
            print(
                "AUDIT_TRACE "
                + json.dumps(
                    {
                        "tokens": tokens,
                        "prompt_digest": prompt_digest,
                        "decode_tps": (len(tokens) - 1) / (last - first)
                        if first and last and last > first
                        else None,
                        "digest": sha(tokens),
                        "complete": complete,
                        "tree_calls": count[0] - start,
                        "row_exact": omlx._STATE["row_exact"],
                        "driver": getattr(self._batch_runner, "driver", None)
                        is not None,
                    }
                ),
                flush=True,
            )

    def traced_guide(self, *a, **kw):
        built = guide(self, *a, **kw)
        print("AUDIT_GUIDE " + json.dumps({"active": built is not None}), flush=True)
        return built

    VLMEngine._runner_events = trace
    VLMEngine._tool_guide = traced_guide
    uvicorn.run(
        "yunshu_gateway.main:app", host="127.0.0.1", port=port, log_level="info"
    )


@contextlib.contextmanager
def server(checkpoint, settings, path):
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("YUNSHU_")}
    env.update(settings)
    env.update(
        AUDIT_TINY_SMOKE="1" if checkpoint == TINY else "0",
        YUNSHU_MODEL=str(checkpoint),
        YUNSHU_AUTH_DISABLED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(ROOT / "python"),
        NO_PROXY="127.0.0.1",
    )
    url = f"http://127.0.0.1:{port}"
    with path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, __file__, "--serve", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError("audit server exited before ready")
                try:
                    models = request(url, "/v1/models", None, timeout=2)["data"]
                    model = models[0]["id"]
                    break
                except (OSError, KeyError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError("server readiness timeout")
            log_text = path.read_text()
            mode = re.findall(
                r"VLM batch runner: [^\n]*draft=(dflash|mtp|off)\b", log_text
            )
            expected = (
                "dflash" if settings.get("YUNSHU_VLM_DRAFT") == str(DRAFT) else "mtp"
            )
            if not mode or mode[-1] != expected:
                raise RuntimeError(f"engaged mode {mode} != {expected}")
            yield url, model, mode[-1]
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()


def trace_rows(path, offset):
    new = path.read_text()[offset:]
    traces = [
        json.loads(line.split("AUDIT_TRACE ", 1)[1])
        for line in new.splitlines()
        if "AUDIT_TRACE " in line
    ]
    guides = [
        json.loads(line.split("AUDIT_GUIDE ", 1)[1])
        for line in new.splitlines()
        if "AUDIT_GUIDE " in line
    ]
    if not traces or not all(row["complete"] for row in traces):
        raise RuntimeError("missing final complete token trace")
    return traces, guides


def quality(response, body):
    from jsonschema import validate

    calls, text = [], ""
    if "content" in response:
        calls = [
            (part.get("name"), part.get("input"))
            for part in response["content"]
            if part.get("type") == "tool_use"
        ]
        text = "".join(part.get("text", "") for part in response["content"])
    elif "output" in response:
        calls = [
            (part.get("name"), part.get("arguments"))
            for part in response["output"]
            if part.get("type") == "function_call"
        ]
        text = json.dumps(response["output"])
    else:
        message = response["choices"][0].get("message") or {}
        calls = [
            (part["function"].get("name"), part["function"].get("arguments"))
            for part in message.get("tool_calls") or []
        ]
        text = message.get("content") or ""
    schemas = {}
    for tool in body.get("tools") or []:
        fn = tool.get("function", tool)
        schemas[fn.get("name")] = fn.get("input_schema", fn.get("parameters", {}))
    malformed = 0
    for name, args in calls:
        try:
            if name not in schemas:
                raise ValueError("unknown tool")
            validate(json.loads(args) if isinstance(args, str) else args, schemas[name])
        except Exception:
            malformed += 1
    leak = bool(re.search(r"<(?:tool_call|function|parameter)\b", text))
    return {
        "calls": len(calls),
        "malformed": malformed,
        "leaked_markup": leak,
        "no_call": not calls,
        "dropped": not calls and leak,
    }


def prompts(checkpoint, contexts):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    result = []
    for corpus in ("code_python", "novel_en"):
        text = (CORPORA / (corpus + ".txt")).read_text()
        ids = tok.encode(text, truncation=False, verbose=False)
        for ctx in contexts:
            prefix = f"AUDIT-{corpus}-{ctx}: "
            body = tok.decode(
                (ids * (ctx // len(ids) + 1))[: ctx - len(tok.encode(prefix)) - 40]
            )
            ask = (
                "\nContinue with more code."
                if corpus == "code_python"
                else "\nDiscuss the themes in your own words without quoting the passage."
            )
            result.append((f"{corpus}-{ctx}", prefix + body + ask))
    return result


def serving_arm(args, area, arm, rep, prepared, captured):
    smoke = args.smoke
    checkpoint = TINY if smoke else MODEL
    setting = {
        "YUNSHU_ROUND_DRIVER": "0",
        "YUNSHU_TOOL_GRAMMAR": "0",
        "YUNSHU_SPEC_TREE": "off",
        "YUNSHU_MTP_ROW_EXACT": "0",
        "YUNSHU_VLM_DRAFT": "mtp" if smoke or area == "driver" else str(DRAFT),
    }
    flag = {
        "tree": "YUNSHU_SPEC_TREE",
        "tools": "YUNSHU_TOOL_GRAMMAR",
        "driver": "YUNSHU_ROUND_DRIVER",
    }[area]
    setting[flag] = (
        ("tree" if area == "tree" else "1")
        if arm
        else ("off" if area == "tree" else "0")
    )
    name = f"{area}-r{rep}-{arm}"
    log = args.out.parent / (name + ".server.log")
    rows = []
    with server(checkpoint, setting, log) as (url, model, mode):
        # Tiny compile warm-up is outside measurements and separate from cases.
        warmup = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "enable_thinking": False,
        }
        request(url, "/v1/chat/completions", warmup)
        cases = [
            (
                label,
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16 if smoke else 128,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                    "enable_thinking": False,
                },
            )
            for label, prompt in prepared
        ]
        if area == "tools":
            cases = []
            for client, route, source in captured:
                for temp in (0.0, 0.7):
                    body = {
                        **source,
                        "model": model,
                        "temperature": temp,
                        "seed": 42,
                        "stream": False,
                    }
                    body["max_output_tokens" if client == "codex" else "max_tokens"] = (
                        32 if smoke else 512
                    )
                    cases.append((f"{client}-{temp}", route, body))
        if area == "driver":
            base = cases[0][2]
            lengths = [512] if smoke else [1024, 32768]
            for length in lengths:
                prompt = next(
                    p for label, p in prepared if label == f"code_python-{length}"
                )
                for batch in [2] if smoke else [2, 4, 8]:
                    cases.append(
                        (
                            f"batch-{length}-{batch}",
                            "/v1/chat/completions",
                            [
                                {
                                    **base,
                                    "messages": [{"role": "user", "content": prompt}],
                                }
                                for _ in range(batch)
                            ],
                        )
                    )
        if area == "driver":
            sample_prompt = prepared[0][1]
            sampled = {
                **cases[0][2],
                "temperature": 0.7,
                "seed": 42,
                "messages": [{"role": "user", "content": sample_prompt}],
            }
            cases.append(("sampled-solo", "/v1/chat/completions", sampled))
            cases.append(
                (
                    "sampled-batch",
                    "/v1/chat/completions",
                    [dict(sampled) for _ in range(2 if smoke else 4)],
                )
            )
        for label, route, body in cases:
            offset = len(log.read_text())
            start = time.perf_counter()
            if isinstance(body, list):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(body)
                ) as pool:
                    response = list(pool.map(lambda b: request(url, route, b), body))
            else:
                response = request(url, route, body)
            wall = time.perf_counter() - start
            traces, guides = trace_rows(log, offset)
            if area == "tree" and arm and not any(t["tree_calls"] for t in traces):
                raise RuntimeError("tree flag set but actual tree search did not run")
            if area == "driver" and arm and not all(t["driver"] for t in traces):
                raise RuntimeError("round driver not engaged")
            if area == "tools" and arm and not any(g["active"] for g in guides):
                raise RuntimeError("tool grammar flag set but per-request guide absent")
            row = {
                "case": label,
                "area": area,
                "arm": arm,
                "rep": rep,
                "engaged_mode": mode,
                "wall_s": wall,
                "traces": traces,
                "response_digest": sha(response),
                "quality": quality(response, body) if area == "tools" else None,
                "usage": [r.get("usage") for r in response]
                if isinstance(response, list)
                else response.get("usage"),
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "traces"}), flush=True
            )
    return rows


def subprocess_arm(args, area, arm, rep, context=None):
    name = f"{area}-r{rep}-{arm}" + (f"-ctx{context}" if context else "")
    path = args.out.parent / (name + ".jsonl")
    if area == "external":
        command = [
            sys.executable,
            str(ROOT / "scripts/research/audit_external_draft.py"),
            "--target",
            str(TEXT),
            "--draft",
            str(TEXT_DRAFT),
            "--out",
            str(path),
            "--tokens",
            "16" if args.smoke else "128",
        ]
        if args.smoke:
            command.append("--smoke")
        # This subprocess itself performs all three interleaved repetitions.
    else:
        command = [
            sys.executable,
            str(ROOT / "scripts/research/sweep_mtp_depth.py"),
            str(TINY if args.smoke else MODEL),
            "6",
            "--context=" + str(context),
            "--tasks=code,prose",
            "--tokens=" + ("16" if args.smoke else "128"),
            "--yunshu-kernels=" + ("row_exact" if arm else "exact"),
        ]
        if not arm:
            command += ["--invariant", "--lane-linear", "--ragged-lane", "--mtp-lane"]
    if args.dry_run:
        command.append("--dry-run")
    with path.open("w") as output:
        # External writes its own JSON output. Keep stdout/log separate.
        log = path.with_suffix(".log") if area == "external" else path
        if area == "external":
            with log.open("w") as external_log:
                rc = subprocess.run(
                    command, cwd=ROOT, stdout=external_log, stderr=subprocess.STDOUT
                ).returncode
        else:
            rc = subprocess.run(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
            ).returncode
    if rc:
        raise RuntimeError(f"{name} rc={rc}; read {log}")
    if area == "external":
        data = json.loads(path.read_text())
        if not data.get("complete"):
            raise RuntimeError("external output incomplete")
        return data["rows"]
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]
    if not records or not records[-1].get("complete"):
        raise RuntimeError("sweep output incomplete")
    if not args.dry_run:
        cells = [r for r in records if "task" in r]
        if len(cells) != 4 or any(
            r.get("parity") is not True or not r.get("token_digest") for r in cells
        ):
            raise RuntimeError("missing cells, full-token parity, or digest")
    return records


def summarize(arms):
    """Evidence table only: never interpret a missing/failed cell as a win."""
    summary = {}
    for area in ("tree", "driver"):
        grouped = {}
        for arm in arms:
            if arm["area"] != area or arm.get("rc") != 0:
                continue
            for row in arm.get("rows", []):
                grouped.setdefault(row["case"], {0: [], 1: []})[arm["arm"]].append(row)
        cells = []
        for case, pair in sorted(grouped.items()):
            parity = bool(pair[0] and pair[1]) and all(
                sorted(
                    (trace["prompt_digest"], trace["digest"]) for trace in row["traces"]
                )
                == sorted(
                    (trace["prompt_digest"], trace["digest"])
                    for trace in pair[0][0]["traces"]
                )
                for rows in pair.values()
                for row in rows
            )
            if not pair[0] or not pair[1]:
                cells.append({"case": case, "eligible": False, "error": "missing arm"})
                continue
            base = statistics.median(row["wall_s"] for row in pair[0])
            candidate = statistics.median(row["wall_s"] for row in pair[1])
            cells.append(
                {
                    "case": case,
                    "runs_off": len(pair[0]),
                    "runs_on": len(pair[1]),
                    "token_parity": parity,
                    "wall_off_s": base,
                    "wall_on_s": candidate,
                    "wall_speedup": base / candidate,
                }
            )
        summary[area] = cells
    summary["tools"] = [
        {
            "arm": arm["arm"],
            "rep": arm["rep"],
            "case": row["case"],
            "wall_s": row["wall_s"],
            **row["quality"],
        }
        for arm in arms
        if arm["area"] == "tools" and arm.get("rc") == 0
        for row in arm.get("rows", [])
    ]
    return summary


def driver_full_validation(args, rep=0):
    # The owner's existing full exit matrix supplies the remaining gates.
    script = ROOT / "scripts/research/validate_round_driver.sh"
    output = args.out.parent / f"driver-full-r{rep}"
    env = dict(
        os.environ,
        M=str(MODEL),
        PY=sys.executable,
        OUT=str(output),
        PHASE="list" if args.dry_run else "all",
        BASE_PORT="18990",
        PYTHONPATH=str(ROOT / "python"),
    )
    log = args.out.parent / f"driver-full-r{rep}.log"
    with log.open("w") as stream:
        rc = subprocess.run(
            ["zsh", str(script)],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode
    text = log.read_text()
    if rc or re.search(r"FAILED|NOT ENGAGED|server .*failed|unknown phase", text):
        raise RuntimeError(f"full driver matrix failed rc={rc}; read {log}")
    if not args.dry_run:
        required = (
            "parity-1k.jsonl",
            "rows-1k.jsonl",
            "parity-32k.jsonl",
            "rows-32k.jsonl",
            "context-batch.jsonl",
            "mixed-load.jsonl",
            "matrix.jsonl",
        )
        if any(not (output / name).is_file() for name in required):
            raise RuntimeError("full driver matrix output missing")
        accuracy = list(output.glob("mmlu-rd?-?.jsonl"))
        if len(accuracy) != 6:
            raise RuntimeError("driver MMLU-Pro 300 paired shards missing")
    return {"rc": rc, "log": str(log), "full_matrix": str(script), "complete": True}


def validate_smoke(receipt, digest, areas):
    expected = sum(1 if area == "external" else 2 for area in areas)
    return (
        receipt.get("complete") is True
        and receipt.get("smoke") is True
        and receipt.get("dry_run") is False
        and receipt.get("source_sha") == digest
        and receipt.get("areas") == areas
        and len(receipt.get("arms", [])) == expected
        and all(arm.get("rc") == 0 and arm.get("rows") for arm in receipt["arms"])
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--areas", nargs="+", choices=AREAS, default=list(AREAS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-smoke", type=Path)
    ap.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve is not None:
        launcher(args.serve)
        return
    if args.out is None:
        ap.error("--out required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = source_sha()
    if not args.smoke and not args.dry_run:
        receipt = (
            json.loads(args.require_smoke.read_text()) if args.require_smoke else {}
        )
        if not validate_smoke(receipt, digest, args.areas):
            raise ValueError(
                "matching successful tiny smoke required before any 27B arm"
            )
    contexts = [512] if args.smoke or args.dry_run else [1024, 8192, 32768, 131072]
    prepared = prompts(TINY if args.smoke else MODEL, contexts)
    captured = captures()
    if args.dry_run:
        from transformers import AutoTokenizer

        from yunshu_engine.tool_call_grammar import compile_tool_grammar

        tok = AutoTokenizer.from_pretrained(TINY, local_files_only=True)
        for client, _, body in captured:
            grammar = compile_tool_grammar(body["tools"], tok, len(tok))
            if grammar is None:
                raise ValueError(
                    f"{client}: tool grammar cannot engage for captured tools"
                )
    result = {
        "complete": False,
        "source_sha": digest,
        "areas": args.areas,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "arms": [],
        "load_start": os.getloadavg(),
        "capture_digests": {client: sha(body) for client, _, body in captured},
        "corpus_sha256": {
            name: hashlib.sha256((CORPORA / (name + ".txt")).read_bytes()).hexdigest()
            for name in ("code_python", "novel_en")
        },
    }
    reps = 1 if args.smoke or args.dry_run else 3
    try:
        for rep in range(reps):
            for area in args.areas:
                if area == "external" and rep:
                    continue
                arms = (0, 1) if rep % 2 == 0 else (1, 0)
                if area == "external":
                    arms = (0,)
                for arm in arms:
                    for ctx in contexts if area == "row_exact" else [None]:
                        receipt = {
                            "area": area,
                            "arm": arm,
                            "rep": rep,
                            "context": ctx,
                            "rc": 1,
                        }
                        try:
                            if area in ("external", "row_exact"):
                                receipt["rows"] = subprocess_arm(
                                    args, area, arm, rep, ctx
                                )
                            elif args.dry_run:
                                receipt["cases"] = (
                                    len(captured) * 2
                                    if area == "tools"
                                    else len(prepared)
                                )
                            else:
                                receipt["rows"] = serving_arm(
                                    args, area, arm, rep, prepared, captured
                                )
                            receipt["rc"] = 0
                        except Exception as exc:
                            receipt["error"] = str(exc)
                        result["arms"].append(receipt)
                        args.out.write_text(json.dumps(result, indent=2) + "\n")
                        print(
                            json.dumps(
                                {k: v for k, v in receipt.items() if k != "rows"}
                            ),
                            flush=True,
                        )
        result["summary"] = summarize(result["arms"])
        if "driver" in args.areas and not args.smoke:
            cells = result["summary"]["driver"]
            singles = [
                c for c in cells if c["case"].startswith(("code_python-", "novel_en-"))
            ]
            if args.dry_run or (
                singles
                and all(
                    c.get("token_parity") and c.get("wall_speedup", 0) >= 1
                    for c in singles
                )
            ):
                try:
                    result["driver_full"] = {
                        "rc": 0,
                        "runs": [
                            driver_full_validation(args, rep)
                            for rep in range(1 if args.dry_run else 3)
                        ],
                    }
                except Exception as exc:
                    result["driver_full"] = {"rc": 1, "error": str(exc)}
            else:
                result["driver_full"] = {
                    "skipped": "single-request speed/parity gate did not pass; current driver is not eligible for promotion"
                }
        result["complete"] = (
            all(arm["rc"] == 0 for arm in result["arms"])
            and result.get("driver_full", {}).get("rc", 0) == 0
        )
        result["load_end"] = os.getloadavg()
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}), flush=True)
    if not result["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
