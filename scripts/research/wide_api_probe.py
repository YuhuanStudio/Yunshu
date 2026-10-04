"""API same-checkpoint main/candidate with raw-ID and engaged-mode evidence."""

import argparse, hashlib, json, signal, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--contexts", nargs="+", type=int, default=[256, 1024, 2048, 4096, 8192])
p.add_argument("--budgets", nargs="+", default=["unset", "4096"])
p.add_argument("--rep-offset", type=int, default=0)
p.add_argument("--tiny", action="store_true")
p.add_argument("--no-plain", action="store_true")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if a.dry_run:
    print(
        json.dumps(
            dict(
                complete=True,
                dry_run=True,
                cells=len(a.contexts) * len(a.budgets) * 2 * a.reps,
            )
        )
    )
    sys.exit()
ROOT = Path("/Volumes/P5Plus/yunshu-build/codex/worktrees/wide-lead")
sys.path.insert(0, str(ROOT / "scripts/research"))
from spec_bench_snapshot import freeze
from bench_copy_width import start_server
import tfbench as b

source, fingerprint = freeze(a.output)
# Fixed main baseline of this worktree, never a moving shared checkout.
base = a.output.with_name(a.output.stem + "-main")
import shutil

shutil.copytree(source, base)
for name in ["yunshu_engine/dflash_fast.py", "yunshu_engine/dflash_copy.py"]:
    (base / name).write_bytes(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "b6a36e3e:python/" + name]
        )
    )
boot = a.output.with_name(a.output.stem + "-boot")
boot.mkdir()
(boot / "sitecustomize.py").write_text("""import json
from yunshu_engine.vlm_batch_runner import VLMBatchRunner
from yunshu_engine import dflash_fast,dflash_copy,mtp_lane
CURRENT={}
from pathlib import Path
CLEAR=Path(__file__).with_name('clear-next')
PLAIN=Path(__file__).with_name('plain-next')
print('wide7_certificate '+json.dumps(dict(minimum=getattr(dflash_fast,'CONTEXT_MIN',512),limit=getattr(dflash_fast,'CONTEXT_LIMIT',1536))),flush=True)
raw=VLMBatchRunner.iter_tokens
fast=dflash_fast.rounds
resume=getattr(dflash_copy,'resume_chain',None)
def trace_fast(*args,**kw):
 key=id(args[2]);entry=dict(context=len(mtp_lane._STATE['context']),maximum=kw['max_tokens']);CURRENT[key]=entry
 print('wide7_fast_enter',flush=True)
 try:yield from fast(*args,**kw)
 finally:
  if CURRENT.get(key) is entry:CURRENT.pop(key)
dflash_fast.rounds=trace_fast
if resume:
 def trace_resume(*args,**kw):
  entry=CURRENT[id(args[3])]
  generated=entry['maximum']-kw['max_tokens']+1
  print('wide7_chain_handoff '+json.dumps(dict(context=entry['context'],generated=generated,total=entry['context']+generated,limit=dflash_fast.CONTEXT_LIMIT)),flush=True)
  yield from resume(*args,**kw)
 dflash_copy.resume_chain=trace_resume
def trace(self,*args,**kw):
 if PLAIN.exists():
  kw['allow_draft']=False;PLAIN.unlink();print('wide7_plain_enter',flush=True)
 if CLEAR.exists():
  if self.apc_manager is not None:self.apc_manager.clear()
  CLEAR.unlink()
 ids=[]
 try:
  for token in raw(self,*args,**kw):ids.append(int(token));yield token
 finally:print('wide7_ids '+json.dumps(ids),flush=True)
VLMBatchRunner.iter_tokens=trace
""")
b.OUT = a.output.parent / (a.output.stem + "-servers")
b.YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(SystemExit(128 + s)))
parity = True
refs = {}
with a.output.open("x") as out:

    def emit(r):
        out.write(json.dumps(r) + "\n")
        out.flush()
        print(
            json.dumps({k: v for k, v in r.items() if k not in ("ids", "_text")}),
            flush=True,
        )

    emit(dict(part="snapshot", baseline="b6a36e3e", **fingerprint))
    for rep in range(a.rep_offset, a.rep_offset + a.reps):
        for arm in ["main", "candidate"] if rep % 2 == 0 else ["candidate", "main"]:
            server = None
            try:
                b.YUNSHU_SRC = str(boot) + ":" + str(base if arm == "main" else source)
                env = {
                    "YUNSHU_VLM_DRAFT": b.D,
                    "YUNSHU_SPEC_TREE": "auto",
                    "YUNSHU_SPEC_COPY_ROWS": "16",
                    "YUNSHU_VLM_APC_DISK": "0",
                    "YUNSHU_AUTH_DISABLED": "1",
                }
                if a.tiny:
                    env["YUNSHU_VLM_DRAFT"] = "off"
                server = (
                    start_server(
                        "yunshu", env, f"wide7-{a.output.stem}-{arm}-{rep}", **{}
                    )
                    if not a.tiny
                    else b.Srv(
                        "yunshu",
                        env,
                        f"wide7-tiny-{arm}-{rep}",
                        model="/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16",
                    )
                )
                b.send(server.url, b.req(server.model, "Say hi.", 8))
                certificate = json.loads(
                    [
                        line.split(" ", 1)[1]
                        for line in server.log.read_text().splitlines()
                        if line.startswith("wide7_certificate ")
                    ][-1]
                )
                for ctx in a.contexts:
                    for task in ["code", "prose"]:
                        # Standard prompts where available, fixed deterministic padding otherwise.
                        file = (
                            Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
                            / f"{task}-{ctx}.txt"
                        )
                        text = (
                            file.read_text()
                            if file.exists()
                            else "".join(
                                f"Sensor {i}: pressure {i * 37 % 1000}.\n"
                                for i in range(max(1, ctx // 12))
                            )
                            + "\n"
                            + (
                                "Write a Python LRU cache class with get, put, delete and resize. Output code only."
                                if task == "code"
                                else "Explain in detail how a refrigerator works, including compressor and evaporator."
                            )
                        )
                        for budget in a.budgets:
                            body = b.req(
                                server.model,
                                text,
                                int(budget) if budget != "unset" else 4096,
                            )
                            if budget == "unset":
                                body.pop("max_tokens")
                            # Spec miss/hit first, then the same initialized target with drafting disabled.
                            for phase in ["miss", "hit"] + (
                                [] if a.no_plain else ["plain"]
                            ):
                                req = dict(body)
                                if phase == "plain":
                                    (boot / "plain-next").touch()
                                if phase == "miss":
                                    (boot / "clear-next").touch()
                                before = server.log.read_text()
                                r = b.send(server.url, req)
                                log = server.log.read_text()
                                tail = log[len(before) :]
                                # iter_tokens finally may be logged just after [DONE].
                                deadline = time.monotonic() + 3
                                while (
                                    "wide7_ids " not in tail
                                    and time.monotonic() < deadline
                                ):
                                    time.sleep(0.02)
                                    log = server.log.read_text()
                                    tail = log[len(before) :]
                                if phase == "plain":
                                    assert "wide7_plain_enter" in tail and not (
                                        r.get("xy") or {}
                                    ).get("speculative"), (
                                        "plain hook did not engage",
                                        r.get("xy"),
                                    )
                                if (
                                    arm == "candidate"
                                    and phase != "plain"
                                    and not a.tiny
                                ):
                                    expected = (
                                        certificate["minimum"]
                                        <= r["pt"] + 1
                                        < certificate["limit"]
                                    )
                                    assert ("wide7_fast_enter" in tail) == expected, (
                                        "unexpected fast admission",
                                        r["pt"],
                                        certificate,
                                    )
                                assert "VLM runner step failed" not in tail, (
                                    "server failed during request"
                                )
                                ids = json.loads(
                                    [
                                        l.split(" ", 1)[1]
                                        for l in tail.splitlines()
                                        if l.startswith("wide7_ids ")
                                    ][-1]
                                )
                                sha = hashlib.sha256(
                                    json.dumps(ids).encode()
                                ).hexdigest()
                                key = (ctx, task, budget)
                                refs.setdefault(key, sha)
                                equal = sha == refs[key]
                                parity &= equal
                                if phase == "miss" and not a.tiny:
                                    assert r["cached"] == 0, (
                                        "not a real miss",
                                        ctx,
                                        r["cached"],
                                    )
                                if phase == "hit" and not a.tiny:
                                    assert r["cached"] > 0, (
                                        "missing APC hit",
                                        ctx,
                                        phase,
                                    )
                                emit(
                                    dict(
                                        part="http",
                                        arm=arm,
                                        rep=rep,
                                        context=ctx,
                                        task=task,
                                        budget=budget,
                                        phase=phase,
                                        ids=ids,
                                        raw_sha=sha,
                                        parity=equal,
                                        fast="wide7_fast_enter" in tail,
                                        handoff="wide7_chain_handoff" in tail,
                                        handoff_points=[
                                            json.loads(line.split(" ", 1)[1])
                                            for line in tail.splitlines()
                                            if line.startswith("wide7_chain_handoff ")
                                        ],
                                        **{k: v for k, v in r.items() if k != "_text"},
                                    )
                                )
                log = server.log.read_text()
                mode = b.engaged_spec_mode("yunshu", log)
                assert mode == ("off" if a.tiny else "dflash")
                emit(
                    dict(
                        part="mode", arm=arm, rep=rep, engaged=mode, log=str(server.log)
                    )
                )
            finally:
                if server is not None:
                    server.kill()
    emit(dict(complete=True, success=parity))
if not parity:
    raise SystemExit(1)
