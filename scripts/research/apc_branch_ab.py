"""APC branch A/B: cached tokens, TTFT and memory when a conversation branches.

Each arm starts a fresh server from its source tree and runs, on one 27B checkpoint:
  build   an agent-style conversation grown turn by turn to ~40K tokens
  a       linear follow-up (extends the last request)
  b       branch at the midpoint (turns 1..4 kept, a new user turn instead of 5..8)
  c-user  two sub-agents sharing a 20K prefix in the first user message
  c-sys   two sub-agents sharing a 20K system prompt
Every measured request uses max_tokens=1, so its wall time is the TTFT. The ideal column is
what a radix cache would hit (the tokens the two requests share). Process footprint, peak
footprint and APC resident bytes are recorded after every step and after idling.

    python scripts/research/apc_branch_ab.py --arm base=/tree --arm cand=/tree \
        --model /path/model --reps 1 --rep-offset 0 --out out.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from memory_ab import GIB, code_doc, metrics  # noqa: E402
from process_memory import process_tree_memory  # noqa: E402

SYSTEM = "You are a coding agent. Answer briefly. Tools: read_file, write_file, run."
TURN_TOKENS = 5000
TURNS = 8


def build_turns(seed: int, turns: int = TURNS, per_turn: int = TURN_TOKENS) -> list:
    """The user messages of the grown conversation (distinct text per turn and seed)."""
    return [
        code_doc(seed * 100 + t, per_turn) + f"\nQuestion {t}: name one function."
        for t in range(turns)
    ]


def branch_messages(history: list, keep_turns: int, seed: int) -> list:
    """The conversation cut after ``keep_turns`` completed turns plus a different user turn."""
    cut = 1 + 2 * keep_turns  # system + (user, assistant) * keep_turns
    return history[:cut] + [
        {"role": "user", "content": f"Retry differently ({seed}): list two imports."}
    ]


def subagent_requests(seed: int, prefix_tokens: int, where: str) -> tuple:
    """Two requests sharing ``prefix_tokens`` of text, either in the user turn or the system."""
    shared = code_doc(seed, prefix_tokens)
    out = []
    for task in ("Task A: count the classes.", "Task B: find the slowest loop."):
        if where == "system":
            out.append(
                [
                    {"role": "system", "content": SYSTEM + "\n" + shared},
                    {"role": "user", "content": task},
                ]
            )
        else:
            out.append(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": shared + "\n" + task},
                ]
            )
    return tuple(out)


def cached_of(usage: dict) -> int:
    return int(
        ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )


def chat(url, messages, max_tokens):
    body = json.dumps(
        dict(model="m", messages=messages, max_tokens=max_tokens, temperature=0)
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    t = time.time()
    data = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = data["choices"][0]["message"]
    return msg.get("content") or "", data.get("usage", {}), time.time() - t


def scenarios(chat_fn, record, seed: int, turns: int, per_turn: int, sub_tokens: int):
    """Run the scenarios through ``chat_fn``; ``record(step, usage, secs, ideal)`` per step."""
    users = build_turns(seed, turns, per_turn)
    history = [{"role": "system", "content": SYSTEM}]
    prompt_at = {}
    for t, u in enumerate(users):
        history.append({"role": "user", "content": u})
        out, usage, secs = chat_fn(history, 16)
        history.append({"role": "assistant", "content": out or "ok"})
        prompt_at[t + 1] = usage.get("prompt_tokens", 0)
        record(f"build{t + 1}", usage, secs, None)
    follow = history + [{"role": "user", "content": "And one more function?"}]
    _, usage, secs = chat_fn(follow, 1)
    record("a-linear", usage, secs, usage.get("prompt_tokens", 0) - 24)
    keep = max(1, turns // 2)
    _, usage, secs = chat_fn(branch_messages(history, keep, seed), 1)
    record("b-branch-mid", usage, secs, prompt_at[keep])
    for where in ("user", "system"):
        first, second = subagent_requests(seed + 7, sub_tokens, where)
        _, u1, s1 = chat_fn(first, 1)
        record(f"c-{where}-first", u1, s1, 0)
        _, u2, s2 = chat_fn(second, 1)
        record(f"c-{where}-second", u2, s2, u2.get("prompt_tokens", 0) - 24)


def run_arm(name, tree, model, port, rep, emit, extra_env, args):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="1",
        YUNSHU_VLM_APC_DISK="0",
    )
    env.update(extra_env or {})
    log = open(f"{os.path.splitext(args.out)[0]}_{name}_{rep}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            model,
            "--port",
            str(port),
        ],
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                fp = process_tree_memory(proc.pid)["physical_footprint_sum_bytes"]
                peak[0] = max(peak[0], fp)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)

    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2)
                break
            except Exception as exc:  # noqa: BLE001
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{name}: server exited rc={proc.returncode}"
                    ) from exc
                time.sleep(1)
        else:
            raise RuntimeError(f"{name}: server not ready")
        threading.Thread(target=sampler, daemon=True).start()

        def record(step, usage=None, secs=None, ideal=None):
            fp = process_tree_memory(proc.pid)["physical_footprint_sum_bytes"]
            row = dict(
                arm=name,
                rep=rep,
                step=step,
                prompt=(usage or {}).get("prompt_tokens"),
                cached=cached_of(usage) if usage else None,
                ideal=ideal,
                secs=None if secs is None else round(secs, 3),
                footprint_gib=round(fp / GIB, 3),
                peak_footprint_gib=round(peak[0] / GIB, 3),
                **metrics(url),
            )
            emit(row)
            if usage is not None and not usage.get("prompt_tokens"):
                raise RuntimeError(f"{name}/{step}: response carries no prompt_tokens")

        record("ready")
        scenarios(chat, record, 11 + rep, args.turns, args.per_turn, args.sub_tokens)
        time.sleep(20)
        record("idle20s")
        time.sleep(15)
        record("idle35s")
    finally:
        stop.set()
        try:
            os.killpg(proc.pid, 2)
            proc.wait(60)
        except Exception:  # noqa: BLE001
            os.killpg(proc.pid, 9)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=tree")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18995)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--turns", type=int, default=TURNS)
    ap.add_argument("--per-turn", type=int, default=TURN_TOKENS)
    ap.add_argument("--sub-tokens", type=int, default=20000)
    ap.add_argument("--arm-env", action="append", default=[], help="name:K=V")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arm]
    arm_env: dict = {}
    for item in a.arm_env:
        who, kv = item.split(":", 1)
        k, v = kv.split("=", 1)
        arm_env.setdefault(who, {})[k] = v
    done = 0
    with open(a.out, "a") as f:

        def emit(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)

        for i in range(a.reps):
            rep = a.rep_offset + i
            for name, tree in arms if rep % 2 == 0 else arms[::-1]:
                run_arm(name, tree, a.model, a.port, rep, emit, arm_env.get(name), a)
                done += 1
        emit(dict(complete=done == a.reps * len(arms)))


if __name__ == "__main__":
    main()
