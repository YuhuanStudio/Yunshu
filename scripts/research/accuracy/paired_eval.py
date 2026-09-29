"""Tier 3 of the accuracy harness: paired downstream evaluation.

The same questions go through a *reference* server (stock ``mlx_vlm.server``, same
checkpoint, no Yunshu import) and a *candidate* (Yunshu serving, default or with an
option changed). Every question is scored per arm; the report pairs them by id and gives
the accuracy difference with a 95% interval and the exact McNemar test on the discordant
pairs. Everything is resumable: one jsonl per (bench, arm), finished ids are skipped, a
job stops starting new questions when its time budget is used, so a queue job of 15-20
minutes can be repeated until the set is done.

    # one job: start the server for the arm, run questions, stop the server
    paired_eval.py run --bench gsm8k --arm ref --model $M
    paired_eval.py run --bench gsm8k --arm default --model $M
    paired_eval.py run --bench mmlu_pro --arm kv8 --env YUNSHU_KV_PRECISION=int8 --model $M
    # queue every job (alternating reference / candidate), lowest priority
    paired_eval.py submit --bench gsm8k --arms ref,default --rounds 12 --priority -2 --model $M
    # IFEval scoring needs lm_eval (kept in its own venv): run with that python
    paired_eval.py score --bench ifeval
    paired_eval.py report --bench gsm8k --ref ref --cand default

Benchmarks: gsm8k (1319), mmlu_pro (seeded random sample, default 2000), ifeval (541,
strict prompt level), needle (RULER-style single / multi-key / multi-value retrieval at
8K / 32K / 120K tokens), bfcl (simple + parallel tool calls). Datasets live in
/Volumes/P5Plus/datasets (see fetch_datasets.sh). Decoding is greedy; thinking is on with
reasoning_effort medium for gsm8k / mmlu_pro / ifeval, off for needle and bfcl.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/research"))
DATASETS = Path(os.environ.get("DATASETS", "/Volumes/P5Plus/datasets"))
OUT = Path(
    os.environ.get("PAIRED_OUT", "/Volumes/P5Plus/yunshu-test-cache/paired-eval")
)
# Dedicated env pinned to the released lock so both arms keep the same mlx / mlx-vlm for
# the whole evaluation (the main .venv moves with development).
MAIN_PY = Path(
    os.environ.get(
        "PAIRED_PY", "/Volumes/P5Plus/yunshu-test-envs/paired-eval/bin/python"
    )
)
DEPS_PY = Path("/Volumes/P5Plus/yunshu-test-envs/paired-eval-deps/bin/python")
BOOKS = Path("/Volumes/P5Plus/yunshu-test-cache/accuracy")
SEED = 20260930
NEEDLE_LENGTHS = (8192, 32768, 120000)


# ── statistics ───────────────────────────────────────────────────────────
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def loss_p(b: int, c: int) -> float:
    """One-sided exact p for 'candidate is worse' (b: ref right cand wrong > c)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(c + 1)) / 2**n


def paired_stats(ref: list[bool], cand: list[bool]) -> dict:
    n = len(ref)
    b = sum(r and not c for r, c in zip(ref, cand, strict=True))
    c_ = sum(c and not r for r, c in zip(ref, cand, strict=True))
    d = (c_ - b) / n if n else 0.0
    # paired variance of the difference of proportions
    var = ((b + c_) - (b - c_) ** 2 / n) / n**2 if n else 0.0
    half = 1.96 * math.sqrt(max(var, 0.0))
    return {
        "n": n,
        "ref_acc": sum(ref) / n if n else 0.0,
        "cand_acc": sum(cand) / n if n else 0.0,
        "b_ref_only": b,
        "c_cand_only": c_,
        "delta": d,
        "ci95": (d - half, d + half),
        "p_two_sided": mcnemar_exact(b, c_),
        "p_cand_worse": loss_p(b, c_),
        "discordant": (b + c_) / n if n else 0.0,
    }


# ── shared text helpers (MMLU-Pro extraction is the soak's, aligned with oMLX) ─
def strip_think(text: str) -> str:
    if "<think>" not in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().split("\n"):
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):  # torn last line
                rows.append(json.loads(line))
    return rows


# ── benchmarks: each returns items (dict with id) and knows request + score ─
class Bench:
    name = ""
    thinking = True
    max_tokens = 8192
    chunk = 100

    def items(self, args) -> list[dict]:
        raise NotImplementedError

    def request(self, item: dict) -> dict:
        """Extra chat-completions fields: messages (+ tools)."""
        raise NotImplementedError

    def score(self, item: dict, resp: dict) -> dict:
        raise NotImplementedError


def last_number(text: str) -> str | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


class GSM8K(Bench):
    name = "gsm8k"
    max_tokens = 8192

    def items(self, args):
        rows = [
            json.loads(x)
            for x in (DATASETS / "gsm8k/gsm8k_test.jsonl").read_text().split("\n")
            if x.strip()
        ]
        return [
            {
                "id": f"gsm8k-{i}",
                "question": r["question"],
                "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
            }
            for i, r in enumerate(rows)
        ]

    def request(self, item):
        p = (
            f"{item['question']}\n\nSolve the problem. End your reply with a line of the "
            "form 'Answer: <number>' (digits only, no units)."
        )
        return {"messages": [{"role": "user", "content": p}]}

    def score(self, item, resp):
        text = strip_think(resp["content"])
        m = re.findall(
            r"answer\s*[:is]*\s*\$?\\?(?:boxed\{)?(-?[\d,]+\.?\d*)", text, re.I
        )
        pred = m[-1].replace(",", "").rstrip(".") if m else last_number(text)
        try:
            ok = pred is not None and abs(float(pred) - float(item["gold"])) < 1e-6
        except ValueError:
            ok = False
        return {"correct": bool(ok), "pred": pred, "gold": item["gold"]}


class MMLUPro(Bench):
    name = "mmlu_pro"
    max_tokens = 16384
    chunk = 50

    def items(self, args):
        rows = [
            json.loads(x)
            for x in (DATASETS / "mmlu_pro/mmlu_pro_test.jsonl").read_text().split("\n")
            if x.strip()
        ]
        random.Random(SEED).shuffle(rows)
        return [
            {
                "id": f"mmlu_pro-{r['id']}",
                **{
                    k: r[k]
                    for k in ("question", "choices", "labels", "answer", "subject")
                },
            }
            for r in rows[: args.mmlu_n]
        ]

    def request(self, item):
        from soak_mmlu_pro import prompt_of

        return {"messages": [{"role": "user", "content": prompt_of(item)}]}

    def score(self, item, resp):
        from soak_mmlu_pro import extract

        pred = extract(strip_think(resp["content"]), item["labels"])
        return {"correct": pred == item["answer"], "pred": pred, "gold": item["answer"]}


class IFEval(Bench):
    name = "ifeval"
    max_tokens = 8192

    def items(self, args):
        rows = [
            json.loads(x)
            for x in (DATASETS / "ifeval/input_data.jsonl").read_text().split("\n")
            if x.strip()
        ]
        return [{"id": f"ifeval-{r['key']}", "doc": r} for r in rows]

    def request(self, item):
        return {"messages": [{"role": "user", "content": item["doc"]["prompt"]}]}

    def score(self, item, resp):
        return {"correct": None}  # scored later by `score` with lm_eval's checker


def ifeval_score_file(path: Path) -> None:
    """Fill correct / correct_loose in place using lm_eval's IFEval checker."""
    os.environ.setdefault("NLTK_DATA", "/Volumes/P5Plus/yunshu-test-cache/nltk")
    from lm_eval.tasks.ifeval import utils as ifeval_utils  # noqa: PLC0415

    ds = {
        f"ifeval-{r['key']}": r
        for r in read_jsonl(DATASETS / "ifeval/input_data.jsonl")
    }
    rows = read_jsonl(path)
    for r in rows:
        if r.get("kind") != "q" or "response" not in r:
            continue
        doc = ds[r["id"]]
        resp = strip_think(r["response"]) if r.get("finish") != "length" else ""
        res = ifeval_utils.process_results(doc, [resp])
        r["correct"] = bool(res["prompt_level_strict_acc"])
        r["correct_loose"] = bool(res["prompt_level_loose_acc"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# needle / RULER-style ------------------------------------------------------
_TOK = None


def _tokenizer(model: str):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(model)
    return _TOK


class Needle(Bench):
    """Haystack of public-domain prose to an exact token length, with needles as
    'The special magic number for <key> is <value>.' sentences.

    * single: one needle, ask for its value
    * multikey: 4 needles with different keys, ask for one
    * multivalue: one key hidden 3 times with different values, ask for all three
    """

    name = "needle"
    thinking = False
    max_tokens = 256
    chunk = 20

    def items(self, args):
        if getattr(args, "no_needle_build", False):
            return []
        tok = _tokenizer(args.model)
        rng = random.Random(SEED)
        paras = []
        for name in ("pg1342", "pg2701", "pg1661"):
            t = (BOOKS / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
            a = t.find("\n", t.find("*** START")) + 1
            t = t[a : t.find("*** END")].replace("\r\n", "\n")
            paras += [p.strip() for p in t.split("\n\n") if len(p.strip()) > 120]
        counts = [len(x) for x in tok(paras, add_special_tokens=False)["input_ids"]]
        adjectives = [
            "red",
            "silent",
            "golden",
            "broken",
            "distant",
            "hollow",
            "amber",
            "quiet",
        ]
        nouns = [
            "harbor",
            "lantern",
            "orchard",
            "compass",
            "meadow",
            "citadel",
            "violin",
            "glacier",
        ]
        items = []
        for length in NEEDLE_LENGTHS:
            for kind, n in (
                ("single", args.needle_n),
                ("multikey", args.needle_n),
                ("multivalue", args.needle_n),
            ):
                for k in range(n):
                    keys = [
                        f"{rng.choice(adjectives)}-{rng.choice(nouns)}-{rng.randint(10, 99)}"
                        for _ in range(4)
                    ]
                    vals = [str(rng.randint(1000000, 9999999)) for _ in range(4)]
                    if kind == "single":
                        needles = [(keys[0], vals[0])]
                        asked, gold = keys[0], [vals[0]]
                    elif kind == "multikey":
                        needles = list(zip(keys, vals, strict=True))
                        j = rng.randrange(4)
                        asked, gold = keys[j], [vals[j]]
                    else:
                        needles = [(keys[0], v) for v in vals[:3]]
                        asked, gold = keys[0], vals[:3]
                    # paragraphs to reach length minus needle + prompt overhead
                    budget = length - 200 - 30 * len(needles)
                    start = rng.randrange(len(paras))
                    chosen, total, i = [], 0, start
                    while budget - total > 60:
                        c = counts[i % len(paras)]
                        if total + c <= budget:
                            chosen.append(paras[i % len(paras)])
                            total += c
                        i += 1
                    for depth_i, (key, val) in enumerate(needles):
                        frac = (depth_i + 1) / (len(needles) + 1) * (0.5 + rng.random())
                        pos = min(len(chosen), max(0, int(frac * len(chosen) * 0.9)))
                        chosen.insert(
                            pos, f"The special magic number for {key} is {val}."
                        )
                    items.append(
                        {
                            "id": f"needle-{length}-{kind}-{k}",
                            "length": length,
                            "kind": kind,
                            "asked": asked,
                            "gold": gold,
                            "context": "\n\n".join(chosen),
                        }
                    )
        return items

    def request(self, item):
        if item["kind"] == "multivalue":
            q = f"What are all the special magic numbers for {item['asked']} mentioned in the text above? List every one."
        else:
            q = f"What is the special magic number for {item['asked']} mentioned in the text above?"
        p = (
            "Some sentences in the following text state special magic numbers. Read the text and "
            f"answer the question at the end.\n\n{item['context']}\n\n{q} Answer briefly with the numbers only."
        )
        return {"messages": [{"role": "user", "content": p}]}

    def score(self, item, resp):
        text = strip_think(resp["content"])
        found = set(re.findall(r"\d{7}", text.replace(",", "")))
        ok = all(g in found for g in item["gold"]) and (
            item["kind"] == "multivalue" or len(found) == 1
        )
        return {"correct": bool(ok), "pred": sorted(found), "gold": item["gold"]}


# tool calls (BFCL simple + parallel) ---------------------------------------
_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
    "int": "integer",
    "bool": "boolean",
}


def _schema(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_MAP.get(v, v)
            elif k in ("properties", "items") or isinstance(v, (dict, list)):
                out[k] = _schema(v)
            else:
                out[k] = v
        return out
    if isinstance(node, list):
        return [_schema(x) for x in node]
    return node


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _arg_ok(val, accepted) -> bool:
    return any(_norm(val) == _norm(a) for a in accepted)


def _call_matches(call: tuple[str, dict], gold: dict) -> bool:
    ((gname, gargs),) = gold.items()
    name, args = call
    if name != gname.replace(".", "_"):
        return False
    for k, v in args.items():
        if k not in gargs or not _arg_ok(v, gargs[k]):
            return False
    return all(k in args or "" in accepted for k, accepted in gargs.items())


class BFCL(Bench):
    name = "bfcl"
    thinking = False
    max_tokens = 1024
    chunk = 200

    def items(self, args):
        out = []
        for kind in ("simple", "parallel"):
            qs = [
                json.loads(x)
                for x in (DATASETS / f"bfcl/BFCL_v3_{kind}.json")
                .read_text()
                .split("\n")
                if x.strip()
            ]
            gt = {}
            for line in (
                (DATASETS / f"bfcl/possible_answer/BFCL_v3_{kind}.json")
                .read_text()
                .split("\n")
            ):
                if line.strip():
                    r = json.loads(line)
                    gt[r["id"]] = r["ground_truth"]
            for q in qs[: args.bfcl_n]:
                out.append(
                    {"id": f"bfcl-{q['id']}", "kind": kind, "q": q, "gold": gt[q["id"]]}
                )
        return out

    def request(self, item):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f["name"].replace(".", "_"),
                    "description": f.get("description", ""),
                    "parameters": _schema(f["parameters"]),
                },
            }
            for f in item["q"]["function"]
        ]
        return {
            "messages": item["q"]["question"][0],
            "tools": tools,
            "tool_choice": "auto",
        }

    def score(self, item, resp):
        calls = []
        for tc in resp.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                a = fn.get("arguments")
                a = json.loads(a) if isinstance(a, str) else (a or {})
                calls.append((fn.get("name", ""), a))
            except json.JSONDecodeError:
                return {"correct": False, "parse_fail": True, "pred": None}
        gold = list(item["gold"])
        ok = len(calls) == len(gold)
        left = list(gold)
        if ok:
            for c in calls:
                hit = next((g for g in left if _call_matches(c, g)), None)
                if hit is None:
                    ok = False
                    break
                left.remove(hit)
        return {
            "correct": ok,
            "parse_fail": not calls,
            "pred": [[n, a] for n, a in calls],
        }


BENCHES: dict[str, Bench] = {
    b.name: b for b in (GSM8K(), MMLUPro(), IFEval(), Needle(), BFCL())
}


# ── servers ──────────────────────────────────────────────────────────────
class Server:
    def __init__(self, arm: str, model: str, port: int, env: dict[str, str], log: Path):
        self.arm, self.model, self.port = arm, model, port
        self.url = f"http://127.0.0.1:{port}"
        e = {k: v for k, v in os.environ.items() if not k.startswith("YUNSHU_")}
        e["HF_HUB_OFFLINE"] = "1"
        e["TRANSFORMERS_OFFLINE"] = "1"
        if arm.startswith("ref"):
            # stock mlx-vlm: no Yunshu on the path, no YUNSHU_* settings
            e.pop("PYTHONPATH", None)
            cmd = [
                str(MAIN_PY),
                "-m",
                "mlx_vlm.server",
                "--model",
                model,
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--max-num-seqs",
                env.pop("MAX_SEQS", "8"),
                "--max-tokens",
                "16384",
            ]
            self.ready_path = "/health"
        else:
            env.pop("MAX_SEQS", None)
            e.update(
                YUNSHU_MODEL=model,
                YUNSHU_AUTH_DISABLED="1",
                PYTHONPATH=str(ROOT / "python"),
            )
            e.update(env)
            cmd = [
                str(MAIN_PY),
                "-m",
                "uvicorn",
                "yunshu_gateway.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            self.ready_path = "/health/ready"
        e.update(env)
        log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            cmd,
            env=e,
            stdout=log.open("a"),
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            start_new_session=True,
        )

    def wait_ready(self, timeout=600) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
                c.request("GET", self.ready_path)
                r = c.getresponse()
                body = r.read().decode(errors="replace")
                c.close()
                if r.status == 200 and (
                    self.arm.startswith("ref")
                    or '"ready":true' in body.replace(" ", "")
                ):
                    return True
            except OSError:
                pass
            time.sleep(2)
        return False

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            self.proc.wait(30)
        except Exception:  # noqa: BLE001
            pass
        with contextlib.suppress(Exception):
            os.killpg(self.proc.pid, signal.SIGKILL)


def chat(url: str, bench: Bench, item: dict, model: str, timeout: float) -> dict:
    u = urllib.parse.urlparse(url)
    body = {
        "model": model,
        "max_tokens": bench.max_tokens,
        "temperature": 0.0,
        "enable_thinking": bench.thinking,
        "chat_template_kwargs": {"enable_thinking": bench.thinking},
        **bench.request(item),
    }
    if bench.thinking:
        body["reasoning_effort"] = "medium"
        body["chat_template_kwargs"]["reasoning_effort"] = "medium"
    else:
        body["reasoning_effort"] = "none"
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    r = conn.getresponse()
    raw = r.read().decode(errors="replace")
    conn.close()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {raw[:300]}"}
    j = json.loads(raw)
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = j.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning_len": len(
            msg.get("reasoning_content") or msg.get("reasoning") or ""
        ),
        "tool_calls": msg.get("tool_calls"),
        "finish": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


# ── commands ─────────────────────────────────────────────────────────────
def result_path(bench: str, arm: str) -> Path:
    return OUT / bench / f"{arm}.jsonl"


def cmd_run(a) -> int:
    bench = BENCHES[a.bench]
    items = bench.items(a)
    path = result_path(a.bench, a.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {
        r["id"] for r in read_jsonl(path) if r.get("kind") == "q" and not r.get("error")
    }
    todo = [x for x in items if x["id"] not in done]
    if a.start or a.n:
        window = {x["id"] for x in items[a.start : a.start + a.n]}
        todo = [x for x in todo if x["id"] in window]
    print(f"{a.bench}/{a.arm}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    env = dict(kv.split("=", 1) for kv in a.env)
    env.setdefault("MAX_SEQS", str(a.concurrency))
    srv = Server(
        a.arm,
        a.model,
        a.port,
        env,
        OUT / "logs" / f"{a.bench}-{a.arm}-{time.strftime('%m%d-%H%M%S')}.log",
    )
    t0 = time.time()
    lock = threading.Lock()
    out = path.open("a")
    n_done = [0]
    try:
        if not srv.wait_ready():
            print("server failed to start", flush=True)
            return 2
        print(f"server up in {time.time() - t0:.0f}s", flush=True)
        with lock:
            out.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "arm": a.arm,
                        "env": a.env,
                        "model": a.model,
                        "t": time.strftime("%FT%T"),
                        "concurrency": a.concurrency,
                    }
                )
                + "\n"
            )
            out.flush()
        deadline = time.time() + a.budget_min * 60
        hard = time.time() + (a.budget_min + 3) * 60
        it = iter(todo)

        def worker():
            while time.time() < deadline:
                with lock:
                    item = next(it, None)
                if item is None:
                    return
                t1 = time.perf_counter()
                try:
                    resp = chat(
                        srv.url, bench, item, a.model_name, max(60, hard - time.time())
                    )
                except Exception as e:  # noqa: BLE001
                    resp = {"error": repr(e)[:300]}
                row = {
                    "kind": "q",
                    "id": item["id"],
                    "wall_s": round(time.perf_counter() - t1, 2),
                }
                for k in ("subject", "length"):
                    if k in item:
                        row[k] = item[k]
                if a.bench in ("needle", "bfcl"):
                    row["sub"] = item["kind"]
                if "error" in resp:
                    row["error"] = resp["error"]
                else:
                    row.update(
                        finish=resp["finish"],
                        completion_tokens=resp["completion_tokens"],
                        prompt_tokens=resp["prompt_tokens"],
                        reasoning_len=resp["reasoning_len"],
                        response=resp["content"],
                    )
                    row.update(bench.score(item, resp))
                    if resp["finish"] == "length":
                        row["truncated"] = True
                        if row.get("correct") is not None:
                            row["correct"] = False
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    n_done[0] += 1
                    print(
                        f"{n_done[0]}/{len(todo)} {item['id']} correct={row.get('correct')} tok={row.get('completion_tokens')} {row['wall_s']}s",
                        flush=True,
                    )

        threads = [
            threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(max(1.0, hard - time.time()))
        return 0
    finally:
        srv.stop()
        out.close()


def load_arm(bench: str, arm: str) -> dict[str, dict]:
    rows = [
        r
        for r in read_jsonl(result_path(bench, arm))
        if r.get("kind") == "q" and not r.get("error")
    ]
    return {r["id"]: r for r in rows}  # last row per id wins


def cmd_score(a) -> int:
    if a.bench == "ifeval":
        try:
            import lm_eval  # noqa: F401, PLC0415
        except ImportError:
            return subprocess.call([str(DEPS_PY), __file__, *sys.argv[1:]])
    for arm_file in (OUT / a.bench).glob("*.jsonl"):
        if a.bench == "ifeval":
            ifeval_score_file(arm_file)
    return 0


def cmd_report(a) -> int:
    ref, cand = load_arm(a.bench, a.ref), load_arm(a.bench, a.cand)
    ids = sorted(set(ref) & set(cand))
    key = a.field
    ids = [
        i for i in ids if ref[i].get(key) is not None and cand[i].get(key) is not None
    ]
    if not ids:
        print(f"{a.bench}: no paired items ({len(ref)} ref, {len(cand)} cand)")
        return 1
    s = paired_stats(
        [bool(ref[i][key]) for i in ids], [bool(cand[i][key]) for i in ids]
    )
    print(
        f"{a.bench} {a.ref} vs {a.cand}: n={s['n']} ref={s['ref_acc']:.4f} cand={s['cand_acc']:.4f} "
        f"delta={s['delta'] * 100:+.2f} pts CI95=[{s['ci95'][0] * 100:+.2f}, {s['ci95'][1] * 100:+.2f}] "
        f"b={s['b_ref_only']} c={s['c_cand_only']} discordant={s['discordant'] * 100:.1f}% "
        f"McNemar p={s['p_two_sided']:.3f} p(cand worse)={s['p_cand_worse']:.3f}"
    )
    for name, arm in ((a.ref, ref), (a.cand, cand)):
        rs = [arm[i] for i in ids]
        trunc = sum(bool(r.get("truncated")) for r in rs)
        toks = sum(r.get("completion_tokens") or 0 for r in rs) / len(rs)
        pf = sum(bool(r.get("parse_fail")) for r in rs)
        print(
            f"  {name}: mean completion tokens {toks:.0f}, truncated {trunc}, parse_fail {pf}"
        )
    groups = sorted(
        {ref[i].get("length") or ref[i].get("sub") or "" for i in ids} - {""}, key=str
    )
    for g in groups:
        sub = [i for i in ids if (ref[i].get("length") or ref[i].get("sub")) == g]
        st = paired_stats(
            [bool(ref[i][key]) for i in sub], [bool(cand[i][key]) for i in sub]
        )
        print(
            f"  [{g}] n={st['n']} ref={st['ref_acc']:.3f} cand={st['cand_acc']:.3f} b={st['b_ref_only']} c={st['c_cand_only']}"
        )
    return 0


def cmd_submit(a) -> int:
    """Queue jobs alternating between arms (drift and thermal state hit both)."""
    arms = a.arms.split(",")
    envs = dict(x.split(":", 1) for x in a.arm_env)  # ARM:K=V;K=V
    gq = ROOT / "scripts/dev/gpuq"
    for rnd in range(a.rounds):
        for arm in arms:
            cmd = [
                str(gq),
                "submit",
                "--label",
                f"paired-{a.bench}-{arm}-r{rnd}",
                "--timeout",
                "20",
                "--stall",
                "12",
                "--priority",
                str(a.priority),
                "--",
                str(MAIN_PY),
                str(Path(__file__)),
                "run",
                "--bench",
                a.bench,
                "--arm",
                arm,
                "--model",
                a.model,
                "--budget-min",
                str(a.budget_min),
                "--concurrency",
                str(a.concurrency),
            ]
            if a.mmlu_n != 2000:
                cmd += ["--mmlu-n", str(a.mmlu_n)]
            for kv in filter(None, envs.get(arm, "").split(";")):
                cmd += ["--env", kv]
            print(
                subprocess.run(
                    cmd, capture_output=True, text=True, cwd=ROOT
                ).stdout.strip(),
                flush=True,
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bench", required=True, choices=sorted(BENCHES))
    common.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    common.add_argument(
        "--model-name", default="Qwen3.8-27B", help="'model' field sent in requests"
    )
    common.add_argument("--mmlu-n", type=int, default=2000)
    common.add_argument(
        "--needle-n", type=int, default=10, help="items per kind and length"
    )
    common.add_argument("--bfcl-n", type=int, default=400, help="max items per kind")
    common.add_argument("--concurrency", type=int, default=8)
    common.add_argument(
        "--budget-min",
        type=float,
        default=15,
        help="stop starting questions after this",
    )
    r = sub.add_parser("run", parents=[common])
    r.add_argument(
        "--arm",
        required=True,
        help="'ref*' = stock mlx_vlm.server; anything else = Yunshu",
    )
    r.add_argument(
        "--env", action="append", default=[], help="K=V for the Yunshu server"
    )
    r.add_argument("--port", type=int, default=18990)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--n", type=int, default=0)
    s = sub.add_parser("submit", parents=[common])
    s.add_argument("--arms", default="ref,default")
    s.add_argument("--arm-env", action="append", default=[], help="ARM:K=V;K=V")
    s.add_argument("--rounds", type=int, default=6)
    s.add_argument("--priority", type=int, default=-2)
    sub.add_parser("score", parents=[common])
    rp = sub.add_parser("report", parents=[common])
    rp.add_argument("--ref", default="ref")
    rp.add_argument("--cand", default="default")
    rp.add_argument("--field", default="correct")
    a = ap.parse_args()
    return {
        "run": cmd_run,
        "submit": cmd_submit,
        "score": cmd_score,
        "report": cmd_report,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
