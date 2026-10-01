"""MMLU-Pro 300 soak over any OpenAI-compatible server (the standard soak).

Same questions, prompt, sampling and answer extraction as oMLX's built-in
MMLU-Pro eval with thinking on (temperature 0, max_tokens 8192, think block
stripped before extraction), so accuracy is comparable to results the user
ran in oMLX / Splash. Long reasoning chains are the memory stress; the
server's process-tree physical footprint is sampled throughout and after an
idle period at the end.

    python scripts/research/soak_mmlu_pro.py --url http://127.0.0.1:18764 \
        --model Qwen3.8 --pid 1234 --output runs/mmlu-soak.jsonl
"""

import argparse
import collections
import concurrent.futures
import http.client
import json
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_memory import apc_resident_gib, process_tree_memory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl"
DEFAULT_IDS = Path.home() / "Downloads/Qwen3.8-27B-oQ4e-mtp_mmlu_pro.json"


def prompt_of(item):
    parts = [
        "Answer the following question. Answer with just the letter.\n",
        f"Question: {item['question']}\n",
    ]
    parts += [
        f"{lab}. {ch}" for lab, ch in zip(item["labels"], item["choices"], strict=False)
    ]
    parts.append("\nAnswer:")
    return "\n".join(parts)


def strip_think(text):
    # oMLX BaseBenchmark._strip_think_tags
    if "<think>" not in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract(resp, letters):
    up = resp.strip().upper()
    pl = "".join(letters)
    m = re.findall(r"(?:answer\s*(?:is|:)\s*)([" + pl + r"])\b", up, re.IGNORECASE)
    if m:
        return m[-1]
    m = re.findall(r"\b([" + pl + r"])\b", up)
    if m:
        return m[-1]
    return up[:1] if up[:1] in letters else ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument(
        "--ids",
        type=Path,
        default=DEFAULT_IDS,
        help="result JSON whose question ids to reuse",
    )
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="first question (with --n: run a slice, e.g. 100-question jobs)",
    )
    # Defaults match the user's oMLX MMLU-Pro run: 16384 max output, medium
    # reasoning effort, 8 questions in flight.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--reasoning-effort", default="medium", help="'' to omit")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--final-idle-s", type=float, default=120)
    ap.add_argument("--footprint-stop-gib", type=float, default=100)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for line in DATASET.read_text().split("\n"):
        if not line.strip():
            continue
        item = json.loads(line)
        data[item["id"]] = item
    ids = [q["id"] for q in json.loads(a.ids.read_text())["questions"]][
        a.start : a.start + a.n
    ]
    think = not a.no_thinking
    u = urllib.parse.urlparse(a.url)
    out = a.output.open("a")
    lock, stop = threading.Lock(), threading.Event()

    def emit(row):
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    def mem():
        try:
            return round(
                process_tree_memory(a.pid)["physical_footprint_sum_bytes"] / 2**30, 3
            )
        except Exception:  # noqa: BLE001
            return None

    def sampler():
        while not stop.is_set():
            emit({"kind": "mem", "t": round(time.time(), 1), "footprint_gib": mem()})
            stop.wait(10)

    start_mem = mem()
    start_apc = apc_resident_gib(a.url)
    emit(
        {
            "kind": "meta",
            "url": a.url,
            "model": a.model,
            "pid": a.pid,
            "n": len(ids),
            "thinking": think,
            "max_tokens": a.max_tokens,
            "reasoning_effort": a.reasoning_effort,
            "concurrency": a.concurrency,
            "note": a.note,
            "start_footprint_gib": start_mem,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    threading.Thread(target=sampler, daemon=True).start()
    t_start = time.time()
    rows = []
    aborted = threading.Event()

    def run_one(qi, qid):
        if aborted.is_set():
            return None
        item = data[qid]
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": prompt_of(item)}],
            "max_tokens": a.max_tokens,
            "temperature": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "enable_thinking": think,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
            body["chat_template_kwargs"]["reasoning_effort"] = a.reasoning_effort
        t0 = time.perf_counter()
        row = {
            "kind": "q",
            "i": qi,
            "id": qid,
            "subject": item["subject"],
            "answer": item["answer"],
        }
        try:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=7200)
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode(errors="replace")
            conn.close()
            row["status"] = resp.status
            j = json.loads(raw) if resp.status == 200 else {}
            ch = (j.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            usage = j.get("usage") or {}
            row.update(
                finish=ch.get("finish_reason"),
                completion_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"),
                reasoning_len=len(
                    msg.get("reasoning_content") or msg.get("reasoning") or ""
                ),
                content=text[-200:],
            )
            if resp.status != 200:
                row["error"] = raw[:300]
            pred = extract(strip_think(text), item["labels"])
            row.update(
                pred=pred,
                correct=pred == item["answer"],
                truncated=ch.get("finish_reason") == "length",
            )
        except Exception as e:  # noqa: BLE001
            row.update(error=repr(e)[:300], pred="", correct=False, truncated=False)
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        row["footprint_gib"] = mem()
        emit(row)
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in (
                        "i",
                        "subject",
                        "answer",
                        "pred",
                        "correct",
                        "finish",
                        "completion_tokens",
                        "wall_s",
                        "footprint_gib",
                    )
                }
            ),
            flush=True,
        )
        if row["footprint_gib"] and row["footprint_gib"] > a.footprint_stop_gib:
            aborted.set()
            emit(
                {
                    "kind": "abort",
                    "reason": f"footprint {row['footprint_gib']} > {a.footprint_stop_gib}",
                }
            )
        return row

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        futures = [pool.submit(run_one, qi, qid) for qi, qid in enumerate(ids, a.start)]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)
    elapsed = time.time() - t_start
    time.sleep(a.final_idle_s)
    stop.set()
    end_mem, end_apc = mem(), apc_resident_gib(a.url)
    mems = [
        json.loads(line)["footprint_gib"]
        for line in a.output.read_text().split("\n")
        if '"kind": "mem"' in line and json.loads(line).get("footprint_gib")
    ]
    cats = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        cats[r["subject"]][0] += 1
        cats[r["subject"]][1] += bool(r.get("correct"))
    tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    summary = {
        "kind": "summary",
        "n": len(rows),
        "correct": sum(bool(r.get("correct")) for r in rows),
        "accuracy": round(
            sum(bool(r.get("correct")) for r in rows) / max(1, len(rows)), 4
        ),
        "truncated": sum(bool(r.get("truncated")) for r in rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "time_s": round(elapsed, 1),
        "completion_tokens": tokens,
        "tok_per_s": round(tokens / elapsed, 1) if elapsed else None,
        "start_footprint_gib": start_mem,
        "max_footprint_gib": max(mems) if mems else None,
        "end_footprint_gib": end_mem,
        "start_apc_gib": start_apc,
        "end_apc_gib": end_apc,
        "start_footprint_ex_apc_gib": None
        if start_mem is None or start_apc is None
        else round(start_mem - start_apc, 3),
        "end_footprint_ex_apc_gib": None
        if end_mem is None or end_apc is None
        else round(end_mem - end_apc, 3),
        "categories": {k: f"{v[1]}/{v[0]}" for k, v in sorted(cats.items())},
    }
    emit(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
