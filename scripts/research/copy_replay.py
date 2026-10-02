"""Offline replay of recorded traffic through the copy drafter (CPU only).

For each cell (prompt tokens, recorded output tokens) it walks the output the way
the speculative lane would and counts verify rounds for:

- model-only: every round commits the cell's measured tokens per round
  (``--tpr``, or ct / rounds from the recorded x_yunshu.speculative);
- copy-only: a copy round whenever a long enough match exists (no gating);
- policy: the lane's round choice (benefit ratio + backoff), model round otherwise.

A copy round with L drafts commits ``accepted + 1`` tokens (accepted = the
matching prefix of the draft against the recorded output). The window is a
parameter (``--window``: verify rows; copy drafts <= window - 1).

Usage: copy_replay.py [--window 8] [--tfnew DIR] [--agent DIR ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from yunshu_engine.copy_drafter import CopyConfig, CopyDrafter  # noqa: E402

MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"


def walk(prompt, out, *, mode, tpr, window, cfg):
    """-> (rounds, copy_rounds, copy_tokens, proposed, accepted)."""
    d = CopyDrafter(cfg, max_draft=window - 1)
    d.extend(prompt)
    i = 0
    rounds = copy_rounds = copy_tokens = 0
    carry = 0.0
    while i < len(out):
        prop = []
        if mode == "copy":
            prop = d.lookup()
            if len(prop) < 2 or d.last_match < cfg.min_match:
                prop = []
        elif mode == "policy":
            prop = d.draft()
        rounds += 1
        if prop:
            a = 0
            while a < len(prop) and i + a < len(out) and prop[a] == out[i + a]:
                a += 1
            n = min(a + 1, len(out) - i)
            copy_rounds += 1
            copy_tokens += n
            d.observe_copy(len(prop), a)
        else:
            carry += tpr
            n = max(1, min(int(carry), len(out) - i))
            carry -= int(carry)
            d.observe_model(n)
        d.extend(out[i : i + n])
        i += n
    return rounds, copy_rounds, copy_tokens, d.proposed, d.accepted


def report(label, prompt, out, tpr, window, cfg, rows):
    res = {
        m: walk(prompt, out, mode=m, tpr=tpr, window=window, cfg=cfg)
        for m in ("model", "copy", "policy")
    }
    n = len(out)
    cells = [label, len(prompt), n, f"{tpr:.2f}"]
    for m in ("model", "copy", "policy"):
        r, cr, ct, _p, _a = res[m]
        cells.append(f"{n / r:.2f}")
        if m != "model":
            cells.append(f"{100 * ct / n:.0f}%")
    rows.append(cells)
    return res


def tfnew_cells(root, window, cfg, rows):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    recs = {}
    for line in Path(f"{root}/out/r0-yunshu-decode.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("part") == "decode":
            recs[(d["ctx"], d["kind"], d["phase"])] = d
    for ctx, kind in [
        (1024, "prose"),
        (1024, "code"),
        (8192, "code"),
        (8192, "prose"),
        (32768, "code"),
        (32768, "prose"),
    ]:
        prompt_text = Path(f"{root}/prompts/{kind}-{ctx}.txt").read_text()
        cold = recs.get((ctx, kind, "cold"))
        t2 = recs.get((ctx, kind, "turn2"))
        for name, rec, hist in (
            ("cold", cold, prompt_text),
            (
                "turn2",
                t2,
                prompt_text
                + "\n"
                + (cold or {}).get("text", "")
                + "\nContinue with the next part, at the same length.",
            ),
        ):
            if not rec or "text" not in rec:
                continue
            xy = rec.get("xy") or {}
            sp = (xy.get("speculative") or {}) if isinstance(xy, dict) else {}
            tpr = rec["ct"] / sp["rounds"] if sp.get("rounds") else 3.0
            p = tok(hist, add_special_tokens=False)["input_ids"]
            o = tok(rec["text"], add_special_tokens=False)["input_ids"][:512]
            report(f"{kind}-{ctx} {name}", p, o, tpr, window, cfg, rows)


def render_request(body):
    parts = [json.dumps(body.get("tools") or [], ensure_ascii=False)]
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append(f"<{m.get('role')}>{c or ''}")
        for tc in m.get("tool_calls") or []:
            parts.append(tc.get("function", {}).get("arguments", ""))
    return "\n".join(parts)


def parse_resp(path):
    """Generated text of an SSE chat completion: reasoning, content, tool-call arguments in order."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("data: ") or line.startswith("data: [DONE]"):
            continue
        try:
            ch = json.loads(line[6:])["choices"][0]["delta"]
        except (ValueError, KeyError, IndexError):
            continue
        for k in ("reasoning_content", "content"):
            if ch.get(k):
                out.append(ch[k])
        for tc in ch.get("tool_calls") or []:
            out.append(tc.get("function", {}).get("arguments") or "")
    return "".join(out)


def agent_cells(dirs, window, cfg, rows, tpr):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    for d in dirs:
        for req in sorted(glob.glob(f"{d}/bodies/*-req.json")):
            resp = req.replace("-req.json", "-resp.txt")
            if not os.path.exists(resp):
                continue
            text = parse_resp(resp)
            if len(text) < 80:
                continue
            body = json.loads(Path(req).read_text())
            p = tok(render_request(body), add_special_tokens=False)["input_ids"]
            o = tok(text, add_special_tokens=False)["input_ids"][:2048]
            report(
                f"{os.path.basename(d)[:24]}/{os.path.basename(req)[:4]}",
                p,
                o,
                tpr,
                window,
                cfg,
                rows,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--tfnew", default="/Volumes/P5Plus/yunshu-build/tfnew")
    ap.add_argument("--agent", action="append", default=[])
    ap.add_argument("--agent-tpr", type=float, default=2.8)
    ap.add_argument("--min-match", type=int, default=8)
    ap.add_argument("--confident", type=int, default=24)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument(
        "--sweep", default="", help="window:min_match,... overrides the single setting"
    )
    a = ap.parse_args()
    settings = [(a.window, a.min_match)]
    if a.sweep:
        settings = [tuple(int(x) for x in s.split(":")) for s in a.sweep.split(",")]
    for window, min_match in settings:
        cfg = CopyConfig(
            min_match=min_match, confident_match=a.confident, benefit_ratio=a.ratio
        )
        rows: list = []
        if a.tfnew:
            tfnew_cells(a.tfnew, window, cfg, rows)
        agent_cells(a.agent, window, cfg, rows, a.agent_tpr)
        head = [
            "cell",
            "prompt",
            "out",
            "tpr",
            "model",
            "copy-only",
            "hit",
            "policy",
            "hit",
        ]
        print(
            f"window={window} min_match={min_match} confident={a.confident} ratio={a.ratio} (tokens/round)"
        )
        print(" | ".join(head))
        for r in rows:
            print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
