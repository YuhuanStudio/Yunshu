"""verdict.json + verdict.md (Traditional Chinese summary + an English PERF_TREND block)."""

from __future__ import annotations

import time

SCHEMA = 1
STAGE_ORDER = ["preflight", "smoke", "identity", "apc", "quality", "speed", "memory"]


def build_verdict(
    *,
    label: str,
    base: dict,
    cand: dict,
    env: dict,
    cand_env: dict,
    base_env: dict,
    suite: dict,
    model: str,
    stages: list,
    planned: list,
    infra_error: str = "",
    started: float = 0.0,
    ended: float = 0.0,
    run_dir: str = "",
) -> dict:
    done = {s["name"]: s for s in stages}
    rows = []
    for name in planned:
        if name in done:
            rows.append(done[name])
        else:
            rows.append(
                {
                    "name": name,
                    "status": "NOT_RUN",
                    "reasons": [],
                    "numbers": {},
                    "jobs": [],
                }
            )
    failed = next((r["name"] for r in rows if r["status"] == "FAIL"), None)
    if infra_error:
        overall, exit_code = "INFRA_ERROR", 2
    elif failed:
        overall, exit_code = "FAIL", 1
    elif all(r["status"] == "PASS" for r in rows):
        overall, exit_code = "PASS", 0
    else:
        overall, exit_code = "INCOMPLETE", 2
    jobs = []
    for r in rows:
        for j in r.get("jobs", []):
            jobs.append(
                {
                    "stage": j[0],
                    "cell": j[1],
                    "job": j[2],
                    "reused": bool(j[3]) if len(j) > 3 else False,
                }
            )
    return {
        "schema": SCHEMA,
        "label": label,
        "overall": overall,
        "exit_code": exit_code,
        "failed_stage": failed,
        "infra_error": infra_error,
        "base": base,
        "cand": cand,
        "env": env,
        "base_env": base_env,
        "cand_env": cand_env,
        "suite": {k: v for k, v in suite.items()},
        "model": model,
        "stages": rows,
        "jobs": jobs,
        "started": started,
        "ended": ended or time.time(),
        "run_dir": run_dir,
    }


def _stage_line_zh(s: dict) -> str:
    mark = {"PASS": "通過", "FAIL": "失敗", "NOT_RUN": "未執行"}.get(
        s["status"], s["status"]
    )
    why = "；".join(str(x)[:160] for x in s["reasons"][:3])
    return f"- {s['name']}：{mark}" + (f"（{why}）" if why else "")


def _numbers_en(s: dict) -> str:
    n = s.get("numbers") or {}
    nm = s["name"]
    if nm == "speed" and n.get("cells"):
        parts = []
        for c in n["cells"]:
            parts.append(
                f"{c['kind']}@{c['ctx']} {c['metric']} {c['base_median']}->{c['cand_median']} "
                f"({c['delta_pct']:+.1f}%, noise +-{c['noise_pct']:.1f}%, reps {c['rep_deltas_pct']})"
            )
        return "; ".join(parts)
    if nm == "memory" and n.get("cells"):
        return "; ".join(
            f"{c['metric']} {c['base_gib']}->{c['cand_gib']} GiB" for c in n["cells"]
        )
    if nm == "quality" and "base_correct" in n:
        return f"n={n['n']} base {n['base_correct']} cand {n['cand_correct']} net {n['net']:+d}"
    if nm == "identity":
        bits = []
        for k in ("base_vs_cand", "spec_on_vs_off"):
            if k in n:
                bits.append(
                    f"{k} {n[k]['compared']} cells, {n[k]['mismatches']} mismatches"
                )
        return "; ".join(bits)
    if nm == "apc" and "compared" in n:
        return f"{n['compared']} cold/warm pairs, hits " + ",".join(
            str(h["cached"]) for h in n.get("hits", [])
        )
    if nm == "preflight":
        return f"{n.get('changed_files', 0)} changed files, {n.get('related_tests', 0)} related test files, pytest rc {n.get('pytest_rc')}"
    if nm == "smoke":
        return f"engaged {n.get('engaged') or 'default'}"
    return ""


def render_md(v: dict) -> str:
    zh = {
        "PASS": "通過",
        "FAIL": "失敗",
        "INFRA_ERROR": "基礎設施錯誤",
        "INCOMPLETE": "未完成",
    }[v["overall"]]
    lines = [
        f"# 驗證結論：{v['label']}",
        "",
        f"- 判定：**{zh}**（exit {v['exit_code']}）",
        f"- 基準：{v['base']['spec']} @ {v['base']['key']}",
        f"- 候選：{v['cand']['spec']} @ {v['cand']['key']}",
        f"- 模型：{v['model']}　suite：{v['suite'].get('name')}",
        f"- 環境變數：共用 {v['env'] or '無'}；候選 {v['cand_env'] or '無'}；基準 {v['base_env'] or '無'}",
    ]
    if v["failed_stage"]:
        lines.append(f"- 失敗階段：{v['failed_stage']}（之後的階段未執行）")
    if v["infra_error"]:
        lines.append(f"- 基礎設施錯誤：{v['infra_error']}")
    lines += ["", "## 各階段", ""] + [_stage_line_zh(s) for s in v["stages"]]
    lines += ["", "## PERF_TREND block", "", "```"]
    lines.append(
        f"verify {v['label']}: base {v['base']['key']} vs cand {v['cand']['key']} on {v['model']}, "
        f"suite {v['suite'].get('name')} -> {v['overall']}"
    )
    if v["cand_env"] or v["env"]:
        lines.append(f"env: shared {v['env']} cand {v['cand_env']}")
    for s in v["stages"]:
        extra = _numbers_en(s)
        why = "; ".join(str(x)[:200] for x in s["reasons"][:3])
        lines.append(
            f"  {s['name']}: {s['status']}"
            + (f" - {extra}" if extra else "")
            + (f" [{why}]" if why else "")
        )
    ids = ", ".join(f"{j['job']}{'*' if j['reused'] else ''}" for j in v["jobs"])
    lines.append(f"jobs: {ids or 'none'}  (* = reused from an earlier invocation)")
    lines.append("```")
    return "\n".join(lines) + "\n"
