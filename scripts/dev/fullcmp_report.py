#!/usr/bin/env python3
"""Full-comparison report: aggregate a fullcmp raw root into one JSON table and a self-contained HTML page.

CPU only (reads finished jsonl files; no MLX, no servers). Reusable for every run:

    python3 scripts/dev/fullcmp_report.py --raw /Volumes/Micron/yunshu-fullcmp/real-1104 \
        --manifest <run>/manifest.jsonl --out <dir>

Writes <out>/fullcmp_rows.json (one row per engine x cell x phase, medians over reps, with every rep kept)
and <out>/fullcmp_report.html (progress, curves per engine, concurrency, labels). Superseded files
(*.superseded-*) are ignored; a cell's newest complete record wins.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from collections import defaultdict

PHASE_ALIAS = {"fu": "turn2"}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_cell_name(name: str) -> dict:
    # B2.conc.tf-1.2.1.1K.c8.code.o512.single.th-off.none.greedy-r0.jsonl
    stem = name[: -len(".jsonl")]
    base, _, rep = stem.rpartition("-r")
    parts = base.split(".")
    batch, family = parts[0], parts[1]
    # engine ids may contain dots (tf-1.2.1); ctx is the first part ending in K
    i = next(k for k in range(2, len(parts)) if parts[k][:-1].isdigit() and parts[k].endswith("K"))
    engine = ".".join(parts[2:i])
    ctx, conc, kind, out = parts[i], parts[i + 1], parts[i + 2], parts[i + 3]
    rest = parts[i + 4 :]
    return {
        "batch": batch,
        "family": family,
        "engine": engine,
        "ctx": ctx,
        "conc": int(conc[1:]) if conc.startswith("c") else conc,
        "kind": kind,
        "out": int(out[1:]) if out.startswith("o") else out,
        "variant": ".".join(rest),
        "cell": base,
        "rep": rep,
    }


def collect(raw: str) -> tuple[list[dict], dict]:
    reps = defaultdict(list)  # (cell, phase) -> list of measurements
    labels, unsupported, engines_meta = {}, {}, {}
    for f in glob.glob(os.path.join(raw, "*", "*.jsonl")):
        name = os.path.basename(f)
        if "telemetry" in name or "superseded" in name or name.endswith("-prof.jsonl"):
            continue
        try:
            meta = parse_cell_name(name)
        except (StopIteration, ValueError, IndexError):
            continue
        try:
            lines = [json.loads(x) for x in open(f, errors="replace") if x.strip()]
        except ValueError:
            continue
        for r in lines:
            part = r.get("part")
            if part == "session":
                engines_meta.setdefault(meta["engine"], {
                    "version": r.get("version"), "git_sha": (r.get("git_sha") or "")[:9] or None,
                    "spec_mode_expected": r.get("spec_mode_expected"),
                })
                if r.get("engaged_label"):
                    labels[meta["cell"]] = r["engaged_label"]
                if r.get("unsupported_cell"):
                    unsupported[meta["cell"]] = str(r.get("unsupported_cell"))[:300]
            if part in ("unsupported",) or r.get("label", "").startswith("unsupported") if isinstance(r.get("label"), str) else False:
                unsupported[meta["cell"]] = str(r.get("label") or r.get("reason"))[:300]
            phase = r.get("phase")
            if phase in ("cold", "warm", "fu", "turn2", "restart", "fu_idle") and _num(r.get("dec_tps")) is not None:
                reps[(meta["cell"], PHASE_ALIAS.get(phase, phase))].append({
                    **meta, "phase": PHASE_ALIAS.get(phase, phase),
                    "ttft_s": _num(r.get("ttft_s")), "dec_tps": _num(r.get("dec_tps")),
                    "cached": r.get("cached"), "pt": r.get("pt"), "ct": r.get("ct"),
                })
            if part in ("conc", "mixed") and _num(r.get("agg_tps")) is not None:
                tt = [t for t in (r.get("ttfts") or []) if _num(t) is not None]
                reps[(meta["cell"], part)].append({
                    **meta, "phase": part, "agg_tps": _num(r.get("agg_tps")),
                    "dec_min": _num(r.get("dec_min")), "dec_max": _num(r.get("dec_max")),
                    "ttft_max": max(tt) if tt else None, "ttft_min": min(tt) if tt else None,
                })
    rows = []
    for (cell, phase), ms in reps.items():
        m0 = ms[0]
        row = {k: m0[k] for k in ("batch", "family", "engine", "ctx", "conc", "kind", "out", "variant", "cell")}
        row["phase"] = phase
        row["n"] = len(ms)
        for key in ("ttft_s", "dec_tps", "agg_tps", "dec_min", "dec_max", "ttft_max", "ttft_min"):
            vals = [x[key] for x in ms if x.get(key) is not None]
            if vals:
                row[key] = round(statistics.median(vals), 4)
                if len(vals) > 1:
                    row[key + "_spread_pct"] = round(100 * (max(vals) - min(vals)) / max(statistics.median(vals), 1e-9), 1)
        cached = [x.get("cached") for x in ms if isinstance(x.get("cached"), (int, float))]
        if cached:
            row["cached"] = int(statistics.median(cached))
        row["label"] = labels.get(cell)
        rows.append(row)
    rows.sort(key=lambda r: (r["batch"], r["family"], r["engine"], r["ctx"], r["kind"], r["phase"]))
    return rows, {"labels": labels, "unsupported": unsupported, "engines": engines_meta}


def progress(manifest: str, raw: str) -> list[dict]:
    if not manifest or not os.path.isfile(manifest):
        return []
    planned = defaultdict(int)
    for x in open(manifest):
        r = json.loads(x)
        if r.get("status") == "unverified" and r.get("batch") in ("B1", "B2") and r.get("engine") in ENGINE_ORDER:
            planned[(r["batch"], r["engine"])] += int(r.get("reps") or 2)
    done = defaultdict(int)
    for f in glob.glob(os.path.join(raw, "*", "*-r[0-9]*.jsonl")):
        name = os.path.basename(f)
        if "telemetry" in name or "superseded" in name:
            continue
        txt = open(f, errors="replace").read()
        if '"part_done"' in txt or '"part": "part_done"' in txt or "part_done" in txt:
            try:
                m = parse_cell_name(name)
                done[(m["batch"], m["engine"])] += 1
            except (StopIteration, ValueError, IndexError):
                pass
    out = []
    for (b, e), n in sorted(planned.items()):
        out.append({"batch": b, "engine": e, "planned": n, "done": done.get((b, e), 0)})
    return out


HTML = r"""<title>Yunshu 全面對比</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: instrument-panel report; summary strip, then curve panels, then the full table */
:root{--bg:#f6f7f5;--panel:#ffffff;--fg:#1d2321;--muted:#5f6b66;--line:#d9dfdb;--accent:#0f6e63;--warn:#a55c00;--bad:#b3261e;
--f-body:"IBM Plex Sans",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,monospace;--f-disp:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121615;--panel:#1a201e;--fg:#e6ebe8;--muted:#9aa6a1;--line:#2c3532;--accent:#4fc1b0;--warn:#f0a94b;--bad:#ff8a80;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121615;--panel:#1a201e;--fg:#e6ebe8;--muted:#9aa6a1;--line:#2c3532;--accent:#4fc1b0;--warn:#f0a94b;--bad:#ff8a80;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:22px}
h1{font-family:var(--f-disp);font-size:26px;margin:0;text-wrap:balance} h2{font-family:var(--f-disp);font-size:18px;margin:0 0 8px}
.sub{color:var(--muted);margin-top:4px;line-height:1.5}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font-family:var(--f-mono);font-size:20px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--muted);font-size:12px;letter-spacing:.03em}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px;color:var(--muted);font-size:12px}
select{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12px;margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.tbl{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),td.l{text-align:left}
th{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--panel)}
.prog{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:6px 16px;font-size:12px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:3px}.bar>i{display:block;height:100%;background:var(--accent)}
.lab{color:var(--warn);font-size:11px} .tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.note{color:var(--muted);font-size:12px;line-height:1.5}
</style>
<div class="wrap">
<header><h1>Yunshu 全面對比 · dev0</h1>
<div class="sub" id="sub"></div></header>
<section class="strip" id="kpis"></section>
<section class="card"><h2>進度</h2><div class="prog" id="prog"></div></section>
<section class="grid2">
 <div class="card"><h2>冷啟動 TTFT 對上下文長度</h2><div class="ctl">內容 <select id="k1"><option>code</option><option>prose</option></select> · 秒，對數軸</div><svg id="c1" width="100%"></svg><div class="legend" id="l1"></div></div>
 <div class="card"><h2>解碼速度對上下文長度</h2><div class="ctl">內容 <select id="k2"><option>code</option><option>prose</option></select> 階段 <select id="p2"><option value="cold">冷</option><option value="warm">暖</option><option value="turn2">第二輪</option></select> · tok/s</div><svg id="c2" width="100%"></svg><div class="legend" id="l2"></div></div>
 <div class="card"><h2>並行總吞吐</h2><div class="ctl">不同 prompt 並行 · 1K · tok/s</div><svg id="c3" width="100%"></svg><div class="legend" id="l3"></div></div>
 <div class="card"><h2>第二輪 TTFT 對上下文長度</h2><div class="ctl">內容 <select id="k4"><option>code</option><option>prose</option></select> · 秒</div><svg id="c4" width="100%"></svg><div class="legend" id="l4"></div></div>
</section>
<section class="card"><h2>所有格子</h2><div class="ctl">批次 <select id="fb"><option value="">全部</option><option>B1</option><option>B2</option></select> 類型 <select id="ff"><option value="">全部</option></select> 引擎 <select id="fe"><option value="">全部</option></select></div>
<div class="tbl"><table id="tb"><thead><tr><th>格子</th><th>引擎</th><th>階段</th><th>TTFT s</th><th>解碼 tok/s</th><th>總吞吐</th><th>快取命中</th><th>次數</th><th>差距%</th><th>標註</th></tr></thead><tbody></tbody></table></div></section>
<p class="note" id="foot"></p>
</div>
<div class="tip" id="tip" hidden></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
const D=JSON.parse(document.getElementById('data').textContent);
const rows=D.rows, ENG=D.engine_order.filter(e=>rows.some(r=>r.engine===e));
const COL=['#0f6e63','#d0742c','#4a6fd1','#a0438f','#5d8a2f','#b8902a','#2f8fa8','#c4453d','#6b5bb5','#7d7d2b','#3c6e8f','#9a5a3a'];
const color=e=>COL[D.engine_order.indexOf(e)%COL.length];
const CTX=['1K','4K','8K','16K','32K','64K','128K'], ctxN=c=>parseInt(c)*1024;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
document.getElementById('sub').textContent=`更新 ${D.generated} · 原始數據 ${D.raw} · ${rows.length} 筆中位數 · 每格 2 次重複，差距大自動補第 3 次。`;
const tot=D.progress.reduce((a,p)=>a+p.planned,0), dn=D.progress.reduce((a,p)=>a+p.done,0);
const kp=[[`${dn} / ${tot}`,'已完成的格子次數'],[`${Math.round(100*dn/Math.max(tot,1))}%`,'總進度'],[ENG.length,'已有數據的引擎'],[Object.keys(D.meta.unsupported).length,'標註為不支援的格子']];
document.getElementById('kpis').innerHTML='';kp.forEach(([v,l])=>{const d=document.createElement('div');d.className='kpi';const b=document.createElement('b');b.textContent=v;const s=document.createElement('span');s.textContent=l;d.append(b,s);document.getElementById('kpis').append(d)});
const pg=document.getElementById('prog');D.progress.forEach(p=>{const d=document.createElement('div');d.textContent=`${p.batch} · ${p.engine} — ${p.done}/${p.planned}`;const b=document.createElement('div');b.className='bar';const i=document.createElement('i');i.style.width=(100*p.done/Math.max(p.planned,1))+'%';b.append(i);d.append(b);pg.append(d)});
const tip=document.getElementById('tip');
function line(svgId,legId,series,{log=false,yl=''}={}){
 const svg=document.getElementById(svgId), W=Math.max(300,svg.clientWidth||svg.parentNode.clientWidth), H=240, m={l:46,r:10,t:10,b:28};
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML='';
 const xs=CTX.filter(c=>series.some(s=>s.pts.some(p=>p.x===c))); if(!xs.length){svg.innerHTML=`<text x="10" y="20" fill="${css('--muted')}">尚無數據</text>`;return}
 const ys=series.flatMap(s=>s.pts.map(p=>p.y)).filter(v=>v>0); let lo=Math.min(...ys),hi=Math.max(...ys);
 if(log){lo=Math.pow(10,Math.floor(Math.log10(lo)));hi=Math.pow(10,Math.ceil(Math.log10(hi)))}else{lo=0;hi=hi*1.08}
 const X=c=>m.l+(xs.indexOf(c)+.5)*(W-m.l-m.r)/xs.length, Y=v=>log?H-m.b-(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(H-m.t-m.b):H-m.b-(v-lo)/(hi-lo)*(H-m.t-m.b);
 const NS='http://www.w3.org/2000/svg',el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);svg.append(e);return e};
 const ticks=log?Array.from({length:Math.round(Math.log10(hi/lo))+1},(_,i)=>lo*Math.pow(10,i)):[0,.25,.5,.75,1].map(f=>lo+f*(hi-lo));
 ticks.forEach(t=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(t),y2:Y(t),stroke:css('--line')});const tx=el('text',{x:m.l-6,y:Y(t)+4,'text-anchor':'end','font-size':11,fill:css('--muted')});tx.textContent=t>=100?Math.round(t):(+t.toFixed(2))});
 xs.forEach(c=>{const tx=el('text',{x:X(c),y:H-8,'text-anchor':'middle','font-size':11,fill:css('--muted')});tx.textContent=c});
 series.forEach(s=>{const pts=s.pts.filter(p=>xs.includes(p.x)&&p.y>0).sort((a,b)=>ctxN(a.x)-ctxN(b.x));if(!pts.length)return;
  el('polyline',{points:pts.map(p=>`${X(p.x)},${Y(p.y)}`).join(' '),fill:'none',stroke:color(s.e),'stroke-width':2});
  pts.forEach(p=>{const c=el('circle',{cx:X(p.x),cy:Y(p.y),r:3.5,fill:color(s.e)});c.addEventListener('mousemove',ev=>{tip.hidden=false;tip.textContent=`${s.e} · ${p.x} · ${p.y.toFixed(2)} ${yl}${p.lab?' · '+p.lab:''}`;tip.style.left=Math.min(ev.clientX+12,innerWidth-260)+'px';tip.style.top=(ev.clientY+12)+'px'});c.addEventListener('mouseleave',()=>tip.hidden=true)})});
 const lg=document.getElementById(legId);lg.innerHTML='';series.filter(s=>s.pts.length).forEach(s=>{const d=document.createElement('span');const i=document.createElement('i');i.style.background=color(s.e);d.append(i,document.createTextNode(s.e));lg.append(d)});
}
const single=(fam)=>r=>(r.family===fam)&&r.conc===1;
function ser(filter,key,phase){return ENG.map(e=>({e,pts:rows.filter(r=>r.engine===e&&r.phase===phase&&filter(r)).map(r=>({x:r.ctx,y:r[key],lab:r.label}))}))}
function coreOrCtx(kind){return r=>(r.family==='core'||r.family==='ctx')&&r.conc===1&&r.kind===kind&&r.out===512}
function draw(){
 line('c1','l1',ser(coreOrCtx(k1.value),'ttft_s','cold'),{log:true,yl:'s'});
 line('c2','l2',ser(coreOrCtx(k2.value),'dec_tps',p2.value),{yl:'tok/s'});
 line('c4','l4',ser(coreOrCtx(k4.value),'ttft_s','turn2'),{log:true,yl:'s'});
 // concurrency: x = conc level as a category on the same helper
 const cs=ENG.map(e=>({e,pts:rows.filter(r=>r.engine===e&&r.phase==='conc'&&r.family==='conc'&&r.ctx==='1K').map(r=>({x:'c'+r.conc,y:r.agg_tps,lab:r.label}))}));
 const save=CTX.slice();CTX.length=0;CTX.push('c2','c4','c8');const old=ctxN;window._c=1;
 line('c3','l3',cs.map(s=>({e:s.e,pts:s.pts})),{yl:'tok/s'});CTX.length=0;save.forEach(c=>CTX.push(c));
 table();
}
const k1=document.getElementById('k1'),k2=document.getElementById('k2'),p2=document.getElementById('p2'),k4=document.getElementById('k4');
[k1,k2,p2,k4].forEach(s=>s.addEventListener('change',draw));
const fb=document.getElementById('fb'),ff=document.getElementById('ff'),fe=document.getElementById('fe');
[...new Set(rows.map(r=>r.family))].sort().forEach(f=>{const o=document.createElement('option');o.textContent=f;ff.append(o)});
ENG.forEach(e=>{const o=document.createElement('option');o.textContent=e;fe.append(o)});
[fb,ff,fe].forEach(s=>s.addEventListener('change',table));
const PH={cold:'冷',warm:'暖',turn2:'第二輪',restart:'重啟',fu_idle:'閒置後',conc:'並行',mixed:'混合'};
function table(){const tb=document.querySelector('#tb tbody');tb.innerHTML='';
 rows.filter(r=>(!fb.value||r.batch===fb.value)&&(!ff.value||r.family===ff.value)&&(!fe.value||r.engine===fe.value)).forEach(r=>{const tr=document.createElement('tr');
  const sp=[r.ttft_s_spread_pct,r.dec_tps_spread_pct,r.agg_tps_spread_pct].filter(x=>x!=null);
  const cells=[`${r.batch}.${r.family} ${r.ctx} c${r.conc} ${r.kind} o${r.out}${r.variant?' '+r.variant:''}`,r.engine,PH[r.phase]||r.phase,r.ttft_s!=null?r.ttft_s.toFixed(3):(r.ttft_max!=null?r.ttft_min.toFixed(2)+'–'+r.ttft_max.toFixed(2):'—'),r.dec_tps!=null?r.dec_tps.toFixed(1):(r.dec_min!=null?r.dec_min.toFixed(1)+'–'+r.dec_max.toFixed(1):'—'),r.agg_tps!=null?r.agg_tps.toFixed(1):'—',r.cached??'—',r.n,sp.length?Math.max(...sp):'—',r.label||''];
  cells.forEach((c,i)=>{const td=document.createElement('td');td.textContent=c;if(i===9){td.className='l lab'}tr.append(td)});tb.append(tr)});
}
const un=Object.entries(D.meta.unsupported);
document.getElementById('foot').textContent=`引擎版本：${Object.entries(D.meta.engines).map(([e,m])=>e+' '+(m.git_sha||m.version||'')).join(' · ')}。不支援的格子 ${un.length} 個（引擎自己的拒絕訊息記在原始數據裡）。這份頁面由 scripts/dev/fullcmp_report.py 從原始數據產生，數字是每格各次重複的中位數。`;
draw(); addEventListener('resize',()=>{clearTimeout(window._rt);window._rt=setTimeout(draw,150)});
})();
</script>
"""

ENGINE_ORDER = ["yunshu", "yunshu-pb2", "tf-1.2.1", "tf-1.1.8", "splash-1.5.0", "omlx-0.7.3", "omlx-0.7.3rc1",
                "mlx-serve-26.11.1", "kapitsa-0.2.26", "mlx-lm-server", "mlx-vlm-server", "llama-cpp-b12061"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, meta = collect(a.raw)
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "raw": os.path.basename(a.raw.rstrip("/")),
        "rows": rows,
        "meta": meta,
        "progress": progress(a.manifest, a.raw),
        "engine_order": ENGINE_ORDER,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "fullcmp_rows.json"), "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "fullcmp_report.html"), "w") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(f"{len(rows)} rows, {sum(p['done'] for p in data['progress'])}/{sum(p['planned'] for p in data['progress'])} cell-reps done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
