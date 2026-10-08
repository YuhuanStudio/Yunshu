// Pure parsers (no I/O) so they can be unit-tested with plain strings.

export type Decision = {
  date: string;
  decision: string;
  quote: string;
  source: string;
  section: string;
  sectionTitle: string;
  superseded: boolean;
};

function cells(line: string): string[] {
  const t = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  const out: string[] = [];
  let cur = "";
  let tick = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (c === "\\" && t[i + 1] === "|") {
      cur += "|";
      i++;
    } else if (c === "`") {
      tick = !tick;
      cur += c;
    } else if (c === "|" && !tick) {
      out.push(cur.trim());
      cur = "";
    } else cur += c;
  }
  out.push(cur.trim());
  return out;
}

export function parseDecisions(md: string): Decision[] {
  const rows: Decision[] = [];
  let section = "";
  let sectionTitle = "";
  for (const line of md.split("\n")) {
    const h = /^##\s+([A-Z])\.\s*(.*)$/.exec(line);
    if (h) {
      section = h[1];
      sectionTitle = h[2].trim();
      continue;
    }
    if (!section || !line.trimStart().startsWith("|")) continue;
    const c = cells(line);
    if (c.length < 4 || /^:?-{2,}/.test(c[0]) || c[0] === "日期") continue;
    const text = c.join(" ");
    rows.push({
      date: c[0],
      decision: c[1],
      quote: c[2],
      source: c.slice(3).join(" | "),
      section,
      sectionTitle,
      superseded: /已取代/.test(c[1]) || c[1].startsWith("~~") || /^\s*~~/.test(text),
    });
  }
  return rows;
}

export type Job = {
  id: string;
  label: string;
  line: string;
  state: string;
  display: string;
  priority: number;
  short: boolean;
  gate: boolean;
  device: string;
  memGb: number | null;
  submitted: number | null;
  started: number | null;
  ended: number | null;
  rc: number | null;
  contended: boolean;
  quiet: boolean;
  minutes: number;
  bad: boolean;
};

/** Reduce a raw gpuq job record to a whitelist; cmd/env/cwd/outputs never leave the server. */
export function slimJob(j: Record<string, unknown>): Job | null {
  if (!j || typeof j.id !== "string") return null;
  const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const state = typeof j.state === "string" ? j.state : "unknown";
  const label = typeof j.label === "string" ? j.label : "";
  const started = num(j.started);
  const ended = num(j.ended);
  let paused = 0;
  if (Array.isArray(j.pauses))
    for (const p of j.pauses as unknown[]) {
      if (Array.isArray(p) && typeof p[0] === "number") paused += (typeof p[1] === "number" ? p[1] : (ended ?? p[0])) - p[0];
    }
  const minutes = started && ended ? Math.max(0, ended - started - paused) / 60 : 0;
  const contended = j.contended === true;
  const quiet = j.quiet === true;
  const bad = ["failed", "timeout", "stalled", "lost"].includes(state) || (state === "done" && contended && quiet);
  let display = state;
  if (state === "running" && j.paused) display = "paused";
  else if (state === "pending" && typeof j.waiting === "string" && j.waiting) display = `wait-${j.waiting}`;
  return {
    id: j.id,
    label,
    line: label.split("-")[0] || "?",
    state,
    display,
    priority: num(j.priority) ?? 0,
    short: j.short === true,
    gate: j.gate === true,
    device: typeof j.device === "string" ? j.device : "m5",
    memGb: num(j.mem_gb),
    submitted: num(j.submitted),
    started,
    ended,
    rc: num(j.rc),
    contended,
    quiet,
    minutes,
    bad,
  };
}

export type QueueStats = {
  windowH: number;
  jobs: number;
  gpuMin: number;
  wastedMin: number;
  perLine: { line: string; jobs: number; bad: number; gpuMin: number; wastedMin: number }[];
};

export function queueStats(jobs: Job[], now: number, hours: number): QueueStats {
  const per = new Map<string, { line: string; jobs: number; bad: number; gpuMin: number; wastedMin: number }>();
  for (const j of jobs) {
    if (!j.started || !j.ended || now - j.ended > hours * 3600) continue;
    const r = per.get(j.line) ?? { line: j.line, jobs: 0, bad: 0, gpuMin: 0, wastedMin: 0 };
    r.jobs++;
    r.gpuMin += j.minutes;
    if (j.bad) {
      r.bad++;
      r.wastedMin += j.minutes;
    }
    per.set(j.line, r);
  }
  const perLine = [...per.values()].sort((a, b) => b.wastedMin - a.wastedMin || b.gpuMin - a.gpuMin);
  return {
    windowH: hours,
    jobs: perLine.reduce((s, r) => s + r.jobs, 0),
    gpuMin: perLine.reduce((s, r) => s + r.gpuMin, 0),
    wastedMin: perLine.reduce((s, r) => s + r.wastedMin, 0),
    perLine,
  };
}

export function titleOf(md: string, fallback: string): string {
  for (const line of md.split("\n")) {
    const m = /^#\s+(.+)$/.exec(line);
    if (m) return m[1].trim();
  }
  return fallback;
}

export type Hit = { id: string; title: string; snippet: string; score: number };

export function searchDocs(docs: { id: string; title: string; text: string }[], query: string, limit = 40): Hit[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return [];
  const hits: Hit[] = [];
  for (const d of docs) {
    const low = d.text.toLowerCase();
    const title = d.title.toLowerCase();
    const id = d.id.toLowerCase();
    let score = 0;
    let first = -1;
    let ok = true;
    for (const t of terms) {
      const inTitle = title.includes(t) || id.includes(t);
      const at = low.indexOf(t);
      if (at < 0 && !inTitle) {
        ok = false;
        break;
      }
      if (first < 0 && at >= 0) first = at;
      score += (inTitle ? 20 : 0) + (at >= 0 ? 1 + Math.min(10, low.split(t).length - 1) : 0);
    }
    if (!ok) continue;
    const a = Math.max(0, first < 0 ? 0 : first - 50);
    const snippet = d.text.slice(a, a + 160).replace(/\s+/g, " ").trim();
    hits.push({ id: d.id, title: d.title, snippet, score });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}
