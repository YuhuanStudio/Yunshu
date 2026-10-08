// Read-only research server: JSON API + SSE + the built SPA. It never writes, and the only
// subprocess it starts is `research_index.py --json -`, which is read-only by construction.
import { execFile } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { Denied, readDoc, resolveInside, assertReadable, type Roots } from "./guard.ts";
import { parseDecisions, queueStats, searchDocs, slimJob, titleOf, type Job } from "./parse.ts";

export type Config = {
  roots: Roots;
  indexScript: string;
  staticDir: string;
  python?: string;
};

type DocEntry = { id: string; title: string; mtime: number; size: number; text: string };

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".woff2": "font/woff2",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
};

function walkMd(root: string, out: string[], rel = "", depth = 0): void {
  if (depth > 6) return;
  let ents: fs.Dirent[];
  try {
    ents = fs.readdirSync(path.join(root, rel), { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of ents) {
    if (e.name.startsWith(".") || e.name === "node_modules" || e.isSymbolicLink()) continue;
    const r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) walkMd(root, out, r, depth + 1);
    else if (e.isFile() && e.name.toLowerCase().endsWith(".md")) out.push(r);
  }
}

export function createApp(cfg: Config) {
  const { roots } = cfg;
  const clients = new Set<http.ServerResponse>();
  const docCache = new Map<string, DocEntry>();
  const jobCache = new Map<string, { mtime: number; job: Job | null }>();
  let linesCache: { at: number; value: unknown } | null = null;
  let linesFlight: Promise<unknown> | null = null;
  const watchers: fs.FSWatcher[] = [];

  function docs(): DocEntry[] {
    const files: string[] = [];
    walkMd(roots.research, files);
    const seen = new Set<string>();
    for (const rel of files) {
      const id = `r/${rel}`;
      seen.add(id);
      try {
        const abs = path.join(roots.research, rel);
        const st = fs.statSync(abs);
        const old = docCache.get(id);
        if (old && old.mtime === st.mtimeMs) continue;
        const fd = fs.openSync(abs, "r");
        const len = Math.min(st.size, 300_000);
        const buf = Buffer.alloc(len);
        fs.readSync(fd, buf, 0, len, 0);
        fs.closeSync(fd);
        const text = buf.toString("utf8");
        docCache.set(id, { id, title: titleOf(text, path.basename(rel, ".md")), mtime: st.mtimeMs, size: st.size, text });
      } catch {
        docCache.delete(id);
      }
    }
    for (const id of [...docCache.keys()]) if (!seen.has(id)) docCache.delete(id);
    return [...docCache.values()];
  }

  function allJobs(now: number): Job[] {
    let names: string[] = [];
    try {
      names = fs.readdirSync(roots.jobs).filter((n) => n.endsWith(".json"));
    } catch {
      return [];
    }
    const out: Job[] = [];
    const live = new Set(names);
    for (const n of names) {
      const f = path.join(roots.jobs, n);
      try {
        const st = fs.statSync(f);
        if (now - st.mtimeMs / 1000 > 72 * 3600) continue;
        let c = jobCache.get(n);
        if (!c || c.mtime !== st.mtimeMs) {
          c = { mtime: st.mtimeMs, job: slimJob(JSON.parse(fs.readFileSync(f, "utf8"))) };
          jobCache.set(n, c);
        }
        if (c.job) out.push(c.job);
      } catch {
        /* a half-written job file: skip this round */
      }
    }
    for (const k of [...jobCache.keys()]) if (!live.has(k)) jobCache.delete(k);
    return out;
  }

  function registeredWorkers(): { name: string; pid: number; prefixes: string[] }[] {
    try {
      const raw = JSON.parse(fs.readFileSync(path.join(roots.codex, "workers.json"), "utf8")) as Record<string, { pid?: number; prefixes?: string[] }>;
      const out = [];
      for (const [name, w] of Object.entries(raw)) {
        if (typeof w.pid !== "number") continue;
        try {
          process.kill(w.pid, 0);
        } catch (e) {
          if ((e as NodeJS.ErrnoException).code !== "EPERM") continue;
        }
        out.push({ name, pid: w.pid, prefixes: Array.isArray(w.prefixes) ? w.prefixes : [] });
      }
      return out;
    } catch {
      return [];
    }
  }

  function runLines(): Promise<unknown> {
    if (linesCache && Date.now() - linesCache.at < 8000) return Promise.resolve(linesCache.value);
    if (linesFlight) return linesFlight;
    linesFlight = new Promise((resolve) => {
      execFile(
        cfg.python ?? "python3",
        [cfg.indexScript, "--json", "-", "--codex", roots.codex, "--jobs", roots.jobs],
        { timeout: 25_000, maxBuffer: 8_000_000 },
        (err, stdout) => {
          let value: unknown;
          try {
            if (err) throw err;
            const j = JSON.parse(stdout) as { generated: number; lines: Record<string, unknown>[] };
            const workers = registeredWorkers();
            for (const l of j.lines) {
              const stem = String(l.branch).split("/").pop() ?? "";
              l.handoff = fs.existsSync(path.join(roots.research, stem, "HANDOFF.md")) ? `r/${stem}/HANDOFF.md` : "";
              const rp = String(l.report_path || "");
              l.report_id = rp && path.dirname(rp) === roots.codex ? `c/${path.basename(rp)}` : "";
              l.registered = workers.filter((w) => w.prefixes.some((p) => p.startsWith(stem + "-"))).map((w) => w.name);
              delete l.report_path;
              delete l.worktree;
            }
            value = { ok: true, generated: j.generated, lines: j.lines };
          } catch (e) {
            value = { ok: false, error: `research_index.py --json failed: ${(e as Error).message.slice(0, 200)}`, lines: [] };
          }
          linesCache = { at: Date.now(), value };
          linesFlight = null;
          resolve(value);
        },
      );
    });
    return linesFlight;
  }

  function broadcast(topic: string) {
    linesCache = topic === "tick" || topic === "research" || topic === "codex" || topic === "jobs" ? null : linesCache;
    for (const c of clients) c.write(`data: ${JSON.stringify({ topic, at: Date.now() })}\n\n`);
  }

  function startWatch() {
    const pending = new Map<string, NodeJS.Timeout>();
    const watch = (dir: string, topic: string, recursive: boolean) => {
      try {
        const w = fs.watch(dir, { recursive }, () => {
          if (pending.has(topic)) return;
          pending.set(
            topic,
            setTimeout(() => {
              pending.delete(topic);
              broadcast(topic);
            }, 500),
          );
        });
        w.on("error", () => {});
        watchers.push(w);
      } catch {
        /* directory missing: the pages show an empty state */
      }
    };
    watch(roots.research, "research", true);
    watch(roots.jobs, "jobs", false);
    watch(roots.codex, "codex", false);
    const tick = setInterval(() => broadcast("tick"), 30_000);
    const beat = setInterval(() => {
      for (const c of clients) c.write(": ping\n\n");
    }, 20_000);
    tick.unref();
    beat.unref();
    watchers.push({ close: () => (clearInterval(tick), clearInterval(beat)) } as fs.FSWatcher);
  }

  const send = (res: http.ServerResponse, status: number, body: unknown) => {
    const text = JSON.stringify(body);
    res.writeHead(status, { "content-type": MIME[".json"], "cache-control": "no-store", "x-content-type-options": "nosniff" });
    res.end(text);
  };

  async function api(url: URL, res: http.ServerResponse) {
    const p = url.pathname;
    const now = Date.now() / 1000;
    if (p === "/api/meta") {
      return send(res, 200, {
        researchRoot: roots.research,
        codexRoot: roots.codex,
        now: Date.now(),
        hasResearch: fs.existsSync(roots.research),
      });
    }
    if (p === "/api/index") {
      const d = readDoc(roots, "r/INDEX.md");
      const m = /<!-- auto:stamp (\d+) -->/.exec(d.text);
      const stamp = m ? Number(m[1]) : null;
      const ageMin = stamp ? (now - stamp) / 60 : null;
      return send(res, 200, { ...d, stamp, ageMin, stale: ageMin === null || ageMin > 60 });
    }
    if (p === "/api/lines") return send(res, 200, await runLines());
    if (p === "/api/decisions") {
      const d = readDoc(roots, "r/DECISIONS.md");
      return send(res, 200, { mtime: d.mtime, rows: parseDecisions(d.text) });
    }
    if (p === "/api/parity") {
      const f = resolveInside(roots.research, "parityboard/board.json");
      const b = JSON.parse(fs.readFileSync(f, "utf8"));
      const items = (b.items ?? []) as Record<string, unknown>[];
      const engines = new Set<string>();
      for (const it of items) for (const e of Object.keys((it.engines as object) ?? {})) engines.add(e);
      return send(res, 200, {
        mtime: fs.statSync(f).mtimeMs,
        verdict: b.verdict,
        summary: b.summary,
        policy: b.policy,
        parity: b.parity,
        total: b.total,
        withData: b.with_data,
        bestOurs: b.best_ours,
        gateOpen: b.gate_open,
        missing: Array.isArray(b.missing) ? b.missing.length : null,
        headToHead: b.head_to_head,
        rejected: b.rejected ?? [],
        methodFlagged: Array.isArray(b.memory_method_flagged) ? b.memory_method_flagged.length : 0,
        engines: [...engines],
        gaps: items
          .filter((it) => it.provisional)
          .map((it) => ({
            model: it.model,
            ctx: it.ctx,
            kind: it.kind,
            metric: it.metric,
            higherIsBetter: it.higher_is_better,
            ...(it.provisional as object),
          })),
      });
    }
    if (p === "/api/gpuq") {
      const jobs = allJobs(now);
      const running = jobs.filter((j) => j.state === "running").map((j) => ({ ...j, runningS: j.started ? now - j.started : null }));
      const pending = jobs
        .filter((j) => j.state === "pending")
        .map((j) => ({ ...j, waitS: j.submitted ? now - j.submitted : null }))
        .sort((a, b) => b.priority - a.priority || (a.submitted ?? 0) - (b.submitted ?? 0));
      const byPriority: Record<string, number> = {};
      const byLine: Record<string, { pending: number; running: number; maxWaitS: number }> = {};
      for (const j of [...running, ...pending]) {
        const l = (byLine[j.line] ??= { pending: 0, running: 0, maxWaitS: 0 });
        if (j.state === "running") l.running++;
        else {
          l.pending++;
          l.maxWaitS = Math.max(l.maxWaitS, (j as { waitS: number | null }).waitS ?? 0);
          byPriority[`p${j.priority}`] = (byPriority[`p${j.priority}`] ?? 0) + 1;
        }
      }
      return send(res, 200, {
        now: Date.now(),
        running,
        pending,
        byPriority,
        byLine,
        stats1h: queueStats(jobs, now, 1),
        stats24h: queueStats(jobs, now, 24),
      });
    }
    if (p === "/api/docs") {
      return send(
        res,
        200,
        docs()
          .map(({ id, title, mtime, size }) => ({ id, title, mtime, size }))
          .sort((a, b) => a.id.localeCompare(b.id)),
      );
    }
    if (p === "/api/reports") {
      const out = fs
        .readdirSync(roots.codex)
        .filter((n) => n.endsWith("_last.md"))
        .map((n) => ({ id: `c/${n}`, title: n, mtime: fs.statSync(path.join(roots.codex, n)).mtimeMs }));
      return send(res, 200, out.sort((a, b) => b.mtime - a.mtime));
    }
    if (p === "/api/doc") {
      const id = url.searchParams.get("id") ?? "";
      const d = readDoc(roots, id);
      return send(res, 200, { ...d, title: titleOf(d.text, path.basename(id)) });
    }
    if (p === "/api/search") {
      const q = (url.searchParams.get("q") ?? "").slice(0, 200);
      return send(res, 200, searchDocs(docs(), q));
    }
    if (p === "/api/events") {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive" });
      res.write(`data: ${JSON.stringify({ topic: "hello", at: Date.now() })}\n\n`);
      clients.add(res);
      res.on("close", () => clients.delete(res));
      return;
    }
    throw new Denied("not found", 404);
  }

  function serveStatic(url: URL, res: http.ServerResponse) {
    let rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    if (!rel) rel = "index.html";
    let file: string;
    try {
      file = resolveInside(cfg.staticDir, rel);
      if (!fs.statSync(file).isFile()) throw new Error("dir");
    } catch {
      if (path.extname(rel)) {
        res.writeHead(404).end("not found");
        return;
      }
      file = path.join(cfg.staticDir, "index.html"); // SPA fallback
      if (!fs.existsSync(file)) {
        res.writeHead(503, { "content-type": "text/plain; charset=utf-8" }).end("UI not built: run `pnpm build` in tools/research-site");
        return;
      }
    }
    res.writeHead(200, {
      "content-type": MIME[path.extname(file)] ?? "application/octet-stream",
      "cache-control": file.includes(`${path.sep}assets${path.sep}`) ? "public, max-age=31536000, immutable" : "no-cache",
      "x-content-type-options": "nosniff",
    });
    fs.createReadStream(file).pipe(res);
  }

  const server = http.createServer((req, res) => {
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, { allow: "GET, HEAD" }).end("read-only");
      return;
    }
    let url: URL;
    try {
      url = new URL(req.url ?? "/", "http://x");
    } catch {
      res.writeHead(400).end("bad url");
      return;
    }
    if (url.pathname.startsWith("/api/")) {
      api(url, res).catch((e) => {
        if (res.headersSent) return res.end();
        if (e instanceof Denied) return send(res, e.status, { error: e.message });
        const code = (e as NodeJS.ErrnoException).code;
        if (code === "ENOENT") return send(res, 404, { error: "not found" });
        send(res, 500, { error: "internal error" });
      });
      return;
    }
    serveStatic(url, res);
  });
  server.on("close", () => watchers.forEach((w) => w.close()));
  return { server, start: startWatch, assertReadable };
}
