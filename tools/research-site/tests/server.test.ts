import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { Denied, readDoc, resolveDoc, resolveInside } from "../server/guard.ts";
import { createApp } from "../server/app.ts";
import { parseDecisions, queueStats, searchDocs, slimJob } from "../server/parse.ts";

let tmp: string;
let roots: { research: string; jobs: string; codex: string };
let secretDir: string;

before(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "rs-"));
  roots = { research: path.join(tmp, "research"), jobs: path.join(tmp, "jobs"), codex: path.join(tmp, "codex") };
  for (const d of Object.values(roots)) fs.mkdirSync(d, { recursive: true });
  secretDir = path.join(tmp, "outside");
  fs.mkdirSync(secretDir);
  fs.writeFileSync(path.join(secretDir, "secret.md"), "TOP SECRET");
  fs.writeFileSync(path.join(roots.research, "INDEX.md"), "# Index\n<!-- auto:stamp 1 -->\n");
  fs.writeFileSync(path.join(roots.research, "notes.md"), "# Notes\nhello speculative world\n");
  fs.writeFileSync(path.join(roots.research, ".env.md"), "KEY=1");
  fs.writeFileSync(path.join(roots.research, "api_token.md"), "x");
  fs.writeFileSync(path.join(roots.research, "bin.exe"), "x");
  fs.symlinkSync(secretDir, path.join(roots.research, "link"));
  fs.symlinkSync(path.join(secretDir, "secret.md"), path.join(roots.research, "alias.md"));
  fs.writeFileSync(path.join(roots.codex, "a_last.md"), "# report");
  fs.writeFileSync(
    path.join(roots.jobs, "j1.json"),
    JSON.stringify({ id: "j1", label: "alpha-x", state: "pending", priority: 1, submitted: Date.now() / 1000 - 90, env: { TOKEN: "SECRETVALUE" }, cmd: ["--api-key", "SECRETVALUE"], cwd: "/x" }),
  );
});
after(() => fs.rmSync(tmp, { recursive: true, force: true }));

describe("path guard", () => {
  const bad = ["../outside/secret.md", "..", "a/../../outside/secret.md", "/etc/passwd", "link/secret.md", "alias.md", "x\0y", "", "a/./b", "..\\outside"];
  for (const rel of bad)
    it(`rejects ${JSON.stringify(rel)}`, () => {
      assert.throws(() => resolveInside(roots.research, rel), Denied);
    });
  it("serves a normal file", () => {
    assert.equal(readDoc(roots, "r/notes.md").text.includes("hello"), true);
    assert.equal(readDoc(roots, "c/a_last.md").text, "# report");
  });
  it("blocks credential-looking names and unknown types", () => {
    assert.throws(() => resolveDoc(roots, "r/.env.md"), Denied);
    assert.throws(() => resolveDoc(roots, "r/api_token.md"), Denied);
    assert.throws(() => resolveDoc(roots, "r/bin.exe"), Denied);
    assert.throws(() => resolveDoc(roots, "x/notes.md"), Denied);
  });
});

describe("parsers", () => {
  it("parses decision tables with sections and supersession", () => {
    const md = `# t\n\n## A. 範圍\n\n| 日期 | 決策 | 原話 | 來源 |\n|---|---|---|---|\n| 2026-10-01 | 用 a \\| b | 「引」 | MEM x |\n| 2026-10-02 | 已取代：舊 | q | CLAUDE |\n\n## B. 其他\n| 日期 | 決策 | 原話 | 來源 |\n|---|---|---|---|\n| 2026-10 | z | q | s |\n`;
    const rows = parseDecisions(md);
    assert.equal(rows.length, 3);
    assert.equal(rows[0].decision, "用 a | b");
    assert.equal(rows[0].section, "A");
    assert.equal(rows[1].superseded, true);
    assert.equal(rows[2].section, "B");
  });
  it("slimJob drops cmd/env/cwd", () => {
    const j = slimJob({ id: "a", label: "l-x", state: "pending", waiting: "slot", cmd: ["s"], env: { A: "1" }, cwd: "/x" })!;
    assert.equal(j.display, "wait-slot");
    assert.ok(!("cmd" in j) && !("env" in j) && !("cwd" in j));
  });
  it("queueStats counts wasted minutes like gpuq stats", () => {
    const now = 10_000;
    const mk = (id: string, state: string, st: number, en: number) => slimJob({ id, label: "l-1", state, started: st, ended: en })!;
    const s = queueStats([mk("a", "done", now - 600, now - 540), mk("b", "failed", now - 300, now - 240), mk("c", "done", 0, 10)], now, 1);
    assert.equal(s.jobs, 2);
    assert.equal(Math.round(s.gpuMin), 2);
    assert.equal(Math.round(s.wastedMin), 1);
  });
  it("search ranks title hits first", () => {
    const hits = searchDocs([{ id: "r/a.md", title: "Other", text: "speculative" }, { id: "r/b.md", title: "Speculative decode", text: "x" }], "speculative");
    assert.equal(hits[0].id, "r/b.md");
  });
});

describe("http", () => {
  let server: http.Server;
  let base: string;
  before(async () => {
    const app = createApp({ roots, indexScript: "/nonexistent.py", staticDir: path.join(tmp, "dist") });
    server = app.server;
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  });
  after(() => server.close());
  it("rejects traversal over HTTP and never leaks the outside file", async () => {
    for (const q of ["r/../outside/secret.md", "r/link/secret.md", "r/alias.md", "r/%2e%2e/outside/secret.md"]) {
      const r = await fetch(`${base}/api/doc?id=${q}`);
      assert.ok(r.status >= 400, q);
      assert.ok(!(await r.text()).includes("TOP SECRET"));
    }
    const t = await fetch(`${base}/..%2f..%2fetc/passwd`);
    assert.ok(t.status >= 400 || !(await t.text()).includes("root:"));
  });
  it("is read-only", async () => {
    for (const m of ["POST", "PUT", "DELETE"]) assert.equal((await fetch(`${base}/api/index`, { method: m })).status, 405);
  });
  it("never exposes job env/cmd", async () => {
    const body = await (await fetch(`${base}/api/gpuq`)).text();
    assert.ok(!body.includes("SECRETVALUE"));
    const j = JSON.parse(body);
    assert.equal(j.pending.length, 1);
    assert.ok(j.pending[0].waitS >= 90);
  });
  it("index reports stale", async () => {
    const j = await (await fetch(`${base}/api/index`)).json();
    assert.equal(j.stale, true);
  });
  it("lines fail closed with an error, not fake data", async () => {
    const j = await (await fetch(`${base}/api/lines`)).json();
    assert.equal(j.ok, false);
    assert.deepEqual(j.lines, []);
  });
  it("missing doc gives 404 JSON", async () => {
    const r = await fetch(`${base}/api/doc?id=r/nope.md`);
    assert.equal(r.status, 404);
  });
  it("search finds a doc", async () => {
    const j = await (await fetch(`${base}/api/search?q=speculative`)).json();
    assert.equal(j[0].id, "r/notes.md");
  });
});
