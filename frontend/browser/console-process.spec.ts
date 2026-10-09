import { expect, test, type Page } from "@playwright/test";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "node:net";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The console is its own process: it serves the web console and docs, proxies the engine API on one
// origin and records the history. These tests run the REAL console process (python -m yunshu_console)
// against a stand-in engine process that they kill and restart, with a browser attached throughout.
//
// Needs a Python with the repo's dependencies: YUNSHU_TEST_PYTHON, else <repo>/.venv/bin/python.

const REPO = resolve(__dirname, "../..");
const PYTHON = process.env.YUNSHU_TEST_PYTHON ?? join(REPO, ".venv/bin/python");
const ENV = {
  ...process.env,
  PYTHONPATH: join(REPO, "python"),
  PYTHONUNBUFFERED: "1",
};

const procs: ChildProcess[] = [];
let home = "";
let dist = "";

function start(
  args: string[],
  extra: Record<string, string> = {},
): ChildProcess {
  const child = spawn(PYTHON, args, {
    env: { ...ENV, HOME: home, ...extra },
    stdio: ["ignore", "ignore", "pipe"],
  });
  child.stderr?.on("data", (d) =>
    process.stderr.write(`[proc ${args[1] ?? args[0]}] ${d}`),
  );
  procs.push(child);
  return child;
}

async function until<T>(
  what: string,
  fn: () => Promise<T | false | null | undefined>,
  ms = 25_000,
): Promise<T> {
  const end = Date.now() + ms;
  let last: unknown;
  while (Date.now() < end) {
    try {
      const v = await fn();
      if (v) return v;
    } catch (e) {
      last = e;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new Error(`timed out waiting for ${what}${last ? `: ${last}` : ""}`);
}

async function freePort(not = 0): Promise<number> {
  for (let port = 18990; port <= 18999; port++) {
    if (port === not) continue;
    const free = await new Promise<boolean>((done) => {
      const s = createServer();
      s.once("error", () => done(false));
      s.listen(port, "127.0.0.1", () => s.close(() => done(true)));
    });
    if (free) return port;
  }
  throw new Error("no free port in 18990-18999");
}

const json = async (url: string, init?: RequestInit) =>
  (await fetch(url, init)).json() as Promise<any>;
const killAndWait = async (p: ChildProcess) => {
  const done = new Promise((r) => p.once("exit", r));
  p.kill("SIGKILL");
  await done;
};

test.describe.configure({ mode: "serial" });

let engineProc: ChildProcess;
let consoleProc: ChildProcess;
let E = 0;
let C = 0;
const consoleUrl = () => `http://127.0.0.1:${C}`;
const engineUrl = () => `http://127.0.0.1:${E}`;

test.beforeAll(async () => {
  test.setTimeout(120_000);
  // Server ports for agents are 18990-18999; other jobs use some of them, so take two that are free.
  E = await freePort();
  C = await freePort(E);
  home = mkdtempSync(join(tmpdir(), "yunshu-e2e-"));
  dist = mkdtempSync(join(tmpdir(), "yunshu-e2e-dist-"));
  execFileSync(
    "npx",
    ["vite", "build", "--outDir", dist, "--emptyOutDir", "--logLevel", "error"],
    {
      cwd: resolve(__dirname, ".."),
      stdio: ["ignore", "ignore", "inherit"],
    },
  );
  engineProc = start([join(__dirname, "support/fake_engine.py"), String(E)]);
  await until(
    "the stand-in engine",
    async () => (await fetch(`${engineUrl()}/health`)).ok,
  );
  consoleProc = start(
    [
      "-m",
      "yunshu_console",
      "--engine",
      engineUrl(),
      "--port",
      String(C),
      "--static-dir",
      dist,
      "--log-level",
      "warning",
    ],
    { YUNSHU_CONSOLE_POLL_S: "0.25" },
  );
  await until(
    "the console process",
    async () => (await json(`${consoleUrl()}/v1/yunshu/console`)).up === true,
  );
});

test.afterAll(async () => {
  for (const p of procs) if (p.exitCode === null) p.kill("SIGKILL");
  rmSync(home, { recursive: true, force: true });
  rmSync(dist, { recursive: true, force: true });
});

async function open(page: Page, hash: string) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.goto(`${consoleUrl()}/console/#/${hash}`, {
    waitUntil: "domcontentloaded",
  });
}

test("requests that finished while no browser was open are in the history afterwards", async ({
  page,
}) => {
  // the engine serves three requests; the console process records them with nobody watching
  await json(`${engineUrl()}/__finish`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ n: 3 }),
  });
  await until("the three requests in the recorded log", async () => {
    const h = await json(`${consoleUrl()}/v1/yunshu/requests/history`);
    return h.count === 3 && h;
  });
  await open(page, "requests");
  await expect(page.getByRole("button", { name: "詳情" })).toHaveCount(3, {
    timeout: 15_000,
  });
});

test("metrics were recorded all along and the charts backfill from them", async ({
  page,
}) => {
  await until("recorded rows", async () => {
    const m = await json(
      `${consoleUrl()}/v1/yunshu/metrics/history?since=${Date.now() / 1000 - 600}`,
    );
    return m.series.t.length >= 4 && m;
  });
  const m = await json(
    `${consoleUrl()}/v1/yunshu/metrics/history?since=${Date.now() / 1000 - 600}`,
  );
  expect(m.tier).toBe("m1");
  expect(m.series.active_gb.at(-1)).toBeCloseTo(12, 0);
  expect(m.series.gpu_w.at(-1)).toBe(9);
  await open(page, "overview");
  await expect(
    page
      .getByTestId("live-panel")
      .or(page.getByTestId("overview-stats"))
      .first(),
  ).toBeVisible();
});

test("killing the engine mid-session: the console stays usable, says so, and keeps recording the gap", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await open(page, "overview");
  await expect(page.getByTestId("overview-stats")).toBeVisible({
    timeout: 15_000,
  });

  await killAndWait(engineProc);

  // the offline state: since when, and when the next try is
  const banner = page.getByRole("status").filter({ hasText: /離線/ }).first();
  await expect(banner).toBeVisible({ timeout: 20_000 });
  await expect(banner).toContainText(/自 \d{1,2}:\d{2}:\d{2} 離線/);
  await expect(banner).toContainText(/秒後重試|正在重試/);
  await expect(page.getByTestId("engine-needed")).toBeVisible();

  // everything that does not need the engine keeps working: the docs, the settings, the shell
  await page.evaluate(() => (location.hash = "#/docs/getting-started/install"));
  await expect(
    page.getByTestId("docs-article").getByRole("heading", { level: 1 }),
  ).toBeVisible({ timeout: 10_000 });
  await page.evaluate(() => (location.hash = "#/settings"));
  await expect(page.getByTestId("settings")).toBeVisible();
  await page.evaluate(() => (location.hash = "#/models"));
  await expect(page.getByTestId("engine-needed")).toBeVisible();
  await page.evaluate(() => (location.hash = "#/overview"));
  await expect(page.getByTestId("overview-stats")).toBeVisible();

  // the console process saw it and recorded it as an event, not as made-up zeros
  const state = await until("the console noticing", async () => {
    const s = await json(`${consoleUrl()}/v1/yunshu/console`);
    return s.up === false && s;
  });
  expect(state.recording).toBe(true);
  expect(errors).toEqual([]);
});

test("restarting the engine: the session reconnects, the data resumes and the gap is on record", async ({
  page,
}) => {
  await open(page, "overview");
  await expect(
    page.getByRole("status").filter({ hasText: /離線/ }).first(),
  ).toBeVisible({ timeout: 20_000 });
  const downAt = Date.now() / 1000;
  await new Promise((r) => setTimeout(r, 2500));
  engineProc = start([join(__dirname, "support/fake_engine.py"), String(E)]);
  await until(
    "the engine back",
    async () => (await fetch(`${engineUrl()}/health`)).ok,
  );
  await json(`${engineUrl()}/__busy/1`, { method: "POST" });

  // the page reconnects by itself: the banner goes, live numbers come back
  await expect(
    page.getByRole("status").filter({ hasText: /離線/ }),
  ).toHaveCount(0, { timeout: 30_000 });
  await expect(page.getByTestId("engine-needed")).toHaveCount(0);

  const m = await until("the recorded history with its gap", async () => {
    const h = await json(
      `${consoleUrl()}/v1/yunshu/metrics/history?since=${downAt - 120}`,
    );
    return (
      h.gaps.length >= 1 &&
      h.events.some((e: any) => e.kind === "engine_reachable") &&
      h
    );
  });
  const kinds = m.events.map((e: any) => e.kind);
  expect(kinds).toContain("engine_unreachable");
  expect(kinds).toContain("engine_reachable");
  expect(kinds).toContain("engine_restarted");
  const gap = m.gaps[0];
  expect(gap[1] - gap[0]).toBeGreaterThan(1.5);
  // the engine's own ring is empty after the restart; the recorded log still lists the earlier requests
  await json(`${engineUrl()}/__busy/0`, { method: "POST" });
  await open(page, "requests");
  await expect(page.getByRole("button", { name: "詳情" })).toHaveCount(3, {
    timeout: 15_000,
  });
  // the recorder resumed
  const after = await json(
    `${consoleUrl()}/v1/yunshu/metrics/history?since=${Date.now() / 1000 - 5}`,
  );
  expect(after.series.t.length).toBeGreaterThan(0);
});

test("a model that failed to load shows its error and keeps load-another usable", async ({
  page,
}) => {
  await json(`${engineUrl()}/__load_error`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: "out of memory while loading org/big-model",
    }),
  });
  await open(page, "overview");
  const banner = page.getByTestId("load-error");
  await expect(banner).toBeVisible({ timeout: 15_000 });
  await expect(banner).toContainText(
    "out of memory while loading org/big-model",
  );
  await banner.getByRole("button", { name: /載入其他模型/ }).click();
  await expect(page).toHaveURL(/#\/models/);
  await json(`${engineUrl()}/__load_error`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: null }),
  });
});

test("the console process is a proxy: one origin reaches the engine API", async () => {
  const status = await json(`${consoleUrl()}/v1/yunshu/status`);
  expect(status.object).toBe("yunshu.status");
  await killAndWait(engineProc);
  const r = await fetch(`${consoleUrl()}/v1/yunshu/status`);
  expect(r.status).toBe(502);
  expect((await r.json()).error.type).toBe("engine_unreachable");
  engineProc = start([join(__dirname, "support/fake_engine.py"), String(E)]);
  await until(
    "the engine back",
    async () => (await fetch(`${engineUrl()}/health`)).ok,
  );
});
