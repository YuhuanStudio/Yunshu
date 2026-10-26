// Motion evidence: records screencast frames (every compositor frame, ~60 fps) for each motion scenario
// (the console's page transitions and reveals), and measures CLS after first paint per page.
//
//   node scripts/motion/evidence.mjs <label> <console-url> <out-dir> [--only a,b] [--themes dark,light] [--vps 1440,390] [--cls]
//
// The console is served by `python -m yunshu_console` against the stand-in engine from
// browser/support/fake_engine.py (started and stopped here on ENGINE_PORT). Scenarios that need data
// the stand-in does not have (chat streaming) are MOCKED in the page and labelled `mock` in
// scenarios.json. Frames: <out>/<label>/<scenario>/<theme>-<vp>/f_<ms>.jpg, plus anim.json
// (CSS animation / transition starts seen during the recording).
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [, , label, url, outRoot, ...rest] = process.argv;
const arg = (k, d) => {
  const i = rest.indexOf(k);
  return i >= 0 ? rest[i + 1] : d;
};
const only = arg("--only", "")?.split(",").filter(Boolean);
const themes = arg("--themes", "dark,light").split(",");
const vps = arg("--vps", "1440,390").split(",").map(Number);
const wantCls = rest.includes("--cls");
const ENGINE_PORT = 18993;
const PY = process.env.PYTHON ?? "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python";
const here = new URL(".", import.meta.url).pathname;
const engineScript = resolve(here, "../../browser/support/fake_engine.py");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let engine = null;
async function startEngine() {
  try {
    if ((await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`)).ok) throw new Error(`port ${ENGINE_PORT} already serves an engine; stop it first`);
  } catch (e) {
    if (String(e).includes("already serves")) throw e;
  }
  engine = spawn(PY, [engineScript, String(ENGINE_PORT)], { stdio: "ignore" });
  for (let i = 0; i < 50; i++) {
    try {
      if ((await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`)).ok) return;
    } catch {}
    await sleep(100);
  }
  throw new Error("stand-in engine did not start");
}
async function stopEngine() {
  if (!engine) return;
  if (engine.exitCode === null) {
    const done = new Promise((r) => engine.once("exit", r));
    engine.kill("SIGKILL");
    await done;
  }
  engine = null;
}
const post = (p, body) =>
  fetch(`http://127.0.0.1:${ENGINE_PORT}${p}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body ?? {}) });

// ---- recording -------------------------------------------------------------------------------
async function newPage(browser, theme, vp) {
  const ctx = await browser.newContext({
    viewport: { width: vp, height: vp === 390 ? 844 : 900 },
    locale: "en-US",
    colorScheme: theme,
    deviceScaleFactor: 1,
    reducedMotion: process.env.REDUCED ? "reduce" : "no-preference",
  });
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem("yunshu.console.theme", matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    } catch {}
    window.__anim = [];
    const add = (kind) => (e) =>
      window.__anim.push({ t: Math.round(performance.now()), kind, name: e.animationName ?? e.propertyName, el: (e.target?.tagName ?? "") + "." + String(e.target?.className ?? "").slice(0, 40) });
    document.addEventListener("animationstart", add("animation"), true);
    document.addEventListener("transitionrun", add("transition"), true);
  });
  const page = await ctx.newPage();
  return { ctx, page };
}

async function record(page, dir, action, ms) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const cdp = await page.context().newCDPSession(page);
  const frames = [];
  let t0 = null;
  cdp.on("Page.screencastFrame", (f) => {
    const ts = f.metadata.timestamp * 1000;
    if (t0 === null) t0 = ts;
    frames.push({ t: Math.round(ts - t0), data: f.data });
    cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
  });
  await page.evaluate(() => (window.__anim.length = 0));
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 80, everyNthFrame: 1 });
  await sleep(120);
  const start = Date.now();
  await action();
  const left = ms - (Date.now() - start);
  if (left > 0) await sleep(left);
  await cdp.send("Page.stopScreencast");
  const anim = await page.evaluate(() => window.__anim.slice());
  await cdp.detach().catch(() => {});
  const { writeFileSync: w } = await import("node:fs");
  for (const f of frames) w(join(dir, `f_${String(f.t).padStart(5, "0")}.jpg`), Buffer.from(f.data, "base64"));
  const names = {};
  for (const a of anim) names[`${a.kind}:${a.name}`] = (names[`${a.kind}:${a.name}`] ?? 0) + 1;
  w(join(dir, "anim.json"), JSON.stringify({ frames: frames.length, spanMs: frames.at(-1)?.t ?? 0, starts: names }, null, 1));
  return frames.length;
}

const click = (page, sel, opts) => page.locator(`${sel}:visible`).first().click({ timeout: 8000, ...opts });
// The console process polls the engine on its own clock; wait until it sees the stand-in engine.
async function waitUp() {
  const base = new URL(url).origin;
  for (let i = 0; i < 120; i++) {
    try {
      if ((await (await fetch(`${base}/v1/yunshu/console`)).json()).up === true) return;
    } catch {}
    await sleep(500);
  }
  throw new Error("console never saw the engine");
}
const goto = async (page, hash) => {
  await page.goto(`${url}${hash}`);
  await page.waitForSelector("[data-testid='page-scroll'], [data-testid='playground'], h1", { timeout: 15000 });
  await sleep(1800);
};
const mobile = (vp) => vp === 390;
async function openNav(page, vp) {
  if (mobile(vp)) await page.getByRole("button", { name: "Open navigation" }).click();
  await sleep(500);
}

const CHAT = `Sure. Here is a short answer in **three parts**.

1. Prefill reads the prompt once and fills the KV cache.
2. Decode then emits one token at a time from that cache.
3. A prefix hit skips most of step 1.

\`\`\`python
def decode(cache, tok):
    return next_token(cache, tok)
\`\`\`

That is the whole loop; the rest is scheduling.`;

// ---- scenarios -----------------------------------------------------------------------------------
// each: { id, mock?, ms, setup(page,vp), action(page,vp) }
const scenarios = [
  {
    id: "page-switch",
    ms: 900,
    setup: (p) => goto(p, "#/overview"),
    action: async (p, vp) => {
      if (mobile(vp)) {
        await openNav(p, vp);
      }
      await p.getByRole("link", { name: "Requests", exact: true }).first().click();
    },
  },
  {
    id: "skeleton-arrival",
    ms: 2600,
    setup: async (p) => {
      await goto(p, "#/models");
      await p.route("**/v1/yunshu/status*", async (r) => {
        await sleep(1000);
        await r.continue();
      });
    },
    action: async (p) => {
      await p.goto(`${url}#/overview`);
      await p.reload();
    },
  },
  {
    id: "live-1hz",
    ms: 4200,
    setup: async (p) => {
      await goto(p, "#/overview");
      await post("/__busy/1");
    },
    action: async () => {
      for (let i = 0; i < 4; i++) {
        await post("/__finish", { n: 1 });
        await sleep(1000);
      }
    },
    after: () => post("/__busy/0"),
  },
  {
    id: "engine-restart",
    ms: 9000,
    setup: (p) => goto(p, "#/overview"),
    action: async () => {
      await stopEngine();
      await sleep(3500);
      await startEngine();
      await sleep(3500);
    },
  },
  {
    id: "island",
    only: (vp) => !mobile(vp),
    ms: 1500,
    setup: (p) => goto(p, "#/overview"),
    action: async (p) => {
      await click(p, "[data-testid='footer-trigger']");
      await sleep(700);
      await p.keyboard.press("Escape");
    },
  },
  {
    id: "menu-popover",
    ms: 1500,
    setup: (p) => goto(p, "#/overview"),
    action: async (p, vp) => {
      await click(p, mobile(vp) ? "button[aria-label='Notifications']" : "button[aria-label^='Language']");
      await sleep(700);
      await p.keyboard.press("Escape");
    },
  },
  {
    id: "dialog",
    ms: 1500,
    setup: (p) => goto(p, "#/overview"),
    action: async (p) => {
      await p.keyboard.press("Meta+k");
      await sleep(700);
      await p.keyboard.press("Escape");
    },
  },
  {
    id: "drawer",
    ms: 1500,
    only: (vp) => mobile(vp),
    setup: (p) => goto(p, "#/overview"),
    action: async (p) => {
      await p.getByRole("button", { name: "Open navigation" }).click();
      await sleep(700);
      await p.getByRole("button", { name: "Close navigation" }).first().click();
    },
  },
  {
    id: "accordion",
    ms: 1400,
    setup: (p) => goto(p, "#/overview"),
    action: async (p) => {
      await p.getByRole("button", { name: "Details", exact: true }).first().click({ timeout: 8000 });
      await sleep(600);
      await p.getByRole("button", { name: "Details", exact: true }).first().click({ timeout: 8000 });
    },
  },
  {
    id: "tab-switch",
    ms: 800,
    setup: (p) => goto(p, "#/overview"),
    action: async (p) => {
      await p.getByRole("tab", { name: "5 min", exact: true }).first().click({ timeout: 8000 });
    },
  },
  {
    id: "theme-switch",
    ms: 900,
    setup: (p) => goto(p, "#/overview"),
    action: async (p, vp) => {
      if (mobile(vp)) await openNav(p, vp);
      await p.getByRole("button", { name: /^Switch to (light|dark)$/ }).first().click({ timeout: 8000 });
    },
  },
  {
    id: "playground-streaming",
    mock: true,
    ms: 5200,
    setup: async (p) => {
      await p.addInitScript((text) => {
        const native = window.fetch.bind(window);
        window.fetch = (input, init = {}) => {
          const u = new URL(typeof input === "string" ? input : input.url, location.href);
          if (!u.pathname.endsWith("/v1/chat/completions")) return native(input, init);
          const enc = new TextEncoder();
          const words = text.split(/(?<=\s)/);
          return Promise.resolve(
            new Response(
              new ReadableStream({
                async start(c) {
                  await new Promise((r) => setTimeout(r, 250));
                  for (const w of words) {
                    c.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: w } }] })}\n\n`));
                    await new Promise((r) => setTimeout(r, 55));
                  }
                  c.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`));
                  c.close();
                },
              }),
              { headers: { "content-type": "text/event-stream" } },
            ),
          );
        };
      }, CHAT);
      await goto(p, "#/playground");
    },
    action: async (p) => {
      await p.getByRole("button", { name: /^Explain what/ }).first().click({ timeout: 8000 });
      await sleep(200);
      await p.getByRole("button", { name: /^send/i }).first().click({ timeout: 8000 });
    },
  },
];

// ---- CLS ------------------------------------------------------------------------------------------
const PAGES = ["overview", "requests", "models", "cache", "diagnostics", "settings", "playground", "logs"];
async function measureCls(browser, theme, vp) {
  const out = {};
  await post("/__finish", { n: 3 });
  for (const pg of PAGES) {
    const { ctx, page } = await newPage(browser, theme, vp);
    await page.addInitScript(() => {
      window.__cls = 0;
      window.__shifts = [];
      new PerformanceObserver((l) => {
        for (const e of l.getEntries())
          if (!e.hadRecentInput) {
            window.__cls += e.value;
            window.__shifts.push({ v: +e.value.toFixed(4), t: Math.round(e.startTime), src: (e.sources ?? []).map((s) => (s.node?.nodeName ?? "") + "." + String(s.node?.className ?? "").slice(0, 30)).slice(0, 2) });
          }
      }).observe({ type: "layout-shift", buffered: true });
    });
    await page.goto(`${url}#/${pg}`);
    await page.waitForSelector("h1, [data-testid='playground']", { timeout: 15000 }).catch(() => {});
    // "after first paint": count only shifts that happen once the first contentful paint has passed.
    const fcp = await page.evaluate(() => performance.getEntriesByName("first-contentful-paint")[0]?.startTime ?? 0);
    await sleep(3500);
    const r = await page.evaluate((f) => ({ total: window.__cls, afterFcp: window.__shifts.filter((s) => s.t >= f).reduce((a, s) => a + s.v, 0), shifts: window.__shifts }), fcp);
    out[pg] = { total: +r.total.toFixed(4), afterFcp: +r.afterFcp.toFixed(4), shifts: r.shifts.slice(0, 6) };
    await ctx.close();
  }
  return out;
}

// ---- main -------------------------------------------------------------------------------------------
const browser = await chromium.launch();
await startEngine();
const summary = { label, url, cls: {}, scenarios: {} };
try {
  for (const theme of themes)
    for (const vp of vps) {
      const tag = `${theme}-${vp}`;
      if (wantCls) summary.cls[tag] = await measureCls(browser, theme, vp);
      for (const s of scenarios) {
        if (only.length && !only.includes(s.id)) continue;
        if (s.only && !s.only(vp)) continue;
        if (!engine) await startEngine();
        await waitUp();
        const { ctx, page } = await newPage(browser, theme, vp);
        try {
          await s.setup(page, vp);
          const dir = join(outRoot, label, s.id, tag);
          const n = await record(page, dir, () => s.action(page, vp), s.ms);
          await s.after?.();
          summary.scenarios[`${s.id}/${tag}`] = { frames: n, mock: !!s.mock };
          console.log(`${label} ${s.id} ${tag}: ${n} frames`);
        } catch (e) {
          summary.scenarios[`${s.id}/${tag}`] = { error: String(e).split("\n")[0] };
          console.log(`${label} ${s.id} ${tag}: FAILED ${String(e).split("\n")[0]}`);
        } finally {
          await ctx.close();
        }
      }
    }
} finally {
  await stopEngine();
  await browser.close();
  mkdirSync(join(outRoot, label), { recursive: true });
  writeFileSync(join(outRoot, label, `summary-${themes.join("+")}-${vps.join("+")}${only.length ? "-" + only.join("+") : ""}.json`), JSON.stringify(summary, null, 1));
}
