// Screenshot every page and interactive state, dark+light, 1440 and 402 wide.
// Usage: BASE=http://127.0.0.1:3990 OUT=/path node tests/shots.mjs
import { chromium } from "@playwright/test";
import fs from "node:fs";

const BASE = process.env.BASE ?? "http://127.0.0.1:3990";
const OUT = process.env.OUT ?? "/Volumes/P5Plus/yunshu-build/researchsite-shots";
fs.mkdirSync(OUT, { recursive: true });
const sizes = { d: { width: 1440, height: 900 }, m: { width: 402, height: 874 } };
const browser = await chromium.launch();
let failures = 0;

async function settle(page) {
  // SSE keeps the network busy forever, so wait for content instead of "networkidle".
  await page.waitForLoadState("load");
  await page.waitForFunction(() => !document.querySelector('[aria-busy="true"]'), null, { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(500);
}
async function shot(page, name, size, theme) {
  const h = await page.evaluate(() => document.getElementById("rs-scroll")?.scrollHeight ?? 0);
  const base = sizes[size];
  await page.setViewportSize({ width: base.width, height: Math.min(Math.max(base.height, h + 60), 2600) });
  await page.waitForTimeout(150);
  await page.screenshot({ path: `${OUT}/${name}-${size}-${theme}.png` });
  await page.setViewportSize(base);
}

const states = [
  ["home", "#/", null],
  ["lines", "#/lines", null],
  ["lines-search", "#/lines", async (p) => p.getByLabel("搜尋研究線").fill("agent")],
  ["lines-empty", "#/lines", async (p) => p.getByLabel("搜尋研究線").fill("zzzz-no-such-line")],
  ["lines-merged", "#/lines", async (p) => p.getByRole("button", { name: "已合併" }).click()],
  ["decisions", "#/decisions", null],
  ["decisions-search", "#/decisions", async (p) => p.getByLabel("搜尋決策").fill("記憶體")],
  ["decisions-filter", "#/decisions", async (p) => {
    await p.getByRole("button", { name: "近 7 天" }).click();
    await p.getByLabel("顯示已取代的決策").click();
  }],
  ["decisions-empty", "#/decisions", async (p) => p.getByLabel("搜尋決策").fill("zzzz-no-such-decision")],
  ["ground", "#/ground", null],
  ["parity", "#/parity", null],
  ["parity-metric", "#/parity", async (p) => p.getByRole("button", { name: /TTFT/ }).first().click()],
  ["gpu", "#/gpu", null],
  ["gpu-24h", "#/gpu", async (p) => p.getByRole("button", { name: "近 24 小時" }).click()],
  ["docs", "#/docs", null],
  ["docs-search", "#/docs", async (p) => p.getByLabel("全文搜尋").fill("speculative")],
  ["docs-doc-tables", "#/docs/r/parityboard/BOARD.md", null],
  ["docs-doc-links", "#/docs/r/notes/BACKLOG.md", null],
  ["docs-missing", "#/docs/r/nope/missing.md", null],
  ["unknown-route", "#/nope", null],
];

for (const theme of ["dark", "light"]) {
  for (const size of ["d", "m"]) {
    const ctx = await browser.newContext({ viewport: sizes[size], colorScheme: theme, reducedMotion: "reduce" });
    await ctx.addInitScript((t) => localStorage.setItem("yunshu.research.theme", t), theme);
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => m.type() === "error" && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
    for (const [name, hash, act] of states) {
      try {
        await page.goto(`${BASE}/${hash}`);
        await page.reload();
        await settle(page);
        if (act) {
          await act(page);
          await settle(page);
        }
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
        if (overflow) {
          console.log(`HORIZONTAL OVERFLOW ${name} ${size} ${theme}`);
          failures++;
        }
        await shot(page, name, size, theme);
      } catch (e) {
        console.log(`FAIL ${name} ${size} ${theme}: ${e}`);
        failures++;
      }
    }
    if (size === "m") {
      await page.goto(`${BASE}/#/`);
      await settle(page);
      await page.getByRole("button", { name: "開啟導覽" }).click();
      await page.waitForTimeout(400);
      await page.screenshot({ path: `${OUT}/nav-open-${size}-${theme}.png` });
    }
    if (errors.length) {
      console.log(`console errors (${size} ${theme}):`, errors.slice(0, 5));
      failures++;
    }
    await ctx.close();
  }
}
await browser.close();
console.log(failures ? `${failures} problem(s)` : "all shots ok", OUT);
process.exit(failures ? 1 : 0);
