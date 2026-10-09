// Walks every console page in one viewport while a real engine is under load, recording a screenshot
// every 500 ms and the layout-shift evidence (see review-analyze.mjs).
//
//   node scripts/review-capture.mjs --base http://127.0.0.1:18992 --out DIR --viewport desktop
//        [--seconds 40] [--pages overview,requests,...]
//
// Viewports: desktop (chromium 1440x900), ipad-portrait / ipad-landscape (WebKit, iPad Pro 11, touch,
// DPR 2), iphone (WebKit, iPhone 15). Output: DIR/<viewport>/<page>/f-0001.jpg ..., shifts.json.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { chromium, webkit, devices } from "@playwright/test";
import { PAGE_RECORDER, summarize } from "./review-analyze.mjs";

export const ALL_PAGES = [
  "overview",
  "requests",
  "logs",
  "diagnostics",
  "models",
  "downloads",
  "cache",
  "playground",
  "api",
  "docs",
  "keys",
  "settings",
];

export function profile(name) {
  const strip = ({ defaultBrowserType, ...rest }) => rest;
  switch (name) {
    case "desktop":
      return {
        browser: chromium,
        context: { viewport: { width: 1440, height: 900 } },
      };
    case "ipad-portrait":
      return { browser: webkit, context: strip(devices["iPad Pro 11"]) };
    case "ipad-landscape":
      return {
        browser: webkit,
        context: strip(devices["iPad Pro 11 landscape"]),
      };
    case "iphone":
      return {
        browser: webkit,
        context: {
          ...strip(devices["iPhone 15"]),
          viewport: { width: 402, height: 874 },
        },
      };
    default:
      throw new Error(`unknown viewport ${name}`);
  }
}

async function main() {
  const { values } = parseArgs({
    options: {
      base: { type: "string" },
      out: { type: "string" },
      viewport: { type: "string" },
      seconds: { type: "string", default: "40" },
      pages: { type: "string", default: ALL_PAGES.join(",") },
      "frame-ms": { type: "string", default: "500" },
    },
  });
  if (!values.base || !values.out || !values.viewport)
    throw new Error("--base, --out and --viewport are required");
  const { browser: engine, context } = profile(values.viewport);
  const seconds = Number(values.seconds);
  const frameMs = Number(values["frame-ms"]);
  const browser = await engine.launch();
  const ctx = await browser.newContext({
    locale: "zh-TW",
    colorScheme: "dark",
    ...context,
  });
  await ctx.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await ctx.addInitScript(PAGE_RECORDER);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message.slice(0, 200)));
  const summary = {};
  for (const name of values.pages.split(",")) {
    const dir = join(values.out, values.viewport, name);
    mkdirSync(dir, { recursive: true });
    await page.goto(`${values.base}/console/#/${name}`, {
      waitUntil: "domcontentloaded",
    });
    await page.waitForTimeout(1500); // first paint settles; the wait is not part of the recording
    await page.evaluate(() => window.__review?.reset());
    const end = Date.now() + seconds * 1000;
    let n = 0;
    while (Date.now() < end) {
      n += 1;
      await page.screenshot({
        path: join(dir, `f-${String(n).padStart(4, "0")}.jpg`),
        type: "jpeg",
        quality: 55,
        scale: "css",
      });
      await page.waitForTimeout(frameMs);
    }
    const rec = await page.evaluate(
      () => window.__review?.stop() ?? { entries: [], frames: [] },
    );
    await page.evaluate(() => window.__review?.reset());
    const s = summarize(rec.entries, rec.frames);
    summary[name] = { ...s, frames: n };
    writeFileSync(
      join(dir, "shifts.json"),
      JSON.stringify(
        { ...s, entries: rec.entries, deltas: rec.frames.length },
        null,
        1,
      ),
    );
    console.log(
      `${values.viewport} ${name}: ${n} frames, CLS ${s.cls}, ${s.moves} box moves`,
    );
  }
  writeFileSync(
    join(values.out, values.viewport, "summary.json"),
    JSON.stringify({ errors, pages: summary }, null, 1),
  );
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
