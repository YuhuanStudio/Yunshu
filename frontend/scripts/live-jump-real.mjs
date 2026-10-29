// Jump measurement against a REAL console + engine (scripts/research/brand_live_jump.py drives the load).
//
//   node scripts/live-jump-real.mjs --base http://127.0.0.1:18992 --out DIR [--seconds 40] [--routes overview,requests]
//
// For every route: open the console, record the box of every visible element every 50 ms, every CSS
// animation / transition / Element.animate the page starts, and every number whose text changes,
// for `--seconds`, plus a frame strip (a screenshot every 500 ms for the first 8 frames, kept small).
// Writes DIR/jump.json and judges it with the same rules as the browser specs (pills do not flip,
// layout shift < 0.02, few animations, every changing number tabular). Exit 1 when a rule fails.

import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { STABILITY_RECORDER, analyze, select } from "./stability-analyze.mjs";
import { clsOf } from "./review-analyze.mjs";
import { ANIM_RECORDER, summarize } from "./anim-count.mjs";
import { NUMBER_RECORDER, notTabular, tooFast } from "./number-audit.mjs";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const base = arg("base");
const out = arg("out");
const seconds = Number(arg("seconds", "40"));
const routes = arg("routes", "overview,requests").split(",");
if (!base || !out) {
  console.error(
    "usage: live-jump-real.mjs --base URL --out DIR [--seconds N] [--routes a,b]",
  );
  process.exit(2);
}
mkdirSync(join(out, "strip"), { recursive: true });

const browser = await chromium.launch();
const result = { base, seconds, routes: {}, failures: [] };
for (const route of routes) {
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    locale: "zh-TW",
  });
  const page = await ctx.newPage();
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.addInitScript(
    STABILITY_RECORDER + ANIM_RECORDER + NUMBER_RECORDER,
  );
  await page.goto(`${base}/console/#/${route}`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForTimeout(5000);
  await page.evaluate(
    "window.__stab.reset(); window.__anim.reset(); window.__num.reset()",
  );
  const t0 = Date.now();
  for (let i = 0; Date.now() - t0 < seconds * 1000; i++) {
    if (i < 8)
      await page.screenshot({
        path: join(out, "strip", `${route}-${String(i).padStart(2, "0")}.jpg`),
        type: "jpeg",
        quality: 55,
      });
    await page.waitForTimeout(500);
  }
  const rec = await page.evaluate("window.__stab.stop()");
  const events = await page.evaluate("window.__anim.stop()");
  const numbers = await page.evaluate("window.__num.report()");
  await ctx.close();
  const all = analyze(rec.frames, rec.keys, { boxed: rec.boxed });
  const band = analyze(
    select(rec.frames, rec.keys, (k) => k.startsWith("status-band")),
    rec.keys,
    { boxed: rec.boxed },
  );
  const pills = band.perElement.filter((e) => e.key.includes("li."));
  const anim = summarize(events, seconds);
  const r = {
    frames: rec.frames.length,
    pillSteps: pills.reduce((n, e) => n + e.steps, 0),
    pillFlips: pills.reduce((n, e) => n + e.flips, 0),
    pills: pills.map((e) => [e.key, e.steps, e.flips]),
    movedFrames: all.movedFrames,
    shoved: all.shoved,
    cls: clsOf(rec.entries),
    animations: {
      gestures: anim.gestures,
      total: anim.total,
      perSecond: anim.perSecond,
      peak: anim.peak,
      byName: anim.byName.slice(0, 6),
    },
    numbers: {
      changing: numbers.length,
      notTabular: notTabular(numbers).map((n) => n.key),
      tooFast: tooFast(numbers, seconds).map((n) => n.key),
    },
  };
  result.routes[route] = r;
  const fail = (m) => result.failures.push(`${route}: ${m}`);
  if (r.frames < seconds * 15) fail(`only ${r.frames} frames recorded`);
  if (r.pillFlips > 1)
    fail(`status pills went A -> B -> A ${r.pillFlips} times`);
  if (r.pillSteps > Math.ceil(seconds / 3))
    fail(`${r.pillSteps} status pill width steps`);
  if (r.cls >= 0.02) fail(`layout shift ${r.cls.toFixed(4)}`);
  if (r.animations.perSecond > 0.5)
    fail(`${r.animations.perSecond.toFixed(2)} animations per second`);
  if (r.numbers.notTabular.length)
    fail(`not tabular: ${r.numbers.notTabular.slice(0, 3).join(" | ")}`);
  if (r.numbers.tooFast.length)
    fail(`faster than 5 Hz: ${r.numbers.tooFast.slice(0, 3).join(" | ")}`);
}
await browser.close();
writeFileSync(join(out, "jump.json"), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
process.exit(result.failures.length ? 1 : 0);
