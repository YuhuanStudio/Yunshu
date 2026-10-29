import { appendFileSync } from "node:fs";
import { devices, expect, test, type Page } from "@playwright/test";
import {
  STABILITY_RECORDER,
  analyze,
  select,
} from "../scripts/stability-analyze.mjs";
import { clsOf } from "../scripts/review-analyze.mjs";
import { ANIM_RECORDER, summarize } from "../scripts/anim-count.mjs";
import { storm } from "./support/storm";

// "It just keeps jumping": the storm feed (browser/support/storm.ts) changes every number on every
// poll for 20 s (>= 200 updates). The recorder samples the box of every visible element every 50 ms.
//   - the status pills hold their width (a number crossing a digit boundary 20 times does not resize
//     the pill 20 times; no pill goes A -> B -> A more than once),
//   - nothing that is not a chart or a lane moves without a width step of something before it,
//   - the layout-shift score stays near 0,
//   - the screen starts (almost) no animation (see live-animation-rate.spec.ts for the count).

interface Recording {
  frames: Parameters<typeof analyze>[0];
  keys: Record<string, string>;
  entries: { value: number; hadRecentInput: boolean }[];
  boxed: Record<string, boolean>;
}
const SECONDS = 20;
const report = (name: string, data: unknown) => {
  if (process.env.JUMP_REPORT)
    appendFileSync(
      process.env.JUMP_REPORT,
      JSON.stringify({ name, ...(data as object) }) + "\n",
    );
};
const ev = <T>(page: Page, fn: string) =>
  page.evaluate(`(${fn})()`) as Promise<T>;

const PROJECTS = [
  ["desktop", { viewport: { width: 1440, height: 900 } }],
  ["ipad", (({ defaultBrowserType: _, ...d }) => d)(devices["iPad Pro 11"])],
] as const;

for (const [name, project] of PROJECTS) {
  for (const route of ["overview", "requests", "cache"]) {
    test(`${name} ${route}: a storm of updates does not make the page jump`, async ({
      browser,
    }) => {
      test.setTimeout(90_000);
      const ctx = await browser.newContext(project);
      const page = await ctx.newPage();
      await storm(page, STABILITY_RECORDER + ANIM_RECORDER);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(6000);
      await ev(page, "() => { window.__stab.reset(); window.__anim.reset(); }");
      await page.waitForTimeout(SECONDS * 1000);
      const rec = await ev<Recording>(page, "() => window.__stab.stop()");
      const events = await ev<{ t: number; name: string }[]>(
        page,
        "() => window.__anim.stop()",
      );
      expect(rec.frames.length).toBeGreaterThan(SECONDS * 15);
      const all = analyze(rec.frames, rec.keys, { boxed: rec.boxed });
      const band = analyze(
        select(rec.frames, rec.keys, (k) => k.startsWith("status-band")),
        rec.keys,
        { boxed: rec.boxed },
      );
      const pills = band.perElement.filter((e) => e.key.includes("li."));
      const anim = summarize(events, SECONDS);
      // room trailing a pill's last glyph: the dead space the user sees as a wide pill
      const dead = all.dead.filter((d) => d.px > 2.5);
      report(`${name} ${route}`, {
        bandSteps: band.steps,
        bandFlips: band.flips,
        pills: pills.map((e) => [e.key, e.steps, e.flips]),
        movedFrames: all.movedFrames,
        shoved: all.shoved,
        top: all.perElement.slice(0, 8).map((e) => [e.key, e.steps, e.flips]),
        changes: all.perElement
          .slice(0, 5)
          .map((e) => [
            e.key,
            e.changes
              .slice(0, 8)
              .map((c) => [c.t, Math.round(c.from), Math.round(c.to)]),
          ]),
        deadFrames: dead.length,
        deadMax: Math.max(0, ...all.dead.map((d) => d.px)),
        cls: clsOf(rec.entries),
        anim: [anim.gestures, anim.total],
      });
      expect(
        pills.filter((e) => e.flips > 1).map((e) => [e.key, e.steps, e.flips]),
        "status pills whose width goes A -> B -> A",
      ).toEqual([]);
      // a pill changes width when a number gets another digit (grow at once) and once more when it is
      // narrower for 8 s: a few steps in 20 s of storm, never one per update
      const pillSteps = pills.reduce((n, e) => n + e.steps, 0);
      expect(
        pillSteps,
        `status pill width steps in ${SECONDS} s of storm: ${JSON.stringify(pills.map((e) => [e.key, e.steps]))}`,
      ).toBeLessThanOrEqual(6);
      expect(
        dead.slice(0, 3).map((d) => [d.key, d.px]),
        "status pills with empty room after their text (px, over the whole storm)",
      ).toEqual([]);
      expect(clsOf(rec.entries)).toBeLessThan(0.02);
      expect(
        anim.gestures,
        JSON.stringify(anim.byName.slice(0, 4)),
      ).toBeLessThanOrEqual(2);
      await ctx.close();
    });
  }
}
