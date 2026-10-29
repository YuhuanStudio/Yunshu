import { appendFileSync } from "node:fs";
import { devices, expect, test } from "@playwright/test";
import { ANIM_RECORDER, summarize } from "../scripts/anim-count.mjs";
import { replay, type LoadKind } from "./support/real-load";
import { storm } from "./support/storm";

// Change animations under a bursty feed. The real engine's stream (11 concurrent requests, prefill
// and decode flips, cancels) is replayed 3x faster, so 20 s of wall time carries ~60 s of updates.
// Every CSS animation, CSS transition and Element.animate the page starts is counted (see
// scripts/anim-count.mjs). A live screen swaps routine ticks in place; only a meaningful change may
// animate, and a burst may not play one motion per update.

const SECONDS = 20;
const report = (name: string, data: unknown) => {
  if (process.env.ANIM_REPORT)
    appendFileSync(
      process.env.ANIM_REPORT,
      JSON.stringify({ name, ...(data as object) }) + "\n",
    );
};
// Most animations the page may start per second on average, and in its busiest second.
const LIMIT = { perSecond: 0.5, peak: 3 };

const PROJECTS = [
  ["desktop", { viewport: { width: 1440, height: 900 } }],
  ["ipad", (({ defaultBrowserType: _, ...d }) => d)(devices["iPad Pro 11"])],
] as const;

for (const [name, project] of PROJECTS) {
  for (const [route, kind] of [
    ["overview", "burst"],
    ["requests", "burst"],
    ["cache", "burst"],
  ] as [string, LoadKind][]) {
    test(`${name} ${route}: a bursty feed starts few animations`, async ({
      browser,
    }) => {
      test.setTimeout(90_000);
      const ctx = await browser.newContext(project);
      const page = await ctx.newPage();
      await replay(page, kind, 3, ANIM_RECORDER);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(4500);
      await page.evaluate(() =>
        (window as unknown as { __anim: { reset(): void } }).__anim.reset(),
      );
      await page.waitForTimeout(SECONDS * 1000);
      const events = (await page.evaluate(() =>
        (window as unknown as { __anim: { stop(): unknown } }).__anim.stop(),
      )) as { t: number; name: string }[];
      const s = summarize(events, SECONDS);
      report(`${name} ${route}`, {
        ...s,
        events: events.map((e) => [e.t, e.name]),
      });
      expect(
        s.perSecond,
        `animations per second; started: ${JSON.stringify(s.byName.slice(0, 6))}`,
      ).toBeLessThanOrEqual(LIMIT.perSecond);
      expect(s.peak, "animations in the busiest second").toBeLessThanOrEqual(
        LIMIT.peak,
      );
      await ctx.close();
    });
  }
}

// A storm (browser/support/storm.ts): a request starts and one ends every 400 ms, speeds and memory
// jitter on every poll, a finished request lands every 300 ms. After the first moments the screen
// must stop animating: values swap in place.
for (const route of ["overview", "requests"]) {
  test(`storm ${route}: animations stop once the feed is a burst`, async ({
    browser,
  }) => {
    test.setTimeout(90_000);
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
    });
    const page = await ctx.newPage();
    await storm(page, ANIM_RECORDER);
    await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(6000);
    await page.evaluate(() =>
      (window as unknown as { __anim: { reset(): void } }).__anim.reset(),
    );
    await page.waitForTimeout(SECONDS * 1000);
    const events = (await page.evaluate(() =>
      (window as unknown as { __anim: { stop(): unknown } }).__anim.stop(),
    )) as { t: number; name: string }[];
    const s = summarize(events, SECONDS);
    report(`storm ${route}`, {
      ...s,
      events: events.map((e) => [e.t, e.name]),
    });
    expect(
      s.gestures,
      `animations in ${SECONDS} s of storm; started: ${JSON.stringify(s.byName.slice(0, 6))}`,
    ).toBeLessThanOrEqual(2);
    await ctx.close();
  });
}
