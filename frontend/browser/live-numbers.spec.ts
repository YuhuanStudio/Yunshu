import { appendFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import {
  NUMBER_RECORDER,
  notTabular,
  tooFast,
} from "../scripts/number-audit.mjs";
import { storm } from "./support/storm";

// Every number on screen that changes under the storm feed (browser/support/storm.ts) is drawn with
// tabular digits and its text changes at a pace a person can read (<= 5 Hz; the status poll is 4 Hz).
// Charts (svg) are excluded: they are allowed to run faster.

const SECONDS = 12;
interface Row {
  key: string;
  changes: number;
  tabular: boolean;
}
for (const route of ["overview", "requests", "cache"]) {
  test(`${route}: every changing number is tabular and paced`, async ({
    browser,
  }) => {
    test.setTimeout(60_000);
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
    });
    const page = await ctx.newPage();
    await storm(page, NUMBER_RECORDER);
    await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(5000);
    await page.evaluate("window.__num.reset()");
    await page.waitForTimeout(SECONDS * 1000);
    const rows = (await page.evaluate("window.__num.report()")) as Row[];
    if (process.env.NUM_REPORT)
      appendFileSync(
        process.env.NUM_REPORT,
        JSON.stringify({ route, rows }) + "\n",
      );
    expect(rows.length, "changing numbers were seen").toBeGreaterThanOrEqual(3);
    expect(
      notTabular(rows).map((r) => [r.key, r.changes]),
      "changing numbers without tabular digits",
    ).toEqual([]);
    expect(
      tooFast(rows, SECONDS).map((r) => [r.key, r.changes]),
      "numbers whose text changes faster than 5 Hz",
    ).toEqual([]);
    await ctx.close();
  });
}
