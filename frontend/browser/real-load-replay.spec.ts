import { appendFileSync, readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { devices, expect, test, type Page } from "@playwright/test";
import {
  PAGE_RECORDER,
  summarize,
  clsOf,
  boxDeltas,
} from "../scripts/review-analyze.mjs";

// What a page does while a REAL engine is busy. fixtures/real-load.jsonl.gz is 60 s of the real
// Qwen3.8-27B engine's /v1/yunshu/status stream and finished-request stream (metadata only) from a
// bursty review run (scripts/research/console_review.py): up to 11 requests at once, prefill and
// decode phases, prefix hits, cancels. It is replayed here at 3x so the 60 s window fits in 20 s.
// Every card, row and heading is sampled every 250 ms; none may move or change size because data
// changed, and the layout-shift score stays at 0.

interface Line {
  t: number;
  status?: Record<string, unknown>;
  requests?: Record<string, unknown>[];
}
const LINES: Line[] = gunzipSync(
  readFileSync(new URL("./fixtures/real-load.jsonl.gz", import.meta.url)),
)
  .toString()
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l));
const STATUS = LINES.filter((l) => l.status);
const SPEED = 3;
const SECONDS = 20;

async function replay(page: Page) {
  const t0 = Date.now();
  const at = () => ((Date.now() - t0) / 1000) * SPEED;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.addInitScript(PAGE_RECORDER);
  let seq = 0;
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") {
      const now = at();
      let cur = STATUS[0];
      for (const l of STATUS) if (l.t <= now) cur = l;
      return route.fulfill({ json: cur.status });
    }
    if (path === "/v1/yunshu/requests/recent") {
      const now = at();
      const rows = LINES.filter((l) => l.requests && l.t <= now).flatMap(
        (l) => l.requests ?? [],
      );
      seq = rows.length;
      return route.fulfill({
        json: {
          object: "list",
          data: rows.slice().reverse().slice(0, 100),
          count: rows.length,
          capacity: 512,
          boot_id: "real",
          latest_seq: seq,
        },
      });
    }
    return route.fulfill({ status: 404, json: {} });
  });
}

for (const [name, project] of [
  ["desktop", { viewport: { width: 1440, height: 900 } }],
  ["ipad", (({ defaultBrowserType: _, ...d }) => d)(devices["iPad Pro 11"])],
  ["iphone", (({ defaultBrowserType: _, ...d }) => d)(devices["iPhone 15"])],
] as const) {
  for (const route of ["overview", "requests"]) {
    test(`${name} ${route}: real load does not move the page`, async ({
      browser,
    }) => {
      const ctx = await browser.newContext(project);
      const page = await ctx.newPage();
      await replay(page);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(1800);
      await page.evaluate(() => window.__review.reset());
      await page.waitForTimeout(SECONDS * 1000);
      const rec = await page.evaluate(() => window.__review.stop());
      const s = summarize(rec.entries, rec.frames);
      if (process.env.REPLAY_RESULTS)
        appendFileSync(
          process.env.REPLAY_RESULTS,
          JSON.stringify({
            test: test.info().title,
            project: test.info().project.name,
            cls: s.cls,
            moves: s.moves,
          }) + "\n",
        );
      expect(
        s.moves,
        `moved: ${JSON.stringify(s.movers.slice(0, 4))} first: ${JSON.stringify(
          boxDeltas(rec.frames)
            .slice(0, 400)
            .map((d) => [d.t, d.key.slice(0, 22), d.dTop, d.dH, d.dLeft]),
        )}`,
      ).toBe(0);
      expect(
        clsOf(rec.entries),
        `CLS ${clsOf(rec.entries).toFixed(4)}; shifts: ${JSON.stringify(rec.entries.filter((e: { value: number }) => e.value > 0.0005).slice(0, 6))}`,
      ).toBeLessThan(0.03);
      await ctx.close();
    });
  }
}
