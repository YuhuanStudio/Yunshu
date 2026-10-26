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
const load = (name: string): Line[] =>
  gunzipSync(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)))
    .toString()
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
// "burst": 11 concurrent requests, prefill and decode; "single": one request decoding for 40 s.
const FIXTURES = {
  burst: load("real-load.jsonl.gz"),
  single: load("real-load-single.jsonl.gz"),
};
const SPEED = 3;
const SECONDS = 20;

async function replay(page: Page, kind: keyof typeof FIXTURES = "burst") {
  const LINES = FIXTURES[kind];
  const STATUS = LINES.filter((l) => l.status);
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
  for (const [route, kind] of [
    ["overview", "burst"],
    ["requests", "burst"],
    ["overview", "single"],
  ] as const) {
    test(`${name} ${route} (${kind}): real load does not move the page`, async ({
      browser,
    }) => {
      // The scripted run is 4.5 s settle + SECONDS of sampling (about 25.4 s alone): the default 30 s
      // budget leaves no room for CPU contention, which ended a run mid-sample ("Test ended").
      test.setTimeout(90_000);
      const ctx = await browser.newContext(project);
      const page = await ctx.newPage();
      await replay(page, kind);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(4500); // first paint and first host sample are not "load"
      await page.evaluate(() => window.__review.reset());
      // The live card's count and its rows must agree: "n active" with an empty row area is a defect.
      const empty: string[] = [];
      let streak = 0;
      let previousRows: Record<string, number> = {};
      const rowMoves: string[] = [];
      let liveRowSamples = 0,
        liveRowComparisons = 0;
      for (let i = 0; i < SECONDS * 2; i++) {
        await page.waitForTimeout(500);
        if (route === "requests") {
          const current = await page.evaluate(() =>
            Object.fromEntries(
              [
                ...document.querySelectorAll<HTMLElement>(
                  "tr[data-live-request][data-request-id]",
                ),
              ].map((el) => [
                el.dataset.requestId!,
                el.getBoundingClientRect().top,
              ]),
            ),
          );
          liveRowSamples += Object.keys(current).length;
          for (const [id, top] of Object.entries(current)) {
            if (previousRows[id] == null) continue;
            liveRowComparisons++;
            if (Math.abs(top - previousRows[id]) > 1.5)
              rowMoves.push(`${id}: ${previousRows[id]} -> ${top}`);
          }
          previousRows = current;
          continue;
        }
        if (route !== "overview") continue;
        const seen = await page.evaluate(() => {
          const h = document.querySelector('[data-testid="live-panel"] h2');
          const n = Number(/\d+/.exec(h?.textContent ?? "")?.[0] ?? 0);
          const rows = [
            ...document.querySelectorAll(
              '[data-testid="live-panel"] li.live-slot .live-row-inner',
            ),
          ].filter((e) => {
            const r = e.getBoundingClientRect();
            return r.height > 10 && Number(getComputedStyle(e).opacity) > 0.5;
          }).length;
          const first = document.querySelector(
            '[data-testid="live-panel"] li:first-child .live-row-inner',
          );
          return { n, rows, first: !!first };
        });
        streak =
          seen.n > 0 && (seen.rows === 0 || !seen.first) ? streak + 1 : 0;
        if (streak >= 2) empty.push(`t=${i / 2}s n=${seen.n} rows=0`);
      }
      expect(empty, "active requests but no visible lane row").toEqual([]);
      expect(
        rowMoves,
        "surviving live requests changed lanes under real load",
      ).toEqual([]);
      if (route === "requests") {
        expect(liveRowSamples, "live rows were observed").toBeGreaterThan(5);
        expect(
          liveRowComparisons,
          "surviving rows were compared",
        ).toBeGreaterThan(5);
      }
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
