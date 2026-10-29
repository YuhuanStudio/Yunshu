import { appendFileSync } from "node:fs";
import { devices, expect, test, type Page } from "@playwright/test";
import {
  STABILITY_RECORDER,
  analyze,
  select,
} from "../scripts/stability-analyze.mjs";
import { clsOf } from "../scripts/review-analyze.mjs";
import { replay, type LoadKind } from "./support/real-load";

// Live readings must not make the page jump. The real engine's bursty stream (11 concurrent
// requests, prefill/decode flips, cancels) is replayed 3x faster, so 20 s of wall time carries about
// 60 s of status updates (well over 200 samples at the console's 4 Hz poll). The recorder samples
// every element of the header, main area and status band every 50 ms:
//   - a status pill never trails empty room after its text (it hugs the content),
//   - no element's width goes A -> B -> A (flip-flop) and none changes width more than a handful of
//     times (a width step happens at a new digit count or a new phase, never per tick),
//   - nothing moves without a reason: elements shoved by a neighbour only follow a real width step,
//   - the layout-shift score stays near 0.

interface Recording {
  frames: Parameters<typeof analyze>[0];
  keys: Record<string, string>;
  entries: { value: number; hadRecentInput: boolean }[];
  boxed: Record<string, boolean>;
}
const record = (page: Page) =>
  page.evaluate(() =>
    (window as unknown as { __stab: { stop(): unknown } }).__stab.stop(),
  ) as Promise<Recording>;
const reset = (page: Page) =>
  page.evaluate(() =>
    (window as unknown as { __stab: { reset(): void } }).__stab.reset(),
  );

const SECONDS = 20;
// frames (of ~400 in 20 s) in which anything visible moved, by page; a new digit count or phase word
// moves things once, a tick never does
const BUDGET: Record<string, number> = {
  overview: 20,
  requests: 10,
  cache: 10,
};
const PROJECTS = [
  ["desktop", { viewport: { width: 1440, height: 900 } }],
  ["ipad", (({ defaultBrowserType: _, ...d }) => d)(devices["iPad Pro 11"])],
] as const;
const report = (name: string, data: unknown) => {
  if (process.env.STAB_REPORT)
    appendFileSync(
      process.env.STAB_REPORT,
      JSON.stringify({ name, ...(data as object) }) + "\n",
    );
};

for (const [name, project] of PROJECTS) {
  for (const [route, kind] of [
    ["overview", "burst"],
    ["requests", "burst"],
    ["cache", "burst"],
  ] as [string, LoadKind][]) {
    test(`${name} ${route}: a bursty live engine does not move widths or neighbours`, async ({
      browser,
    }) => {
      test.setTimeout(90_000);
      const ctx = await browser.newContext(project);
      const page = await ctx.newPage();
      await replay(page, kind, 3, STABILITY_RECORDER);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(4500);
      await reset(page);
      await page.waitForTimeout(SECONDS * 1000);
      const rec = await record(page);
      expect(rec.frames.length).toBeGreaterThan(SECONDS * 15);
      const a = analyze(rec.frames, rec.keys, { boxed: rec.boxed });
      report(`${name} ${route}`, {
        steps: a.steps,
        flips: a.flips,
        movedFrames: a.movedFrames,
        moved: a.moved,
        top: a.perElement
          .slice(0, 10)
          .map((e) => [e.key, e.steps, e.flips, e.boxed]),
        shoved: a.shoved,
        boxedChanges: a.perElement
          .filter((e) => e.boxed)
          .slice(0, 4)
          .map((e) => [
            e.key,
            e.changes.map((c) => [c.t, Math.round(c.from), Math.round(c.to)]),
          ]),
        cls: clsOf(rec.entries),
      });
      // a boxed element (pill, card, chip, button) whose width goes A -> B -> A twice is a visible flicker
      // (once is a settled change taken back; the hold then doubles)
      const flickers = a.perElement.filter((e) => e.boxed && e.flips > 1);
      expect(
        flickers.slice(0, 5).map((e) => [e.key.slice(0, 50), e.steps, e.flips]),
        "boxed elements whose width goes A -> B -> A",
      ).toEqual([]);
      expect(
        a.movedFrames,
        `frames in which something moved (${a.moved} moves): ${JSON.stringify(a.shoved.slice(0, 5))}`,
      ).toBeLessThanOrEqual(BUDGET[route]);
      expect(clsOf(rec.entries)).toBeLessThan(0.02);
      await ctx.close();
    });
  }
}

// ---- Synthetic feeds that hit the failure modes on purpose --------------------------------------

type Item = Record<string, unknown>;
interface Feed {
  items: Item[];
  active?: number;
  queued?: number;
  memGb?: number;
  tps?: number | null;
}
const status = (f: Feed) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [
    {
      id: "org/Qwen3.8-27B-oQ4e-mtp",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 16,
    },
  ],
  memory: {
    active_gb: f.memGb ?? 24,
    cache_gb: 2,
    peak_gb: 30,
    total_gb: 128,
  },
  requests: {
    active: f.active ?? f.items.length,
    queued: f.queued ?? 0,
    prefill: f.items.filter((i) => i.phase === "prefill").length,
    decode: f.items.filter((i) => i.phase === "decode").length,
    items: f.items,
  },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: f.tps ?? null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
});
async function feed(page: Page, make: (ms: number) => Feed) {
  const t0 = Date.now();
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.addInitScript(STABILITY_RECORDER);
  await page.route("**/v1/**", (route) => {
    if (new URL(route.request().url()).pathname === "/v1/yunshu/status")
      return route.fulfill({ json: status(make(Date.now() - t0)) });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}
const pills = (rec: Recording, keyHas = "li.") =>
  Object.entries(rec.keys)
    .filter(([, k]) => k.includes(keyHas) && k.startsWith("status-band"))
    .map(([id]) => id);

test("an idle engine: every status pill hugs its text (no dead room, no padding word)", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await feed(page, () => ({ items: [] }));
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2500);
  await reset(page);
  await page.waitForTimeout(1500);
  const rec = await record(page);
  const a = analyze(rec.frames, rec.keys);
  const bad = a.dead.filter((d) => d.px > 2.5);
  expect(a.dead.length, "pills were measured").toBeGreaterThan(20);
  expect(
    bad.slice(0, 4),
    "pills with trailing empty room (px after the last glyph)",
  ).toEqual([]);
});

test("a steady decode: every status pill hugs its text and nothing moves", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await feed(page, (ms) => ({
    items: [
      {
        request_id: "r",
        elapsed_s: ms / 1000,
        phase: "decode",
        completion_tokens: 50 + Math.floor(ms / 30),
        tokens_per_second: 80 + ((ms / 250) % 7),
      },
    ],
    tps: 80 + ((ms / 250) % 7),
    memGb: 24 + ((ms / 250) % 5) / 10,
  }));
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(3000);
  await reset(page);
  await page.waitForTimeout(4000);
  const rec = await record(page);
  const a = analyze(rec.frames, rec.keys);
  expect(
    a.dead.filter((d) => d.px > 2.5).slice(0, 4),
    "trailing empty room",
  ).toEqual([]);
  const band = analyze(
    select(rec.frames, rec.keys, (k) => k.startsWith("status-band")),
    rec.keys,
  );
  expect(band.steps, "pill widths while only digits change").toBe(0);
  // charts slide with the clock by design (LiveScroll); everything else holds still
  const still = analyze(
    select(rec.frames, rec.keys, (k) => !/metric-plot|chart|spark/.test(k)),
    rec.keys,
  );
  expect(still.moved, `moved: ${JSON.stringify(still.shoved)}`).toBe(0);
});

test("phase words and counts changing do not shove the pills before them", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  // idle -> prefill -> decode -> idle, repeated; counts cross a digit boundary (9 <-> 10) and memory
  // crosses 10 GB, every 250 ms.
  await feed(page, (ms) => {
    const cycle = (ms % 9000) / 1000;
    const flick = Math.floor(ms / 250) % 2;
    const base = {
      active: cycle < 1 ? 0 : 9 + flick,
      queued: cycle < 1 ? 0 : flick,
      memGb: 9.9 + flick * 0.2,
    };
    if (cycle < 1) return { ...base, items: [] };
    if (cycle < 3)
      return {
        ...base,
        items: [
          {
            request_id: "r",
            elapsed_s: cycle,
            phase: "prefill",
            prompt_tokens: 32000,
            cached_tokens: 0,
            processed_tokens: Math.round((cycle - 1) * 12000),
            percent: Math.min(99, (cycle - 1) * 40),
          },
        ],
      };
    if (cycle < 7)
      return {
        ...base,
        items: [
          {
            request_id: "r",
            elapsed_s: cycle,
            phase: "decode",
            completion_tokens: 20 + Math.floor(ms / 25),
            tokens_per_second: cycle < 5 ? 87 : 123,
          },
        ],
        tps: cycle < 5 ? 87 : 123,
      };
    return { ...base, items: [] };
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2500);
  await reset(page);
  await page.waitForTimeout(18000);
  const rec = await record(page);
  const ids = pills(rec);
  expect(ids.length, "pills in the band").toBeGreaterThanOrEqual(3);
  const ordered = ids
    .filter((id) => rec.frames.some((f) => f.boxes[id]))
    .sort(
      (x, y) =>
        rec.frames.find((f) => f.boxes[x])!.boxes[x][0] -
        rec.frames.find((f) => f.boxes[y])!.boxes[y][0],
    );
  // The last pill is the phase word and changes width with the phase; the pills before it hold
  // still: a count crossing a digit boundary or memory crossing 10 GB grows a pill once and the
  // flicker after that does not resize it.
  const before = ordered.slice(0, -1);
  const lefts = before.map((id) => ({
    key: rec.keys[id],
    seen: new Set(
      rec.frames
        .filter((f) => f.boxes[id])
        .map((f) => Math.round(f.boxes[id][0])),
    ),
  }));
  expect(
    lefts
      .filter((l) => l.seen.size > 3)
      .map((l) => `${l.key}: ${[...l.seen].join(",")}`),
    "pills before the phase pill whose left edge wandered",
  ).toEqual([]);
  const mine = analyze(
    select(rec.frames, rec.keys, (k) =>
      before.some((id) => rec.keys[id] === k),
    ),
    rec.keys,
  );
  expect(
    {
      flips: mine.flips,
      steps: mine.steps,
      top: mine.perElement.slice(0, 2).map((e) => [e.key, e.steps, e.flips]),
    },
    "width flip-flops and steps of the pills before the phase pill",
  ).toMatchObject({ flips: 0 });
  expect(mine.steps, "width steps in 18 s").toBeLessThanOrEqual(8);
});

test("the open status island does not reflow while a bursty engine works", async ({
  browser,
}) => {
  test.setTimeout(90_000);
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await ctx.newPage();
  await replay(page, "burst", 3, STABILITY_RECORDER);
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(4500);
  await page.getByTestId("footer-trigger").click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.waitForTimeout(1500);
  await reset(page);
  await page.waitForTimeout(15_000);
  const rec = await record(page);
  const inside = select(
    rec.frames,
    rec.keys,
    (k) => k.startsWith("island") || k.includes("island"),
  );
  const a = analyze(rec.frames, rec.keys, { boxed: rec.boxed });
  const b = analyze(inside, rec.keys, { boxed: rec.boxed });
  report("island", {
    steps: b.steps,
    flips: b.flips,
    movedFrames: b.movedFrames,
    shoved: b.shoved,
    top: b.perElement
      .slice(0, 10)
      .map((e) => [e.key, e.steps, e.flips, e.boxed]),
    allMovedFrames: a.movedFrames,
  });
  expect(
    b.perElement.filter((e) => e.boxed && e.flips > 1).map((e) => e.key),
    "island boxes that flip",
  ).toEqual([]);
  expect(
    b.movedFrames,
    `island frames with movement: ${JSON.stringify(b.shoved.slice(0, 5))}`,
  ).toBeLessThanOrEqual(10);
  await ctx.close();
});
