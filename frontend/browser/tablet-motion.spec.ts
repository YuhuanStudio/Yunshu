import { devices, expect, test, type Page } from "@playwright/test";

// Live charts on tablets (WebKit, touch, DPR 2, portrait and landscape): the time axis is a pure
// function of the clock, so nothing jumps when a sample lands; the value axis holds still and only
// changes with a glide; idle is drawn as 0, never ramped in from a value that was not measured; and
// a monotone curve never leaves the range of the samples.

const PROFILES = [
  "iPad Pro 11",
  "iPad Pro 11 landscape",
  "iPad Mini",
  "iPad Mini landscape",
] as const;

const BASE_STATUS = {
  object: "yunshu.status",
  version: "fixture",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [
    {
      id: "Qwen-VL",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 16,
    },
  ],
  memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
  last: null,
};

/** Busy for 9 s (rate rising 20 to 90, a late spike), idle for 8 s, busy again: both transitions. */
function statusAt(t: number) {
  const phase = t % 17_000;
  const busy = phase < 9_000;
  const tps = busy
    ? 20 + Math.min(70, phase / 90) + (phase > 6_000 && phase < 7_000 ? 25 : 0)
    : null;
  return {
    ...BASE_STATUS,
    requests: {
      active: busy ? 1 : 0,
      queued: 0,
      prefill: 0,
      decode: busy ? 1 : 0,
      items: busy
        ? [
            {
              request_id: "r",
              elapsed_s: 1,
              phase: "decode",
              completion_tokens: 10,
              tokens_per_second: tps,
            },
          ]
        : [],
    },
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: tps,
      mean_prefill_tps: null,
      mean_decode_tps: null,
    },
  };
}

async function open(page: Page) {
  const t0 = Date.now();
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: statusAt(Date.now() - t0) });
    return route.fulfill({ status: 404, json: {} });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
}

interface Frame {
  t: number;
  ticks: Record<string, number>;
  yLabels: string[];
  yLines: number[];
  paths: {
    d: string;
    box: { top: number; bottom: number; left: number; right: number };
  }[];
  plot: { top: number; bottom: number } | null;
  dpr: number;
  width: number;
}

/** Every animation frame for `ms`: tick positions by label, y-axis labels and lines, series paths. */
function record(page: Page, ms: number): Promise<{ frames: Frame[] }> {
  return page.evaluate(
    (duration) =>
      new Promise<{ frames: Frame[] }>((resolve) => {
        const root = document.querySelector(".live-scroll");
        if (!root) return resolve({ frames: [] });
        root.scrollIntoView({ block: "center" });
        const frames: Frame[] = [];
        const end = performance.now() + duration;
        const tick = () => {
          const svg = root.querySelector("svg");
          if (svg) {
            const ticks: Record<string, number> = {};
            for (const el of root.querySelectorAll('svg text[data-axis="x"]'))
              ticks[el.textContent ?? ""] = el.getBoundingClientRect().left;
            const yTexts = [
              ...root.querySelectorAll('svg text[data-axis="y"]'),
            ];
            const clip = root
              .querySelector("clipPath rect")
              ?.getBoundingClientRect();
            frames.push({
              t: performance.now(),
              ticks,
              yLabels: yTexts.map((e) => e.textContent ?? ""),
              yLines: yTexts.map((e) => e.getBoundingClientRect().top),
              paths: [
                ...root.querySelectorAll("svg [clip-path] path[stroke]"),
              ].map((p) => {
                const r = p.getBoundingClientRect();
                return {
                  d: p.getAttribute("d") ?? "",
                  box: {
                    top: r.top,
                    bottom: r.bottom,
                    left: r.left,
                    right: r.right,
                  },
                };
              }),
              plot: clip ? { top: clip.top, bottom: clip.bottom } : null,
              dpr: window.devicePixelRatio,
              width: window.innerWidth,
            });
          }
          if (performance.now() < end) requestAnimationFrame(tick);
          else resolve({ frames });
        };
        requestAnimationFrame(tick);
      }),
    ms,
  );
}

for (const name of PROFILES) {
  test.describe(name, () => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { defaultBrowserType, ...profile } = devices[name];
    test.use(profile);

    test("time axis: absolute ticks move left monotonically, never jump, across sample boundaries", async ({
      page,
      browserName,
    }) => {
      test.skip(browserName !== "webkit", "tablet profiles are WebKit");
      test.setTimeout(120_000);
      await open(page);
      await page.waitForTimeout(16_000); // five samples are needed before the chart draws a line
      const { frames } = await record(page, 9_000);
      expect(frames.length, "frames recorded").toBeGreaterThan(200);
      expect(frames[0]!.dpr).toBe(2);
      // follow every tick label that exists in two consecutive frames
      let worstUp = 0;
      let moved = 0;
      let samples = 0;
      for (let i = 1; i < frames.length; i++)
        for (const [label, x] of Object.entries(frames[i]!.ticks)) {
          const before = frames[i - 1]!.ticks[label];
          if (before == null) continue;
          samples++;
          const dx = x - before;
          worstUp = Math.max(worstUp, dx);
          if (dx < -0.001) moved++;
        }
      expect(samples).toBeGreaterThan(100);
      expect(
        worstUp,
        `a tick moved right by ${worstUp.toFixed(2)} px between frames`,
      ).toBeLessThan(0.3);
      expect(moved, "ticks slide with the clock").toBeGreaterThan(
        samples * 0.5,
      );
      // a tick passes a sample boundary: its largest single-frame step stays tiny (a re-base would be several px)
      let worstStep = 0;
      for (let i = 1; i < frames.length; i++)
        for (const [label, x] of Object.entries(frames[i]!.ticks)) {
          const before = frames[i - 1]!.ticks[label];
          if (before != null)
            worstStep = Math.max(worstStep, Math.abs(x - before));
        }
      expect(
        worstStep,
        `largest one-frame tick step ${worstStep.toFixed(2)} px`,
      ).toBeLessThan(2.5);
    });

    test("value axis holds still; any change is a glide, not a jump", async ({
      page,
      browserName,
    }) => {
      test.skip(browserName !== "webkit");
      test.setTimeout(120_000);
      await open(page);
      await page.waitForTimeout(16_000);
      const { frames } = await record(page, 9_000);
      const top = (f: Frame) =>
        f.yLabels
          .map((l) => Number(l.replace(/[^\d.]/g, "")))
          .reduce((a, b) => Math.max(a, b), 0);
      let changes = 0;
      for (let i = 1; i < frames.length; i++)
        if (top(frames[i]!) !== top(frames[i - 1]!)) changes++;
      // labels follow the gliding maximum: a handful of frames per change, never one per sample
      expect(changes).toBeLessThan(40);
      // gridline positions never leap more than a quarter of the plot between two frames
      for (let i = 1; i < frames.length; i++) {
        const a = frames[i - 1]!;
        const b = frames[i]!;
        if (!a.plot || a.yLines.length !== b.yLines.length) continue;
        const height = a.plot.bottom - a.plot.top;
        for (let k = 0; k < a.yLines.length; k++)
          expect(
            Math.abs(b.yLines[k]! - a.yLines[k]!),
            "y gridline leap",
          ).toBeLessThan(height * 0.3);
      }
    });

    test("the curve stays inside the plot and idle is drawn as zero in one unbroken line", async ({
      page,
      browserName,
    }) => {
      test.skip(browserName !== "webkit");
      test.setTimeout(150_000);
      await open(page);
      await page.waitForTimeout(24_000); // a busy stretch, an idle stretch and the next busy one are in the window
      const { frames } = await record(page, 3_000);
      const last = frames.at(-1)!;
      expect(last.paths.length).toBeGreaterThan(0);
      for (const f of frames.slice(-20))
        for (const p of f.paths) {
          expect(f.plot).not.toBeNull();
          expect(p.box.top, "above the plot").toBeGreaterThanOrEqual(
            f.plot!.top - 1,
          );
          expect(
            p.box.bottom,
            "below the baseline (negative overshoot)",
          ).toBeLessThanOrEqual(f.plot!.bottom + 1);
        }
      // idle between the busy stretches is 0, so the decode series is one segment, not broken into pieces
      const segments = last.paths
        .filter((p) => /^M/.test(p.d))
        .map((p) => (p.d.match(/M/g) ?? []).length);
      expect(
        Math.max(...segments),
        "the decode line has no gap where the engine was idle",
      ).toBe(1);
    });
  });
}
