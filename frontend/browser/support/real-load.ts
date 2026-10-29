import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import type { Page } from "@playwright/test";
import { PAGE_RECORDER } from "../../scripts/review-analyze.mjs";

// fixtures/real-load*.jsonl.gz is the REAL Qwen3.8-27B engine's status stream and finished
// request stream (metadata only) from a bursty review run (scripts/research/console_review.py).

interface Line {
  t: number;
  status?: Record<string, unknown>;
  requests?: Record<string, unknown>[];
}
const load = (name: string): Line[] =>
  gunzipSync(readFileSync(new URL(`../fixtures/${name}`, import.meta.url)))
    .toString()
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
// "burst": 11 concurrent requests, prefill and decode; "single": one request decoding for 40 s.
export const FIXTURES = {
  burst: load("real-load.jsonl.gz"),
  single: load("real-load-single.jsonl.gz"),
};
export type LoadKind = keyof typeof FIXTURES;

/** Serve the recorded stream to the page, `speed` times faster than it was recorded. */
export async function replay(
  page: Page,
  kind: LoadKind = "burst",
  speed = 3,
  recorder: string = PAGE_RECORDER,
) {
  const LINES = FIXTURES[kind];
  const STATUS = LINES.filter((l) => l.status);
  const t0 = Date.now();
  const at = () => ((Date.now() - t0) / 1000) * speed;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.addInitScript(recorder);
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
