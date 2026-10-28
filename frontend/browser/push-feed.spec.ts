import { createServer, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { expect, test, type Page } from "@playwright/test";

// The console's change feed (GET /v1/yunshu/events) against a real HTTP server that the test
// controls byte by byte: a pushed status reaches the screen without polling, and when the stream is
// refused or dies the hooks poll again. The server speaks CORS because the page is served by vite.

type Json = Record<string, unknown>;

const idle: Json = {
  object: "yunshu.status",
  version: "fixture",
  state: "running",
  uptime_s: 500,
  load_error: null,
  models: [
    {
      id: "Qwen3.8-27B",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: false,
    },
  ],
  memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
};
const prefilling: Json = {
  ...idle,
  requests: {
    active: 1,
    queued: 0,
    prefill: 1,
    decode: 0,
    items: [
      {
        request_id: "push-1",
        elapsed_s: 1,
        phase: "prefill",
        prompt_tokens: 1000,
        processed_tokens: 340,
        tokens_per_second: 170,
      },
    ],
  },
};

class Engine {
  server!: Server;
  url = "";
  status: Json = idle;
  statusGets = 0;
  eventOpens = 0;
  eventsMode: "stream" | "refuse" = "stream";
  streams = new Set<ServerResponse>();
  async start() {
    this.server = createServer((req, res) => {
      const cors = {
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "authorization, content-type",
      };
      if (req.method === "OPTIONS") {
        res.writeHead(204, cors).end();
        return;
      }
      const path = new URL(req.url ?? "/", "http://x").pathname;
      if (path === "/v1/yunshu/status") {
        this.statusGets++;
        res.writeHead(200, { ...cors, "content-type": "application/json" });
        res.end(JSON.stringify(this.status));
      } else if (path === "/v1/yunshu/events") {
        this.eventOpens++;
        if (this.eventsMode === "refuse") {
          res.writeHead(404, cors).end("no");
          return;
        }
        res.writeHead(200, { ...cors, "content-type": "text/event-stream" });
        this.streams.add(res);
        res.on("close", () => this.streams.delete(res));
        this.send(res, "hello", { boot_id: "B1", topics: ["status"] });
        this.send(res, "status", { v: 1, skipped: 0, d: this.status });
      } else {
        res.writeHead(404, cors).end("{}");
      }
    });
    await new Promise<void>((r) => this.server.listen(0, "127.0.0.1", r));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }
  send(res: ServerResponse, event: string, data: unknown) {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }
  push(status: Json) {
    this.status = status;
    for (const s of this.streams)
      this.send(s, "status", { v: 2, skipped: 0, d: status });
  }
  dropStreams() {
    for (const s of [...this.streams]) s.end();
  }
  async stop() {
    this.dropStreams();
    this.server.closeAllConnections();
    await new Promise((r) => this.server.close(r));
  }
}

async function open(page: Page, engine: Engine) {
  await page.addInitScript((url) => {
    localStorage.setItem("yunshu.console.url", url);
    localStorage.setItem("yunshu.console.lang", "en");
  }, engine.url);
  await page.goto("/console/?lang=en#/overview");
  await expect(page.getByTestId("live-phase")).toBeVisible();
}

test.describe("console change feed", () => {
  test("a pushed status reaches the screen at once and nothing polls while the stream is live", async ({
    page,
  }) => {
    const engine = new Engine();
    await engine.start();
    try {
      await open(page, engine);
      await expect.poll(() => engine.streams.size).toBe(1);
      await expect(page.getByTestId("live-phase")).toContainText(/idle|閒置/i);
      const polls = engine.statusGets;
      const sent = Date.now();
      engine.push(prefilling);
      await expect(page.getByTestId("live-phase")).toContainText(
        /prefill|預填/i,
      );
      expect(Date.now() - sent).toBeLessThan(1500);
      // 6 s is two idle poll periods (3 s): a polling console would have asked at least once.
      await page.waitForTimeout(6000);
      expect(engine.statusGets - polls).toBeLessThanOrEqual(1);
      expect(engine.eventOpens).toBe(1);
    } finally {
      await engine.stop();
    }
  });

  test("an engine without the feed is polled as before", async ({ page }) => {
    const engine = new Engine();
    engine.eventsMode = "refuse";
    await engine.start();
    try {
      await open(page, engine);
      await expect(page.getByTestId("live-phase")).toContainText(/idle|閒置/i);
      engine.status = prefilling;
      await expect(page.getByTestId("live-phase")).toContainText(
        /prefill|預填/i,
        {
          timeout: 8000,
        },
      );
      expect(engine.statusGets).toBeGreaterThan(1);
      expect(engine.eventOpens).toBe(1); // a refusing engine is not hammered
    } finally {
      await engine.stop();
    }
  });

  test("a dropped stream reconnects and its first event is a full snapshot", async ({
    page,
  }) => {
    const engine = new Engine();
    await engine.start();
    try {
      await open(page, engine);
      await expect.poll(() => engine.streams.size).toBe(1);
      engine.status = prefilling; // changes while the tab is cut off
      engine.dropStreams();
      await expect(page.getByTestId("live-phase")).toContainText(
        /prefill|預填/i,
        { timeout: 8000 },
      );
      expect(engine.eventOpens).toBeGreaterThanOrEqual(2);
      await expect.poll(() => engine.streams.size).toBe(1);
    } finally {
      await engine.stop();
    }
  });

  test("ten tabs of one browser share one stream (HTTP/1.1 allows six connections per origin)", async ({
    browser,
  }) => {
    const engine = new Engine();
    await engine.start();
    const context = await browser.newContext({ locale: "en-US" });
    try {
      await context.addInitScript((url) => {
        localStorage.setItem("yunshu.console.url", url);
      }, engine.url);
      const pages = [];
      for (let i = 0; i < 10; i++) {
        const page = await context.newPage();
        await page.goto("/console/?lang=en#/overview", { timeout: 20000 });
        pages.push(page);
      }
      for (const page of pages)
        await expect(page.getByTestId("live-phase")).toBeVisible();
      await expect.poll(() => engine.streams.size).toBe(1);
      expect(engine.eventOpens).toBe(1);
      engine.push(prefilling);
      for (const page of pages)
        await expect(page.getByTestId("live-phase")).toContainText(
          /prefill|預填/i,
          { timeout: 4000 },
        );
      const polls = engine.statusGets;
      await page0Wait(pages[0]);
      expect(engine.statusGets - polls).toBeLessThanOrEqual(2);
    } finally {
      await context.close();
      await engine.stop();
    }
  });
});

const page0Wait = (page: Page) => page.waitForTimeout(6000);
