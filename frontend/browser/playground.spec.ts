import { expect, test, type Page, type Route } from "@playwright/test";

type ModelFixture = {
  id: string;
  type: string;
  loaded: boolean;
  loading: boolean;
  pinned: boolean;
  size_gb: number;
  idle_s: number | null;
  keep_alive_s: number | null;
  expires_in_s: number | null;
};
type RequestFixture = {
  request_id: string;
  elapsed_s: number;
  phase: string;
  model: string;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  tokens_per_second: number;
  percent?: number;
};

function createApiFixture() {
  const token = "playwright-only-token";
  const models: ModelFixture[] = [
    {
      id: "Qwen3.8-27B",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: true,
      size_gb: 17.6,
      idle_s: 0,
      keep_alive_s: null,
      expires_in_s: null,
    },
    {
      id: "Qwen3.5-9B",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 5.8,
      idle_s: null,
      keep_alive_s: 600,
      expires_in_s: null,
    },
  ];
  let requests: RequestFixture[] = [
    {
      request_id: "qa-active-1",
      elapsed_s: 2.4,
      phase: "decode",
      model: "Qwen3.8-27B",
      prompt_tokens: 128,
      cached_tokens: 64,
      completion_tokens: 12,
      tokens_per_second: 35.5,
    },
  ];
  const statusCalls: Array<{ sourcePath: string; sample: number }> = [];
  const apiCalls: Array<{
    method: string;
    path: string;
    sourcePath: string;
    authorized: boolean;
  }> = [];
  const expectedResponses: Array<{
    status: number;
    method: string;
    sourcePath: string;
  }> = [];
  let failNextLoad401 = false;
  let conflictNextUnloadId: string | null = null;
  let warmupCount = 0;
  let cancelCount = 0;

  const json = (route: Route, payload: unknown, status = 200) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  const statusBody = (sourcePath: string) => {
    const sample =
      statusCalls.filter((call) => call.sourcePath === sourcePath).length + 1;
    statusCalls.push({ sourcePath, sample });
    const count = (phase: string) =>
      requests.filter((row) => row.phase === phase).length;
    return {
      object: "yunshu.status",
      version: "playwright-fixture",
      state: "running",
      uptime_s: 3_600 + sample,
      load_error: null,
      models: models.map((model) => ({ ...model })),
      memory: {
        active_gb: 18 + sample / 10,
        cache_gb: 2.1,
        peak_gb: 20.5,
        total_gb: 64,
        pressure: 0.3,
      },
      requests: {
        active: requests.length,
        queued: count("queued"),
        prefill: count("prefill"),
        decode: count("decode"),
        items: requests.map((row) => ({ ...row })),
      },
      last: {
        request_id: "qa-last-1",
        prompt_tokens: 1_024,
        completion_tokens: 256,
        cached_tokens: 512,
        prefill_tps: 810 + sample,
        decode_tps: 41 + sample / 10,
        ttft_ms: 620,
        t: 1_710_000_000 + sample,
      },
      throughput: {
        window_s: 60,
        requests: 8 + sample,
        prompt_tokens: 8_400 + sample * 10,
        completion_tokens: 1_200 + sample * 3,
        live_decode_tps: 34 + sample / 10,
        mean_prefill_tps: 780 + sample,
        mean_decode_tps: 39 + sample / 10,
      },
    };
  };

  async function routeApi(route: Route) {
    const request = route.request();
    const url = new URL(request.url());
    const marker = url.pathname.lastIndexOf("/v1/");
    const path = marker >= 0 ? url.pathname.slice(marker + 3) : url.pathname;
    const method = request.method();
    const authorized =
      (request.headers()["authorization"] ?? "") === `Bearer ${token}`;
    apiCalls.push({ method, path, sourcePath: url.pathname, authorized });

    if (!authorized) {
      expectedResponses.push({ status: 401, method, sourcePath: url.pathname });
      return json(
        route,
        { detail: "The Playwright auth fixture requires a bearer token." },
        401,
      );
    }
    if (method === "GET" && path === "/yunshu/status")
      return json(route, statusBody(url.pathname));
    if (method === "GET" && path.startsWith("/models/")) {
      const id = decodeURIComponent(path.slice("/models/".length));
      const model = models.find((item) => item.id === id);
      return model
        ? json(route, {
            ...model,
            card: { capabilities: ["text"], fixture: true },
          })
        : json(route, { detail: `Unknown model ${id}` }, 404);
    }
    if (method === "POST" && path === "/models/load") {
      const input = JSON.parse(request.postData() ?? "{}");
      if (failNextLoad401) {
        failNextLoad401 = false;
        expectedResponses.push({
          status: 401,
          method,
          sourcePath: url.pathname,
        });
        return json(route, { detail: "Fixture rejected model load." }, 401);
      }
      const model = models.find((item) => item.id === input.model);
      if (!model)
        return json(route, { detail: `Unknown model ${input.model}` }, 404);
      model.loaded = true;
      return json(route, { status: "loaded", model: model.id });
    }
    if (method === "POST" && path.startsWith("/models/unload/")) {
      const id = decodeURIComponent(path.slice("/models/unload/".length));
      if (conflictNextUnloadId === id) {
        conflictNextUnloadId = null;
        expectedResponses.push({
          status: 409,
          method,
          sourcePath: url.pathname,
        });
        return json(
          route,
          { detail: `Model ${id} has an active Playwright fixture request.` },
          409,
        );
      }
      const model = models.find((item) => item.id === id);
      if (!model) return json(route, { detail: `Unknown model ${id}` }, 404);
      if (model.pinned)
        return json(
          route,
          { detail: "The fixture's single-model entry remains pinned." },
          409,
        );
      model.loaded = false;
      return json(route, { status: "unloaded", model: id });
    }
    if (method === "POST" && path === "/yunshu/warmup") {
      warmupCount += 1;
      const input = JSON.parse(request.postData() ?? "{}");
      return json(route, {
        object: "yunshu.warmup",
        model: input.model,
        generated: false,
        warmup_ms: 12,
        fixture: true,
      });
    }
    if (method === "DELETE" && path.startsWith("/requests/")) {
      const id = decodeURIComponent(path.slice("/requests/".length));
      cancelCount += 1;
      requests = requests.filter((row) => row.request_id !== id);
      return json(route, {
        object: "yunshu.request",
        id,
        status: "cancelling",
        fixture: true,
      });
    }
    return json(
      route,
      { detail: `Unhandled Playwright fixture route: ${method} ${path}` },
      404,
    );
  }

  return {
    token,
    models,
    statusCalls,
    apiCalls,
    expectedResponses,
    attach: (page: Page) => page.route("**/v1/**", routeApi),
    requireNextLoad401: () => {
      failNextLoad401 = true;
    },
    conflictNextUnload: (id: string) => {
      conflictNextUnloadId = id;
    },
    get warmupCount() {
      return warmupCount;
    },
    get cancelCount() {
      return cancelCount;
    },
  };
}

async function installCompareSse(page: Page) {
  await page.addInitScript(() => {
    const state = ((
      window as unknown as {
        __cmp?: {
          calls: Array<{ model: string; temperature: number }>;
          active: number;
          maxActive: number;
        };
      }
    ).__cmp = { calls: [], active: 0, maxActive: 0 });
    const nativeFetch = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const event = (data: unknown) => `data: ${JSON.stringify(data)}\n\n`;
    window.fetch = async (input, init = {}) => {
      const rawUrl =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      const url = new URL(rawUrl, location.href);
      if (!url.pathname.endsWith("/v1/chat/completions"))
        return nativeFetch(input, init);
      const payload = JSON.parse(String(init.body ?? "{}"));
      state.calls.push({
        model: payload.model,
        temperature: payload.temperature,
      });
      state.active += 1;
      state.maxActive = Math.max(state.maxActive, state.active);
      const big = payload.model === "Qwen3.8-27B";
      const diverge = String(payload.messages?.at(-1)?.content ?? "").includes(
        "DIVERGE",
      );
      const text =
        big || !diverge
          ? ["The answer ", "is four."]
          : ["The answer ", "is five."];
      return new Response(
        new ReadableStream({
          async start(controller) {
            const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
            await wait(big ? 60 : 120);
            for (const piece of text) {
              controller.enqueue(
                encoder.encode(
                  event({ choices: [{ delta: { content: piece } }] }),
                ),
              );
              await wait(40);
            }
            controller.enqueue(
              encoder.encode(
                event({ choices: [{ delta: {}, finish_reason: "stop" }] }),
              ),
            );
            controller.enqueue(
              encoder.encode(
                event({
                  choices: [],
                  usage: {
                    prompt_tokens: 1024,
                    completion_tokens: 20,
                    prompt_tokens_details: { cached_tokens: 512 },
                  },
                }),
              ),
            );
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
            state.active -= 1;
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    };
  });
}

async function openPlayground(
  page: Page,
  api: ReturnType<typeof createApiFixture>,
) {
  await installCompareSse(page);
  await page.goto("/console/");
  await page.getByRole("button", { name: "設定", exact: true }).click();
  await page.getByLabel("存取權杖").fill(api.token);
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  await page.getByRole("link", { name: "推理測試", exact: true }).click();
  return page.getByTestId("playground");
}

async function installDiagnostics(
  page: Page,
  expected: Array<{ status: number; method: string; sourcePath: string }>,
  label: string,
) {
  const unexpectedConsole: string[] = [];
  const unexpectedHttp: string[] = [];
  const failures: string[] = [];
  const external: string[] = [];
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    if (/status of 401|status of 409/.test(message.text())) return;
    unexpectedConsole.push(message.text());
  });
  page.on("requestfailed", (request) => {
    const path = new URL(request.url()).pathname;
    if (
      /ERR_ABORTED|cancelled/i.test(request.failure()?.errorText ?? "") &&
      path.endsWith("/v1/yunshu/status")
    )
      return;
    failures.push(`${request.url()} ${request.failure()?.errorText}`);
  });
  page.on("request", (request) => {
    if (
      new URL(request.url()).origin !==
      new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:3971").origin
    )
      external.push(request.url());
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const index = expected.findIndex(
      (item) =>
        item.status === response.status() &&
        item.method === response.request().method() &&
        item.sourcePath === new URL(response.url()).pathname,
    );
    if (index >= 0) expected.splice(index, 1);
    else unexpectedHttp.push(`${response.status()} ${response.url()}`);
  });
  return async () => {
    expect(pageErrors, `${label} page errors`).toEqual([]);
    expect(unexpectedConsole, `${label} unexpected console errors`).toEqual([]);
    expect(failures, `${label} request failures`).toEqual([]);
    expect(external, `${label} external requests`).toEqual([]);
    expect(unexpectedHttp, `${label} unexpected HTTP errors`).toEqual([]);
    // React StrictMode can abort the first unauthenticated status request after
    // the route handler records its fixture response but before response delivery.
    // Authentication is separately asserted through the visible unauthorized UI.
    const remaining = expected.filter(
      (item) =>
        !(
          item.status === 401 &&
          item.method === "GET" &&
          item.sourcePath.endsWith("/v1/yunshu/status")
        ),
    );
    expect(remaining, `${label} unobserved operation error responses`).toEqual(
      [],
    );
  };
}

test("chat reply shows per-reply stats with TTFT and cache badge", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "stats",
  );
  const playground = await openPlayground(page, api);
  await playground.locator("textarea").first().fill("stats please");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  await expect(playground).toContainText("is four.");
  const stats = playground.getByTestId("reply-stats");
  await expect(stats).toContainText("20");
  await expect(stats).toContainText("tok/s");
  await expect(stats).toContainText(/TTFT \d/);
  await expect(stats).toContainText("快取 512/1,024");
  await verifyClean();
});

test("compare mode runs sequentially and reports identical greedy output", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "compare",
  );
  const playground = await openPlayground(page, api);
  await page.getByRole("button", { name: "比較", exact: true }).click();
  await playground.getByRole("button", { name: "貪婪 T=0" }).nth(0).click();
  await playground.getByRole("button", { name: "貪婪 T=0" }).nth(1).click();
  await playground.locator("textarea").first().fill("same prompt");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  const delta = playground.getByTestId("compare-delta");
  await expect(delta).toContainText("輸出完全一致");
  await expect(delta).toContainText("Δ tok/s");
  await expect(delta).toContainText("Δ TTFT");
  await expect(playground.getByTestId("compare-col-a")).toContainText(
    "is four.",
  );
  await expect(playground.getByTestId("compare-col-b")).toContainText(
    "快取 512/1,024",
  );
  const cmp = await page.evaluate(
    () =>
      (
        window as unknown as {
          __cmp: {
            calls: Array<{ model: string; temperature: number }>;
            maxActive: number;
          };
        }
      ).__cmp,
  );
  expect(cmp.maxActive).toBe(1);
  expect(cmp.calls.map((c) => c.model)).toEqual(["Qwen3.8-27B", "Qwen3.5-9B"]);
  expect(cmp.calls.map((c) => c.temperature)).toEqual([0, 0]);
  await verifyClean();
});

test("compare mode shows the first divergence offset and never claims identity", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "diverge",
  );
  const playground = await openPlayground(page, api);
  await page.getByRole("button", { name: "比較", exact: true }).click();
  await playground.getByRole("button", { name: "貪婪 T=0" }).nth(0).click();
  await playground.getByRole("button", { name: "貪婪 T=0" }).nth(1).click();
  await playground.locator("textarea").first().fill("DIVERGE now");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  const delta = playground.getByTestId("compare-delta");
  await expect(delta).toContainText("首次分歧於字元偏移 15");
  await expect(delta).not.toContainText("輸出完全一致");
  await verifyClean();
});

test("compare mode with sampling does not claim determinism", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "sampled",
  );
  const playground = await openPlayground(page, api);
  await page.getByRole("button", { name: "比較", exact: true }).click();
  await playground.locator("textarea").first().fill("same prompt");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  const delta = playground.getByTestId("compare-delta");
  await expect(delta).toContainText("文字相同（取樣非貪婪");
  await expect(delta).not.toContainText("輸出完全一致");
  await verifyClean();
});
