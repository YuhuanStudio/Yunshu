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
      loaded: false,
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

async function installRouteSse(page: Page) {
  await page.addInitScript(() => {
    const state = ((
      window as unknown as {
        __yunshuSse?: { calls: unknown[]; cancels: number };
      }
    ).__yunshuSse = { calls: [], cancels: 0 });
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
      const hold = String(payload.messages?.at(-1)?.content ?? "").includes(
        "STOP QA STREAM",
      );
      state!.calls.push({
        model: payload.model,
        stream: payload.stream,
        hold,
        authorized: new Headers(init.headers).has("Authorization"),
      });
      let timer: number | undefined;
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                event({
                  choices: [
                    { delta: { reasoning_content: "Playwright reasoning. " } },
                  ],
                }),
              ),
            );
            controller.enqueue(
              encoder.encode(
                event({
                  choices: [
                    {
                      delta: {
                        content: hold
                          ? "Playwright partial response. "
                          : "Playwright response. ",
                      },
                    },
                  ],
                }),
              ),
            );
            if (!hold)
              timer = window.setTimeout(() => {
                controller.enqueue(
                  encoder.encode(
                    event({
                      choices: [{ delta: { content: "Stream completed." } }],
                    }),
                  ),
                );
                controller.enqueue(
                  encoder.encode(
                    event({ choices: [{ delta: {}, finish_reason: "stop" }] }),
                  ),
                );
                controller.enqueue(encoder.encode("data: [DONE]\n\n"));
                controller.close();
              }, 120);
          },
          cancel() {
            state!.cancels += 1;
            if (timer !== undefined) window.clearTimeout(timer);
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    };
  });
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

test("auth, model lifecycle, warmup and request cancellation use the real /v1 API contract", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  await installRouteSse(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "engine integration",
  );
  await page.goto("/console/");
  await expect(page.getByTestId("overview")).toBeVisible();
  await expect(page.getByText("需要有效的存取權杖")).toBeVisible();

  await page.getByRole("button", { name: "設定", exact: true }).click();
  const tokenInput = page.getByLabel("存取權杖");
  await tokenInput.fill("wrong-playwright-token");
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  await page.getByRole("link", { name: "引擎總覽", exact: true }).click();
  await expect(page.getByText("需要有效的存取權杖")).toBeVisible();

  await page.getByRole("button", { name: "設定", exact: true }).click();
  await tokenInput.fill(api.token);
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  await page.getByRole("link", { name: "引擎總覽", exact: true }).click();
  await expect(page.getByText("已連線", { exact: true })).toBeVisible();
  const storage = await page.evaluate(() =>
    JSON.stringify({
      local: Object.values(localStorage),
      session: Object.values(sessionStorage),
    }),
  );
  expect(storage).not.toContain(api.token);
  expect(api.apiCalls.some((call) => call.authorized)).toBe(true);
  await expect
    .poll(
      () =>
        api.statusCalls.filter((call) =>
          call.sourcePath.endsWith("/v1/yunshu/status"),
        ).length,
      { timeout: 8_000 },
    )
    .toBeGreaterThanOrEqual(2);
  await expect(page.getByTestId("overview")).toContainText(
    /本頁開啟後採樣 · [2-9]\d* 筆/,
  );

  await page.getByRole("link", { name: "模型庫", exact: true }).click();
  const models = page.getByTestId("models");
  const detailButton = models.getByRole("button", {
    name: "Qwen3.8-27B",
    exact: true,
  });
  await detailButton.focus();
  await detailButton.click();
  const details = page.getByRole("dialog");
  await expect(
    details.getByRole("heading", { name: "Qwen3.8-27B", exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(details).toBeHidden();
  await expect(detailButton).toBeFocused();

  const qwenSmall = models.getByRole("row").filter({ hasText: "Qwen3.5-9B" });
  api.requireNextLoad401();
  await qwenSmall.getByRole("button", { name: "載入", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Authentication failed");
  await qwenSmall.getByRole("button", { name: "載入", exact: true }).click();
  await expect(qwenSmall.getByText("已載入", { exact: true })).toBeVisible();
  await qwenSmall.getByRole("button", { name: "預熱", exact: true }).click();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "模型已載入；此類型未執行文字預熱。" }),
  ).toBeVisible();
  expect(api.warmupCount).toBe(1);

  api.conflictNextUnload("Qwen3.5-9B");
  await qwenSmall.getByRole("button", { name: "卸載", exact: true }).click();
  const unloadDialog = page.getByRole("dialog");
  await unloadDialog.getByRole("button", { name: "卸載", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(
    "active Playwright fixture request",
  );
  await expect(qwenSmall.getByText("已載入", { exact: true })).toBeVisible();
  await qwenSmall.getByRole("button", { name: "卸載", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "卸載", exact: true })
    .click();
  await expect(qwenSmall.getByText("未載入", { exact: true })).toBeVisible();

  await page.getByRole("link", { name: "請求與效能", exact: true }).click();
  const activeRow = page.getByRole("row").filter({ hasText: "qa-active-1" });
  await activeRow.getByRole("button", { name: "取消", exact: true }).click();
  const cancelDialog = page.getByRole("dialog");
  await cancelDialog
    .getByRole("button", { name: "確認取消", exact: true })
    .click();
  await expect(
    page.getByRole("status").filter({ hasText: "已送出取消請求" }),
  ).toBeVisible();
  await expect(
    page.getByRole("row").filter({ hasText: "qa-active-1" }),
  ).toHaveCount(0);
  expect(api.cancelCount).toBe(1);
  await verifyClean();
});

test("stream send/stop and changing service URL resets sampled history", async ({
  page,
}) => {
  const api = createApiFixture();
  await api.attach(page);
  await installRouteSse(page);
  const verifyClean = await installDiagnostics(
    page,
    api.expectedResponses,
    "stream integration",
  );
  await page.goto("/console/");
  await page.getByRole("button", { name: "設定", exact: true }).click();
  await page.getByLabel("存取權杖").fill(api.token);
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  await page.getByRole("link", { name: "推理測試", exact: true }).click();
  const playground = page.getByTestId("playground");
  const composer = playground.locator("textarea").first();
  await composer.fill("正常 route SSE QA");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  await expect(playground).toContainText("Playwright response.");
  await expect(playground).toContainText("Stream completed.");
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as unknown as { __yunshuSse?: { calls: unknown[] } })
            .__yunshuSse?.calls.length ?? 0,
      ),
    )
    .toBe(1);

  await composer.fill("STOP QA STREAM");
  await page.getByRole("button", { name: "傳送測試", exact: true }).click();
  await expect(playground).toContainText("Playwright partial response.");
  await page.getByRole("button", { name: "停止生成", exact: true }).click();
  await expect(
    playground.getByRole("status").filter({ hasText: "已停止生成" }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as unknown as { __yunshuSse?: { cancels: number } })
            .__yunshuSse?.cancels ?? 0,
      ),
    )
    .toBe(1);

  await page.getByRole("button", { name: "設定", exact: true }).click();
  await page
    .getByLabel("服務位址")
    .fill(
      `${new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:3971").origin}/qa-connection`,
    );
  await page.getByLabel("存取權杖").fill(api.token);
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  await expect
    .poll(
      () =>
        api.statusCalls.filter((call) =>
          call.sourcePath.includes("/qa-connection/v1/yunshu/status"),
        ).length,
      { timeout: 8_000 },
    )
    .toBeGreaterThanOrEqual(1);
  await page.getByRole("link", { name: "引擎總覽", exact: true }).click();
  await expect(page.getByTestId("overview")).toContainText(
    "本頁開啟後採樣 · 1 筆 · 中斷期間不補資料",
  );
  await verifyClean();
});
