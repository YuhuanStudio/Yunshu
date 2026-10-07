import { expect, test, type Page, type Route } from "@playwright/test";

type ModelFixture = {
  id: string;
  type: string;
  loaded: boolean;
  loading: boolean;
  pinned: boolean;
  size_gb: number;
  expires_in_s: number | null;
};

type RecordedRequest = {
  method: string;
  path: string;
  body: unknown;
};

function createApiFixture(page: Page, withActiveRequest = false) {
  const models: ModelFixture[] = [
    {
      id: "org/qwen-mlx",
      type: "text",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 8,
      expires_in_s: 900,
    },
    {
      id: "org/vision-vlm",
      type: "vision-vlm",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 10,
      expires_in_s: 900,
    },
    {
      id: "org/pinned-mlx",
      type: "text",
      loaded: true,
      loading: false,
      pinned: true,
      size_gb: 4,
      expires_in_s: null,
    },
  ];
  const requests: RecordedRequest[] = [];
  const unexpected: string[] = [];
  const unexpectedMutations: string[] = [];
  let detailCalls = 0;

  const record = async (route: Route) => {
    const request = route.request();
    let body: unknown;
    try {
      body = request.postDataJSON();
    } catch {
      body = undefined;
    }
    const url = new URL(request.url());
    const entry = { method: request.method(), path: url.pathname, body };
    requests.push(entry);
    return entry;
  };

  const json = (route: Route, status: number, body?: unknown) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: body === undefined ? "" : JSON.stringify(body),
    });

  const engineStatus = () => ({
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "ready",
    uptime_s: 3_600,
    load_error: null,
    models: models.map((model) => ({ ...model })),
    memory: {
      active_gb: 12,
      cache_gb: 2,
      peak_gb: 14,
      total_gb: 64,
      pressure: 0.2,
    },
    requests: {
      active: withActiveRequest ? 1 : 0,
      queued: 0,
      prefill: 0,
      decode: withActiveRequest ? 1 : 0,
      items: withActiveRequest
        ? [
            {
              request_id: "qa-active-01",
              elapsed_s: 5,
              phase: "decode",
              model: "org/qwen-mlx",
              prompt_tokens: 111,
              cached_tokens: 22,
              completion_tokens: 3,
              tokens_per_second: 8,
            },
          ]
        : [],
    },
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
  });

  async function handleV1(route: Route) {
    const call = await record(route);
    if (call.method === "GET" && call.path === "/v1/yunshu/status")
      return json(route, 200, engineStatus());

    if (
      call.method === "GET" &&
      call.path === "/v1/requests/qa-active-01" &&
      withActiveRequest
    ) {
      detailCalls += 1;
      if (detailCalls === 3)
        return json(route, 404, { detail: "request is no longer active" });
      if (detailCalls > 3) {
        unexpected.push(`${call.method} ${call.path} after terminal 404`);
        return json(route, 500, { detail: "unexpected poll after 404" });
      }
      return json(route, 200, {
        request_id: "qa-active-01",
        phase: "decode",
        model: "org/qwen-mlx",
        elapsed_s: 42 + detailCalls,
        prompt_tokens: 777,
        cached_tokens: 333,
        completion_tokens: 55,
        ttft_ms: 88,
        decode_tps: 61,
        prefill_tps: 700,
        t: 1_800_000_000,
      });
    }

    if (call.method === "POST" && call.path === "/v1/yunshu/warmup")
      return json(route, 200, { generated: false, fixture: true });

    if (call.method === "POST" && call.path === "/v1/chat/completions") {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: { "cache-control": "no-cache" },
        body:
          'data: {"choices":[{"delta":{"content":"{\\"ok\\":true}"},"finish_reason":null}]}\n\n' +
          'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n' +
          "data: [DONE]\n\n",
      });
      return;
    }

    // Older engines have no memory ledger; the console falls back quietly.
    // Optional Yunshu-native reads (memory ledger, config, recent requests) are absent
    // on older engines; the console falls back, so the fixture answers 404 without flagging.
    if (call.method === "GET" && call.path.startsWith("/v1/yunshu/"))
      return json(route, 404, { detail: "Not Found" });

    if (call.method !== "GET")
      unexpectedMutations.push(`${call.method} ${call.path}`);
    unexpected.push(`${call.method} ${call.path}`);
    return json(route, 404, { detail: "fixture endpoint not found" });
  }

  async function handleApi(route: Route) {
    const call = await record(route);
    const body = call.body as Record<string, unknown> | undefined;
    if (call.method === "POST" && call.path === "/api/pull") {
      if (body && typeof body.model === "string") {
        models.push({
          id: body.model,
          type: "text",
          loaded: false,
          loading: false,
          pinned: false,
          size_gb: 3,
          expires_in_s: null,
        });
      }
      return json(route, 200);
    }
    if (call.method === "POST" && call.path === "/api/copy") {
      const source = models.find((model) => model.id === body?.source);
      if (source && typeof body?.destination === "string")
        models.push({
          ...source,
          id: body.destination,
          pinned: false,
          expires_in_s: null,
        });
      return json(route, 200);
    }
    if (call.method === "DELETE" && call.path === "/api/delete") {
      const id = body?.model;
      const index = models.findIndex((model) => model.id === id);
      if (index >= 0) models.splice(index, 1);
      return json(route, 200);
    }
    if (call.method !== "GET")
      unexpectedMutations.push(`${call.method} ${call.path}`);
    unexpected.push(`${call.method} ${call.path}`);
    return json(route, 404, { detail: "fixture endpoint not found" });
  }

  async function handleDebug(route: Route) {
    await record(route);
    return json(route, 404, { detail: "diagnostics disabled in fixture" });
  }

  async function handleOpenApi(route: Route) {
    await record(route);
    return json(route, 200, {
      paths: {
        "/v1/models": {
          get: {
            operationId: "listModels",
            summary: "列出模型",
            tags: ["模型"],
          },
        },
        "/v1/chat/completions": {
          post: {
            operationId: "createChatCompletion",
            summary: "建立對話回應",
            tags: ["推理"],
          },
        },
        "/api/pull": {
          post: {
            operationId: "pullModel",
            summary: "匯入 MLX 模型",
            tags: ["模型管理"],
          },
        },
      },
    });
  }

  return {
    models,
    requests,
    unexpected,
    unexpectedMutations,
    detailCalls: () => detailCalls,
    install: async () => {
      await page.route("**/v1/**", handleV1);
      await page.route("**/api/**", handleApi);
      await page.route("**/debug/**", handleDebug);
      await page.route("**/openapi.json", handleOpenApi);
    },
  };
}

async function open(page: Page, hash: string) {
  const fixture = createApiFixture(page);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await fixture.install();
  await page.goto(`/console/#/${hash}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("main")).toBeVisible();
  await expect(page.getByTestId(hash === "api" ? "api" : hash)).toBeVisible();
  await expect
    .poll(
      () =>
        fixture.requests.filter(
          (request) => request.path === "/v1/yunshu/status",
        ).length,
    )
    .toBeGreaterThan(0);
  return { fixture, pageErrors };
}

test.describe("fixture-only API controls", () => {
  test("imports without streaming, copies an alias, and deletes only after exact ID confirmation", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await open(page, "models");
    const models = page.getByTestId("models");

    await models.getByRole("button", { name: "匯入模型", exact: true }).click();
    await page
      .getByRole("textbox", { name: "儲存庫 ID" })
      .fill("fixture-owner/tiny-mlx");
    await page.getByRole("button", { name: "開始下載", exact: true }).click();
    await expect(
      models.getByRole("row").filter({ hasText: "tiny-mlx" }),
    ).toBeVisible();
    const pull = fixture.requests.find(
      (request) => request.path === "/api/pull",
    );
    expect(pull).toMatchObject({
      method: "POST",
      body: { model: "fixture-owner/tiny-mlx", stream: false },
    });

    const sourceRow = models.getByRole("row").filter({ hasText: "qwen-mlx" });
    await sourceRow
      .getByRole("button", { name: "org/qwen-mlx 的更多操作", exact: true })
      .click();
    await page.getByRole("menuitem", { name: "建立別名", exact: true }).click();
    await page
      .getByRole("textbox", { name: "新模型 ID" })
      .fill("team/qwen-alias");
    await page
      .getByRole("dialog", { name: "建立模型別名" })
      .getByRole("button", { name: "建立別名", exact: true })
      .click();
    await expect(
      models.getByRole("row").filter({ hasText: "qwen-alias" }),
    ).toBeVisible();
    expect(
      fixture.requests.find((request) => request.path === "/api/copy"),
    ).toMatchObject({
      method: "POST",
      body: { source: "org/qwen-mlx", destination: "team/qwen-alias" },
    });

    const aliasRow = models.getByRole("row").filter({ hasText: "qwen-alias" });
    await aliasRow
      .getByRole("button", { name: "team/qwen-alias 的更多操作", exact: true })
      .click();
    await page.getByRole("menuitem", { name: "刪除模型", exact: true }).click();
    const deleteDialog = page.getByRole("dialog", { name: "刪除模型與權重？" });
    const deleteButton = deleteDialog.getByRole("button", {
      name: "刪除模型",
      exact: true,
    });
    await expect(deleteButton).toBeDisabled();
    await deleteDialog
      .getByRole("textbox", { name: "確認模型 ID" })
      .fill("team/qwen-alias");
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();
    await expect(
      models.getByRole("row").filter({ hasText: "qwen-alias" }),
    ).toHaveCount(0);
    await expect(
      models.getByRole("row").filter({ hasText: "qwen-mlx" }),
    ).toHaveCount(1);
    expect(
      fixture.requests.find((request) => request.path === "/api/delete"),
    ).toMatchObject({
      method: "DELETE",
      body: { model: "team/qwen-alias" },
    });
    expect(fixture.unexpectedMutations).toEqual([]);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("sends the selected keep-alive duration through the fixture warmup endpoint", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await open(page, "settings");
    const keepAlive = page.getByRole("combobox", { name: "閒置保留時間" });
    await keepAlive.click();
    await page.getByRole("option", { name: "15 分鐘", exact: true }).click();
    await page.getByRole("button", { name: "套用並預熱", exact: true }).click();
    await expect(
      page
        .getByRole("status")
        .filter({ hasText: "模型已載入；此類型未執行文字預熱。" }),
    ).toBeVisible();
    expect(
      fixture.requests.find((request) => request.path === "/v1/yunshu/warmup"),
    ).toMatchObject({
      method: "POST",
      body: { model: "org/qwen-mlx", keep_alive: "15m", max_tokens: 1 },
    });
    expect(fixture.unexpectedMutations).toEqual([]);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("shows unavailable diagnostics and filters the fetched OpenAPI catalog", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await open(page, "diagnostics");
    const diagnostics = page.getByTestId("diagnostics");
    await expect(diagnostics.getByTestId("debug-disabled")).toContainText(
      "YUNSHU_DEBUG_ROUTES",
    );
    // One probe only: no per-group 404 cards and no placeholder tabs.
    await expect(
      diagnostics.getByRole("button", { name: "快取", exact: true }),
    ).toHaveCount(0);
    const debugPaths = fixture.requests
      .filter((request) => request.path.startsWith("/debug/"))
      .map((request) => request.path);
    expect(new Set(debugPaths)).toEqual(new Set(["/debug/system"]));

    await page.goto("/console/#/api", { waitUntil: "domcontentloaded" });
    const catalog = page.getByTestId("api-catalog");
    await expect(
      catalog.getByText("此服務的完整 API", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("textbox", { name: "搜尋 API" })
      .fill("/v1/chat/completions");
    await expect(
      catalog.getByRole("row").filter({ hasText: "/v1/chat/completions" }),
    ).toBeVisible();
    await expect(
      catalog.getByRole("row").filter({ hasText: "/api/pull" }),
    ).toHaveCount(0);
    expect(
      fixture.requests.some((request) => request.path === "/openapi.json"),
    ).toBe(true);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("updates active request details, preserves the last sample on 404, and stops polling when closed", async ({
    page,
  }) => {
    const fixture = createApiFixture(page, true);
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await fixture.install();
    await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
    const requests = page.getByTestId("requests");
    const row = requests.getByRole("row").filter({ hasText: "qa-active-01" });
    await expect(row).toBeVisible();

    const detailsButton = row.getByRole("button", {
      name: "詳情",
      exact: true,
    });
    await detailsButton.click();
    // Below xl the detail is a Sheet (dialog); from xl it is the inspector column (complementary).
    const dialog = page
      .getByRole("dialog", { name: "請求詳情" })
      .or(page.getByRole("complementary", { name: "請求詳情" }));
    await expect(dialog.getByText("777", { exact: true })).toBeVisible();
    await expect(dialog.getByText("333", { exact: true })).toBeVisible();
    await expect(dialog.getByText("55", { exact: true })).toBeVisible();
    expect(fixture.detailCalls()).toBe(1);

    await page.keyboard.press("Escape");
    await expect(dialog.getByText("777", { exact: true })).toHaveCount(0);
    await expect(detailsButton).toBeFocused();
    await page.waitForTimeout(1_700);
    expect(fixture.detailCalls()).toBe(1); // Closing the Sheet cleared its poll timer.

    await detailsButton.click();
    await expect(dialog.getByText("777", { exact: true })).toBeVisible();
    await expect(
      dialog.getByRole("status").filter({
        hasText: "此請求已結束或已不在活動清單；下方保留最近採樣。",
      }),
    ).toBeVisible();
    await expect(dialog.getByText("777", { exact: true })).toBeVisible();
    expect(fixture.detailCalls()).toBe(3);
    await page.waitForTimeout(1_700);
    expect(fixture.detailCalls()).toBe(3); // A terminal 404 must not start another poll.
    expect(
      fixture.requests
        .filter((request) => request.path.startsWith("/v1/requests/"))
        .map((request) => `${request.method} ${request.path}`),
    ).toEqual([
      "GET /v1/requests/qa-active-01",
      "GET /v1/requests/qa-active-01",
      "GET /v1/requests/qa-active-01",
    ]);

    await page.keyboard.press("Escape");
    await expect(dialog.getByText("777", { exact: true })).toHaveCount(0);
    expect(fixture.unexpectedMutations).toEqual([]);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("sends VLM image parts, thinking and JSON options only to the intercepted streaming route", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await open(page, "models");
    const models = page.getByTestId("models");
    await models
      .getByRole("row")
      .filter({ hasText: "vision-vlm" })
      .getByRole("button", { name: "測試", exact: true })
      .click();
    const playground = page.getByTestId("playground");
    await expect(playground).toBeVisible();

    await playground
      .getByRole("button", { name: "加入圖片", exact: true })
      .click();
    await expect(page.getByRole("dialog", { name: "圖片輸入" })).toBeVisible();
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jW1sAAAAASUVORK5CYII=",
      "base64",
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: "fixture.png",
      mimeType: "image/png",
      buffer: png,
    });
    await expect(page.getByRole("dialog", { name: "圖片輸入" })).toHaveCount(0);
    const prompt = playground.getByPlaceholder("輸入測試提示詞…");
    await prompt.fill("Describe this fixture image");

    await playground
      .getByRole("button", { name: "生成參數", exact: true })
      .click();
    const settings = page.getByRole("dialog", { name: "生成參數" });
    await settings.getByRole("button", { name: "關閉", exact: true }).click();
    await settings.getByRole("button", { name: "JSON", exact: true }).click();
    await settings
      .getByRole("button", { name: "關閉生成參數", exact: true })
      .click();

    await playground
      .getByRole("button", { name: "傳送測試", exact: true })
      .click();
    await expect(playground).toContainText('{"ok":true}');
    await expect(playground).toContainText("已達輸出上限");
    const completion = fixture.requests.find(
      (request) => request.path === "/v1/chat/completions",
    );
    expect(completion?.method).toBe("POST");
    const body = completion?.body as Record<string, unknown>;
    expect(body).toMatchObject({
      model: "org/vision-vlm",
      stream: true,
      enable_thinking: false,
      response_format: { type: "json_object" },
    });
    const messages = body.messages as Array<{ role: string; content: unknown }>;
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: [
        { type: "text", text: "Describe this fixture image" },
        { type: "image_url", image_url: { url: /^data:image\/png;base64,/ } },
      ],
    });
    expect(fixture.unexpectedMutations).toEqual([]);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
