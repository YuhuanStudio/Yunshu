import { expect, test, type Page } from "@playwright/test";

const model = (id: string, type: string) => ({
  id,
  type,
  loaded: true,
  loading: false,
  pinned: false,
  size_gb: 5,
});
const status = (models: unknown[]) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models,
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
});

// A 1x1 transparent PNG.
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

async function install(
  page: Page,
  models: unknown[],
  ocr: { code: number; body: unknown },
) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: status(models) });
    if (path === "/v1/models")
      return route.fulfill({ json: { object: "list", data: [] } });
    if (path === "/v1/ocr")
      return route.fulfill({ status: ocr.code, json: ocr.body });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

test("image to text: preview beside the text, no boxes and no confidence claimed", async ({
  page,
}) => {
  await install(page, [model("glm-ocr", "OCREngine")], {
    code: 200,
    body: {
      text: "INVOICE 0042\nTotal 128.00",
      model: "glm-ocr",
      confidence: 0,
      usage: {
        prompt_tokens: 120,
        completion_tokens: 8,
        total_tokens: 128,
        image_tokens: 100,
      },
    },
  });
  await page.goto("/console/#/playground", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "圖片辨識" }).click();
  const dialog = page.getByTestId("ocr-dialog");
  await dialog
    .locator('input[type="file"]')
    .setInputFiles({ name: "scan.png", mimeType: "image/png", buffer: PNG });
  await expect(dialog.getByAltText("scan.png")).toBeVisible();
  await dialog.getByRole("button", { name: "辨識", exact: true }).click();
  await expect(page.getByTestId("ocr-text")).toContainText("Total 128.00");
  await expect(dialog).toContainText("沒有每段文字的位置");
  await expect(dialog).toContainText("輸入 120 / 輸出 8 token");
  await expect(dialog).not.toContainText("信心 0");
});

test("image to text: an engine error is shown, not swallowed", async ({
  page,
}) => {
  await install(page, [model("Qwen-VL", "VLMEngine")], {
    code: 400,
    body: { detail: "not an image" },
  });
  await page.goto("/console/#/playground", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "圖片辨識" }).click();
  const dialog = page.getByTestId("ocr-dialog");
  await dialog
    .locator('input[type="file"]')
    .setInputFiles({ name: "x.png", mimeType: "image/png", buffer: PNG });
  await dialog.getByRole("button", { name: "辨識", exact: true }).click();
  await expect(page.getByTestId("ocr-error")).toBeVisible();
});

test("image to text: disabled with the reason when no OCR or vision model is loaded", async ({
  page,
}) => {
  await install(page, [model("Qwen3.8-27B", "LLM")], { code: 404, body: {} });
  await page.goto("/console/#/playground", { waitUntil: "domcontentloaded" });
  const button = page.getByRole("button", { name: "圖片辨識" });
  await expect(button).toBeDisabled();
  await expect(button).toHaveAttribute("title", /沒有載入 OCR 或視覺模型/);
});
