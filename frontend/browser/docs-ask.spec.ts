import { test, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";

for (const locale of ["en", "zh-TW", "zh-CN"]) {
  test(`local Ask AI sources, follow-up, feedback and copy ${locale}`, async ({
    page,
  }) => {
    await page.addInitScript(
      (locale) => localStorage.setItem("yunshu.console.locale", locale),
      locale,
    );
    await page.route("**/console/api/docs/state", (route) =>
      route.fulfill({
        json: { ready: true, model: "local", question: 2000, history: 8 },
      }),
    );
    const bodies: { question: string; history: unknown[] }[] = [];
    await page.route("**/console/api/docs/ask", (route) => {
      bodies.push(route.request().postDataJSON());
      return route.fulfill({
        contentType: "text/event-stream",
        body: 'event: sources\ndata: {"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","model":"local","sources":[{"id":"/docs/guides/prompt-caching#apc","slug":"guides/prompt-caching","anchor":"apc","title":"Prompt caching","heading":"APC"}]}\n\nevent: token\ndata: {"text":"Use **APC**. [1]\\n\\n```sh\\nyunshu serve\\n```"}\n\nevent: done\ndata: {"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n\n',
      });
    });
    let feedback: unknown;
    await page.route("**/console/api/docs/feedback", (route) => {
      feedback = route.request().postDataJSON();
      return route.fulfill({ json: { saved: true } });
    });
    await page.goto("/console/#/docs");
    const title =
      locale === "en" ? "Ask AI" : locale === "zh-TW" ? "AI 問答" : "AI 问答";
    await page
      .getByTestId("docs")
      .getByRole("button", { name: title, exact: true })
      .click();
    const dialog = page.getByTestId("docs-ask");
    await expect(dialog).toBeVisible();
    const input = dialog.locator("textarea");
    await input.fill("How do I enable APC?");
    await input.press("Enter");
    await expect(
      dialog.getByTestId("docs-ask-answer").locator("strong"),
    ).toHaveText("APC");
    await expect(
      dialog.getByRole("link", { name: /Prompt caching/ }),
    ).toHaveAttribute("href", "#/docs/guides/prompt-caching?h=apc");
    await expect(dialog.locator("pre")).toContainText("yunshu serve");
    if (locale === "en") {
      await page
        .context()
        .grantPermissions(["clipboard-read", "clipboard-write"]);
      await dialog
        .getByRole("button", { name: "Copy answer", exact: true })
        .click();
      await expect
        .poll(() => page.evaluate(() => navigator.clipboard.readText()))
        .toContain("APC");
    }
    await dialog
      .getByRole("button", {
        name:
          locale === "en"
            ? "Helpful answer"
            : "回答有幫助".replace(
                "幫助",
                locale === "zh-CN" ? "帮助" : "幫助",
              ),
      })
      .click();
    expect(feedback).toEqual({ ask_id: "a".repeat(32), opinion: "good" });
    await input.fill("What about SSD?");
    await input.press("Enter");
    await expect(dialog.getByTestId("docs-ask-answer")).toHaveCount(2);
    expect(bodies[1].history).toHaveLength(2);
    mkdirSync("evidence/docsask", { recursive: true });
    await page.screenshot({ path: `evidence/docsask/ask-${locale}.png` });
  });
}

test("no engine and paused states show clear guidance; palette opens Ask AI", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.locale", "en"),
  );
  let code = "no_engine";
  await page.route("**/console/api/docs/state", (route) =>
    route.fulfill({
      json: {
        ready: false,
        code,
        command: "yunshu serve -m /path/to/mlx-model",
        question: 2000,
        history: 8,
      },
    }),
  );
  await page.goto("/console/#/docs");
  await page.keyboard.press("Meta+k");
  await page.getByRole("dialog").locator("input").first().fill("Ask AI");
  await page.getByRole("option").filter({ hasText: "Ask AI" }).first().click();
  await expect(page.getByTestId("docs-ask")).toContainText(
    "Start an engine to ask the docs.",
  );
  await expect(page.getByTestId("docs-ask").locator("code")).toContainText(
    "yunshu serve",
  );
  await expect(page.getByTestId("docs-ask").locator("textarea")).toBeDisabled();
  await page.keyboard.press("Escape");
  code = "measurement_paused";
  await page
    .getByTestId("docs")
    .getByRole("button", { name: "Ask AI", exact: true })
    .click();
  await expect(page.getByTestId("docs-ask")).toContainText(
    "Ask AI is paused during measurement.",
  );
});

test("stop cancels a pending answer and prevents a second request in the tab", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.locale", "en"),
  );
  await page.route("**/console/api/docs/state", (route) =>
    route.fulfill({ json: { ready: true, question: 2000, history: 8 } }),
  );
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requests = 0;
  await page.route("**/console/api/docs/ask", async (route) => {
    requests++;
    await held;
    await route.abort().catch(() => {});
  });
  await page.goto("/console/#/docs");
  await page
    .getByTestId("docs")
    .getByRole("button", { name: "Ask AI", exact: true })
    .click();
  const dialog = page.getByTestId("docs-ask");
  const input = dialog.locator("textarea");
  await input.fill("How do I start?");
  await input.press("Enter");
  await expect(
    dialog.getByRole("button", { name: "Stop answer", exact: true }),
  ).toBeVisible();
  await input.fill("Second question");
  await input.press("Enter");
  expect(requests).toBe(1);
  await dialog
    .getByRole("button", { name: "Stop answer", exact: true })
    .click();
  await expect(
    dialog.getByRole("button", { name: "Send question", exact: true }),
  ).toBeVisible();
  await expect(dialog.getByRole("alert")).toHaveCount(0);
  release();
});

for (const theme of ["light", "dark"]) {
  test(`mobile Ask AI ${theme} reduced motion and availability recovery`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 402, height: 874 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.addInitScript((theme) => {
      localStorage.setItem("yunshu.console.locale", "en");
      localStorage.setItem("yunshu.console.theme", theme);
    }, theme);
    let ready = false;
    await page.route("**/console/api/docs/state", (route) =>
      route.fulfill({
        json: { ready, code: "no_engine", question: 2000, history: 8 },
      }),
    );
    await page.goto("/console/#/docs");
    await page
      .getByTestId("docs")
      .getByRole("button", { name: "Ask AI", exact: true })
      .click();
    const dialog = page.getByTestId("docs-ask");
    await expect(dialog.locator("textarea")).toBeDisabled();
    ready = true;
    await dialog
      .getByRole("button", { name: "Check again", exact: true })
      .click();
    await expect(dialog.locator("textarea")).toBeEnabled();
    const box = await dialog.boundingBox();
    expect(box?.x).toBeGreaterThanOrEqual(0);
    expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(402);
    mkdirSync("evidence/docsask", { recursive: true });
    await page.screenshot({ path: `evidence/docsask/ask-mobile-${theme}.png` });
  });
}
