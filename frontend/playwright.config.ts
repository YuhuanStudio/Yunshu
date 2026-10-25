import { defineConfig, devices } from "@playwright/test";
const port = process.env.PLAYWRIGHT_PORT ?? "18999";
const origin = `http://127.0.0.1:${port}`;
export default defineConfig({
  testDir: "./browser",
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR ?? "test-results",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  // GitHub's runner occasionally loses a WebKit process ("internal error"): one retry there, none
  // locally. A test that passes only on retry is reported as flaky (scripts/report-flaky.mjs) and is
  // a defect to root-cause, not a pass to forget.
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI
    ? [["list"], ["json", { outputFile: "playwright-report.json" }]]
    : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? origin,
    // The default language is zh-TW; specs read its wording unless they set another locale.
    locale: "zh-TW",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: `pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
        url: `${origin}/console/`,
        // Never adopt whatever already answers on the port: the resident console listens on 18999 too, and reusing
        // it ran the suite against a live engine and another build (32 failures, 2026-10-25). Opt in explicitly.
        reuseExistingServer: process.env.PLAYWRIGHT_REUSE === "1",
        timeout: 30_000,
      },
});
