import { defineConfig, devices } from "@playwright/test";
const port = process.env.PLAYWRIGHT_PORT ?? "18999";
const origin = `http://127.0.0.1:${port}`;
export default defineConfig({
  testDir: "./browser",
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR ?? "test-results",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
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
        reuseExistingServer: !process.env.CI,
        timeout: 30_000,
      },
});
