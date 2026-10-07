import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./browser",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:3971", trace: "retain-on-failure" },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://127.0.0.1:3971/console/",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
