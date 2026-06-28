/**
 * Wave 458: Vitest config for WebUI.
 *
 * Adds jsdom environment + React plugin so component-rendering tests
 * (admin TenantManager, AuditLogViewer) can mount against a DOM and
 * exercise event handlers. Pure-logic tests (smoke/utils) still pass
 * unchanged because jsdom is a strict superset of the node env they
 * relied on previously.
 */
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/__tests__/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
