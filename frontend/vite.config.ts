import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Dev proxy target: the engine to develop against (YUNSHU_CONSOLE_ENGINE), else the CLI default.
const engine = process.env.YUNSHU_CONSOLE_ENGINE ?? "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  base: "/console/",
  resolve: { dedupe: ["react", "react-dom"] },
  build: {
    outDir: "../python/yunshu_gateway/console_static",
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          // Vendor code is grouped by which routes use it, so a library that
          // only some lazy pages need stays out of the entry chunk.
          includeDependenciesRecursively: false,
          groups: [
            { name: "vendor", test: /node_modules/, entriesAware: true },
          ],
        },
      },
    },
    license: { fileName: "licenses/bundled-dependencies.md" },
  },
  server: {
    proxy: {
      "/v1": engine,
      "/health": engine,
      "/api": engine,
      "/debug": engine,
      "/openapi.json": engine,
      "/docs": engine,
    },
  },
});
