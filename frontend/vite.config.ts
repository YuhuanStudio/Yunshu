import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
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
      "/v1": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
      "/debug": "http://127.0.0.1:8000",
      "/openapi.json": "http://127.0.0.1:8000",
      "/docs": "http://127.0.0.1:8000",
    },
  },
});
