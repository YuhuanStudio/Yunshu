import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "/console/",
  resolve: { dedupe: ["react", "react-dom"] },
  build: {
    outDir: "../python/yunshu_gateway/console_static",
    emptyOutDir: true,
    license: { fileName: "licenses/bundled-dependencies.md" },
  },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
