import type { NextConfig } from "next";
import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

// GitHub Pages serves a project site under /<repo>; the workflow sets this.
const basePath = process.env.DOCS_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  // Let phones on the LAN use the dev server.
  allowedDevOrigins: ["192.168.*.*", "10.*.*.*", "172.*.*.*", "*.local"],
  // YunUI ships ESM with "use client" chunks; let Next transpile it.
  transpilePackages: ["@yuhuanowo/yunui"],
  turbopack: { root: import.meta.dirname },
};

export default withMDX(nextConfig);
