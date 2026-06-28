/** @type {import('next').NextConfig} */

const backendUrl = process.env.YUNSHU_BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // Pin the workspace root to this dir so Next/Turbopack doesn't warn about a
  // stray parent pnpm-lock.yaml and pick the wrong root.
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: "/v1/:path*",
        destination: `${backendUrl}/v1/:path*`,
      },
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/health/:path*",
        destination: `${backendUrl}/health/:path*`,
      },
      {
        source: "/health",
        destination: `${backendUrl}/health`,
      },
    ];
  },
};

module.exports = nextConfig;
