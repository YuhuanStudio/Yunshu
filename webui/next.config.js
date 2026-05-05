/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // Proxy API calls to Yunshu backend
      {
        source: "/v1/:path*",
        destination: "http://localhost:8000/v1/:path*",
      },
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/health",
        destination: "http://localhost:8000/health",
      },
    ];
  },
};

module.exports = nextConfig;
