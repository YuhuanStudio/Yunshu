// Runs the Playwright suite as N parallel shards (`playwright test --shard=k/N`), each with its own
// vite port and output directory, so the 24-minute serial run finishes in about 1/N of the time.
// Every spec still runs, in both projects, with the same assertions; only the processes change.
//   node scripts/browser-shards.mjs [SHARDS] [-- playwright args]
// SHARDS defaults to PLAYWRIGHT_SHARDS or 4. Ports come from PLAYWRIGHT_PORT_BASE (default: free ports
// found by binding; the engine servers of the specs use 18990-18999, so never those). Exit code is the
// worst shard's. Never start more than the machine can take: timing specs (layout stability, motion)
// are CPU-sensitive, so keep SHARDS at or below the performance-core count.
import { spawn } from "node:child_process";
import { createServer } from "node:net";

export function shardArgs(index, total, extra = []) {
  if (!Number.isInteger(total) || total < 1)
    throw new Error(`bad shard count ${total}`);
  if (!Number.isInteger(index) || index < 1 || index > total)
    throw new Error(`bad shard ${index}/${total}`);
  return ["exec", "playwright", "test", `--shard=${index}/${total}`, ...extra];
}

export function shardEnv(index, port, env = process.env) {
  return {
    ...env,
    PLAYWRIGHT_PORT: String(port),
    PLAYWRIGHT_OUTPUT_DIR: `test-results/shard-${index}`,
  };
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.once("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function main() {
  const argv = process.argv.slice(2);
  const dash = argv.indexOf("--");
  const head = dash < 0 ? argv : argv.slice(0, dash);
  const extra = dash < 0 ? [] : argv.slice(dash + 1);
  const total = Number(head[0] ?? process.env.PLAYWRIGHT_SHARDS ?? 4);
  const ports = [];
  const base = process.env.PLAYWRIGHT_PORT_BASE
    ? Number(process.env.PLAYWRIGHT_PORT_BASE)
    : 0;
  for (let i = 0; i < total; i++)
    ports.push(base ? base + i : await freePort());
  const t0 = Date.now();
  const codes = await Promise.all(
    ports.map(
      (port, i) =>
        new Promise((resolve) => {
          const p = spawn("pnpm", shardArgs(i + 1, total, extra), {
            stdio: "inherit",
            env: shardEnv(i + 1, port),
          });
          p.on("exit", (code) => resolve(code ?? 1));
        }),
    ),
  );
  console.log(
    `browser-shards: ${total} shards, exit codes ${codes.join(",")}, ${Math.round((Date.now() - t0) / 1000)} s`,
  );
  process.exit(Math.max(...codes));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
