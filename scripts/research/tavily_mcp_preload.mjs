/** Test-only preload for the unmodified upstream tavily-mcp stdio server.
 * Upstream hard-codes https://api.tavily.com/<endpoint>; this replaces axios' network
 * adapter with the socket-free ASGI fixture bridge (path kept, host dropped, /tavily prefix).
 * The package source is not patched. Needs TAVILY_NODE_MODULES and TAVILY_PYTHON.
 */
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const dir = dirname(fileURLToPath(import.meta.url));
const axios = (await import(pathToFileURL(join(process.env.TAVILY_NODE_MODULES, 'axios/index.js')))).default;
const child = spawn(process.env.TAVILY_PYTHON, [join(dir, 'tavily_fixture_bridge.py')], { env: process.env, stdio: ['pipe', 'pipe', 'inherit'] });
const waiting = [];
createInterface({ input: child.stdout }).on('line', line => waiting.shift()?.resolve(JSON.parse(line)));
process.on('exit', () => child.kill());
axios.defaults.adapter = async config => {
  const url = new URL(config.url, config.baseURL);
  const path = '/tavily' + url.pathname + url.search;
  const reply = await new Promise(resolve => {
    waiting.push({ resolve });
    child.stdin.write(JSON.stringify({ path, body: typeof config.data === 'string' ? JSON.parse(config.data) : config.data, method: config.method.toUpperCase(), headers: config.headers.toJSON?.() ?? config.headers }) + '\n');
  });
  const data = JSON.parse(reply.body);
  if (reply.status >= 400) {
    const error = new Error(`Request failed with status code ${reply.status}`);
    error.isAxiosError = true; error.response = { status: reply.status, data, headers: reply.headers }; error.config = config;
    throw error;
  }
  return { data, status: reply.status, statusText: '', headers: reply.headers, config };
};
