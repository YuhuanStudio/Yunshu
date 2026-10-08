/** Real upstream tavily-mcp (stdio) driven by the official MCP client against the offline fixture.
 * usage: TAVILY_NODE_MODULES=... TAVILY_PYTHON=... node tavily_mcp_upstream.mjs out.jsonl
 */
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = process.env.TAVILY_NODE_MODULES;
const require = createRequire(join(root, '../package.json'));
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const dir = dirname(fileURLToPath(import.meta.url));
const rows = [];
const check = (name, pass, extra = {}) => { if (!pass) throw new Error(name); rows.push({ check: name, pass: true, ...extra }); };
const client = new Client({ name: 'websearch-upstream-mcp', version: '1' });
const transport = new StdioClientTransport({
  command: process.execPath,
  args: ['--import', join(dir, 'tavily_mcp_preload.mjs'), join(root, 'tavily-mcp/build/index.js')],
  env: { ...process.env, TAVILY_API_KEY: 'tvly-fixture' },
});
try {
  await client.connect(transport);
  const tools = (await client.listTools()).tools.map(t => t.name);
  check('tavily-mcp.tools', ['tavily_search', 'tavily_extract', 'tavily_crawl', 'tavily_map', 'tavily_research'].every(n => tools.includes(n)), { tools });
  const call = async (name, args) => { const r = await client.callTool({ name, arguments: args }); return r; };
  let r = await call('tavily_search', { query: 'Paris weather', max_results: 3, include_favicon: true });
  check('tavily-mcp.search', !r.isError && /fixture\.example/.test(r.content[0].text), { sample: r.content[0].text.slice(0, 120) });
  r = await call('tavily_extract', { urls: ['https://fixture.example/'] });
  check('tavily-mcp.extract', !r.isError && /Paris/.test(r.content[0].text));
  r = await call('tavily_map', { url: 'https://fixture.example/' });
  check('tavily-mcp.map', !r.isError);
  r = await call('tavily_crawl', { url: 'https://fixture.example/', limit: 2 });
  check('tavily-mcp.crawl', !r.isError);
  r = await call('tavily_research', { input: 'Paris weather', model: 'mini' });
  check('tavily-mcp.research', !r.isError, { sample: r.content[0].text.slice(0, 120) });
  rows.push({ complete: true, pass: true, fixture: true, upstream_unpatched: true });
} finally {
  await client.close().catch(() => {});
  writeFileSync(process.argv[2], rows.map(x => JSON.stringify(x)).join('\n') + '\n');
}
