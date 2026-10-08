/** Official @tavily/core and MCP client through a socket-free offline ASGI transport.
 * Set TAVILY_NODE_MODULES and TAVILY_PYTHON to the isolated environments.
 * No client package source is patched. Upstream tavily-mcp's base-URL blocker is recorded.
 */
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
const root = process.env.TAVILY_NODE_MODULES;
if (!root) throw new Error('TAVILY_NODE_MODULES must point at the isolated node_modules');
const require = createRequire(join(root, '../package.json'));
const axios = require('axios');
const dir = dirname(fileURLToPath(import.meta.url));
const child = spawn(process.env.TAVILY_PYTHON, [join(dir, 'tavily_fixture_bridge.py')], { env: process.env, stdio: ['pipe', 'pipe', 'inherit'] });
const waiting = [];
createInterface({ input: child.stdout }).on('line', line => waiting.shift()?.resolve(JSON.parse(line)));
child.on('exit', code => { while (waiting.length) waiting.shift().reject(new Error(`bridge exited ${code}`)); });
const bridge = (path, body, method = 'POST', headers = {}) => new Promise((resolve, reject) => {
  waiting.push({resolve, reject}); child.stdin.write(JSON.stringify({path, body, method, headers}) + '\n');
});
axios.defaults.adapter = async config => {
  const url = new URL(config.url, config.baseURL);
  // The upstream JS core URL builder preserves a path prefix in apiBaseURL.
  if (!url.pathname.startsWith('/tavily/')) throw new Error(`base URL prefix lost: ${url}`);
  const reply = await bridge(url.pathname + url.search, typeof config.data === 'string' ? JSON.parse(config.data) : config.data, config.method.toUpperCase(), config.headers.toJSON?.() ?? config.headers);
  const data = config.responseType === 'stream' ? Readable.from([Buffer.from(reply.body)]) : JSON.parse(reply.body);
  if (reply.status >= 400) throw new Error(JSON.stringify(data));
  return {data, status: reply.status, statusText: '', headers: reply.headers, config};
};
const { tavily } = require('@tavily/core');
const rows = [];
const check = (name, pass) => { if (!pass) throw new Error(name); rows.push({check:name, pass:true}); };
try {
  const client = tavily({apiKey:'tvly-fixture', apiBaseURL:'http://fixture/tavily'});
  const search = await client.search('Paris weather', {includeUsage:true, includeImages:true, chunksPerSource:2});
  check('js.search', Array.isArray(search.images) && Array.isArray(search.results) && search.requestId && typeof search.results[0].score === 'number');
  const extract = await client.extract(['https://fixture.example/', 'https://fixture.example/fail'], {includeUsage:true});
  check('js.extract', extract.failedResults.length === 1 && extract.results.length === 1);
  const crawl = await client.crawl('https://fixture.example/', {limit:2});
  check('js.crawl', Array.isArray(crawl.results[0].images));
  const map = await client.map('https://fixture.example/', {limit:2});
  check('js.map', typeof map.results[0] === 'string');
  const task = await client.research('Paris weather', {model:'mini'});
  let research;
  for (let i=0; i<100; i++) {
    research = await client.getResearch(task.requestId, {includeUsage:true});
    if (['completed','failed'].includes(research.status)) break;
    await new Promise(r => setTimeout(r, 5));
  }
  check('js.research.poll', research.status === 'completed' && research.sources.length > 0);
  let stream = '';
  for await (const chunk of await client.research('Paris weather', {stream:true})) stream += chunk.toString();
  check('js.research.sse', stream.includes('event: done') && stream.includes('chat.completion.chunk'));
  const feedback = await client.feedback({requestId:search.requestId, humanScore:1});
  check('js.feedback', feedback.success && feedback.feedbackId);
  // Official MCP client, with transport-level fixture injection only.
  const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
  const { StreamableHTTPClientTransport } = require('@modelcontextprotocol/sdk/client/streamableHttp.js');
  const mcp = new Client({name:'websearch-parity', version:'1'});
  const transport = new StreamableHTTPClientTransport(new URL('http://fixture/tavily/mcp'), {
    fetch: async (url, init = {}) => {
      const reply = await bridge(new URL(url).pathname, init.body ? JSON.parse(init.body) : undefined, init.method || 'GET', Object.fromEntries(new Headers(init.headers)));
      return new Response(reply.status === 202 ? null : reply.body, {status:reply.status, headers:reply.headers});
    }
  });
  await mcp.connect(transport);
  const tools = await mcp.listTools();
  check('mcp.native.tools', tools.tools.some(t => t.name === 'tavily_search') && tools.tools.some(t => t.name === 'tavily-search'));
  const answer = await mcp.callTool({name:'tavily_search', arguments:{query:'Paris weather'}});
  check('mcp.native.call', !answer.isError && JSON.parse(answer.content[0].text).results.length > 0);
  await mcp.close();
  const upstream = readFileSync(join(root, 'tavily-mcp/build/index.js'), 'utf8');
  rows.push({check:'tavily-mcp.base_url', status:'BLOCKED_UPSTREAM', reason:'Official stdio client hard-codes api.tavily.com and exposes no base URL setting', hardcoded:upstream.includes('https://api.tavily.com/search')});
  rows.push({complete:true, pass:true, fixture:true, versions:{core:JSON.parse(readFileSync(join(root,'@tavily/core/package.json'))).version, mcp:JSON.parse(readFileSync(join(root,'tavily-mcp/package.json'))).version}});
} finally {
  child.stdin.end();
  writeFileSync(process.argv[2], rows.map(row => JSON.stringify(row)).join('\n') + '\n');
}
