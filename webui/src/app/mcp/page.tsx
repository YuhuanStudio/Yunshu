"use client";

import { useMemo, useRef, useState } from "react";
import {
  Button,
  Badge,
  Alert,
  Card,
  Spinner,
  EmptyState,
  StatusIndicator,
  SearchInput,
  Textarea,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  cn,
  toast,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Server, Wrench, Play, Copy, ChevronRight, Boxes, Plug } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, usePolling, ApiError } from "@/lib/api";

/* ---- page-local types (see the MCP surface of the Yunshu backend) ---------- */

interface McpTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

/** Outbound MCP client status (Yunshu connecting *out* to external servers). */
interface McpClientStatus {
  enabled: boolean;
  connected_servers?: number;
  total_tools?: number;
  servers?: Array<{ name: string; connected?: boolean; tool_count?: number; url?: string }>;
  [k: string]: unknown;
}

interface McpClientTool {
  name: string;
  description?: string;
  server?: string;
  [k: string]: unknown;
}

interface McpContentBlock {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
}

interface McpCallResult {
  content?: McpContentBlock[];
  isError?: boolean;
}

interface JsonRpcResponse {
  jsonrpc: string;
  id: number;
  result?: McpCallResult;
  error?: { code: number; message: string; data?: unknown };
}

export default function MCPPage() {
  // Built-in server tools exposed by Yunshu itself.
  const toolsResp = usePolling<{ tools: McpTool[] }>((s) => api.get("/v1/mcp/tools", s), 15000);
  // Outbound client (Yunshu → external MCP servers).
  const clientStatus = usePolling<McpClientStatus>(
    (s) => api.get("/v1/mcp/client/status", s),
    8000,
  );
  const clientTools = usePolling<{ tools: McpClientTool[]; openai_format?: unknown }>(
    (s) => api.get("/v1/mcp/client/tools", s),
    15000,
  );

  const tools = useMemo(() => toolsResp.data?.tools ?? [], [toolsResp.data]);

  const [tab, setTab] = useState("tools");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const [argsText, setArgsText] = useState("{}");
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [result, setResult] = useState<McpCallResult | null>(null);
  const [rawResult, setRawResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const idRef = useRef(0);

  const filteredTools = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return tools;
    return tools.filter(
      (t) => t.name.toLowerCase().includes(q) || (t.description ?? "").toLowerCase().includes(q),
    );
  }, [tools, query]);

  const selectedTool = useMemo(
    () => tools.find((t) => t.name === selected) ?? null,
    [tools, selected],
  );

  const pickTool = (name: string) => {
    setSelected(name);
    setResult(null);
    setRawResult(null);
    setError(null);
    setTab("runner");
  };

  const execute = async () => {
    setError(null);
    setResult(null);
    setRawResult(null);
    if (!selected) {
      setError("Select a tool first.");
      return;
    }
    let parsedArgs: unknown;
    try {
      parsedArgs = argsText.trim() ? JSON.parse(argsText) : {};
    } catch (e) {
      setError(`Invalid JSON arguments: ${(e as Error).message}`);
      return;
    }
    setRunning(true);
    try {
      const res = await api.post<JsonRpcResponse>("/v1/mcp", {
        jsonrpc: "2.0",
        id: ++idRef.current,
        method: "tools/call",
        params: { name: selected, arguments: parsedArgs },
      });
      setRawResult(JSON.stringify(res, null, 2));
      if (res.error) {
        setError(`JSON-RPC error ${res.error.code}: ${res.error.message}`);
        return;
      }
      setResult(res.result ?? { content: [], isError: false });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const copyRaw = async () => {
    if (!rawResult) return;
    try {
      await navigator.clipboard.writeText(rawResult);
      toast.success("Copied raw JSON-RPC response");
    } catch {
      toast.error("Copy failed");
    }
  };

  const status = clientStatus.data;
  const outboundEnabled = status?.enabled ?? false;
  const outboundTools = clientTools.data?.tools ?? [];

  return (
    <PageShell
      title="MCP"
      description="Inspect Yunshu's built-in Model Context Protocol tools, run them over JSON-RPC, and see connected external MCP servers."
      width="wide"
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="tools">Server tools</TabsTrigger>
          <TabsTrigger value="runner">Tool runner</TabsTrigger>
          <TabsTrigger value="servers">Connected servers</TabsTrigger>
        </TabsList>

        {/* Server tools ------------------------------------------------------ */}
        <TabsContent value="tools" className="mt-4">
          <SearchInput
            value={query}
            onChange={setQuery}
            placeholder="Filter tools by name or description…"
            className="mb-4 max-w-md"
          />
          {toolsResp.error && tools.length === 0 ? (
            <Alert variant="error" title="Failed to load tools">
              {toolsResp.error.message}
            </Alert>
          ) : toolsResp.loading && tools.length === 0 ? (
            <div className="flex justify-center py-12">
              <Spinner />
            </div>
          ) : filteredTools.length === 0 ? (
            <EmptyState
              icon={<Wrench className="h-6 w-6" />}
              title="No tools"
              description={query ? "No tools match your search." : "No MCP tools are available."}
            />
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {filteredTools.map((t) => (
                <ServerToolCard key={t.name} tool={t} onRun={() => pickTool(t.name)} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Tool runner ------------------------------------------------------- */}
        <TabsContent value="runner" className="mt-4">
          <Card className="space-y-4 p-5">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Tool</label>
              <Select value={selected} onValueChange={setSelected}>
                <SelectTrigger aria-label="Tool" className="max-w-md">
                  <SelectValue placeholder="Select a tool…" />
                </SelectTrigger>
                <SelectContent>
                  {tools.map((t) => (
                    <SelectItem key={t.name} value={t.name}>
                      {t.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedTool?.description && (
              <p className="text-sm text-muted-foreground">{selectedTool.description}</p>
            )}

            {selectedTool?.inputSchema && (
              <Collapsible open={schemaOpen} onOpenChange={setSchemaOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <ChevronRight
                      className={cn("h-4 w-4 transition-transform", schemaOpen && "rotate-90")}
                    />
                    {schemaOpen ? "Hide input schema" : "Show input schema"}
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-muted p-4 font-mono text-xs">
                    {JSON.stringify(selectedTool.inputSchema, null, 2)}
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            )}

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Arguments (JSON)</label>
              <Textarea aria-label="Arguments (JSON)"
                value={argsText}
                onChange={(e) => setArgsText(e.target.value)}
                rows={6}
                spellCheck={false}
                className="font-mono text-sm"
                placeholder='{ "key": "value" }'
              />
              <p className="text-xs text-muted-foreground">
                Sent as a JSON-RPC 2.0{" "}
                <code className="font-mono">tools/call</code> request:{" "}
                <code className="font-mono">
                  {"{ method: \"tools/call\", params: { name, arguments } }"}
                </code>
                .
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button onClick={execute} disabled={running || !selected}>
                {running ? <Spinner size="sm" /> : <Play className="h-4 w-4" />}
                Call tool
              </Button>
            </div>

            {error && (
              <Alert variant="error" title="Tool call failed">
                {error}
              </Alert>
            )}

            {result && <ToolResult result={result} />}

            {rawResult !== null && (
              <Collapsible>
                <div className="flex items-center justify-between">
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" size="sm">
                      Raw JSON-RPC response
                    </Button>
                  </CollapsibleTrigger>
                  <Button variant="secondary" size="sm" onClick={copyRaw}>
                    <Copy className="h-4 w-4" /> Copy
                  </Button>
                </div>
                <CollapsibleContent>
                  <pre className="mt-2 max-h-96 overflow-auto rounded-lg bg-muted p-4 font-mono text-xs">
                    {rawResult}
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            )}
          </Card>
        </TabsContent>

        {/* Connected servers ------------------------------------------------- */}
        <TabsContent value="servers" className="mt-4 space-y-6">
          {clientStatus.loading && !status ? (
            <div className="flex justify-center py-12">
              <Spinner />
            </div>
          ) : !outboundEnabled ? (
            <Alert variant="info" title="Outbound MCP client disabled">
              Yunshu is not configured to connect out to external MCP servers. Enable the MCP
              client in the backend configuration to discover and use their tools here.
            </Alert>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <StatCard
                  icon={Plug}
                  label="Client"
                  value={outboundEnabled ? "Enabled" : "Disabled"}
                  tone="green"
                />
                <StatCard
                  icon={Server}
                  label="Connected servers"
                  value={String(status?.connected_servers ?? 0)}
                  tone="blue"
                />
                <StatCard
                  icon={Boxes}
                  label="Total tools"
                  value={String(status?.total_tools ?? outboundTools.length)}
                  tone="purple"
                />
              </div>

              {Array.isArray(status?.servers) && status.servers.length > 0 && (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {status.servers.map((s) => (
                    <Card key={s.name} className="p-5">
                      <div className="flex items-center justify-between gap-3">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium">
                          {s.name}
                        </span>
                        <StatusIndicator status={s.connected ? "online" : "offline"}>
                          {s.connected ? "connected" : "disconnected"}
                        </StatusIndicator>
                      </div>
                      {typeof s.tool_count === "number" && (
                        <div className="mt-3">
                          <Badge variant="info">{s.tool_count} tools</Badge>
                        </div>
                      )}
                      {s.url && (
                        <p className="mt-3 truncate font-mono text-xs text-muted-foreground">
                          {s.url}
                        </p>
                      )}
                    </Card>
                  ))}
                </div>
              )}

              <div>
                <h3 className="mb-3 text-sm font-medium">Tools from connected servers</h3>
                {outboundTools.length === 0 ? (
                  <EmptyState
                    icon={<Wrench className="h-6 w-6" />}
                    title="No external tools"
                    description="Connected MCP servers have not advertised any tools."
                  />
                ) : (
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    {outboundTools.map((t, i) => (
                      <Card key={`${t.server ?? ""}:${t.name}:${i}`} className="p-4">
                        <div className="flex items-center justify-between gap-2">
                          <span className="min-w-0 flex-1 truncate font-mono text-sm font-medium">
                            {t.name}
                          </span>
                          {t.server && <Badge variant="default">{String(t.server)}</Badge>}
                        </div>
                        {t.description && (
                          <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">
                            {t.description}
                          </p>
                        )}
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}

/* ---- pieces --------------------------------------------------------------- */

function ServerToolCard({ tool, onRun }: { tool: McpTool; onRun: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="min-w-0 flex-1 truncate font-mono text-sm font-medium">{tool.name}</span>
        <Button variant="secondary" size="sm" onClick={onRun}>
          <Play className="h-4 w-4" /> Run
        </Button>
      </div>
      {tool.description && (
        <p className="mt-2 text-sm text-muted-foreground">{tool.description}</p>
      )}
      {tool.inputSchema && (
        <Collapsible open={open} onOpenChange={setOpen} className="mt-2">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm">
              <ChevronRight className={cn("h-4 w-4 transition-transform", open && "rotate-90")} />
              {open ? "Hide schema" : "Show schema"}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-60 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs">
              {JSON.stringify(tool.inputSchema, null, 2)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      )}
    </Card>
  );
}

function ToolResult({ result }: { result: McpCallResult }) {
  const blocks = result.content ?? [];
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Result</span>
        {result.isError ? (
          <Badge variant="error">isError</Badge>
        ) : (
          <Badge variant="success">ok</Badge>
        )}
      </div>

      {result.isError && (
        <Alert variant="error" title="The tool reported an error">
          The MCP tool returned <code className="font-mono">isError: true</code>. See the content
          below for details.
        </Alert>
      )}

      {blocks.length === 0 ? (
        <p className="text-sm text-muted-foreground">The tool returned no content.</p>
      ) : (
        blocks.map((b, i) => <ContentBlock key={i} block={b} />)
      )}
    </div>
  );
}

function ContentBlock({ block }: { block: McpContentBlock }) {
  if (block.type === "image" && block.data) {
    const mime = block.mimeType ?? "image/png";
    return (
      <img
        src={`data:${mime};base64,${block.data}`}
        alt="MCP tool image result"
        className="max-h-96 w-auto rounded-lg border border-border"
      />
    );
  }
  if (block.type === "audio" && block.data) {
    const mime = block.mimeType ?? "audio/wav";
    return (
      // eslint-disable-next-line jsx-a11y/media-has-caption
      <audio controls src={`data:${mime};base64,${block.data}`} className="w-full" />
    );
  }
  // text (and any unknown block) -> show its text or a JSON dump.
  const text = block.text ?? JSON.stringify(block, null, 2);
  return (
    <pre className="max-h-96 overflow-auto rounded-lg bg-muted p-4 font-mono text-xs whitespace-pre-wrap">
      {text}
    </pre>
  );
}
