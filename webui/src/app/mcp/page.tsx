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
import { Server, Wrench, Play, Copy, ChevronRight } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, usePolling, ApiError } from "@/lib/api";
import type { MCPServer, MCPTool } from "@/lib/types";

export default function MCPPage() {
  const status = usePolling<{ servers: MCPServer[] }>(
    (s) => api.get("/v1/mcp/client/status", s),
    8000,
  );
  const toolsResp = usePolling<{ tools: MCPTool[] }>((s) => api.get("/v1/mcp/tools", s), 15000);

  const servers = status.data?.servers ?? [];
  const tools = useMemo(() => toolsResp.data?.tools ?? [], [toolsResp.data]);

  const [tab, setTab] = useState("servers");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const [argsText, setArgsText] = useState("{}");
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const idRef = useRef(0);

  const filteredTools = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return tools;
    return tools.filter((t) => t.name.toLowerCase().includes(q));
  }, [tools, query]);

  const selectedTool = useMemo(
    () => tools.find((t) => t.name === selected) ?? null,
    [tools, selected],
  );

  const pickTool = (name: string) => {
    setSelected(name);
    setTab("execute");
  };

  const execute = async () => {
    setError(null);
    setResult(null);
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
      const res = await api.post<unknown>("/v1/mcp", {
        jsonrpc: "2.0",
        id: ++idRef.current,
        method: "tools/call",
        params: { name: selected, arguments: parsedArgs },
      });
      setResult(JSON.stringify(res, null, 2));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const copyResult = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result);
      toast.success("Copied result to clipboard");
    } catch {
      toast.error("Copy failed");
    }
  };

  return (
    <PageShell title="MCP" description="Model Context Protocol client." width="wide">
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="servers">Servers</TabsTrigger>
          <TabsTrigger value="tools">Tools</TabsTrigger>
          <TabsTrigger value="execute">Execute</TabsTrigger>
        </TabsList>

        {/* Servers */}
        <TabsContent value="servers" className="mt-4">
          {status.loading && servers.length === 0 ? (
            <div className="flex justify-center py-12">
              <Spinner />
            </div>
          ) : servers.length === 0 ? (
            <EmptyState
              icon={<Server className="h-6 w-6" />}
              title="No MCP servers"
              description="No Model Context Protocol servers are connected."
            />
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {servers.map((s) => (
                <Card key={s.name} className="p-5">
                  <div className="flex items-center justify-between gap-3">
                    <span className="min-w-0 flex-1 truncate text-sm font-medium">{s.name}</span>
                    <StatusIndicator status={s.status === "connected" ? "online" : "offline"}>
                      {s.status}
                    </StatusIndicator>
                  </div>
                  {typeof s.tool_count === "number" && (
                    <div className="mt-3">
                      <Badge variant="info">{s.tool_count} tools</Badge>
                    </div>
                  )}
                  {s.url && (
                    <p className="mt-3 truncate font-mono text-xs text-muted-foreground">{s.url}</p>
                  )}
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Tools */}
        <TabsContent value="tools" className="mt-4">
          <SearchInput
            value={query}
            onChange={setQuery}
            placeholder="Filter tools by name…"
            className="mb-4 max-w-md"
          />
          {toolsResp.loading && tools.length === 0 ? (
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
                <Card
                  key={`${t.server ?? ""}:${t.name}`}
                  hover
                  role="button"
                  tabIndex={0}
                  onClick={() => pickTool(t.name)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      pickTool(t.name);
                    }
                  }}
                  className="cursor-pointer p-4"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 flex-1 truncate font-mono text-sm font-medium">
                      {t.name}
                    </span>
                    {t.server && <Badge variant="default">{t.server}</Badge>}
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
        </TabsContent>

        {/* Execute */}
        <TabsContent value="execute" className="mt-4">
          <Card className="space-y-4 p-5">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Tool</label>
              <Select value={selected} onValueChange={setSelected}>
                <SelectTrigger className="max-w-md">
                  <SelectValue placeholder="Select a tool…" />
                </SelectTrigger>
                <SelectContent>
                  {tools.map((t) => (
                    <SelectItem key={`${t.server ?? ""}:${t.name}`} value={t.name}>
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
                    {schemaOpen ? "Hide schema" : "Show schema"}
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
              <Textarea
                value={argsText}
                onChange={(e) => setArgsText(e.target.value)}
                rows={6}
                spellCheck={false}
                className="font-mono text-sm"
                placeholder='{ "key": "value" }'
              />
            </div>

            <div className="flex items-center gap-2">
              <Button onClick={execute} disabled={running || !selected}>
                {running ? <Spinner className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                Execute
              </Button>
            </div>

            {error && (
              <Alert variant="error" title="Execution failed">
                {error}
              </Alert>
            )}

            {result !== null && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Result</span>
                  <Button variant="secondary" size="sm" onClick={copyResult}>
                    <Copy className="h-4 w-4" /> Copy
                  </Button>
                </div>
                <pre className="max-h-96 overflow-auto rounded-lg bg-muted p-4 font-mono text-xs">
                  {result}
                </pre>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}
