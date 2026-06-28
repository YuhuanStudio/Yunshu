"use client";

import { useState, useCallback } from "react";
import {
  Wrench,
  Search,
  Play,
  RefreshCw,
  Server,
  AlertCircle,
  Copy,
} from "lucide-react";

const API_BASE =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:8000";

interface MCPServer {
  name: string;
  status: string;
  tool_count?: number;
  url?: string;
}

interface MCPTool {
  name: string;
  description?: string;
  server?: string;
  inputSchema?: Record<string, unknown>;
}

export default function MCPPage() {
  const [tab, setTab] = useState<"servers" | "tools" | "execute">("servers");
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [search, setSearch] = useState("");
  const [selectedTool, setSelectedTool] = useState<string>("");
  const [toolParams, setToolParams] = useState("{}");
  const [toolResult, setToolResult] = useState("");
  const [toolError, setToolError] = useState("");
  const [loading, setLoading] = useState(false);

  const loadServers = useCallback(async () => {
    try {
      // Gateway has no /servers endpoint — use /client/status which returns the
      // configured MCP client servers (Wave 431 fix).
      const res = await fetch(`${API_BASE}/v1/mcp/client/status`);
      if (res.ok) {
        const data = await res.json();
        setServers(data.servers || []);
      }
    } catch {}
  }, []);

  const loadTools = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/v1/mcp/tools`);
      if (res.ok) {
        const data = await res.json();
        setTools(data.tools || []);
      }
    } catch {
    } finally {
      setLoading(false);
    }
  }, []);

  const executeTool = async () => {
    setToolError("");
    setToolResult("");
    setLoading(true);
    try {
      const params = JSON.parse(toolParams);
      // Gateway uses JSON-RPC 2.0 at /v1/mcp; there is no per-tool URL.
      // Format the tools/call request per spec 2024-11-05 (Wave 431 fix).
      const res = await fetch(`${API_BASE}/v1/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: Date.now(),
          method: "tools/call",
          params: { name: selectedTool, arguments: params },
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        setToolError(err.detail || "Execution failed");
        return;
      }
      const data = await res.json();
      setToolResult(JSON.stringify(data, null, 2));
    } catch (e: any) {
      setToolError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const filteredTools = tools.filter(
    (t) =>
      !search ||
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      (t.description || "").toLowerCase().includes(search.toLowerCase())
  );

  const selectedToolDef = tools.find((t) => t.name === selectedTool);

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--color-border)]">
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <Wrench className="w-5 h-5 text-[var(--color-accent)]" />
          MCP Client
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Model Context Protocol — tool servers, discovery, and execution
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Tab Bar */}
        <div className="flex border-b border-[var(--color-border)]">
          {[
            { id: "servers" as const, label: "Servers", icon: Server },
            { id: "tools" as const, label: "Tools", icon: Search },
            { id: "execute" as const, label: "Execute", icon: Play },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                tab === id
                  ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                  : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* Servers Tab */}
        {tab === "servers" && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <button
                onClick={loadServers}
                className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 flex items-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Refresh
              </button>
            </div>
            {servers.length === 0 ? (
              <div className="text-center py-12 text-[var(--color-text-secondary)]">
                <Server className="w-12 h-12 mx-auto mb-3 opacity-40" />
                <p className="text-sm">No MCP servers configured</p>
                <p className="text-xs mt-1">
                  Configure via YUNSHU_MCP_SERVERS env var or mcp.json
                </p>
              </div>
            ) : (
              <div className="grid gap-3">
                {servers.map((s) => (
                  <div
                    key={s.name}
                    className="p-4 border border-[var(--color-border)] rounded-xl bg-[var(--color-bg-primary)]"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="font-medium text-sm">{s.name}</span>
                        <span
                          className={`text-xs px-2 py-0.5 rounded ${
                            s.status === "connected"
                              ? "bg-[var(--color-success)]/20 text-[var(--color-success)]"
                              : "bg-[var(--color-danger)]/20 text-[var(--color-danger)]"
                          }`}
                        >
                          {s.status}
                        </span>
                      </div>
                      {s.tool_count != null && (
                        <span className="text-xs text-[var(--color-text-secondary)]">
                          {s.tool_count} tools
                        </span>
                      )}
                    </div>
                    {s.url && (
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1 font-mono">
                        {s.url}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tools Tab */}
        {tab === "tools" && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="flex-1 relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-secondary)]" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search tools..."
                  className="w-full pl-9 pr-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm outline-none focus:border-[var(--color-accent)]"
                />
              </div>
              <button
                onClick={loadTools}
                className="px-3 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 flex items-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Load Tools
              </button>
            </div>
            {loading ? (
              <div className="text-center py-8 text-[var(--color-text-secondary)]">
                Loading...
              </div>
            ) : filteredTools.length === 0 ? (
              <div className="text-center py-8 text-[var(--color-text-secondary)] text-sm">
                {tools.length === 0 ? "Click 'Load Tools' to discover available tools" : "No matching tools"}
              </div>
            ) : (
              <div className="grid gap-2">
                {filteredTools.map((t) => (
                  <div
                    key={t.name}
                    className="p-3 border border-[var(--color-border)] rounded-lg bg-[var(--color-bg-primary)] cursor-pointer hover:border-[var(--color-accent)] transition-colors"
                    onClick={() => {
                      setSelectedTool(t.name);
                      setToolParams(
                        t.inputSchema?.properties
                          ? JSON.stringify(
                              Object.fromEntries(
                                Object.keys(t.inputSchema.properties).map((k) => [k, ""])
                              ),
                              null,
                              2
                            )
                          : "{}"
                      );
                      setTab("execute");
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm text-[var(--color-accent)]">
                        {t.name}
                      </span>
                      {t.server && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]">
                          {t.server}
                        </span>
                      )}
                    </div>
                    {t.description && (
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                        {t.description}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Execute Tab */}
        {tab === "execute" && (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] mb-1 block">
                Tool
              </label>
              <select
                value={selectedTool}
                onChange={(e) => setSelectedTool(e.target.value)}
                onClick={loadTools}
                className="w-full px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm outline-none"
              >
                <option value="">Select a tool...</option>
                {tools.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                  </option>
                ))}
              </select>
            </div>

            {selectedToolDef?.description && (
              <p className="text-xs text-[var(--color-text-secondary)]">
                {selectedToolDef.description}
              </p>
            )}

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-[var(--color-text-secondary)]">
                  Parameters (JSON)
                </label>
                {selectedToolDef?.inputSchema && (
                  <button
                    onClick={() =>
                      setToolParams(
                        JSON.stringify(selectedToolDef.inputSchema, null, 2)
                      )
                    }
                    className="text-xs text-[var(--color-accent)] hover:underline"
                  >
                    Show Schema
                  </button>
                )}
              </div>
              <textarea
                value={toolParams}
                onChange={(e) => setToolParams(e.target.value)}
                rows={8}
                className="w-full p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm font-mono resize-none outline-none focus:border-[var(--color-accent)]"
              />
            </div>

            <button
              onClick={executeTool}
              disabled={!selectedTool || loading}
              className="px-4 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
            >
              <Play className="w-3.5 h-3.5" />
              Execute
            </button>

            {toolError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-[var(--color-danger)] text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {toolError}
              </div>
            )}

            {toolResult && (
              <div className="border border-[var(--color-border)] rounded-xl p-4 bg-[var(--color-bg-primary)]">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Result</span>
                  <button
                    onClick={() => navigator.clipboard.writeText(toolResult)}
                    className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
                <pre className="text-xs font-mono whitespace-pre-wrap break-words text-[var(--color-text-primary)] max-h-96 overflow-auto">
                  {toolResult}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
