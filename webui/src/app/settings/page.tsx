"use client";

import { useEffect, useMemo, useState, useCallback, type ReactNode, type ElementType } from "react";
import {
  Button,
  Input,
  Card,
  Alert,
  Spinner,
  Separator,
  StatusIndicator,
  toast,
} from "yunui";
import { SettingRow, CodeBlock, StatCard } from "yunui/patterns";
import {
  SlidersHorizontal,
  Plug,
  Terminal,
  Info,
  Save,
  Cpu,
  MemoryStick,
  Boxes,
} from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, ApiError } from "@/lib/api";
import { fmtBytes, fmtPct } from "@/lib/format";
import type { SystemStats } from "@/lib/types";

type EngineConfig = Record<string, unknown>;

const ENGINE_PATH = "/api/v1/admin/config/engine";
const SYSTEM_PATH = "/api/v1/monitoring/system";

/** Stringify a config value for editing in a text input. */
function toInput(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** Coerce an edited string back to the type of the original value. */
function fromInput(original: unknown, next: string): unknown {
  if (typeof original === "number") {
    const n = Number(next);
    return Number.isFinite(n) ? n : next;
  }
  if (typeof original === "boolean") {
    return next.trim().toLowerCase() === "true";
  }
  if (typeof original === "object" && original !== null) {
    try {
      return JSON.parse(next);
    } catch {
      return next;
    }
  }
  return next;
}

function SectionCard({
  icon: Icon,
  title,
  action,
  children,
}: {
  icon: ElementType;
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card className="p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">{title}</span>
        </div>
        {action}
      </div>
      {children}
    </Card>
  );
}

export default function SettingsPage() {
  // --- Engine configuration -------------------------------------------------
  const [config, setConfig] = useState<EngineConfig | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [saving, setSaving] = useState(false);

  const loadConfig = useCallback(async (signal?: AbortSignal) => {
    setLoadingConfig(true);
    try {
      const data = await api.get<EngineConfig>(ENGINE_PATH, signal);
      if (signal?.aborted) return;
      setConfig(data);
      setDrafts(Object.fromEntries(Object.entries(data).map(([k, v]) => [k, toInput(v)])));
      setLoadError(null);
    } catch (e) {
      if (signal?.aborted) return;
      setLoadError(e instanceof ApiError ? e.message : "Failed to load engine configuration.");
    } finally {
      if (!signal?.aborted) setLoadingConfig(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadConfig(controller.signal);
    return () => controller.abort();
  }, [loadConfig]);

  const changedKeys = useMemo(() => {
    if (!config) return [] as string[];
    return Object.keys(config).filter((k) => drafts[k] !== toInput(config[k]));
  }, [config, drafts]);

  const saveConfig = async () => {
    if (!config || changedKeys.length === 0) return;
    setSaving(true);
    try {
      const body: EngineConfig = {};
      for (const k of changedKeys) body[k] = fromInput(config[k], drafts[k]);
      const updated = await api.patch<EngineConfig>(ENGINE_PATH, body);
      // Prefer the server echo; fall back to a local merge.
      const merged = updated && typeof updated === "object" ? updated : { ...config, ...body };
      setConfig(merged);
      setDrafts(Object.fromEntries(Object.entries(merged).map(([k, v]) => [k, toInput(v)])));
      toast.success("Engine configuration saved");
    } catch (e) {
      toast.error("Save failed", e instanceof ApiError ? e.message : undefined);
    } finally {
      setSaving(false);
    }
  };

  // --- Connection -----------------------------------------------------------
  const endpoints: { label: string; path: string }[] = [
    { label: "Base URL", path: "/v1" },
    { label: "Admin", path: "/api/v1" },
    { label: "Health", path: "/health" },
  ];
  const [conn, setConn] = useState<"online" | "offline" | null>(null);
  const [testing, setTesting] = useState(false);

  const testConnection = async () => {
    setTesting(true);
    try {
      const ok = await api.health();
      setConn(ok ? "online" : "offline");
      if (ok) toast.success("Connected", "The engine is reachable.");
      else toast.error("Offline", "The engine did not respond.");
    } finally {
      setTesting(false);
    }
  };

  // --- About / system info --------------------------------------------------
  const [system, setSystem] = useState<SystemStats | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<SystemStats>(SYSTEM_PATH, controller.signal)
      .then((s) => {
        if (!controller.signal.aborted) setSystem(s);
      })
      .catch(() => {
        /* non-fatal: about section simply shows placeholders */
      });
    return () => controller.abort();
  }, []);

  const curlSnippet = `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer no-key" \\
  -d '{
    "model": "your-model-id",
    "messages": [
      { "role": "user", "content": "Hello!" }
    ]
  }'`;

  const pythonSnippet = `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key",  # any string; the local engine ignores it
)

resp = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)`;

  return (
    <PageShell
      title="Settings"
      description="Tune the inference engine, verify connectivity and get started."
      width="narrow"
    >
      <div className="space-y-4">
        {/* 1. Engine configuration */}
        <SectionCard
          icon={SlidersHorizontal}
          title="Engine configuration"
          action={
            <Button size="sm" onClick={saveConfig} disabled={saving || changedKeys.length === 0}>
              {saving ? <Spinner size="sm" /> : <Save className="h-4 w-4" />}
              Save{changedKeys.length > 0 ? ` (${changedKeys.length})` : ""}
            </Button>
          }
        >
          {loadingConfig ? (
            <div className="flex justify-center py-8">
              <Spinner />
            </div>
          ) : loadError ? (
            <Alert variant="error" title="Could not load configuration">
              {loadError}
            </Alert>
          ) : config && Object.keys(config).length > 0 ? (
            <div className="divide-y divide-border">
              {Object.keys(config).map((key) => (
                <SettingRow
                  key={key}
                  title={<span className="font-mono text-sm">{key}</span>}
                  control={
                    <Input
                      className="w-56 font-mono text-sm"
                      value={drafts[key] ?? ""}
                      onChange={(e) => setDrafts((d) => ({ ...d, [key]: e.target.value }))}
                    />
                  }
                />
              ))}
            </div>
          ) : (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No engine configuration keys exposed.
            </p>
          )}
        </SectionCard>

        {/* 2. Connection */}
        <SectionCard
          icon={Plug}
          title="Connection"
          action={
            <div className="flex items-center gap-3">
              {conn && (
                <StatusIndicator status={conn}>
                  <span className={conn === "online" ? "text-success" : "text-error"}>
                    {conn === "online" ? "Online" : "Offline"}
                  </span>
                </StatusIndicator>
              )}
              <Button variant="secondary" size="sm" onClick={testConnection} disabled={testing}>
                {testing ? <Spinner size="sm" /> : null}
                Test connection
              </Button>
            </div>
          }
        >
          <div className="divide-y divide-border">
            {endpoints.map((ep) => (
              <div key={ep.label} className="flex items-center justify-between gap-3 py-2.5">
                <span className="text-sm text-muted-foreground">{ep.label}</span>
                <code className="font-mono text-sm">{ep.path}</code>
              </div>
            ))}
          </div>
        </SectionCard>

        {/* 3. Quick start */}
        <SectionCard icon={Terminal} title="Quick start">
          <div className="space-y-4">
            <CodeBlock code={curlSnippet} language="bash" filename="chat.sh" />
            <CodeBlock code={pythonSnippet} language="python" filename="chat.py" />
          </div>
        </SectionCard>

        {/* 4. About / system info */}
        <SectionCard icon={Info} title="About">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard icon={Boxes} label="MLX version" value={system?.mlx_version ?? "—"} compact />
            <StatCard
              icon={Boxes}
              label="Python"
              value={system?.python_version ?? "—"}
              compact
              tone="blue"
            />
            <StatCard
              icon={Cpu}
              label="CPU"
              value={system ? fmtPct(system.cpu_percent) : "—"}
              compact
              tone="purple"
            />
            <StatCard
              icon={MemoryStick}
              label="RAM used"
              value={system ? fmtBytes(system.memory_used_bytes) : "—"}
              subtext={system ? `of ${fmtBytes(system.memory_total_bytes)}` : undefined}
              compact
              tone="emerald"
            />
          </div>
          <Separator className="my-4" />
          <p className="text-sm text-muted-foreground">
            Yunshu is an OpenAI-compatible MLX inference server. Point any OpenAI client at the base
            URL above to start generating.
          </p>
        </SectionCard>
      </div>
    </PageShell>
  );
}
