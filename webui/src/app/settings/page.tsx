"use client";

import { useEffect, useState, type ReactNode, type ElementType } from "react";
import {
  Button,
  PasswordInput,
  Card,
  Spinner,
  Separator,
  StatusIndicator,
  Badge,
  toast,
} from "yunui";
import { CodeBlock, StatCard } from "yunui/patterns";
import { Plug, Terminal, Info, Cpu, MemoryStick, Boxes, KeyRound, Save } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api } from "@/lib/api";
import { getToken, setToken, onTokenChange } from "@/lib/auth";
import { fmtBytes, fmtPct, fmtDuration } from "@/lib/format";
import type { SystemStats, HealthStatus, VersionInfo } from "@/lib/types";

const SYSTEM_PATH = "/api/v1/gw/monitoring/system";

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
  // --- Backend authentication ----------------------------------------------
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState("");

  useEffect(() => {
    const sync = () => {
      const t = getToken();
      setSaved(t);
      setDraft(t);
    };
    sync();
    return onTokenChange(sync);
  }, []);

  const saveToken = () => {
    setToken(draft.trim());
    toast.success(draft.trim() ? "Token saved" : "Token cleared");
  };
  const clearToken = () => {
    setDraft("");
    setToken("");
    toast.success("Token cleared");
  };
  const dirty = draft.trim() !== saved;

  // --- Health / version -----------------------------------------------------
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [version, setVersion] = useState<VersionInfo | null>(null);
  const [conn, setConn] = useState<"online" | "offline" | null>(null);
  const [testing, setTesting] = useState(false);

  const probe = async (signal?: AbortSignal) => {
    setTesting(true);
    try {
      const [h, v] = await Promise.allSettled([
        api.get<HealthStatus>("/health", signal),
        api.get<VersionInfo>("/version", signal),
      ]);
      if (signal?.aborted) return;
      if (h.status === "fulfilled") {
        setHealth(h.value);
        setConn("online");
      } else {
        setHealth(null);
        setConn("offline");
      }
      setVersion(v.status === "fulfilled" ? v.value : null);
    } finally {
      if (!signal?.aborted) setTesting(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    probe(controller.signal);
    return () => controller.abort();
    // re-probe whenever the saved token changes
  }, [saved]);

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
  }, [saved]);

  const endpoints: { label: string; path: string }[] = [
    { label: "Base URL", path: "/v1" },
    { label: "Gateway", path: "/api/v1/gw" },
    { label: "Health", path: "/health" },
  ];

  const curlSnippet = `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $YUNSHU_AUTH_TOKEN" \\
  -d '{
    "model": "your-model-id",
    "messages": [
      { "role": "user", "content": "Hello!" }
    ]
  }'`;

  const pythonSnippet = `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YUNSHU_AUTH_TOKEN",  # the server's static bearer token
)

resp = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)`;

  return (
    <PageShell
      title="Settings"
      description="Authenticate to the backend, verify connectivity and get started."
      width="narrow"
    >
      <div className="space-y-4">
        {/* 1. Backend authentication */}
        <SectionCard
          icon={KeyRound}
          title="Backend authentication"
          action={
            conn && (
              <StatusIndicator status={conn}>
                <span className={conn === "online" ? "text-success" : "text-error"}>
                  {conn === "online" ? "Online" : "Offline"}
                </span>
              </StatusIndicator>
            )
          }
        >
          <p className="mb-3 text-sm text-muted-foreground">
            When the server runs with <code className="font-mono">YUNSHU_AUTH_TOKEN</code> set, every
            request must carry it as a bearer token. Authenticated surfaces — model details,
            monitoring and load/unload — need this. Stored locally in your browser.
          </p>
          <div className="flex flex-col gap-2 sm:flex-row">
            <PasswordInput
              className="sm:flex-1 font-mono text-sm"
              placeholder="Paste YUNSHU_AUTH_TOKEN…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveToken()}
            />
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={saveToken} disabled={!dirty}>
                <Save className="h-4 w-4" /> Save
              </Button>
              <Button variant="secondary" size="sm" onClick={clearToken} disabled={!saved && !draft}>
                Clear
              </Button>
            </div>
          </div>
          <Separator className="my-4" />
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard
              icon={Plug}
              label="Server state"
              value={health?.server_state ?? "—"}
              compact
            />
            <StatCard
              icon={Boxes}
              label="Engine"
              value={health?.engine?.loaded === undefined ? "—" : health.engine.loaded ? "Loaded" : "Idle"}
              compact
              tone="blue"
            />
            <StatCard
              icon={Info}
              label={version?.service ?? "Service"}
              value={version?.version ?? "—"}
              compact
              tone="emerald"
            />
            <StatCard
              icon={Info}
              label="Uptime"
              value={health?.uptime_seconds === undefined ? "—" : fmtDuration(health.uptime_seconds)}
              compact
              tone="purple"
            />
          </div>
        </SectionCard>

        {/* 2. Connection */}
        <SectionCard
          icon={Plug}
          title="Connection"
          action={
            <Button variant="secondary" size="sm" onClick={() => probe()} disabled={testing}>
              {testing ? <Spinner size="sm" /> : null}
              Test connection
            </Button>
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
          {saved && (
            <div className="mt-3">
              <Badge variant="success">Token attached</Badge>
            </div>
          )}
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
            <StatCard
              icon={Boxes}
              label="MLX version"
              value={system?.gpu.mlx_version ?? "—"}
              compact
            />
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
              value={system ? fmtPct(system.cpu.percent) : "—"}
              compact
              tone="purple"
            />
            <StatCard
              icon={MemoryStick}
              label="RAM used"
              value={system ? fmtBytes(system.memory.used_bytes) : "—"}
              subtext={system ? `of ${fmtBytes(system.memory.total_bytes)}` : undefined}
              compact
              tone="emerald"
            />
          </div>
          <Separator className="my-4" />
          <p className="text-sm text-muted-foreground">
            {version?.description ??
              "Yunshu is an OpenAI-compatible MLX inference server. Point any OpenAI client at the base URL above to start generating."}
          </p>
        </SectionCard>
      </div>
    </PageShell>
  );
}
