"use client";

import { fmtBytes } from "@/lib/utils";
import { useEffect, useState, useCallback } from "react";
import {
  Settings2,
  Wifi,
  WifiOff,
  Terminal,
  Info,
  Check,
  Copy,
  Save,
  Loader2,
  ExternalLink,
} from "lucide-react";

export default function SettingsPage() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [editing, setEditing] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [connStatus, setConnStatus] = useState<"idle" | "checking" | "ok" | "fail">("idle");
  const [systemInfo, setSystemInfo] = useState<Record<string, string>>({});

  useEffect(() => {
    fetch("/api/v1/admin/config/engine")
      .then((r) => r.json())
      .then((data) => {
        setConfig(data);
        setEditing(Object.fromEntries(Object.entries(data).map(([k, v]) => [k, String(v)])));
      })
      .catch(() => {});

    fetch("/api/v1/monitoring/system")
      .then((r) => r.json())
      .then((d) =>
        setSystemInfo({
          MLX: d.mlx_version ?? "—",
          Python: d.python_version ?? "—",
          CPU: `${d.cpu_percent?.toFixed(1) ?? "—"}%`,
          RAM: fmtBytes(d.memory_total_bytes ?? 0),
        })
      )
      .catch(() => {});
  }, []);

  const saveConfig = async () => {
    setSaving(true);
    const updates: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(editing)) {
      const numVal = Number(value);
      updates[key] = isNaN(numVal) ? value : numVal;
    }
    try {
      const res = await fetch("/api/v1/admin/config/engine", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (res.ok) {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      }
    } catch {
    } finally {
      setSaving(false);
    }
  };

  const testConnection = useCallback(async () => {
    setConnStatus("checking");
    try {
      const res = await fetch("/health");
      setConnStatus(res.ok ? "ok" : "fail");
    } catch {
      setConnStatus("fail");
    }
    setTimeout(() => setConnStatus("idle"), 3000);
  }, []);

  const [copiedIdx, setCopiedIdx] = useState(-1);
  const copyCode = (idx: number, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(-1), 2000);
  };

  const codeBlocks = [
    {
      label: "Start server",
      lang: "bash",
      code: "yunshu serve --model Qwen3.5-9B-MLX-4bit",
    },
    {
      label: "OpenAI SDK",
      lang: "python",
      code: `from openai import OpenAI\nclient = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")\nresp = client.chat.completions.create(\n    model="Qwen3.5-9B-MLX-4bit",\n    messages=[{"role": "user", "content": "Hello!"}]\n)`,
    },
    {
      label: "Yunshu SDK",
      lang: "python",
      code: `from yunshu_sdk import YunshuClient\nclient = YunshuClient("http://localhost:8000")\nresp = client.chat.completions.create(\n    model="Qwen3.5-9B-MLX-4bit",\n    messages=[{"role": "user", "content": "Hello!"}]\n)`,
    },
  ];

  const endpoints = [
    { label: "Gateway", url: "http://localhost:8000" },
    { label: "OpenAI API", url: "http://localhost:8000/v1" },
    { label: "Health", url: "http://localhost:8000/health" },
    { label: "Admin API", url: "http://localhost:8000/api/v1" },
  ];

  return (
    <div className="p-6 space-y-6 max-w-4xl page-enter">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        <Settings2 className="w-6 h-6 text-[var(--color-accent)]" />
        Settings
      </h2>

      {/* Engine Config */}
      <Section icon={Settings2} title="Engine Configuration">
        {config ? (
          <div className="space-y-3">
            {Object.entries(editing).map(([key, value]) => (
              <div key={key} className="flex items-center gap-3">
                <label className="text-sm text-[var(--color-text-secondary)] min-w-[200px]">
                  {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </label>
                <input
                  type="text"
                  value={value}
                  onChange={(e) => setEditing((p) => ({ ...p, [key]: e.target.value }))}
                  className="flex-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-[var(--color-accent)]"
                />
              </div>
            ))}
            <button
              onClick={saveConfig}
              disabled={saving}
              className="flex items-center gap-2 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              {saved ? "Saved!" : "Save Changes"}
            </button>
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-secondary)]">
            Could not load configuration. Make sure the server is running.
          </p>
        )}
      </Section>

      {/* Connection */}
      <Section icon={Wifi} title="Connection">
        <div className="space-y-3">
          {endpoints.map((ep) => (
            <div key={ep.label} className="flex items-center gap-3 text-sm">
              <span className="text-[var(--color-text-secondary)] min-w-[120px]">{ep.label}</span>
              <code className="bg-[var(--color-bg-tertiary)] px-2 py-0.5 rounded text-xs flex-1">
                {ep.url}
              </code>
            </div>
          ))}
          <button
            onClick={testConnection}
            className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
          >
            {connStatus === "checking" ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : connStatus === "ok" ? (
              <Check className="w-4 h-4 text-[var(--color-success)]" />
            ) : connStatus === "fail" ? (
              <WifiOff className="w-4 h-4 text-[var(--color-danger)]" />
            ) : (
              <Wifi className="w-4 h-4" />
            )}
            Test Connection
          </button>
        </div>
      </Section>

      {/* Quick Start */}
      <Section icon={Terminal} title="Quick Start">
        <div className="space-y-4">
          {codeBlocks.map((block, i) => (
            <div key={i}>
              <div className="text-xs text-[var(--color-text-secondary)] mb-1.5">{block.label}</div>
              <div className="relative group">
                <pre className="bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg p-3 text-sm overflow-x-auto">
                  <code>{block.code}</code>
                </pre>
                <button
                  onClick={() => copyCode(i, block.code)}
                  className="absolute top-2 right-2 p-1 rounded bg-[var(--color-bg-secondary)] border border-[var(--color-border)] opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  {copiedIdx === i ? (
                    <Check className="w-3 h-3 text-[var(--color-success)]" />
                  ) : (
                    <Copy className="w-3 h-3 text-[var(--color-text-secondary)]" />
                  )}
                </button>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* About */}
      <Section icon={Info} title="About">
        <div className="space-y-2 text-sm">
          <p className="text-[var(--color-text-secondary)] leading-relaxed">
            Yunshu is a production-grade MLX inference platform for Apple Silicon clusters.
            Based on mlx-lm with continuous batching, oMLX-pattern streaming, and deep Metal integration.
          </p>
          <div className="grid grid-cols-2 gap-3 mt-3">
            {Object.entries(systemInfo).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2">
                <span className="text-[var(--color-text-secondary)] min-w-[80px]">{k}</span>
                <span className="font-mono text-xs">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </Section>
    </div>
  );
}

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof Settings2;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)]">
      <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center gap-2">
        <Icon className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="font-semibold text-sm">{title}</h3>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}
