"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Settings2,
  Key,
  HardDrive,
  FileText,
  Cpu,
  Download,
  Upload,
  Save,
  Loader2,
  Check,
  RefreshCw,
  Trash2,
  Plus,
  Copy,
  Shield,
} from "lucide-react";

type Tab = "models" | "keys" | "logs" | "cache" | "hardware";

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <div className="p-6 space-y-6 max-w-5xl page-enter">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        <Shield className="w-6 h-6 text-[var(--color-accent)]" />
        Admin Panel
      </h2>

      {/* Tabs */}
      <div className="flex gap-1 bg-[var(--color-bg-tertiary)] rounded-lg p-1 w-fit">
        {([
          { id: "models", label: "Model Settings", icon: Settings2 },
          { id: "keys", label: "API Keys", icon: Key },
          { id: "logs", label: "Logs", icon: FileText },
          { id: "cache", label: "Cache", icon: HardDrive },
          { id: "hardware", label: "Hardware", icon: Cpu },
        ] as const).map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === t.id
                ? "bg-[var(--color-accent)] text-white shadow-sm"
                : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
            }`}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {tab === "models" && <ModelSettings />}
      {tab === "keys" && <ApiKeyManager />}
      {tab === "logs" && <LogViewer />}
      {tab === "cache" && <CacheManager />}
      {tab === "hardware" && <HardwareProfile />}
    </div>
  );
}

// ── Model Settings ──

function ModelSettings() {
  const [models, setModels] = useState<{ id: string; loaded?: boolean }[]>([]);
  const [settings, setSettings] = useState<Record<string, Record<string, unknown>>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/v1/models")
      .then((r) => r.json())
      .then((d) => setModels(d.data || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    models.forEach((m) => {
      fetch(`/api/v1/admin/models/${encodeURIComponent(m.id)}/settings`)
        .then((r) => r.json())
        .then((d) => setSettings((prev) => ({ ...prev, [m.id]: d.settings ?? {} })))
        .catch(() => {});
    });
  }, [models]);

  const updateSetting = (modelId: string, key: string, value: unknown) => {
    setSettings((prev) => ({
      ...prev,
      [modelId]: { ...(prev[modelId] || {}), [key]: value },
    }));
  };

  const saveSettings = async (modelId: string) => {
    setSaving(modelId);
    try {
      await fetch(`/api/v1/admin/models/${encodeURIComponent(modelId)}/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings[modelId] || {}),
      });
    } catch {}
    setSaving(null);
  };

  if (loading) {
    return <div className="text-[var(--color-text-secondary)] text-sm">Loading models...</div>;
  }

  if (models.length === 0) {
    return <div className="text-[var(--color-text-secondary)] text-sm">No models registered.</div>;
  }

  return (
    <div className="space-y-4">
      {models.map((m) => {
        const s = settings[m.id] || {};
        return (
          <div key={m.id} className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)]">
            <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-[var(--color-accent)]" />
                <span className="font-semibold text-sm">{m.id}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                  m.loaded ? "bg-[var(--color-success)]/15 text-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                }`}>
                  {m.loaded ? "Loaded" : "Registered"}
                </span>
              </div>
              <button
                onClick={() => saveSettings(m.id)}
                disabled={saving === m.id}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50 transition-colors"
              >
                {saving === m.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                Save
              </button>
            </div>
            <div className="p-4 grid grid-cols-2 lg:grid-cols-3 gap-4">
              <SettingInput label="Max Tokens" value={s.max_tokens ?? 32768} onChange={(v) => updateSetting(m.id, "max_tokens", v)} type="number" />
              <SettingInput label="Temperature" value={s.temperature ?? 1.0} onChange={(v) => updateSetting(m.id, "temperature", v)} type="number" step={0.1} />
              <SettingInput label="Top P" value={s.top_p ?? 0.95} onChange={(v) => updateSetting(m.id, "top_p", v)} type="number" step={0.05} />
              <SettingInput label="Top K" value={s.top_k ?? 0} onChange={(v) => updateSetting(m.id, "top_k", v)} type="number" />
              <SettingToggle label="Pinned" value={!!s.pinned} onChange={(v) => updateSetting(m.id, "pinned", v)} />
              <SettingToggle label="Default" value={!!s.is_default} onChange={(v) => updateSetting(m.id, "is_default", v)} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SettingInput({ label, value, onChange, type = "text", step = 1 }: {
  label: string; value: unknown; onChange: (v: number) => void; type?: string; step?: number;
}) {
  return (
    <div>
      <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide block mb-1">{label}</label>
      <input
        type={type}
        value={String(value)}
        step={step}
        onChange={(e) => onChange(type === "number" ? parseFloat(e.target.value) || 0 : 0)}
        className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-[var(--color-accent)]"
      />
    </div>
  );
}

function SettingToggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-3">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} className="rounded" />
      <label className="text-sm">{label}</label>
    </div>
  );
}

// ── API Key Manager ──

function ApiKeyManager() {
  const [keys, setKeys] = useState<{ name: string; role: string; is_active: boolean; expires_at: number | null }[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [created, setCreated] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch("/api/v1/admin/keys")
      .then((r) => r.json())
      .then((d) => setKeys(d.keys || []))
      .catch(() => {});
  }, []);

  const createKey = async () => {
    if (!newKeyName.trim()) return;
    try {
      const resp = await fetch("/api/v1/admin/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newKeyName }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setCreated(data.key);
        setKeys((prev) => [...prev, { name: newKeyName, role: data.role || "user", is_active: true, expires_at: data.expires_at || null }]);
        setNewKeyName("");
      }
    } catch {}
  };

  const deleteKey = async (keyName: string) => {
    try {
      const resp = await fetch(`/api/v1/admin/keys/${encodeURIComponent(keyName)}`, {
        method: "DELETE",
      });
      if (resp.ok) {
        setKeys((prev) => prev.filter((k) => k.name !== keyName));
      }
    } catch {}
  };

  return (
    <div className="space-y-4">
      {/* Create new key */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
        <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Create API Key
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && createKey()}
            placeholder="Key name (e.g., 'my-app')"
            className="flex-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[var(--color-accent)]"
          />
          <button onClick={createKey} disabled={!newKeyName.trim()} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50 transition-colors">
            Create
          </button>
        </div>
        {created && (
          <div className="mt-3 p-3 bg-[var(--color-success)]/10 border border-[var(--color-success)]/30 rounded-lg">
            <p className="text-xs text-[var(--color-success)] mb-1">Key created. Copy it now — it won't be shown again.</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-sm font-mono bg-[var(--color-bg-tertiary)] px-2 py-1 rounded">{created}</code>
              <button onClick={() => { navigator.clipboard.writeText(created); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="p-1.5 rounded hover:bg-[var(--color-bg-tertiary)]">
                {copied ? <Check className="w-4 h-4 text-[var(--color-success)]" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Key list */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)]">
        <div className="px-4 py-3 border-b border-[var(--color-border)]">
          <h3 className="font-semibold text-sm">Active Keys ({keys.length})</h3>
        </div>
        {keys.length === 0 ? (
          <div className="p-4 text-sm text-[var(--color-text-secondary)]">No API keys configured.</div>
        ) : (
          <div className="divide-y divide-[var(--color-border)]">
            {keys.map((k) => (
              <div key={k.name} className="flex items-center gap-3 px-4 py-3">
                <Key className="w-4 h-4 text-[var(--color-accent)]" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm">{k.name}</div>
                  <div className="text-xs text-[var(--color-text-secondary)]">
                    <span className="inline-block px-1.5 py-0.5 rounded bg-[var(--color-accent)]/10 text-[var(--color-accent)] font-mono">{k.role}</span>
                    {!k.is_active && <span className="ml-2 text-[var(--color-danger)]">inactive</span>}
                  </div>
                </div>
                <button onClick={() => deleteKey(k.name)} className="p-1.5 rounded text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10 transition-colors">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Log Viewer ──

function LogViewer() {
  const [logs, setLogs] = useState<{timestamp?: string; level?: string; message?: string; logger?: string}[]>([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [level, setLevel] = useState("all");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(`/api/v1/admin/logs?level=${level}&lines=100`);
      if (resp.ok) {
        const data = await resp.json();
        if (mountedRef.current) setLogs(data.logs || []);
      }
    } catch {}
    if (mountedRef.current) setLoading(false);
  }, [level]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(fetchLogs, 3000);
    return () => clearInterval(id);
  }, [autoRefresh, fetchLogs]);

  const getLogColor = (entry: {level?: string}) => {
    const lvl = entry.level?.toLowerCase() ?? "";
    if (lvl === "error" || lvl === "critical") return "text-[var(--color-danger)]";
    if (lvl === "warning" || lvl === "warn") return "text-[var(--color-warning)]";
    if (lvl === "info") return "text-[var(--color-text-primary)]";
    if (lvl === "debug") return "text-[var(--color-text-secondary)]";
    return "text-[var(--color-text-secondary)]";
  };

  const formatLogLine = (entry: {timestamp?: string; level?: string; logger?: string; message?: string}) => {
    const ts = entry.timestamp ?? "";
    const lvl = (entry.level ?? "").toUpperCase().padEnd(8);
    const logger = entry.logger ? `[${entry.logger}] ` : "";
    return `${ts} ${lvl} ${logger}${entry.message ?? ""}`;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <select value={level} onChange={(e) => setLevel(e.target.value)} className="bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm">
          <option value="all">All Levels</option>
          <option value="error">Error</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
        </select>
        <button onClick={fetchLogs} disabled={loading} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm border border-[var(--color-border)] hover:bg-[var(--color-bg-tertiary)] transition-colors">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
        <label className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
          Auto-refresh (3s)
        </label>
        <span className="text-xs text-[var(--color-text-secondary)] ml-auto">{logs.length} lines</span>
      </div>

      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] overflow-hidden">
        <div className="h-[500px] overflow-auto font-mono text-xs p-4">
          {logs.length === 0 ? (
            <div className="text-[var(--color-text-secondary)]">No logs available.</div>
          ) : (
            logs.map((entry, i) => (
              <div key={i} className={`${getLogColor(entry)} whitespace-pre-wrap break-all leading-5`}>
                {formatLogLine(entry)}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// ── Cache Manager ──

function CacheManager() {
  const [cacheInfo, setCacheInfo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const fetchCache = async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/v1/admin/cache/status");
      if (resp.ok) setCacheInfo(await resp.json());
    } catch {}
    setLoading(false);
  };

  useEffect(() => { fetchCache(); }, []);

  const clearCache = async () => {
    setClearing(true);
    try {
      await fetch("/api/v1/admin/cache/clear", { method: "POST" });
      await fetchCache();
    } catch {}
    setClearing(false);
  };

  if (loading) return <div className="text-[var(--color-text-secondary)] text-sm">Loading cache info...</div>;

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
            Cache Status
          </h3>
          <button onClick={clearCache} disabled={clearing} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--color-danger)]/15 text-[var(--color-danger)] hover:bg-[var(--color-danger)]/25 transition-colors">
            {clearing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
            Clear Cache
          </button>
        </div>

        {cacheInfo ? (
          <div className="space-y-4">
            {/* Per-model KV caches */}
            {(cacheInfo.caches as Array<{ model_id: string; stats: Record<string, unknown> }>)?.map?.((entry) => (
              <div key={entry.model_id} className="border border-[var(--color-border)] rounded-lg p-3">
                <div className="text-xs font-medium mb-2">{entry.model_id}</div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                  {Object.entries(entry.stats).map(([key, value]) => (
                    <div key={key}>
                      <div className="text-[10px] text-[var(--color-text-secondary)]">{key.replace(/_/g, " ")}</div>
                      <div className="font-medium tabular-nums text-xs">{typeof value === "object" ? JSON.stringify(value) : String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            {(cacheInfo.caches as unknown[])?.length === 0 && (
              <div className="text-sm text-[var(--color-text-secondary)]">No KV caches active.</div>
            )}
            <div className="text-xs text-[var(--color-text-secondary)]">
              Total cache entries: {String(cacheInfo.total ?? 0)}
            </div>
          </div>
        ) : (
          <div className="text-sm text-[var(--color-text-secondary)]">Cache info not available.</div>
        )}
      </div>

      {/* GPU Memory */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
        <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
          GPU Memory
        </h3>
        <GpuMemoryBars />
      </div>
    </div>
  );
}

// ── Hardware Profile ──

function HardwareProfile() {
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/v1/admin/hardware-profile")
      .then((r) => r.json())
      .then((d) => setProfile(d))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-[var(--color-text-secondary)] text-sm">Loading hardware profile...</div>;
  if (!profile || profile.error) return <div className="text-sm text-[var(--color-text-secondary)]">Hardware profile not available. {String(profile?.error || "")}</div>;

  return (
    <div className="space-y-4">
      {/* Chip Info */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
        <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
          Apple Silicon Profile
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          {Object.entries(profile).map(([key, value]) => {
            if (typeof value === "object" && value !== null) return null;
            return (
              <div key={key}>
                <div className="text-[10px] text-[var(--color-text-secondary)] uppercase tracking-wide">
                  {key.replace(/_/g, " ")}
                </div>
                <div className="font-medium text-sm tabular-nums">{String(value)}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* GPU Memory */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
        <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
          GPU Memory
        </h3>
        <GpuMemoryBars />
      </div>
    </div>
  );
}

function GpuMemoryBars() {
  const [gpu, setGpu] = useState<{ total: number; active: number; peak: number; cache: number; available: number } | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    fetch("/api/v1/monitoring/system")
      .then((r) => r.json())
      .then((d) => {
        if (mountedRef.current && d.gpu) {
          setGpu({
            total: d.gpu.total_bytes,
            active: d.gpu.active_bytes,
            peak: d.gpu.peak_bytes,
            cache: d.gpu.cache_bytes,
            available: d.gpu.available_bytes,
          });
        }
      })
      .catch(() => {});
    return () => { mountedRef.current = false; };
  }, []);

  if (!gpu) return <div className="text-sm text-[var(--color-text-secondary)]">GPU info not available.</div>;

  const fmt = (b: number) => {
    const u = ["B", "KB", "MB", "GB"];
    let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return `${b.toFixed(1)} ${u[i]}`;
  };

  return (
    <div className="space-y-2">
      {[
        { label: "Active", value: gpu.active, color: "bg-blue-500" },
        { label: "Peak", value: gpu.peak, color: "bg-amber-500" },
        { label: "Cache", value: gpu.cache, color: "bg-emerald-500" },
        { label: "Available", value: gpu.available, color: "bg-purple-500" },
      ].map(({ label, value, color }) => {
        const pct = gpu.total > 0 ? (value / gpu.total) * 100 : 0;
        return (
          <div key={label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-[var(--color-text-secondary)]">{label}</span>
              <span className="tabular-nums">{fmt(value)} ({pct.toFixed(1)}%)</span>
            </div>
            <div className="w-full bg-[var(--color-bg-tertiary)] rounded-full h-1.5">
              <div className={`${color} h-1.5 rounded-full transition-all duration-500`} style={{ width: `${Math.min(pct, 100)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
