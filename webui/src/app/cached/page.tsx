"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Button,
  Input,
  Textarea,
  NumberInput,
  Card,
  Badge,
  Alert,
  EmptyState,
  Spinner,
  cn,
  toast,
} from "yunui";
import { Database, Plus, Trash2, Clock, Save, X, RefreshCw } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, usePolling, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model } from "@/lib/types";

/* ---- page-local types ----------------------------------------------------- */

/**
 * A Gemini-style cached content entry: a warmed, reusable prompt prefix (the KV
 * cache of a system instruction + seed contents) that chat requests can point at
 * by `name` to skip re-encoding it.
 */
interface CachedEntry {
  name: string;
  model?: string;
  token_count?: number;
  ttl_seconds?: number;
  ttl?: string | number;
  expire_time?: string;
  display_name?: string;
  [k: string]: unknown;
}

const TTL_MIN = 1;
const TTL_MAX = 86400;
const DEFAULT_TTL = 3600;

/** The list endpoint may return an array or wrap it under one of several keys. */
function normalizeList(data: unknown): CachedEntry[] {
  if (Array.isArray(data)) return data as CachedEntry[];
  if (data && typeof data === "object") {
    const obj = data as Record<string, unknown>;
    for (const key of ["cachedContents", "cached_contents", "data", "items", "caches"]) {
      if (Array.isArray(obj[key])) return obj[key] as CachedEntry[];
    }
  }
  return [];
}

/** Extract the `{id}` used by the item endpoints from a (possibly namespaced) name. */
function cacheId(name: string): string {
  const last = name.split("/").filter(Boolean).pop();
  return encodeURIComponent(last ?? name);
}

function ttlLabel(entry: CachedEntry): string | null {
  if (typeof entry.ttl_seconds === "number") return `${fmtNumber(entry.ttl_seconds)}s TTL`;
  if (typeof entry.ttl === "number") return `${fmtNumber(entry.ttl)}s TTL`;
  if (typeof entry.ttl === "string" && entry.ttl) return entry.ttl;
  if (entry.expire_time) return `expires ${entry.expire_time}`;
  return null;
}

export default function CachedContentsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [systemInstruction, setSystemInstruction] = useState("");
  const [content, setContent] = useState("");
  const [ttl, setTtl] = useState(DEFAULT_TTL);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const list = usePolling<unknown>((s) => api.get("/v1/cachedContents", s), 15000);
  const entries = useMemo(() => normalizeList(list.data), [list.data]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => {
        const rows = res.data ?? [];
        setModels(rows);
        setModel((cur) => cur || rows[0]?.id || "");
      })
      .catch(() => {
        /* the create form still works with a manually chosen model */
      });
    return () => controller.abort();
  }, []);

  const create = async () => {
    if (!model) {
      setError("Pick a model first.");
      return;
    }
    if (!systemInstruction.trim() && !content.trim()) {
      setError("Provide a system instruction and/or content to cache.");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { model, ttl_seconds: ttl };
      if (displayName.trim()) body.display_name = displayName.trim();
      if (systemInstruction.trim()) body.system_instruction = systemInstruction;
      if (content.trim()) body.contents = content;
      const entry = await api.post<CachedEntry>("/v1/cachedContents", body);
      toast.success("Cache created", entry.name);
      setDisplayName("");
      setSystemInstruction("");
      setContent("");
      list.refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  return (
    <PageShell
      title="Cached contents"
      description="Create and manage reusable, warmed prompt prefixes that chat requests can reference to skip re-encoding a shared system instruction or context."
      width="wide"
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* Create form -------------------------------------------------------- */}
        <div className="lg:col-span-2">
          <Card className="space-y-4 p-5">
            <div className="flex items-center gap-2">
              <Plus className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">Create a cache</h2>
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Model</label>
              <ModelPicker models={models} value={model} onChange={setModel} />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Display name</label>
              <Input aria-label="Display name"
                placeholder="Optional label, e.g. “Support persona”"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">System instruction</label>
              <Textarea aria-label="System instruction"
                rows={4}
                placeholder="Shared system prompt to warm into the cache…"
                value={systemInstruction}
                onChange={(e) => setSystemInstruction(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Content</label>
              <Textarea aria-label="Content"
                rows={5}
                placeholder="Seed context / documents to cache as a reusable prefix…"
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">TTL (seconds)</label>
              <NumberInput aria-label="TTL (seconds)"
                value={ttl}
                onChange={setTtl}
                min={TTL_MIN}
                max={TTL_MAX}
                step={60}
              />
              <p className="text-xs text-muted-foreground">
                How long the warmed prefix lives before eviction ({TTL_MIN}–
                {fmtNumber(TTL_MAX)}s).
              </p>
            </div>

            {error && (
              <Alert variant="error" title="Could not create cache">
                {error}
              </Alert>
            )}

            <div className="flex justify-end">
              <Button onClick={create} disabled={creating || !model}>
                {creating ? <Spinner size="sm" /> : <Plus className="h-4 w-4" />}
                Create cache
              </Button>
            </div>
          </Card>

          <Alert variant="info" title="Referencing a cache from chat" className="mt-4">
            Pass a cache&rsquo;s <code className="font-mono">name</code> as{" "}
            <code className="font-mono">&quot;cached_content&quot;: &quot;&lt;name&gt;&quot;</code>{" "}
            in a chat/completions request to reuse its warmed KV prefix instead of re-sending the
            system instruction and seed context every turn.
          </Alert>
        </div>

        {/* Existing caches ---------------------------------------------------- */}
        <div className="space-y-4 lg:col-span-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">
              Existing caches{entries.length > 0 && ` (${entries.length})`}
            </h2>
            <Button variant="ghost" size="sm" onClick={list.refresh}>
              <RefreshCw className={cn("h-4 w-4", list.loading && "animate-spin")} /> Refresh
            </Button>
          </div>

          {list.error && entries.length === 0 ? (
            <Alert variant="error" title="Failed to load caches">
              {list.error.message}
            </Alert>
          ) : list.loading && entries.length === 0 ? (
            <div className="flex justify-center py-12">
              <Spinner />
            </div>
          ) : entries.length === 0 ? (
            <EmptyState
              icon={<Database className="h-8 w-8" />}
              title="No cached contents"
              description="Create a cache on the left to warm a reusable prompt prefix."
            />
          ) : (
            <div className="space-y-3">
              {entries.map((entry) => (
                <CacheRow
                  key={entry.name}
                  entry={entry}
                  onChanged={list.refresh}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}

/* ---- one cache row -------------------------------------------------------- */

function CacheRow({ entry, onChanged }: { entry: CachedEntry; onChanged: () => void }) {
  const [editing, setEditing] = useState(false);
  const [ttl, setTtl] = useState(
    typeof entry.ttl_seconds === "number" ? entry.ttl_seconds : DEFAULT_TTL,
  );
  const [busy, setBusy] = useState<"save" | "delete" | null>(null);

  const label = ttlLabel(entry);

  const saveTtl = async () => {
    setBusy("save");
    try {
      await api.patch(`/v1/cachedContents/${cacheId(entry.name)}`, { ttl_seconds: ttl });
      toast.success("TTL updated");
      setEditing(false);
      onChanged();
    } catch (e) {
      toast.error("Update failed", e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const remove = async () => {
    setBusy("delete");
    try {
      const res = await api.delete<{ deleted?: boolean; name?: string }>(
        `/v1/cachedContents/${cacheId(entry.name)}`,
      );
      toast.success("Cache deleted", res.name ?? entry.name);
      onChanged();
    } catch (e) {
      toast.error("Delete failed", e instanceof ApiError ? e.message : (e as Error).message);
      setBusy(null);
    }
  };

  return (
    <Card className="space-y-3 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {entry.display_name && (
            <p className="truncate text-sm font-medium">{entry.display_name}</p>
          )}
          <p className="truncate font-mono text-xs text-muted-foreground">{entry.name}</p>
        </div>
        <div className="flex shrink-0 gap-2">
          {!editing && (
            <Button variant="ghost" size="sm" onClick={() => setEditing(true)}>
              <Clock className="h-4 w-4" /> Edit TTL
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={remove} disabled={busy !== null}>
            {busy === "delete" ? (
              <Spinner size="sm" />
            ) : (
              <Trash2 className="h-4 w-4 text-error" />
            )}
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {entry.model && <Badge variant="info">{entry.model}</Badge>}
        {typeof entry.token_count === "number" && (
          <Badge variant="default">{fmtNumber(entry.token_count)} tokens</Badge>
        )}
        {label && <Badge variant="default">{label}</Badge>}
      </div>

      {editing && (
        <div className="flex flex-wrap items-end gap-2 border-t border-border pt-3">
          <div className="space-y-1.5">
            <label className="text-xs font-medium">New TTL (seconds)</label>
            <NumberInput aria-label="New TTL (seconds)"
              value={ttl}
              onChange={setTtl}
              min={TTL_MIN}
              max={TTL_MAX}
              step={60}
              className="w-40"
            />
          </div>
          <Button size="sm" onClick={saveTtl} disabled={busy !== null}>
            {busy === "save" ? <Spinner size="sm" /> : <Save className="h-4 w-4" />}
            Save
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setEditing(false)}
            disabled={busy !== null}
          >
            <X className="h-4 w-4" /> Cancel
          </Button>
        </div>
      )}
    </Card>
  );
}
