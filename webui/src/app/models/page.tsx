"use client";

import { useCallback, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Input,
  SegmentedSelect,
  Skeleton,
  StatusIndicator,
  EmptyState,
  toast,
} from "yunui";
import { Boxes, Download, Pin, RefreshCw } from "lucide-react";
import { api, usePolling } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import {
  guessModelType,
  ModelTypeChip,
  ModelTypeTile,
  MODEL_TYPES,
  type ModelType,
} from "@/lib/model-type";
import { ModelManagerCard, type ModelManagerField } from "yunui/ai";
import type { Model, MonitoringModel } from "@/lib/types";
import { PageShell } from "@/components/page-shell";

type Filter = "ALL" | ModelType;

const FILTER_OPTIONS: { value: Filter; label: string }[] = [
  { value: "ALL", label: "All" },
  ...(Object.keys(MODEL_TYPES) as ModelType[]).map((k) => ({
    value: k,
    label: MODEL_TYPES[k].label,
  })),
];

/** Small in-flight tracker: keys currently running an action, plus a runner. */
function useBusy() {
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const run = useCallback(
    async (
      key: string,
      fn: () => Promise<void>,
      opts?: { success?: string; errorTitle?: string },
    ) => {
      setBusy((p) => new Set(p).add(key));
      try {
        await fn();
        if (opts?.success) toast.success(opts.success);
      } catch (e) {
        toast.error(opts?.errorTitle ?? "Request failed", (e as Error).message);
      } finally {
        setBusy((p) => {
          const n = new Set(p);
          n.delete(key);
          return n;
        });
      }
    },
    [],
  );
  return { busy, run };
}

export default function ModelsPage() {
  const models = usePolling<{ data: Model[] }>((s) => api.get("/v1/models", s), 5000);
  // Pinned state only surfaces from the (authenticated) monitoring endpoint;
  // it degrades silently when there's no token.
  const monitored = usePolling<{ models: MonitoringModel[] }>(
    (s) => api.get("/api/v1/gw/monitoring/models", s),
    5000,
  );
  const { busy, run } = useBusy();

  const [filter, setFilter] = useState<Filter>("ALL");
  const [loadInput, setLoadInput] = useState("");

  const modelList = models.data?.data ?? [];
  const pinnedIds = new Set(
    (monitored.data?.models ?? []).filter((m) => m.pinned).map((m) => m.model_id),
  );
  // Anonymous callers get only `id` back — no type/size/loaded fields at all.
  const anonymous =
    modelList.length > 0 &&
    modelList.every((m) => m.type === undefined && m.size_gb === undefined && m.loaded === undefined);

  const filtered = modelList.filter(
    (m) => filter === "ALL" || (m.type ?? guessModelType(m.id)) === filter,
  );

  const refresh = useCallback(() => {
    models.refresh();
    monitored.refresh();
  }, [models, monitored]);

  const loadModel = useCallback(
    (id: string, pin = false) =>
      run(
        id,
        async () => {
          await api.post("/v1/models/load", { model: id, pin });
        },
        {
          success: pin ? `Pinned & loaded: ${id}` : `Load requested: ${id}`,
          errorTitle: "Load failed",
        },
      ).then(refresh),
    [run, refresh],
  );

  const unloadModel = useCallback(
    (id: string) =>
      run(
        id,
        async () => {
          await api.post(`/v1/models/unload/${encodeURIComponent(id)}`);
        },
        { success: `Unloaded: ${id}`, errorTitle: "Unload failed" },
      ).then(refresh),
    [run, refresh],
  );

  const headerActions = (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        className="w-56"
        placeholder="Load a model by id…"
        value={loadInput}
        onChange={(e) => setLoadInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && loadInput.trim()) loadModel(loadInput.trim());
        }}
      />
      <Button
        size="sm"
        onClick={() => loadInput.trim() && loadModel(loadInput.trim())}
        disabled={!loadInput.trim() || busy.has(loadInput.trim())}
      >
        <Download className="h-4 w-4" /> Load
      </Button>
    </div>
  );

  return (
    <PageShell
      title="Models"
      description="Load, pin and unload the models the engine serves."
      width="wide"
      actions={headerActions}
    >
      {models.error && (
        <Alert variant="error" title="Failed to load models" className="mb-4">
          {models.error.message}
        </Alert>
      )}

      {anonymous && (
        <Alert variant="info" className="mb-4">
          Showing model ids only. Add a backend token in Settings to see type, size and load status.
        </Alert>
      )}

      {/* Type filter */}
      <div className="mb-4 flex items-center justify-between gap-3">
        <SegmentedSelect<Filter> options={FILTER_OPTIONS} value={filter} onChange={setFilter} />
        <Button variant="ghost" size="sm" onClick={refresh}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      {/* Grid */}
      {models.loading && modelList.length === 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-40 w-full rounded-xl" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Boxes className="h-10 w-10" />}
          title={modelList.length === 0 ? "No models" : "No matching models"}
          description={
            modelList.length === 0
              ? "Load a model by id using the field above."
              : "No models match the selected type filter."
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((m) => (
            <ModelCard
              key={m.id}
              m={m}
              anonymous={anonymous}
              pinned={pinnedIds.has(m.id)}
              busy={busy.has(m.id)}
              onLoad={() => loadModel(m.id)}
              onPin={() => loadModel(m.id, true)}
              onUnload={() => unloadModel(m.id)}
            />
          ))}
        </div>
      )}
    </PageShell>
  );
}

function ModelCard({
  m,
  anonymous,
  pinned,
  busy,
  onLoad,
  onPin,
  onUnload,
}: {
  m: Model;
  anonymous: boolean;
  pinned: boolean;
  busy: boolean;
  onLoad: () => void;
  onPin: () => void;
  onUnload: () => void;
}) {
  const type = m.type ?? guessModelType(m.id);

  const fields: ModelManagerField[] = [
    { label: "Type", value: <ModelTypeChip type={type} /> },
    {
      label: "Status",
      value: m.loaded ? (
        <span className="font-medium text-success">Loaded</span>
      ) : (
        <span className="text-muted-foreground">Available</span>
      ),
    },
    ...(m.size_gb ? [{ label: "Size", value: fmtBytes(m.size_gb * 1e9) }] : []),
    ...(anonymous
      ? [{ full: true, label: "", value: <span className="text-xs text-muted-foreground">Connect a token for details</span> } as ModelManagerField]
      : []),
  ];

  return (
    <ModelManagerCard
      icon={<ModelTypeTile type={type} />}
      name={m.id}
      nameBadges={
        <span className="inline-flex items-center gap-1.5">
          <StatusIndicator status={m.loaded ? "online" : "neutral"} />
          {pinned && <Badge variant="info">Pinned</Badge>}
        </span>
      }
      actions={
        m.loaded ? (
          <div className="flex items-center gap-1">
            {!pinned && (
              <Button variant="ghost" size="sm" onClick={onPin} disabled={busy}>
                <Pin className="h-4 w-4" /> Pin
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={onUnload} disabled={busy}>
              Unload
            </Button>
          </div>
        ) : (
          <Button variant="secondary" size="sm" onClick={onLoad} disabled={busy}>
            Load
          </Button>
        )
      }
      fields={fields}
    />
  );
}
