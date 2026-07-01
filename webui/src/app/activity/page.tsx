"use client";

import { useState, type ReactNode } from "react";
import { Card, Button, Badge, Alert, Spinner, EmptyState, cn } from "yunui";
import { StatCard } from "yunui/patterns";
import { Activity, X, Ban, RefreshCw } from "lucide-react";
import { api, usePolling, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { PageShell } from "@/components/page-shell";

/**
 * Page-local mirrors of the backend in-flight-generation contract (kept here
 * rather than in lib/types so this page owns its own shapes). Routes live under
 * `/v1`; auth is applied automatically by `api`. Non-admin callers see only
 * their own generations and cannot cancel-all.
 *   GET  /v1/active-generations → { object: "list", data: [{ request_id, model, ... }], active?, count }
 *   POST /v1/cancel { request_id }   → { status: "cancelled", request_id }   (404 if not found)
 *   POST /v1/cancel { cancel_all: true } → { status: "cancelled", count }    (403 if not admin/owner)
 */
interface ActiveGeneration {
  request_id: string;
  model?: string;
  [key: string]: unknown;
}
interface ActiveGenerationsResult {
  object?: string;
  data?: ActiveGeneration[];
  active?: ActiveGeneration[];
  count?: number;
}
interface CancelOneResult {
  status: string;
  request_id: string;
}
interface CancelAllResult {
  status: string;
  count: number;
}

function errMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Request failed";
}

/** Render an arbitrary field value generically so unknown backend fields still show. */
function renderValue(v: unknown): ReactNode {
  if (v == null) return <span className="text-muted-foreground">—</span>;
  if (typeof v === "boolean")
    return <Badge variant={v ? "success" : "default"}>{v ? "true" : "false"}</Badge>;
  if (typeof v === "number")
    return <span className="tabular-nums">{Number.isInteger(v) ? fmtNumber(v) : v.toFixed(2)}</span>;
  if (typeof v === "string") return v;
  return <span className="font-mono text-xs">{JSON.stringify(v)}</span>;
}

function GenerationRow({
  gen,
  busy,
  onCancel,
}: {
  gen: ActiveGeneration;
  busy: boolean;
  onCancel: (id: string) => void;
}) {
  // Show request_id + model prominently; any remaining fields as small chips.
  const extras = Object.entries(gen).filter(([k]) => k !== "request_id" && k !== "model");
  return (
    <Card className="flex flex-wrap items-start justify-between gap-4 p-4">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-medium">{gen.request_id}</span>
          {gen.model ? <Badge variant="info">{gen.model}</Badge> : null}
        </div>
        {extras.length > 0 && (
          <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3">
            {extras.map(([k, v]) => (
              <div key={k} className="min-w-0">
                <div className="truncate text-xs uppercase tracking-wide text-muted-foreground">
                  {k.replace(/_/g, " ")}
                </div>
                <div className="truncate text-sm font-medium">{renderValue(v)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
      <Button
        variant="destructive"
        size="sm"
        disabled={busy}
        onClick={() => onCancel(gen.request_id)}
      >
        {busy ? <Spinner size="sm" /> : <X className="h-4 w-4" />} Cancel
      </Button>
    </Card>
  );
}

export default function ActivityPage() {
  const gens = usePolling<ActiveGenerationsResult>(
    (s) => api.get("/v1/active-generations", s),
    1500,
  );

  const [cancelling, setCancelling] = useState<Record<string, boolean>>({});
  const [cancellingAll, setCancellingAll] = useState<boolean>(false);
  const [confirmAll, setConfirmAll] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Backend may return the list under `data` or `active`; prefer whichever has rows.
  const list = gens.data?.data?.length ? gens.data.data : (gens.data?.active ?? gens.data?.data ?? []);
  const count = gens.data?.count ?? list.length;

  const cancelOne = async (id: string) => {
    if (cancelling[id]) return;
    setCancelling((c) => ({ ...c, [id]: true }));
    setError(null);
    setNotice(null);
    try {
      await api.post<CancelOneResult>("/v1/cancel", { request_id: id });
      setNotice(`Cancelled ${id}.`);
      gens.refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setNotice(`${id} was already finished or not found.`);
        gens.refresh();
      } else {
        setError(errMessage(e));
      }
    } finally {
      setCancelling((c) => {
        const next = { ...c };
        delete next[id];
        return next;
      });
    }
  };

  const cancelAll = async () => {
    if (cancellingAll) return;
    setCancellingAll(true);
    setConfirmAll(false);
    setError(null);
    setNotice(null);
    try {
      const res = await api.post<CancelAllResult>("/v1/cancel", { cancel_all: true });
      setNotice(`Cancelled ${fmtNumber(res.count)} generation(s).`);
      gens.refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) {
        setError("Cancel-all requires admin/owner privileges.");
      } else {
        setError(errMessage(e));
      }
    } finally {
      setCancellingAll(false);
    }
  };

  return (
    <PageShell
      title="Activity"
      description="Live view of in-flight generations. Cancel individual requests or all of them."
      actions={
        <Button variant="secondary" size="sm" onClick={() => gens.refresh()}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatCard
            icon={Activity}
            label="In-flight"
            value={fmtNumber(count)}
            tone={count > 0 ? "emerald" : undefined}
            subtext={gens.loading && !gens.data ? "loading…" : "refreshed every 1.5s"}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">Active generations</span>
            <Badge variant={count > 0 ? "info" : "default"}>{fmtNumber(count)}</Badge>
            {gens.loading && !gens.data ? <Spinner size="sm" /> : null}
          </div>
          {confirmAll ? (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Cancel all {fmtNumber(count)}?</span>
              <Button
                variant="destructive"
                size="sm"
                loading={cancellingAll}
                onClick={cancelAll}
              >
                <Ban className="h-4 w-4" /> Confirm
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setConfirmAll(false)}>
                Keep
              </Button>
            </div>
          ) : (
            <Button
              variant="destructive"
              size="sm"
              disabled={cancellingAll || list.length === 0}
              onClick={() => setConfirmAll(true)}
            >
              <Ban className="h-4 w-4" /> Cancel all
            </Button>
          )}
        </div>

        {notice && <Alert variant="success">{notice}</Alert>}
        {error && <Alert variant="error">{error}</Alert>}
        {gens.error && (
          <Alert variant="warning">
            Could not load active generations: {errMessage(gens.error)}
          </Alert>
        )}

        {list.length === 0 ? (
          <Card className="p-4">
            <EmptyState
              icon={<Activity className="h-6 w-6" />}
              title="No in-flight generations"
              description="Requests currently being served will appear here in real time."
            />
          </Card>
        ) : (
          <div className={cn("space-y-3")}>
            {list.map((gen) => (
              <GenerationRow
                key={gen.request_id}
                gen={gen}
                busy={!!cancelling[gen.request_id]}
                onCancel={cancelOne}
              />
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}
