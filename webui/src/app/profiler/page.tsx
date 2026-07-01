"use client";

import { useState, type ReactNode } from "react";
import {
  Card,
  Button,
  Badge,
  Alert,
  Spinner,
  StatusIndicator,
  NumberInput,
  Input,
  Checkbox,
  cn,
} from "yunui";
import { Play, Square, Camera, Cpu, RefreshCw } from "lucide-react";
import { api, usePolling, ApiError } from "@/lib/api";
import { fmtDuration } from "@/lib/format";
import { PageShell } from "@/components/page-shell";

/**
 * Page-local mirrors of the backend Metal-capture profiler contract (kept here
 * rather than in lib/types so this page owns its own shapes). All routes live
 * under `/v1` and are admin-sensitive; auth is applied automatically by `api`.
 *   POST /v1/start_profile  → { status: "started", output_path, duration_seconds }
 *   POST /v1/stop_profile   → { status: "stopped", elapsed_seconds }
 *   GET  /v1/profile/status → { active, elapsed_seconds }
 *   GET  /v1/profile/engine → { engines: [{ model_id, profiler?, auto_tuner?, slo?, ... }] }
 */
interface ProfileStatus {
  active: boolean;
  elapsed_seconds: number;
}
interface StartProfileResult {
  status: string;
  output_path: string;
  duration_seconds: number;
}
interface StopProfileResult {
  status: string;
  elapsed_seconds: number;
}
interface ProfileEngine {
  model_id: string;
  profiler?: unknown;
  auto_tuner?: unknown;
  slo?: unknown;
  scheduler_profiling?: unknown;
  [key: string]: unknown;
}
interface ProfileEngineResult {
  engines: ProfileEngine[];
}

function errMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Request failed";
}

/** Human help text for the well-known profiler error statuses. */
function explainError(e: unknown): { title: string; body: string } {
  const msg = errMessage(e);
  if (e instanceof ApiError) {
    if (e.status === 400)
      return {
        title: "Metal capture is not enabled",
        body:
          "The server must be launched with MTL_CAPTURE_ENABLED=1 to allow GPU trace capture, " +
          "and the output must be a .gputrace path under /tmp/yunshu_profiles. " +
          `(${msg})`,
      };
    if (e.status === 409)
      return {
        title: "Capture state conflict",
        body:
          "A capture is already active, or there is no active capture to stop. " +
          `Refresh the status and try again. (${msg})`,
      };
    if (e.status === 501)
      return {
        title: "Profiler unavailable",
        body:
          "Metal capture is not available in this build or on this platform. " + `(${msg})`,
      };
    if (e.status === 403)
      return { title: "Not permitted", body: msg };
  }
  return { title: "Request failed", body: msg };
}

function renderValue(v: unknown): ReactNode {
  if (v == null) return <span className="text-muted-foreground">—</span>;
  if (typeof v === "boolean")
    return <Badge variant={v ? "success" : "default"}>{v ? "on" : "off"}</Badge>;
  if (typeof v === "number")
    return <span className="tabular-nums">{Number.isInteger(v) ? v : v.toFixed(2)}</span>;
  if (typeof v === "string") return v;
  return (
    <pre className="whitespace-pre-wrap break-words rounded-md bg-muted px-2 py-1 text-xs">
      {JSON.stringify(v, null, 2)}
    </pre>
  );
}

/** Render an engine's per-subsystem info blocks (profiler / auto_tuner / slo / …). */
function EngineDetail({ engine }: { engine: ProfileEngine }) {
  const blocks = Object.entries(engine).filter(([k]) => k !== "model_id");
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2">
        <Cpu className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">{engine.model_id}</span>
      </div>
      {blocks.length === 0 ? (
        <p className="text-sm text-muted-foreground">No profiling metadata for this engine.</p>
      ) : (
        <div className="space-y-3">
          {blocks.map(([key, value]) => (
            <div key={key} className="min-w-0">
              <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                {key.replace(/_/g, " ")}
              </div>
              {value && typeof value === "object" && !Array.isArray(value) ? (
                <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3">
                  {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
                    <div key={k} className="min-w-0">
                      <div className="truncate text-xs text-muted-foreground">
                        {k.replace(/_/g, " ")}
                      </div>
                      <div className="truncate text-sm font-medium">{renderValue(v)}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm font-medium">{renderValue(value)}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

export default function ProfilerPage() {
  const status = usePolling<ProfileStatus>((s) => api.get("/v1/profile/status", s), 1000);
  const engines = usePolling<ProfileEngineResult>((s) => api.get("/v1/profile/engine", s), 5000);

  const [durationSeconds, setDurationSeconds] = useState<number>(10);
  const [useDuration, setUseDuration] = useState<boolean>(false);
  const [outputPath, setOutputPath] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<{ title: string; body: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const active = status.data?.active ?? false;
  const elapsed = status.data?.elapsed_seconds ?? 0;

  const start = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const body: { duration_seconds?: number; output_path?: string } = {};
      if (useDuration && durationSeconds > 0) body.duration_seconds = durationSeconds;
      if (outputPath.trim()) body.output_path = outputPath.trim();
      const res = await api.post<StartProfileResult>("/v1/start_profile", body);
      setNotice(`Capture started → ${res.output_path}`);
      status.refresh();
    } catch (e) {
      setError(explainError(e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.post<StopProfileResult>("/v1/stop_profile");
      setNotice(`Capture stopped after ${fmtDuration(res.elapsed_seconds)}.`);
      status.refresh();
    } catch (e) {
      setError(explainError(e));
    } finally {
      setBusy(false);
    }
  };

  const engineList = engines.data?.engines ?? [];

  return (
    <PageShell
      title="Profiler"
      description="Capture a Metal GPU trace (.gputrace) of the inference engine for offline analysis."
      actions={
        <Button variant="secondary" size="sm" onClick={() => status.refresh()}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      }
    >
      <div className="space-y-4">
        <Alert variant="warning" title="Inference stalls during capture">
          Starting a Metal capture serializes GPU work — active and queued generations will slow
          down or stall for the duration of the capture. Use only on a quiet server or a dedicated
          profiling run.
        </Alert>

        {/* Capture state */}
        <Card className="flex flex-wrap items-center justify-between gap-4 p-5">
          <div className="flex items-center gap-3">
            <StatusIndicator status={active ? "busy" : "neutral"} pulse={active} />
            <div>
              <div className="text-sm font-medium">
                {active ? "Capturing" : "Idle"}
                {status.loading && !status.data ? (
                  <Spinner size="sm" className="ml-2 inline-block align-middle" />
                ) : null}
              </div>
              <div className="mt-0.5 text-xs text-muted-foreground">
                {active
                  ? `Elapsed ${fmtDuration(elapsed)} (${elapsed.toFixed(1)}s)`
                  : "No capture in progress"}
              </div>
            </div>
          </div>
          {active && (
            <div className="text-3xl font-semibold tabular-nums text-foreground">
              {elapsed.toFixed(1)}
              <span className="ml-1 text-base font-normal text-muted-foreground">s</span>
            </div>
          )}
        </Card>

        {status.error && (
          <Alert variant="warning">
            Could not read capture status: {errMessage(status.error)}
          </Alert>
        )}

        {/* Start form */}
        <Card className="p-5">
          <div className="mb-4 flex items-center gap-2">
            <Camera className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Start a capture</span>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-2 flex items-center gap-2 text-sm font-medium">
                <Checkbox checked={useDuration} onCheckedChange={setUseDuration} />
                Auto-stop after (seconds)
              </label>
              <NumberInput
                min={1}
                step={1}
                value={durationSeconds}
                onChange={setDurationSeconds}
                disabled={!useDuration || active}
              />
              <p className="mt-1.5 text-xs text-muted-foreground">
                Leave unchecked to capture until you press Stop.
              </p>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium">Output path (optional)</label>
              <Input
                placeholder="/tmp/yunshu_profiles/capture.gputrace"
                value={outputPath}
                onChange={(e) => setOutputPath(e.target.value)}
                disabled={active}
              />
              <p className="mt-1.5 text-xs text-muted-foreground">
                Must be a <code>.gputrace</code> path under{" "}
                <code>/tmp/yunshu_profiles</code>. Defaults to a server-chosen path.
              </p>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <Button onClick={start} disabled={busy || active}>
              {busy && !active ? <Spinner size="sm" /> : <Play className="h-4 w-4" />} Start capture
            </Button>
            <Button variant="destructive" onClick={stop} disabled={busy || !active}>
              {busy && active ? <Spinner size="sm" /> : <Square className="h-4 w-4" />} Stop capture
            </Button>
          </div>

          {notice && (
            <Alert variant="success" className="mt-4">
              {notice}
            </Alert>
          )}
          {error && (
            <Alert variant="error" title={error.title} className="mt-4">
              {error.body}
            </Alert>
          )}
        </Card>

        {/* Engine profiling metadata */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-medium">Engines</h2>
            {engines.loading && !engines.data ? <Spinner size="sm" /> : null}
          </div>
          {engines.error ? (
            <Alert variant="warning">
              Could not load engine profiling info: {errMessage(engines.error)}
            </Alert>
          ) : engineList.length === 0 ? (
            <Card className="p-5">
              <p className="text-sm text-muted-foreground">No engines reported.</p>
            </Card>
          ) : (
            <div className={cn("grid grid-cols-1 gap-4", engineList.length > 1 && "lg:grid-cols-2")}>
              {engineList.map((engine) => (
                <EngineDetail key={engine.model_id} engine={engine} />
              ))}
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
