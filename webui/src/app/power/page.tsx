"use client";

import { useState, type ReactNode } from "react";
import { Button, Card, Badge, Alert, Spinner, cn, toast } from "yunui";
import { Pause, Power, MoonStar, Sun, type LucideIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, ApiError, usePolling } from "@/lib/api";

/* ------------------------------------------------------------------ *
 * Page-local types
 * ------------------------------------------------------------------ */

interface SleepStatus {
  sleeping: boolean;
  level: number;
  transitioning: boolean;
}

interface SleepLevelMeta {
  level: 0 | 1 | 2;
  title: string;
  description: string;
  icon: LucideIcon;
}

const SLEEP_LEVELS: SleepLevelMeta[] = [
  {
    level: 0,
    title: "Pause",
    description: "Freeze request handling; weights stay resident. Fastest to wake, keeps VRAM held.",
    icon: Pause,
  },
  {
    level: 1,
    title: "Unload weights",
    description: "Offload model weights to free most VRAM. Wakes by reloading weights from cache.",
    icon: MoonStar,
  },
  {
    level: 2,
    title: "Deep sleep",
    description: "Full teardown of the engine and buffers. Lowest footprint; slowest to wake.",
    icon: Power,
  },
];

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default function PowerPage() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: status, loading, refresh } = usePolling<SleepStatus>(
    (signal) => api.get<SleepStatus>("/sleep/status", signal),
    2000,
  );

  const transitioning = status?.transitioning ?? false;
  const sleeping = status?.sleeping ?? false;
  const level = status?.level ?? 0;

  const sleep = async (target: 0 | 1 | 2) => {
    setBusy(true);
    setError(null);
    try {
      await api.post("/v1/sleep", { level: target });
      toast.success(`Sleeping (L${target})`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Sleep request failed";
      setError(msg);
      toast.error("Could not sleep", msg);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const wake = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.post<{ previous_level?: number }>("/v1/wake-up");
      toast.success("Awake", r?.previous_level != null ? `Resumed from L${r.previous_level}` : undefined);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Wake request failed";
      setError(msg);
      toast.error("Could not wake", msg);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  // Can't act while a transition is in flight or a local request is pending.
  const locked = busy || transitioning;

  const stateLabel: ReactNode = transitioning
    ? "Transitioning…"
    : sleeping
      ? `Sleeping · L${level}`
      : "Awake";

  const stateTone = transitioning
    ? "text-warning"
    : sleeping
      ? "text-muted-foreground"
      : "text-success";

  return (
    <PageShell
      title="Power"
      description="Suspend or wake the inference engine to save VRAM and power when idle."
      width="narrow"
    >
      {/* Big status indicator */}
      <Card className="mb-6 flex flex-col items-center gap-3 p-8 text-center">
        <span
          className={cn(
            "flex h-16 w-16 items-center justify-center rounded-full border-2",
            transitioning
              ? "border-warning/40 text-warning"
              : sleeping
                ? "border-border text-muted-foreground"
                : "border-success/40 text-success",
          )}
        >
          {transitioning ? (
            <Spinner className="h-7 w-7" />
          ) : sleeping ? (
            <MoonStar className="h-7 w-7" />
          ) : (
            <Sun className="h-7 w-7" />
          )}
        </span>
        <div>
          <p className={cn("text-2xl font-semibold", stateTone)}>
            {loading && !status ? "…" : stateLabel}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {transitioning
              ? "The engine is changing power state — controls are locked until it settles."
              : sleeping
                ? "The engine is suspended. Wake it to resume serving requests."
                : "The engine is serving requests normally."}
          </p>
        </div>
      </Card>

      {error && (
        <Alert variant="error" title="Power change failed" className="mb-6">
          {error}
        </Alert>
      )}

      {/* Wake control */}
      <Card className="mb-6 flex flex-wrap items-center gap-4 p-5">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-success/10 text-success">
          <Sun className="h-5 w-5" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold">Wake up</h2>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Restore the engine to the serving state. Available only while sleeping.
          </p>
        </div>
        <Button
          variant="primary"
          onClick={wake}
          disabled={locked || !sleeping}
        >
          {busy ? (
            <>
              <Spinner className="h-4 w-4" /> Working…
            </>
          ) : (
            "Wake"
          )}
        </Button>
      </Card>

      {/* Sleep controls */}
      <div className="grid gap-4">
        {SLEEP_LEVELS.map(({ level: lv, title, description, icon: Icon }) => {
          const activeLevel = sleeping && level === lv;
          return (
            <Card key={lv} className="flex flex-wrap items-center gap-4 p-5">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
                <Icon className="h-5 w-5" />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold">
                    L{lv} · {title}
                  </h2>
                  {activeLevel && <Badge variant="info">current</Badge>}
                </div>
                <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
              </div>
              <Button
                variant="secondary"
                onClick={() => sleep(lv)}
                disabled={locked || activeLevel}
              >
                {busy ? (
                  <>
                    <Spinner className="h-4 w-4" /> Working…
                  </>
                ) : (
                  "Sleep"
                )}
              </Button>
            </Card>
          );
        })}
      </div>

      <p className="mt-6 text-xs text-muted-foreground">
        Sleeping is rejected (409) while there are active requests or another transition is in
        progress — the error appears above if so.
      </p>
    </PageShell>
  );
}
