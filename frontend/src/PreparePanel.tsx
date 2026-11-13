import { useCallback, useEffect, useState } from "react";
import { Button, Progress, StatusIndicator } from "@yuhuanowo/yunui";
import { Rocket } from "lucide-react";
import { t } from "./i18n/index.ts";
import type { Connection } from "./api.ts";
import {
  cancelOrRemovePrepare,
  diskShortfall,
  getPrepare,
  startPrepare,
  UNSUPPORTED,
  type PrepareStatus,
} from "./admin-models-api.ts";
import { ByteValue } from "./ByteValue.tsx";
import { elapsed, SectionCard } from "./ui";

/** Faster cold start: the one-time PLE repack of a Flash-Next model, with progress, cancel and remove. */
export function PreparePanel({
  connection,
  modelId,
  online,
  loaded,
}: {
  connection: Connection;
  modelId: string;
  online: boolean;
  loaded: boolean;
}) {
  const [status, setStatus] = useState<PrepareStatus | null>(null),
    [error, setError] = useState<string | null>(null),
    [hidden, setHidden] = useState(false);
  const running = status?.job?.state === "running";
  const refresh = useCallback(async () => {
    try {
      const s = await getPrepare(connection, modelId);
      if (s === UNSUPPORTED || !s.applicable) setHidden(true);
      else setStatus(s);
    } catch {
      // a failed poll keeps the last view
    }
  }, [connection, modelId]);
  useEffect(() => {
    if (!online) return;
    void refresh();
    if (!running) return;
    const timer = setInterval(() => void refresh(), 1500);
    return () => clearInterval(timer);
  }, [online, running, refresh]);
  if (hidden || !status) return null;
  const act = async (call: typeof startPrepare) => {
    setError(null);
    try {
      setStatus(await call(connection, modelId));
    } catch (e) {
      const short = diskShortfall(e);
      setError(
        short
          ? t("models.prepare.noDisk", {
              needed: Math.ceil((short.needed ?? 0) / 2 ** 30),
              free: Math.floor((short.free ?? 0) / 2 ** 30),
            })
          : e instanceof Error
            ? e.message
            : String(e),
      );
    }
  };
  const job = status.job,
    pct =
      job && job.bytesTotal > 0
        ? Math.min(100, (job.bytesDone / job.bytesTotal) * 100)
        : null;
  return (
    <SectionCard
      icon={Rocket}
      title={t("models.prepare.title")}
      description={t("models.prepare.description")}
      data-testid="prepare-card"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <StatusIndicator status={status.prepared ? "online" : "neutral"}>
            <span className="text-foreground" data-testid="prepare-badge">
              {status.prepared
                ? t("models.prepare.prepared")
                : t("models.prepare.notPrepared")}
            </span>
          </StatusIndicator>
          <span className="text-sm text-muted-foreground">
            {t("models.prepare.storage", { storage: status.storage ?? "—" })}
          </span>
          <div className="ml-auto flex gap-2">
            {running ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void act(cancelOrRemovePrepare)}
              >
                {t("models.prepare.cancel")}
              </Button>
            ) : status.prepared ? (
              <Button
                variant="secondary"
                size="sm"
                disabled={!online || loaded}
                title={loaded ? t("models.prepare.unloadFirst") : undefined}
                onClick={() => void act(cancelOrRemovePrepare)}
              >
                {t("models.prepare.remove")}
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={!online}
                onClick={() => void act(startPrepare)}
              >
                {t("models.prepare.start")}
              </Button>
            )}
          </div>
        </div>
        {running && (
          <div className="space-y-1.5">
            <Progress
              className="h-1.5"
              {...(pct != null ? { value: pct } : { indeterminate: true })}
              label={t("models.prepare.progressAria")}
            />
            <p className="text-xs text-muted-foreground tabular-nums">
              <ByteValue bytes={job?.bytesDone} className="text-foreground" />
              {" / "}
              <ByteValue bytes={job?.bytesTotal} />
              {" · "}
              {t("models.prepare.eta", {
                time: job?.etaS == null ? "—" : elapsed(job.etaS),
              })}
            </p>
          </div>
        )}
        {!status.prepared && !running && (
          <p className="text-xs text-muted-foreground">
            {t("models.prepare.cost", {
              size: Math.round((status.tableBytes ?? 0) / 1e9),
            })}
          </p>
        )}
        {job?.state === "failed" && !running && (
          <p role="alert" className="text-sm text-error">
            {t("models.prepare.failed", { error: job.error ?? "—" })}
          </p>
        )}
        {error && (
          <p role="alert" className="text-sm text-error">
            {error}
          </p>
        )}
      </div>
    </SectionCard>
  );
}
