import { useEffect, useState } from "react";
import { TraceTimeline } from "@yuhuanowo/yunui/patterns";
import type { Connection } from "./api";
import {
  fetchRecentLatency,
  LATENCY_STAGES,
  type RecentLatency,
  type RequestLatency,
} from "./latency-api";
import { t, tr, useLocale } from "./i18n/index.ts";
import { formatMs } from "./RequestTimeline";

const CACHE_MS = 5_000;
const cache = new Map<string, { at: number; value: RecentLatency }>();

/** The engine's recent-request latency table, shared by every open detail for a few seconds. */
function useRecentLatency(connection: Connection): RecentLatency | null {
  const key = `${connection.baseUrl}|${connection.token}`;
  const [value, setValue] = useState<RecentLatency | null>(
    () => cache.get(key)?.value ?? null,
  );
  useEffect(() => {
    const hit = cache.get(key);
    if (hit && Date.now() - hit.at < CACHE_MS) {
      setValue(hit.value);
      return;
    }
    const controller = new AbortController();
    void fetchRecentLatency(connection, controller.signal)
      .then((v) => {
        if (controller.signal.aborted) return;
        cache.set(key, { at: Date.now(), value: v });
        setValue(v);
      })
      .catch(() => {
        // A transient failure shows nothing rather than a false "unsupported".
      });
    return () => controller.abort();
  }, [key, connection]);
  return value;
}

/**
 * Where one request's time went, stage by stage (model lease, gateway admit, queue, template and
 * media preparation, prefix-cache restore, prefill, first decode, first send). Stages the engine
 * did not observe are listed as 未回報, never drawn as zero.
 */
export function RequestWaterfall({
  connection,
  requestId,
}: {
  connection: Connection;
  requestId: string;
}) {
  useLocale();
  const recent = useRecentLatency(connection);
  if (!recent) return null;
  if (recent.kind === "unsupported")
    return (
      <p
        className="text-xs text-muted-foreground"
        data-testid="waterfall-unsupported"
      >
        {t("requests.waterfall.unsupported")}
      </p>
    );
  const latency = recent.byId.get(requestId);
  if (!latency)
    return (
      <p
        className="text-xs text-muted-foreground"
        data-testid="waterfall-missing"
      >
        {t("requests.waterfall.missing")}
      </p>
    );
  return <Waterfall latency={latency} requestId={requestId} />;
}

export function Waterfall({
  latency,
  requestId,
}: {
  latency: RequestLatency;
  requestId: string;
}) {
  const stageName = (id: string) => tr(`requests.waterfall.stage.${id}`);
  const unreported = LATENCY_STAGES.filter(
    (s) => latency.durations[s.id] == null,
  );
  const end = Math.max(0, ...latency.spans.map((s) => s.end));
  return (
    <div className="space-y-2" data-testid="request-waterfall">
      <p className="text-xs text-muted-foreground">
        {t("requests.waterfall.title")}
      </p>
      {end > 0 ? (
        <TraceTimeline
          label={t("requests.waterfall.label", { id: requestId })}
          tracks={latency.spans.map((s) => ({
            id: s.id,
            label: stageName(s.id),
          }))}
          spans={latency.spans.map((s) => ({
            id: s.id,
            track: s.id,
            start: s.start,
            end: s.end,
            label: `${stageName(s.id)} ${formatMs(s.end - s.start)}`,
            tone:
              s.id === "prefill"
                ? ("info" as const)
                : s.id === "engine_queue" || s.id === "model_lease"
                  ? ("warning" as const)
                  : ("neutral" as const),
          }))}
          markers={[]}
          duration={end}
          formatTime={formatMs}
          labels={{
            view: t("requests.timeline.view"),
            timeline: t("requests.timeline.timeline"),
            table: t("requests.timeline.table"),
            track: t("requests.timeline.trackCol"),
            span: t("requests.timeline.span"),
            start: t("requests.timeline.start"),
            end: t("requests.timeline.end"),
            duration: t("requests.timeline.duration"),
            marker: t("requests.timeline.marker"),
            running: t("requests.timeline.running"),
            expand: t("requests.timeline.expand"),
          }}
        />
      ) : (
        <p className="text-xs text-muted-foreground">
          {t("requests.waterfall.empty")}
        </p>
      )}
      {unreported.length > 0 && (
        <p
          className="text-xs text-muted-foreground"
          data-testid="waterfall-null"
        >
          {t("requests.waterfall.unreported", {
            stages: unreported
              .map((s) => stageName(s.id))
              .join(t("requests.waterfall.sep")),
          })}
        </p>
      )}
      <p className="text-xs text-muted-foreground">
        {t("requests.waterfall.note")}
      </p>
    </div>
  );
}
