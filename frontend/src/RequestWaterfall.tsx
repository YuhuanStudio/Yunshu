import { TraceTimeline } from "@yuhuanowo/yunui/patterns";
import type { Row } from "./RequestTrace";
import {
  LATENCY_STAGES,
  type RequestEnergy,
  type RequestLatency,
} from "./latency-api";
import { t, tr, useLocale } from "./i18n/index.ts";
import { formatMs } from "./RequestTimeline";
import { number } from "./ui";

/**
 * Where one request's time went, stage by stage (model lease, gateway admit, queue, template and
 * media preparation, prefix-cache restore, prefill, first decode, first send). Stages the engine
 * did not observe are listed as 未回報, never drawn as zero.
 */
export function RequestWaterfall({ row }: { row: Row }) {
  useLocale();
  // Only rows from the engine's ring carry stage data; a row the page assembled from status polls has none to show.
  if (row.source !== "ring") return null;
  if (row.latency === undefined && row.energy === undefined)
    return (
      <p
        className="text-xs text-muted-foreground"
        data-testid="waterfall-unsupported"
      >
        {t("requests.waterfall.unsupported")}
      </p>
    );
  if (!row.latency && !row.energy)
    return (
      <p
        className="text-xs text-muted-foreground"
        data-testid="waterfall-missing"
      >
        {t("requests.waterfall.missing")}
      </p>
    );
  return (
    <div className="space-y-4">
      {row.latency && <Waterfall latency={row.latency} requestId={row.id} />}
      {row.energy && <EnergyReceipt energy={row.energy} />}
    </div>
  );
}

const jFmt = (v: number | null) =>
  v == null
    ? "—"
    : v >= 100
      ? number(v, 0)
      : v >= 10
        ? number(v, 1)
        : v >= 1
          ? number(v, 2)
          : number(v, 3);

/**
 * Host energy of this request's prefill and decode windows. It is an ESTIMATE of GPU + DRAM
 * energy over the request's time, including other processes and idle power; concurrent requests
 * share it. Never labelled as the request's own power or battery use.
 */
export function EnergyReceipt({ energy }: { energy: RequestEnergy }) {
  const phases = [
    ["prefill", energy.prefill],
    ["decode", energy.decode],
  ] as const;
  return (
    <div className="space-y-1.5" data-testid="request-energy">
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        {t("requests.energy.title")}
        <span
          className="rounded-full border border-border px-1.5 py-0.5 text-[10px] leading-none"
          title={t("requests.energy.tip")}
        >
          {t("requests.energy.estimate")}
        </span>
      </p>
      {energy.state === "unknown" && !energy.prefill && !energy.decode ? (
        <p className="text-xs text-muted-foreground">
          {t("requests.energy.none")}
          {energy.reason && (
            <span className="ml-1 font-mono">{energy.reason}</span>
          )}
        </p>
      ) : (
        <ul className="space-y-1 text-sm tabular-nums">
          {phases.map(([id, p]) => (
            <li key={id} className="flex flex-wrap gap-x-3">
              <span className="w-[3.75rem] text-muted-foreground">
                {t(`requests.energy.${id}`)}
              </span>
              {p && p.state === "estimated" ? (
                <>
                  <span>{jFmt(p.joules)} J</span>
                  <span className="text-muted-foreground">
                    {p.joulesPerToken == null
                      ? t("requests.energy.noPerToken")
                      : `${jFmt(p.joulesPerToken)} J/token`}
                  </span>
                  {p.coverageRatio != null && (
                    <span className="basis-full pl-[3.75rem] text-xs text-muted-foreground">
                      {t("requests.energy.coverage", {
                        pct: number(p.coverageRatio * 100, 0),
                      })}
                      {p.extrapolatedS
                        ? t("requests.energy.extrapolated", {
                            s: number(p.extrapolatedS, 2),
                          })
                        : ""}
                    </span>
                  )}
                </>
              ) : (
                <span
                  className="text-muted-foreground"
                  title={p?.reason ?? undefined}
                >
                  {t("requests.energy.phaseUnknown")}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
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
