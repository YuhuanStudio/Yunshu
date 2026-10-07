import { t } from "./i18n/index.ts";
import { TraceTimeline } from "@yuhuanowo/yunui/patterns";
import type { Row } from "./RequestTrace";
import { number, Readout } from "./ui";

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export const formatMs = (ms: number) =>
  ms >= 10_000
    ? `${number(ms / 1000, 1)} s`
    : ms >= 1000
      ? `${number(ms / 1000, 2)} s`
      : `${number(ms, ms < 100 ? 1 : 0)} ms`;

/** Wall-clock pieces of one finished request, all measured by the server; null when unreported. */
export function breakdown(row: Row) {
  const o = row.offsets_ms ?? {},
    admit = num(o.admit),
    first = num(o.first_token),
    last = num(o.last_token),
    done = num(o.done);
  return {
    queue: admit ?? num(row.queue_wait_ms),
    ttft: num(row.ttft_ms) ?? first,
    decode:
      first != null && (last ?? done) != null ? (last ?? done)! - first : null,
    total: done,
    admit,
    first,
    last,
    done,
  };
}

/** Time-proportional queue / prefill / decode lanes from the server's `offsets_ms`. */
export function RequestTimeline({ row }: { row: Row }) {
  const b = breakdown(row),
    reload = num(row.cache?.reload_ms);
  const end = b.done ?? b.last ?? b.first ?? b.admit;
  if (end == null || end <= 0)
    return (
      <p className="text-xs text-muted-foreground">
        {t("requests.timeline.none")}
      </p>
    );
  const spans = [];
  if (b.admit != null && b.admit > 0)
    spans.push({
      id: "queue",
      track: "t",
      start: 0,
      end: b.admit,
      label: t("requests.timeline.queue"),
      tone: "neutral" as const,
    });
  if (b.admit != null && b.first != null && b.first > b.admit)
    spans.push({
      id: "prefill",
      track: "t",
      start: b.admit,
      end: b.first,
      label: t("requests.timeline.prefill"),
      tone: "info" as const,
      detail: (
        <p className="text-xs">
          {reload != null
            ? t("requests.timeline.prefillDetailReload", {
                cached: number(row.cached_tokens, 0),
                fresh: number(
                  Math.max(
                    (row.prompt_tokens ?? 0) - (row.cached_tokens ?? 0),
                    0,
                  ),
                  0,
                ),
                ms: formatMs(reload),
                tier: row.cache?.tier ?? "—",
              })
            : t("requests.timeline.prefillDetail", {
                cached: number(row.cached_tokens, 0),
                fresh: number(
                  Math.max(
                    (row.prompt_tokens ?? 0) - (row.cached_tokens ?? 0),
                    0,
                  ),
                  0,
                ),
              })}
        </p>
      ),
    });
  const decodeEnd = b.last ?? b.done;
  if (b.first != null && decodeEnd != null && decodeEnd > b.first)
    spans.push({
      id: "decode",
      track: "t",
      start: b.first,
      end: decodeEnd,
      label: t("requests.timeline.decode"),
      tone: "success" as const,
      detail: (
        <p className="text-xs">
          {t("requests.timeline.decodeDetail", {
            tokens: number(row.completion_tokens, 0),
            tps: number(row.decode_tps),
          })}
        </p>
      ),
    });
  return (
    <TraceTimeline
      label={t("requests.timeline.label", { id: row.id })}
      tracks={[{ id: "t", label: t("requests.timeline.track") }]}
      spans={spans}
      markers={
        b.first != null
          ? [
              {
                id: "ft",
                at: b.first,
                label: t("requests.timeline.firstToken"),
              },
            ]
          : []
      }
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
  );
}

/** Header numbers: queue, TTFT, decode, total; cached vs new tokens; spec acceptance. */
export function RequestBreakdown({ row }: { row: Row }) {
  const b = breakdown(row),
    prompt = row.prompt_tokens ?? null,
    cached = row.cached_tokens ?? 0,
    spec = row.speculative;
  const ms = (v: number | null) => (v == null ? "—" : formatMs(v));
  return (
    <div className="space-y-4">
      <div className="grid min-h-[3.5rem] grid-cols-2 gap-x-5 gap-y-4">
        <Readout label={t("requests.breakdown.queue")} value={ms(b.queue)} />
        <Readout label={t("requests.breakdown.ttft")} value={ms(b.ttft)} />
        <Readout label={t("requests.breakdown.decode")} value={ms(b.decode)} />
        <Readout label={t("requests.breakdown.total")} value={ms(b.total)} />
      </div>
      <div className="grid min-h-[3.5rem] grid-cols-2 gap-x-5 gap-y-4">
        <Readout
          label={t("requests.breakdown.cachedTokens")}
          value={number(prompt == null ? null : cached, 0)}
        />
        <Readout
          label={t("requests.breakdown.newTokens")}
          value={number(
            prompt == null ? null : Math.max(prompt - cached, 0),
            0,
          )}
        />
        <Readout
          label={t("requests.breakdown.outputTokens")}
          value={number(row.completion_tokens, 0)}
        />
        <Readout
          label={t("requests.breakdown.specAccept")}
          value={
            spec?.acceptance_rate == null
              ? "—"
              : `${number(spec.acceptance_rate * 100, 0)}%`
          }
          hint={spec ? undefined : t("requests.breakdown.specOff")}
        />
      </div>
    </div>
  );
}
