import { Progress, SegmentedBar, StatusIndicator } from "@yuhuanowo/yunui";
import { t } from "./i18n/index.ts";
import type { EngineHistoryPoint } from "./useEngine";
import { number, phaseDot, Readout } from "./ui";
import { diagnose, type Cause } from "./request-insight";
import { formatMs } from "./RequestTimeline";

/** One request as the console sees it: a live item, a detail poll or a `last` record. */
export type Row = {
  id: string;
  phase: string;
  model?: string;
  elapsed_s?: number;
  prompt_tokens?: number;
  cached_tokens?: number;
  processed_tokens?: number;
  completion_tokens?: number;
  percent?: number;
  tokens_per_second?: number | null;
  eta_s?: number | null;
  queue_position?: number;
  queue_est_wait_ms?: number;
  ttft_ms?: number | null;
  decode_tps?: number | null;
  prefill_tps?: number | null;
  speculative?: {
    mode?: string;
    acceptance_rate?: number | null;
    rounds?: number;
  } | null;
  t?: number;
  path?: string;
  /** Finished-request ring fields (`GET /v1/yunshu/requests/recent`). */
  outcome?: Outcome;
  status_code?: number | null;
  finish_reason?: string | null;
  offsets_ms?: Offsets | null;
  cache?: { tier?: string | null; reload_ms?: number | null } | null;
  queue_wait_ms?: number | null;
  stream?: boolean | null;
  /** Epoch seconds the request arrived. */
  t0_wall?: number | null;
  /** Where a finished row came from: the server ring or the page's own status samples. */
  source?: "ring" | "sampled";
};
export type Offsets = {
  arrive?: number | null;
  admit?: number | null;
  first_token?: number | null;
  last_token?: number | null;
  done?: number | null;
};
export type Outcome = "completed" | "cancelled" | "error";

/** Label for a reported phase; an unknown phase is shown as reported. */
export const phaseLabel = (phase: string): string =>
  phase === "queued"
    ? t("requests.phase.queued")
    : phase === "starting"
      ? t("requests.phase.starting")
      : phase === "prefill"
        ? t("requests.phase.prefill")
        : phase === "decode"
          ? t("requests.phase.decode")
          : phase === "complete"
            ? t("requests.phase.complete")
            : phase;
export const isLive = (phase: string) =>
  phase === "decode" || phase === "prefill" || phase === "starting";

/** Prefill progress in percent, from what the API actually reports. */
export function prefillPercent(row: Row): number | null {
  if (row.percent != null) return Math.max(0, Math.min(100, row.percent));
  const prompt = row.prompt_tokens ?? 0;
  return prompt > 0 && row.processed_tokens != null
    ? Math.max(0, Math.min(100, (row.processed_tokens / prompt) * 100))
    : null;
}

/** Distinct finished requests in the sampled history, oldest first (dedup by id). */
export function finishedFromHistory(
  history: readonly EngineHistoryPoint[],
): Row[] {
  const seen = new Map<string, Row>();
  for (const sample of history) {
    const last = sample.status.last;
    if (last && !seen.has(last.request_id))
      seen.set(last.request_id, {
        ...last,
        id: last.request_id,
        phase: "complete",
      } as Row);
  }
  return [...seen.values()];
}

export function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b),
    mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export const speculativeText = (row: Row) => {
  const spec = row.speculative;
  if (!spec) return null;
  const rate =
    spec.acceptance_rate == null
      ? null
      : `${number(spec.acceptance_rate * 100, 0)}%`;
  return [
    spec.mode ?? "speculative", // i18n-ignore
    rate && t("requests.trace.speculativeAccept", { rate }),
  ]
    .filter(Boolean)
    .join(" · ");
};

const stages = ["queued", "prefill", "decode"] as const;
const stageIndex = (phase: string) =>
  phase === "queued"
    ? 0
    : phase === "starting" || phase === "prefill"
      ? 1
      : phase === "decode"
        ? 2
        : phase === "complete"
          ? 3
          : -1;

/** Queue → prefill → decode, marked done / current / pending from the reported phase only. */
export function StageRail({ phase }: { phase: string }) {
  const at = stageIndex(phase);
  return (
    <ol
      className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs"
      aria-label={t("requests.trace.stagesLabel")}
    >
      {stages.map((stage, index) => {
        const state =
          at < 0
            ? "pending"
            : index < at
              ? "done"
              : index === at
                ? "current"
                : "pending";
        return (
          <li
            key={stage}
            aria-current={state === "current" ? "step" : undefined}
          >
            <StatusIndicator
              status={
                state === "done"
                  ? "online"
                  : state === "current"
                    ? phaseDot(phase)
                    : "neutral"
              }
            >
              <span
                className={
                  state === "pending"
                    ? "text-muted-foreground"
                    : "text-foreground"
                }
              >
                {phaseLabel(stage)}
                {state === "current" && phase === "starting"
                  ? t("requests.trace.startingNote")
                  : ""}
              </span>
            </StatusIndicator>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * Token proportions of one request: cache-hit prompt, newly computed prompt,
 * generated output. Proportions are token counts; the API gives no per-phase
 * durations, so this is deliberately not a time axis.
 */
export function TokenTrace({ row }: { row: Row }) {
  const prompt = row.prompt_tokens,
    cached = Math.min(row.cached_tokens ?? 0, prompt ?? 0),
    output = row.completion_tokens ?? 0;
  if (prompt == null || prompt <= 0)
    return (
      <p className="text-xs text-muted-foreground">
        {t("requests.trace.noPromptTokens")}
      </p>
    );
  const computed = prompt - cached;
  return (
    <div className="space-y-2">
      <SegmentedBar
        label={t("requests.trace.compositionLabel", {
          cached,
          computed,
          output,
        })}
        height={10}
        legend
        formatValue={(value) => `${number(value, 0)} tok`}
        segments={[
          { value: cached, tone: "info", label: t("requests.trace.segCached") },
          {
            value: computed,
            tone: "neutral",
            label: t("requests.trace.segComputed"),
          },
          {
            value: output,
            tone: "success",
            label: t("requests.trace.segOutput"),
          },
        ]}
      />
      <p className="text-xs text-muted-foreground">
        {t("requests.trace.proportionNote")}
      </p>
    </div>
  );
}

export function PrefillMeter({ row }: { row: Row }) {
  const percent = prefillPercent(row);
  if (percent == null) return null;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">
          {t("requests.trace.prefillProgress")}
        </span>
        <span className="tabular-nums">
          {number(percent)}%
          {row.eta_s != null && (
            <span className="ml-2 text-muted-foreground">
              {t("requests.trace.eta", { value: number(row.eta_s) })}
            </span>
          )}
        </span>
      </div>
      <Progress value={percent} label={t("requests.trace.prefillProgress")} />
    </div>
  );
}

/** 41,000 reads as "41K" above ten thousand, so a long prompt is one glance. */
const tokens = (n: number) =>
  n >= 10_000 ? `${number(n / 1000, 0)}K` : number(n, 0);

/** The one-sentence cause of a finished request; rules and thresholds live in request-insight.ts. */
export function causeText(cause: Cause): string {
  switch (cause.kind) {
    case "unknown":
      return t("requests.cause.unknown");
    case "error":
      return t("requests.cause.error", {
        code: cause.code ?? "—",
        reason: cause.reason ?? "—",
      });
    case "cancelled":
      return cause.totalMs == null
        ? t("requests.cause.cancelledNoTime")
        : t("requests.cause.cancelled", { total: formatMs(cause.totalMs) });
    case "fast":
      return t("requests.cause.fast", { total: formatMs(cause.totalMs) });
    case "queue":
      return t("requests.cause.queue", {
        wait: formatMs(cause.ms),
        share: number(cause.share * 100, 0),
      });
    case "prefillMiss":
      return t("requests.cause.prefillMiss", {
        fresh: tokens(cause.fresh),
        hit: number(cause.hitPercent, 0),
        time: formatMs(cause.ms),
      });
    case "prefillReload":
      return t("requests.cause.prefillReload", {
        reload: formatMs(cause.reloadMs),
        tier: cause.tier ?? "—",
        time: formatMs(cause.ms),
      });
    case "prefill":
      return t("requests.cause.prefill", {
        fresh: tokens(cause.fresh),
        time: formatMs(cause.ms),
      });
    case "decodeSpec":
      return t("requests.cause.decodeSpec", {
        rate: number(cause.acceptPercent, 0),
        time: formatMs(cause.ms),
      });
    case "decodeSlow":
      return t("requests.cause.decodeSlow", {
        tps: number(cause.tps),
        time: formatMs(cause.ms),
      });
    case "decodeLong":
      return t("requests.cause.decodeLong", {
        tokens: tokens(cause.tokens),
        time: formatMs(cause.ms),
      });
    case "balanced":
      return t("requests.cause.balanced", { total: formatMs(cause.totalMs) });
  }
}

/** The cause sentence for a finished row, with the dominant-stage wording or an honest "unknown". */
export const causeOf = (row: Row) => causeText(diagnose(row));
