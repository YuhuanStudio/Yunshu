import type { EngineStatus, RequestRow } from "./api";
import { t } from "./i18n/index.ts";
import { fixed, number } from "./i18n/format.ts";

/**
 * What the engine is doing right now, and which speed figure means what.
 * One place decides the wording, so the top bar, the overview strip, the cards
 * and the CSV cannot disagree. Glossary terms (AUDIT section 8):
 *   即時合計  sum over requests decoding right now
 *   近 N 秒均值  mean per finished request over the engine's window
 *   最近一筆  the last finished request
 */
export type ActivityPhase = "idle" | "queued" | "prefill" | "decode";

// Getters, not strings: the table is read at render time, so a language switch shows at once.
export const phaseLabels: Record<ActivityPhase, string> = {
  get idle() {
    return t("shell.engine.phase.idle");
  },
  get queued() {
    return t("shell.engine.phase.queued");
  },
  get prefill() {
    return t("shell.engine.phase.prefill");
  },
  get decode() {
    return t("shell.engine.phase.decode");
  },
};

export const speedTerms = {
  get live() {
    return t("shell.engine.speed.live");
  },
  get last() {
    return t("shell.engine.speed.last");
  },
  window: (windowS: number) =>
    t("shell.engine.speed.window", { seconds: Math.round(windowS) }),
};

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

export const windowSeconds = (status: EngineStatus | null | undefined) =>
  finite(status?.throughput.window_s) && status!.throughput.window_s > 0
    ? status!.throughput.window_s
    : 60;

/** "近 60 秒", from the engine's own `throughput.window_s`; never a guess. */
export const windowLabel = (status: EngineStatus | null | undefined) =>
  t("shell.engine.window.label", {
    seconds: Math.round(windowSeconds(status)),
  });

const isPrefillRow = (row: RequestRow) =>
  row.phase === "prefill" || row.phase === "starting";

export function prefillPercent(row: RequestRow): number | null {
  if (finite(row.percent)) return row.percent;
  const prompt = row.prompt_tokens ?? 0;
  return prompt > 0 && finite(row.processed_tokens)
    ? (row.processed_tokens / prompt) * 100
    : null;
}

export interface Activity {
  /** The phase that leads the display. */
  phase: ActivityPhase;
  /** Every phase that has requests right now, so the strip can light them all. */
  lit: ActivityPhase[];
  counts: { queued: number; prefill: number; decode: number; active: number };
  /** The request whose progress is shown (first one prefilling). */
  prefilling: RequestRow | null;
  /** True when the prefilling request is still `starting` (準備中). */
  preparing: boolean;
  /** 0-100 for the prefilling request; null when the engine did not report it. */
  progress: number | null;
  /** 即時合計 decode tok/s, null while nothing decodes. */
  decodeNow: number | null;
  /** Tokens written so far by the first decoding request. */
  generated: number | null;
}

export function activity(status: EngineStatus): Activity {
  const { queued, prefill, decode, active } = status.requests;
  const items = status.requests.items;
  const prefilling = items.find(isPrefillRow) ?? null;
  const decoding = items.find((r) => r.phase === "decode");
  // Counts drive the phase; rows only add detail. A status with active > 0 but
  // no phase counts (older engines) is shown as decoding, not as idle.
  const known = queued + prefill + decode;
  const nDecode = decode + (known === 0 && active > 0 ? active : 0);
  const lit: ActivityPhase[] = [];
  if (queued > 0) lit.push("queued");
  if (prefill > 0 || (prefilling && prefill === 0 && known === 0))
    lit.push("prefill");
  if (nDecode > 0) lit.push("decode");
  if (!lit.length) lit.push("idle");
  // A prompt being read leads: it is what delays the first token.
  const phase: ActivityPhase = lit.includes("prefill")
    ? "prefill"
    : lit.includes("decode")
      ? "decode"
      : lit.includes("queued")
        ? "queued"
        : "idle";
  return {
    phase,
    lit,
    counts: { queued, prefill, decode: nDecode, active },
    prefilling,
    preparing: prefilling?.phase === "starting",
    progress: prefilling ? prefillPercent(prefilling) : null,
    decodeNow: finite(status.throughput.live_decode_tps)
      ? status.throughput.live_decode_tps
      : null,
    generated: finite(decoding?.completion_tokens)
      ? decoding!.completion_tokens!
      : null,
  };
}

/** The three decode figures, each kept apart. */
export interface DecodeFigures {
  /** 即時合計 */
  live: number | null;
  /** 近 N 秒均值 (per finished request) */
  windowMean: number | null;
  /** 最近一筆 */
  last: number | null;
  windowS: number;
}

export function decodeFigures(status: EngineStatus): DecodeFigures {
  const n = (v: unknown) => (finite(v) ? v : null);
  return {
    live: n(status.throughput.live_decode_tps),
    windowMean: n(status.throughput.mean_decode_tps),
    last: n(status.last?.decode_tps),
    windowS: windowSeconds(status),
  };
}

export interface Headline {
  /** Which of the three figures (or none) the value is. */
  kind: "live" | "last" | "none";
  value: number | null;
  /** The label that must sit with the number. */
  label: string;
  /** One honest sentence about the state. */
  note: string;
  /** One-line variant of the note (the note itself goes to the tooltip). */
  short?: string;
}

/** Decode headline: live while decoding, otherwise the last request, labelled so. */
export function decodeHeadline(status: EngineStatus): Headline {
  const a = activity(status);
  const f = decodeFigures(status);
  if (f.live != null && a.counts.decode > 0)
    return {
      kind: "live",
      value: f.live,
      label: speedTerms.live,
      note: t("shell.engine.headline.decoding", { count: a.counts.decode }),
    };
  if (a.counts.active > 0) {
    // Busy but not decoding: say which phase, never "閒置".
    const [why, short] =
      a.phase === "queued"
        ? [
            t("shell.engine.headline.whyQueued"),
            t("shell.engine.headline.shortQueued"),
          ]
        : a.preparing
          ? [
              t("shell.engine.headline.whyStarting"),
              t("shell.engine.headline.shortStarting"),
            ]
          : [
              t("shell.engine.headline.whyPrefill"),
              t("shell.engine.headline.shortPrefill"),
            ];
    return f.last != null
      ? {
          kind: "last",
          value: f.last,
          label: speedTerms.last,
          note: why,
          short,
        }
      : { kind: "none", value: null, label: speedTerms.live, note: why, short };
  }
  return f.last != null
    ? {
        kind: "last",
        value: f.last,
        label: speedTerms.last,
        note: t("shell.engine.headline.idle"),
      }
    : {
        kind: "none",
        value: null,
        label: speedTerms.last,
        note: t("shell.engine.headline.none"),
      };
}

/** Prefill headline, same rules as decode. */
export function prefillHeadline(
  status: EngineStatus,
  liveTps: number | null,
): Headline {
  const a = activity(status);
  if (liveTps != null && a.counts.prefill > 0)
    return {
      kind: "live",
      value: liveTps,
      label: speedTerms.live,
      note: t("shell.engine.headline.prefilling", { count: a.counts.prefill }),
    };
  const last = finite(status.last?.prefill_tps)
    ? status.last!.prefill_tps
    : null;
  if (a.counts.prefill > 0)
    return {
      kind: last != null ? "last" : "none",
      value: last,
      label: last != null ? speedTerms.last : speedTerms.live,
      note: t("shell.engine.headline.prefillNoSpeed"),
    };
  return last != null
    ? {
        kind: "last",
        value: last,
        label: speedTerms.last,
        note: a.counts.active > 0 ? "" : t("shell.engine.headline.idle"),
      }
    : {
        kind: "none",
        value: null,
        label: speedTerms.last,
        note: t("shell.engine.headline.none"),
      };
}

export interface LivePill {
  /** Short phase word, always one of the glossary terms or a connection state. */
  phase: string;
  /** Number or progress that goes with it; "" when there is none. */
  detail: string;
  tone: "online" | "away" | "offline" | "neutral";
}

const fix1 = (v: number) => fixed(v, 1);
const fix0 = (v: number) => number(v, 0);

/**
 * The always-visible live pill. Idle says 閒置 (the last request lives in the
 * overview, labelled 最近一筆); busy says the phase. A bare tok/s that is not
 * live never appears here.
 */
export function livePill(
  connection: "connecting" | "online" | "offline" | "unauthorized",
  status: EngineStatus | null,
): LivePill {
  if (connection === "offline")
    return {
      phase: t("shell.engine.live.offline"),
      detail: "",
      tone: "offline",
    };
  if (connection === "unauthorized")
    return {
      phase: t("shell.engine.live.unauthorized"),
      detail: "",
      tone: "offline",
    };
  if (!status)
    return {
      phase: t("shell.engine.live.connecting"),
      detail: "",
      tone: "away",
    };
  const a = activity(status);
  // A model that is loading is work the engine is doing: never "idle".
  const loading = status.models.find((m) => m.loading);
  if (a.phase === "idle" && loading)
    return {
      phase: t("shell.engine.live.loading"),
      detail: loading.id.split("/").filter(Boolean).at(-1) ?? loading.id,
      tone: "away",
    };
  if (a.phase === "idle")
    return { phase: phaseLabels.idle, detail: "", tone: "neutral" };
  if (a.phase === "queued")
    return {
      phase: phaseLabels.queued,
      detail: t("shell.engine.live.queued", { count: a.counts.queued }),
      tone: "away",
    };
  if (a.phase === "prefill") {
    const pct = a.progress != null ? `${fix0(a.progress)}%` : "";
    const decode =
      a.counts.decode > 0 && a.decodeNow != null
        ? t("shell.engine.live.decode", { tps: fix1(a.decodeNow) })
        : "";
    return {
      phase: a.preparing
        ? t("shell.engine.phase.starting")
        : phaseLabels.prefill,
      detail: [pct, decode].filter(Boolean).join(" · "),
      tone: "away",
    };
  }
  return {
    phase: phaseLabels.decode,
    detail: a.decodeNow != null ? `${fix1(a.decodeNow)} tok/s` : "",
    tone: "online",
  };
}

/** Tab title: only a live figure ever goes in it. */
export function tabTitle(status: EngineStatus | null, page: string): string {
  if (status) {
    const a = activity(status);
    if (a.phase === "decode" && a.decodeNow != null)
      return t("shell.engine.tab.decode", {
        tps: fix1(a.decodeNow),
        brand: t("shell.brand.name"),
      });
    if (a.phase === "prefill")
      return a.progress != null
        ? t("shell.engine.tab.prefillPct", {
            pct: fix0(a.progress),
            brand: t("shell.brand.name"),
          })
        : t("shell.engine.tab.prefill", { brand: t("shell.brand.name") });
  }
  return t("shell.engine.tab.page", { page, brand: t("shell.brand.name") });
}

export interface Totals {
  since: number;
  requests: number;
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
  /** Tokens read per second over requests that reported a prefill speed. */
  prefillTps: number | null;
  /** Tokens written per second over requests that reported a decode speed. */
  decodeTps: number | null;
}

/**
 * The cumulative line behind "自 HH:MM 起 N 筆請求…": totals over the finished
 * requests this page has observed, so it is a count of what was seen, not of
 * what the engine served. Rates are token-weighted (tokens / summed time), not
 * a mean of per-request rates.
 */
export function totalsFrom(
  finished: readonly {
    prompt_tokens: number;
    cached_tokens: number;
    completion_tokens: number;
    prefill_tps: number | null;
    decode_tps: number | null;
    firstObservedAt: number;
  }[],
): Totals | null {
  if (!finished.length) return null;
  let promptTokens = 0,
    cachedTokens = 0,
    completionTokens = 0,
    readTokens = 0,
    readSeconds = 0,
    writeTokens = 0,
    writeSeconds = 0;
  for (const r of finished) {
    promptTokens += r.prompt_tokens;
    cachedTokens += r.cached_tokens;
    completionTokens += r.completion_tokens;
    const fresh = r.prompt_tokens - r.cached_tokens;
    if (finite(r.prefill_tps) && r.prefill_tps > 0 && fresh > 0) {
      readTokens += fresh;
      readSeconds += fresh / r.prefill_tps;
    }
    if (finite(r.decode_tps) && r.decode_tps > 0 && r.completion_tokens > 0) {
      writeTokens += r.completion_tokens;
      writeSeconds += r.completion_tokens / r.decode_tps;
    }
  }
  return {
    since: finished[0].firstObservedAt,
    requests: finished.length,
    promptTokens,
    cachedTokens,
    completionTokens,
    prefillTps: readSeconds > 0 ? readTokens / readSeconds : null,
    decodeTps: writeSeconds > 0 ? writeTokens / writeSeconds : null,
  };
}
