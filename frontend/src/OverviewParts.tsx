import { Card, Progress, StatusIndicator } from "@yuhuanowo/yunui";
import { StatCard } from "@yuhuanowo/yunui/patterns";
import { BookOpenText, Zap } from "lucide-react";
import type { EngineStatus } from "./api";
import {
  activity,
  decodeFigures,
  decodeHeadline,
  phaseLabels,
  prefillHeadline,
  speedTerms,
  windowLabel,
  type ActivityPhase,
  type Headline,
  type Totals,
} from "./engineView";
import { livePrefillTps } from "./series";
import { clock, elapsed, fixed, number, Slot } from "./ui";

const order: ActivityPhase[] = ["idle", "queued", "prefill", "decode"];
const dot = (phase: ActivityPhase, lit: boolean) =>
  !lit
    ? "neutral"
    : phase === "decode"
      ? "online"
      : phase === "idle"
        ? "neutral"
        : "away";

/**
 * Every phase is always on screen and the active ones are lit, so a request
 * moving from queue to prefill to decode changes emphasis, never layout. One
 * progress bar follows the prompt being read.
 */
export function StateStrip({ status }: { status: EngineStatus | null }) {
  const a = status ? activity(status) : null;
  const last = status?.last ?? null;
  const countOf = (id: ActivityPhase) =>
    !a || id === "idle" ? 0 : a.counts[id as "queued" | "prefill" | "decode"];
  let left = "等待引擎回應";
  let right = "";
  let progress: number | null = null;
  if (status && a) {
    if (a.phase === "idle") {
      left = last
        ? `等待請求 · 最近一筆 ${clock(last.t * 1000)} · ${number(last.prompt_tokens, 0)} 輸入 / ${number(last.completion_tokens, 0)} 輸出 token`
        : "等待請求";
      right =
        last?.decode_tps != null
          ? `${speedTerms.last} 解碼 ${fixed(last.decode_tps)} tok/s`
          : "";
    } else if (a.phase === "queued") {
      left = `${a.counts.queued} 個請求排隊中`;
    } else if (a.phase === "prefill") {
      const row = a.prefilling;
      progress = a.progress;
      left = a.preparing
        ? "準備中，尚未開始讀取提示詞"
        : `預填 ${a.progress != null ? number(a.progress, 0) + "%" : ""} · ${number(row?.processed_tokens, 0)} / ${number(row?.prompt_tokens, 0)} token`;
      right =
        row?.tokens_per_second != null
          ? `${number(row.tokens_per_second, 0)} tok/s${row.eta_s != null ? ` · 約 ${elapsed(row.eta_s)} 後開始輸出` : ""}`
          : "";
    } else {
      left = `解碼中${a.generated != null ? ` · 已輸出 ${number(a.generated, 0)} token` : ""}`;
      right =
        a.decodeNow != null
          ? `${fixed(a.decodeNow)} tok/s ${speedTerms.live}`
          : "";
    }
  }
  return (
    <Card
      className="relative flex flex-wrap items-center gap-x-4 gap-y-1 overflow-hidden px-4 py-2"
      data-testid="state-strip"
    >
      <ul className="flex shrink-0 items-center gap-0.5" aria-label="引擎階段">
        {order.map((id) => {
          const lit = !!a && a.lit.includes(id);
          return (
            <li
              key={id}
              data-phase={id}
              data-lit={lit ? "true" : "false"}
              aria-current={a?.phase === id ? "true" : undefined}
              className={`inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-xs transition-opacity duration-150 ${lit ? "bg-(--bg-elevated) text-foreground" : "text-muted-foreground opacity-50"}`}
            >
              <StatusIndicator status={dot(id, lit)} />
              {phaseLabels[id]}
              {id !== "idle" && (
                <Slot ch={1} align="right" className="tabular-nums">
                  {lit ? countOf(id) : ""}
                </Slot>
              )}
            </li>
          );
        })}
      </ul>
      <p
        className="h-6 min-w-0 flex-1 truncate text-sm leading-6 tabular-nums"
        data-testid="state-strip-detail"
      >
        {left}
      </p>
      <Slot
        ch={22}
        align="right"
        className="min-w-0 truncate text-xs text-muted-foreground"
      >
        {right}
      </Slot>
      {/* The bar belongs to prefill only; it sits on the card edge, so it never moves the row. */}
      {a?.phase === "prefill" && (
        <Progress
          className="absolute inset-x-0 bottom-0 h-0.5 rounded-none"
          value={progress == null ? 0 : Math.max(0, Math.min(100, progress))}
          label={
            progress == null
              ? "預填進度，尚無進度回報"
              : `預填 ${number(progress, 0)}%`
          }
        />
      )}
    </Card>
  );
}

/**
 * Decode and prefill as two stat cards of the same grid. Each number carries its
 * own label (即時合計 / 最近一筆); the window mean is a separate, labelled line.
 */
export function SpeedPair({ status }: { status: EngineStatus }) {
  const f = decodeFigures(status);
  const decode = decodeHeadline(status);
  const prefill = prefillHeadline(status, livePrefillTps(status));
  const windowText = speedTerms.window(f.windowS);
  const card = (
    term: string,
    icon: typeof Zap,
    headline: Headline,
    windowMean: number | null,
    testId: string,
  ) => (
    <div className="contents" data-testid={testId}>
      <StatCard
        compact
        valueFirst
        icon={icon}
        label={term}
        value={
          <>
            {headline.value == null ? "—" : fixed(headline.value)}
            <span className="ml-1 text-xs font-normal text-muted-foreground">
              tok/s
            </span>
          </>
        }
        subtext={
          <>
            <span className="block truncate" data-testid={testId + "-label"}>
              <span className="text-foreground">{headline.label}</span>
              {headline.note ? ` · ${headline.note}` : ""}
            </span>
            <span className="block truncate">
              {windowText} {windowMean == null ? "—" : fixed(windowMean)} tok/s
            </span>
          </>
        }
      />
    </div>
  );
  return (
    <div className="contents" data-testid="speed-pair">
      {card("解碼", Zap, decode, f.windowMean, "speed-decode")}
      {card(
        "預填",
        BookOpenText,
        prefill,
        status.throughput.mean_prefill_tps,
        "speed-prefill",
      )}
    </div>
  );
}

const compact = (n: number) =>
  n >= 1_000_000
    ? `${number(n / 1_000_000, 1)}M`
    : n >= 10_000
      ? `${number(n / 1000, 0)}k`
      : number(n, 0);

/** "自 HH:MM 起 N 筆請求…": what this page has seen finish, with token-weighted rates. */
export function TotalsLine({ totals }: { totals: Totals | null }) {
  return (
    <p
      className="mt-3 min-h-8 text-xs leading-4 text-muted-foreground"
      data-testid="totals-line"
      title="只計入本頁開啟後觀測到的已結束請求，不是引擎的全部流量。"
    >
      {totals
        ? `自 ${clock(totals.since)} 起 ${number(totals.requests, 0)} 筆請求 · 輸入 ${compact(totals.promptTokens)} token（命中 ${compact(totals.cachedTokens)}）${totals.prefillTps != null ? `，預填 ${number(totals.prefillTps, 0)} tok/s` : ""} · 輸出 ${compact(totals.completionTokens)} token${totals.decodeTps != null ? `，解碼 ${fixed(totals.decodeTps)} tok/s` : ""}`
        : "本頁開啟後尚未觀測到已結束的請求。"}
    </p>
  );
}
