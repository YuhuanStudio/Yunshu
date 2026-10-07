import { Card, Progress, StatusIndicator } from "@yuhuanowo/yunui";
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
import { livePrefillTps, type SeriesPoint } from "./series";
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
    <Card className="p-4" data-testid="state-strip">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <ul className="flex items-center gap-1" aria-label="引擎階段">
          {order.map((id) => {
            const lit = !!a && a.lit.includes(id);
            return (
              <li
                key={id}
                data-phase={id}
                data-lit={lit ? "true" : "false"}
                aria-current={a?.phase === id ? "true" : undefined}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-opacity duration-150 ${lit ? "bg-(--bg-elevated) text-foreground" : "text-muted-foreground opacity-50"}`}
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
        <Slot
          ch={22}
          align="right"
          className="min-w-0 truncate text-xs text-muted-foreground"
        >
          {right}
        </Slot>
      </div>
      <p
        className="mt-3 h-5 truncate text-sm tabular-nums"
        data-testid="state-strip-detail"
      >
        {left}
      </p>
      <Progress
        className="mt-2 h-1.5"
        value={progress == null ? 0 : Math.max(0, Math.min(100, progress))}
        label={
          progress == null
            ? "預填進度，尚無進度回報"
            : `預填 ${number(progress, 0)}%`
        }
      />
    </Card>
  );
}

/**
 * Two series on one time axis and one value scale, broken wherever a sample is
 * missing. Idle stretches are gaps, not zeros.
 */
export function PairSparkline({
  points,
  label,
}: {
  points: readonly SeriesPoint[];
  label: string;
}) {
  const W = 200,
    H = 40;
  const t0 = points.length ? points[0].at : 0;
  const t1 = points.length ? points[points.length - 1].at : 1;
  const span = Math.max(1, t1 - t0);
  let max = 0;
  for (const p of points) {
    if (p.decode != null && p.decode > max) max = p.decode;
    if (p.prefill != null && p.prefill > max) max = p.prefill;
  }
  const path = (pick: (p: SeriesPoint) => number | null) => {
    let d = "";
    let open = false;
    for (const p of points) {
      const v = pick(p);
      if (v == null || max <= 0) {
        open = false;
        continue;
      }
      const x = ((p.at - t0) / span) * W;
      const y = H - 2 - (v / max) * (H - 4);
      d += `${open ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)} `;
      open = true;
    }
    return d;
  };
  const decode = path((p) => p.decode),
    prefill = path((p) => p.prefill);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={label}
      className="h-10 w-full"
      data-testid="speed-sparkline"
    >
      <line
        x1="0"
        x2={W}
        y1={H - 1}
        y2={H - 1}
        stroke="currentColor"
        strokeOpacity=".15"
        vectorEffect="non-scaling-stroke"
      />
      {prefill && (
        <path
          d={prefill}
          fill="none"
          stroke="currentColor"
          strokeOpacity=".4"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          className="text-muted-foreground"
        />
      )}
      {decode && (
        <path
          d={decode}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          className="text-foreground"
        />
      )}
    </svg>
  );
}

function Figure({
  term,
  headline,
  windowMean,
  windowText,
  testId,
}: {
  term: string;
  headline: Headline;
  windowMean: number | null;
  windowText: string;
  testId: string;
}) {
  return (
    <div className="min-w-0" data-testid={testId}>
      <p className="text-xs text-muted-foreground">{term}</p>
      <p className="mt-1 yunui-stat-value text-2xl tabular-nums">
        <Slot ch={6}>
          {headline.value == null ? "—" : fixed(headline.value)}
        </Slot>
        <span className="ml-1.5 text-xs font-normal text-muted-foreground">
          tok/s
        </span>
      </p>
      <p
        className="mt-0.5 truncate text-xs text-muted-foreground"
        data-testid={testId + "-label"}
      >
        <span className="text-foreground">{headline.label}</span>
        {headline.note ? ` · ${headline.note}` : ""}
      </p>
      <p className="truncate text-xs text-muted-foreground">
        {windowText} {windowMean == null ? "—" : fixed(windowMean)} tok/s
      </p>
    </div>
  );
}

/**
 * Decode and prefill side by side in one card with one shared sparkline. Each
 * number carries its own label (即時合計 / 最近一筆); window means are a
 * separate, labelled line, never the headline.
 */
export function SpeedPair({
  status,
  points,
}: {
  status: EngineStatus;
  points: readonly SeriesPoint[];
}) {
  const f = decodeFigures(status);
  const decode = decodeHeadline(status);
  const prefill = prefillHeadline(status, livePrefillTps(status));
  const windowText = speedTerms.window(f.windowS);
  return (
    <div className="card p-4" data-testid="speed-pair">
      <div className="grid grid-cols-2 gap-4">
        <Figure
          term="解碼"
          headline={decode}
          windowMean={f.windowMean}
          windowText={windowText}
          testId="speed-decode"
        />
        <Figure
          term="預填"
          headline={prefill}
          windowMean={status.throughput.mean_prefill_tps}
          windowText={windowText}
          testId="speed-prefill"
        />
      </div>
      <div className="mt-3">
        <PairSparkline
          points={points}
          label={`${windowLabel(status)}內解碼與預填的即時合計速度走勢，空白處代表當時沒有請求`}
        />
      </div>
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
