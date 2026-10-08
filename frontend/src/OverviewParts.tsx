import {
  Button,
  Card,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Progress,
  StatusIndicator,
} from "@yuhuanowo/yunui";
import { StatCard } from "@yuhuanowo/yunui/patterns";
import { ArrowRight, BookOpenText, ChevronDown, Zap } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { EngineStatus } from "./api";
import {
  activity,
  decodeFigures,
  decodeHeadline,
  prefillHeadline,
  type ActivityPhase,
  type Headline,
  type Totals,
} from "./engineView";
import { livePrefillTps } from "./series";
import { t, tr, useLocale } from "./i18n/index.ts";
import { clock, elapsed, fixed, number, Slot, useMinWidth } from "./ui";
import { HEALTH_TARGET, type Verdict } from "./health";

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
  useLocale();
  const a = status ? activity(status) : null;
  const last = status?.last ?? null;
  const countOf = (id: ActivityPhase) =>
    !a || id === "idle" ? 0 : a.counts[id as "queued" | "prefill" | "decode"];
  let left = t("overview.strip.waitingEngine");
  let right = "";
  let progress: number | null = null;
  if (status && a) {
    if (a.phase === "idle") {
      left = last
        ? t("overview.strip.idleLast", {
            t: clock(last.t * 1000),
            input: number(last.prompt_tokens, 0),
            output: number(last.completion_tokens, 0),
          })
        : t("overview.strip.waitingRequest");
      right =
        last?.decode_tps != null
          ? t("overview.strip.idleDecode", {
              label: t("overview.speed.last"),
              tps: fixed(last.decode_tps),
            })
          : "";
    } else if (a.phase === "queued") {
      left = t("overview.strip.queued", { n: a.counts.queued });
    } else if (a.phase === "prefill") {
      const row = a.prefilling;
      progress = a.progress;
      left = a.preparing
        ? t("overview.strip.preparing")
        : t("overview.strip.prefill", {
            pct: a.progress != null ? number(a.progress, 0) + "%" : "",
            done: number(row?.processed_tokens, 0),
            total: number(row?.prompt_tokens, 0),
          });
      right =
        row?.tokens_per_second != null
          ? row.eta_s != null
            ? t("overview.strip.prefillRate", {
                tps: number(row.tokens_per_second, 0),
                eta: elapsed(row.eta_s),
              })
            : t("overview.strip.prefillRateOnly", {
                tps: number(row.tokens_per_second, 0),
              })
          : "";
    } else {
      left =
        a.generated != null
          ? t("overview.strip.decodingN", { n: number(a.generated, 0) })
          : t("overview.strip.decoding");
      right =
        a.decodeNow != null
          ? t("overview.strip.decodeLive", {
              tps: fixed(a.decodeNow),
              live: t("overview.speed.live"),
            })
          : "";
    }
  }
  return (
    <Card
      className="relative flex flex-wrap items-center gap-x-4 gap-y-1 overflow-hidden px-4 py-2 max-sm:pb-3"
      data-testid="state-strip"
    >
      <ul
        className="flex shrink-0 items-center gap-0.5 max-sm:w-full max-sm:justify-between"
        aria-label={t("overview.strip.aria")}
      >
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
              {tr(`overview.phase.${id}`)}
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
        className="h-6 min-w-0 flex-1 truncate text-sm leading-6 tabular-nums max-sm:h-auto max-sm:basis-full max-sm:whitespace-normal"
        data-testid="state-strip-detail"
      >
        {left}
      </p>
      <Slot
        ch={22}
        align="right"
        className="min-w-0 truncate text-xs text-muted-foreground max-sm:hidden"
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
              ? t("overview.strip.prefillBarNone")
              : t("overview.strip.prefillBar", { pct: number(progress, 0) })
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
  useLocale();
  const f = decodeFigures(status);
  const decode = decodeHeadline(status);
  const prefill = prefillHeadline(status, livePrefillTps(status));
  const windowText = t("overview.speed.window", { s: Math.round(f.windowS) });
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
            <span className="block min-h-9" data-testid={testId + "-label"}>
              <span className="text-foreground">{headline.label}</span>
              {headline.note ? ` · ${headline.note}` : ""}
            </span>
            <span className="block">
              {windowText} {windowMean == null ? "—" : fixed(windowMean)}
            </span>
          </>
        }
      />
    </div>
  );
  return (
    <div className="contents" data-testid="speed-pair">
      {card(
        t("overview.speed.decodeTitle"),
        Zap,
        decode,
        f.windowMean,
        "speed-decode",
      )}
      {card(
        t("overview.speed.prefillTitle"),
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
  useLocale();
  return (
    <p
      className="mt-3 min-h-8 text-xs leading-4 text-muted-foreground"
      data-testid="totals-line"
      title={t("overview.totals.hint")}
    >
      {totals
        ? t("overview.totals.line", {
            t: clock(totals.since),
            n: totals.requests,
            input: compact(totals.promptTokens),
            cached: compact(totals.cachedTokens),
            prefill:
              totals.prefillTps != null
                ? t("overview.totals.prefill", {
                    tps: number(totals.prefillTps, 0),
                  })
                : "",
            output: compact(totals.completionTokens),
            decode:
              totals.decodeTps != null
                ? t("overview.totals.decode", { tps: fixed(totals.decodeTps) })
                : "",
          })
        : t("overview.totals.none")}
    </p>
  );
}

/**
 * The overview verdict, one line: 健康 / 注意 / 異常 and the reasons, worst
 * first (thresholds live in health.ts). A reason links to the page that fixes it.
 */
export function HealthLine({
  verdict,
  checking,
  navigate,
}: {
  verdict: Verdict;
  /** No status yet: say nothing about health rather than guess. */
  checking: boolean;
  navigate: (page: string) => void;
}) {
  useLocale();
  const { level, reasons } = verdict;
  const shown = reasons.slice(0, 3);
  const more = reasons.length - shown.length;
  const text = checking
    ? t("overview.health.checking")
    : level === "ok"
      ? t("overview.health.okDetail")
      : shown
          // i18n-keys: overview.health.reason.
          .map((r) => tr(`overview.health.reason.${r.code}`, r.vars))
          .join(t("overview.health.sep")) +
        (more > 0 ? t("overview.health.more", { n: more }) : "");
  const target = reasons[0] ? HEALTH_TARGET[reasons[0].code] : null;
  return (
    <div
      role="status"
      data-testid="health-verdict"
      data-level={checking ? "checking" : level}
      className="flex min-h-8 flex-wrap items-center gap-x-3 gap-y-1 text-sm"
    >
      <StatusIndicator
        status={
          checking || level === "ok"
            ? checking
              ? "neutral"
              : "online"
            : level === "watch"
              ? "away"
              : "offline"
        }
      />
      <span
        className={`text-sm font-semibold ${level === "bad" && !checking ? "text-error" : ""}`}
      >
        {/* i18n-keys: overview.health. */}
        {checking ? "—" : tr(`overview.health.${level}`)}
      </span>
      <span className="min-w-0 text-muted-foreground">{text}</span>
      {!checking && level !== "ok" && target && (
        <Button size="sm" variant="ghost" onClick={() => navigate(target)}>
          {t("overview.health.open")}
          <ArrowRight size={13} />
        </Button>
      )}
    </div>
  );
}

/**
 * A group of overview sections. From 768 px up it is plain content; below, it
 * folds behind one header row so the phone page is the verdict, the live state
 * and the stat grid first, with the rest a tap away.
 */
export function FoldSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  const wide = useMinWidth(768);
  const [open, setOpen] = useState(false);
  if (wide) return <>{children}</>;
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <Button
          variant="outline"
          className="h-11 w-full justify-between rounded-xl px-4"
        >
          {title}
          <ChevronDown
            size={15}
            className={`text-muted-foreground transition-transform duration-150 ${open ? "rotate-180" : ""}`}
          />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-5 pt-5">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}
