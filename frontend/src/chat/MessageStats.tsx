import { Fragment, type ReactNode } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@yuhuanowo/yunui";
import { HugWidth } from "../HugWidth";
import { LiveNumber } from "../LiveNumber";
import { fixed, number } from "../i18n/format.ts";
import { t } from "../i18n/index.ts";
import type { RunStats } from "../playground-metrics";

const sec = (ms: number, digits = 1) => `${number(ms / 1000, digits)} s`;
const ttftText = (ms: number) =>
  ms < 1000 ? `${number(ms, 0)} ms` : sec(ms, 2);

/** A message's numbers on the header line, beside the model name: muted text split by "·" (input, output,
 *  speed, first token, total, thinking time), like the mainstream chat apps. While `live`, the changing
 *  numbers are LiveNumber in HugWidth boxes (a readable pace, no jitter); the whole line is focusable and
 *  its tooltip carries the full detail (prefix hit, exact milliseconds, speculative mode, estimates). */
export function MessageStats({
  stats,
  live,
  thinkMs,
  extra,
  jumpKey,
}: {
  stats: RunStats;
  live: boolean;
  thinkMs?: number;
  /** One more fact for the tooltip (speculative decoding as the engine reported it). */
  extra?: string;
  jumpKey: string | number;
}) {
  const parts: ReactNode[] = [];
  if (thinkMs != null && thinkMs >= 100 && !live)
    parts.push(t("playground.stat.think", { s: number(thinkMs / 1000, 1) }));
  if (stats.promptTokens != null)
    parts.push(t("playground.stat.in", { n: number(stats.promptTokens, 0) }));
  if (stats.tokens > 0)
    parts.push(
      live ? (
        <HugWidth key="out">
          {t("playground.stat.out", { n: "" }).trimEnd()}{" "}
          {stats.estimated ? "~" : ""}
          <LiveNumber value={stats.tokens} jumpKey={jumpKey} />
        </HugWidth>
      ) : (
        t("playground.stat.out", {
          n: `${stats.estimated ? "~" : ""}${number(stats.tokens, 0)}`,
        })
      ),
    );
  if (stats.tokensPerSecond !== undefined)
    parts.push(
      live ? (
        <HugWidth key="speed">
          <LiveNumber
            value={stats.tokensPerSecond}
            format={(v) => fixed(v)}
            jumpKey={jumpKey}
          />{" "}
          tok/s
        </HugWidth>
      ) : (
        `${fixed(stats.tokensPerSecond)} tok/s`
      ),
    );
  if (stats.ttftMs !== undefined)
    parts.push(t("playground.stat.ttftShort", { v: ttftText(stats.ttftMs) }));
  parts.push(
    live ? (
      <HugWidth key="total">
        <LiveNumber
          value={stats.latencyMs}
          format={(v) => sec(v)}
          jumpKey={jumpKey}
        />
      </HugWidth>
    ) : (
      sec(stats.latencyMs)
    ),
  );

  const detail: [string, string][] = [];
  if (thinkMs != null)
    detail.push([t("playground.stat.thinkTime"), sec(thinkMs, 2)]);
  if (stats.promptTokens != null)
    detail.push([t("playground.stat.prompt"), number(stats.promptTokens, 0)]);
  if (stats.cachedTokens != null)
    detail.push([
      t("playground.stat.cached"),
      stats.promptTokens
        ? `${number(stats.cachedTokens, 0)} / ${number(stats.promptTokens, 0)}`
        : number(stats.cachedTokens, 0),
    ]);
  detail.push([
    t("playground.stat.completion"),
    `${stats.estimated ? "~" : ""}${number(stats.tokens, 0)}`,
  ]);
  if (stats.tokensPerSecond !== undefined)
    detail.push([
      t("playground.stat.speed"),
      `${fixed(stats.tokensPerSecond, 2)} tok/s`,
    ]);
  if (stats.ttftMs !== undefined)
    detail.push([t("playground.stat.ttft"), `${number(stats.ttftMs, 0)} ms`]);
  detail.push([t("playground.stat.total"), `${number(stats.latencyMs, 0)} ms`]);

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            tabIndex={0}
            role="group"
            aria-label={t("playground.stat.details")}
            data-testid="reply-stats"
            className="inline-flex min-w-0 flex-wrap items-center gap-x-1.5 rounded-sm text-xs font-normal tabular-nums text-muted-foreground"
          >
            {live && stats.ttftMs === undefined && (
              <>
                <span>{t("playground.stat.waitingFirst")}</span>
                <span aria-hidden="true">·</span>
              </>
            )}
            {parts.map((part, i) => (
              <Fragment key={i}>
                {i > 0 && <span aria-hidden="true">·</span>}
                <span className="whitespace-nowrap">{part}</span>
              </Fragment>
            ))}
          </span>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          align="start"
          className="space-y-1 text-xs tabular-nums"
          data-testid="reply-stats-detail"
        >
          {detail.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-6">
              <span className="text-muted-foreground">{k}</span>
              <span>{v}</span>
            </div>
          ))}
          {extra && <div className="text-muted-foreground">{extra}</div>}
          {stats.estimated && (
            <div className="text-muted-foreground">
              {t("playground.stat.estimated")}
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
