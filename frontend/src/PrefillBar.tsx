import { StackedProgress } from "@yuhuanowo/yunui";
import type { RequestRow } from "./api";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format.ts";
import { prefillSplit } from "./prefill-split";
import { useTween } from "./tween";

/**
 * One request's prefill as a bar: the cache hit is its own muted segment (tooltip 「快取命中 N tokens」),
 * then the computed part fills in after it. A cache hit therefore never makes the bar start part-way
 * along. The segment widths are in tokens of the whole prompt; `tweened` is a 0..1 factor the caller
 * animates between polls so the fill glides instead of jumping.
 */
export function PrefillBar({
  row,
  height = 4,
  className,
  doneTokens,
  caption = false,
}: {
  row: Pick<
    RequestRow,
    "prompt_tokens" | "cached_tokens" | "processed_tokens" | "percent"
  >;
  height?: number;
  className?: string;
  /** Tokens computed so far as drawn right now (a tweened value); defaults to the reported figure. */
  doneTokens?: number;
  /** A visible legend line under the bar when there is a cache hit (a tooltip does not exist on a phone). */
  caption?: boolean;
}) {
  useLocale();
  const s = prefillSplit(row);
  // The computed part glides between samples (a few a second) instead of stepping.
  const glide = useTween(
    s ? (s.prompt > 0 ? s.done : (s.percentOfComputed ?? 0)) : null,
    200,
  );
  if (!s) return null;
  const prompt = s.prompt > 0 ? s.prompt : 100;
  const done =
    doneTokens ?? (s.prompt > 0 ? s.done : (s.percentOfComputed ?? 0));
  const cached = s.prompt > 0 ? s.cached : 0;
  const valueText =
    s.cached > 0
      ? t("overview.prefill.value", {
          cached: number(s.cached, 0),
          done: number(s.done, 0),
          prompt: number(s.prompt, 0),
        })
      : s.prompt > 0
        ? t("overview.prefill.valueNoCache", {
            done: number(s.done, 0),
            prompt: number(s.prompt, 0),
          })
        : undefined;
  const bar = (
    <StackedProgress
      className={className}
      height={height}
      total={prompt}
      label={t("requests.trace.prefillProgress")}
      valueText={valueText}
      segments={[
        {
          value: cached,
          tone: "info",
          label:
            cached > 0
              ? t("overview.prefill.cached", { n: number(cached, 0) })
              : undefined,
        },
        {
          value: done,
          tone: "neutral",
          label:
            s.prompt > 0
              ? t("overview.prefill.computed", { n: number(s.done, 0) })
              : undefined,
        },
      ]}
    />
  );
  if (!caption || s.cached <= 0) return bar;
  return (
    <div className="w-full space-y-1">
      {bar}
      <p
        className="m-0 flex items-center gap-1.5 text-xs text-muted-foreground"
        data-testid="prefill-cache-legend"
      >
        <span
          aria-hidden
          className="size-1.5 shrink-0 rounded-full bg-(--info)"
        />
        {t("overview.prefill.cached", { n: number(s.cached, 0) })}
      </p>
    </div>
  );
}
