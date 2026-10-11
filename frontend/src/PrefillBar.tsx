import { LiveBar } from "./motion/LiveBar";
import type { RequestRow } from "./api";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format.ts";
import { prefillSplit } from "./prefill-split";

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
  caption = false,
}: {
  row: Pick<
    RequestRow,
    "prompt_tokens" | "cached_tokens" | "processed_tokens" | "percent"
  > & { request_id?: string };
  height?: number;
  className?: string;
  /** A visible legend line under the bar when there is a cache hit (a tooltip does not exist on a phone). */
  caption?: boolean;
}) {
  useLocale();
  const s = prefillSplit(row);
  if (!s) return null;
  const prompt = s.prompt > 0 ? s.prompt : 100;
  const done = s.prompt > 0 ? s.done : (s.percentOfComputed ?? 0);
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
    <LiveBar
      className={className}
      height={height}
      resetKey={row.request_id}
      label={t("requests.trace.prefillProgress")}
      valueText={valueText}
      fixed={
        cached > 0
          ? [
              {
                fraction: cached / prompt,
                tone: "info",
                label: t("overview.prefill.cached", { n: number(cached, 0) }),
              },
            ]
          : []
      }
      liveLabel={
        s.prompt > 0
          ? t("overview.prefill.computed", { n: number(s.done, 0) })
          : undefined
      }
      live={done / prompt}
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
