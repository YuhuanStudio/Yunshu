import { Button, Card, Input } from "@yuhuanowo/yunui";
import { Banner } from "@yuhuanowo/yunui/patterns";
import { Check, Copy, RefreshCw, type LucideIcon } from "lucide-react";
import {
  Suspense,
  lazy,
  useEffect,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import type { EngineStatus } from "./api";
import type { useEngine } from "./useEngine";
import { offlineCause } from "./errors";
export type Engine = ReturnType<typeof useEngine>;
export type Model = EngineStatus["models"][number];
// Intl formatters are expensive to build and a poll formats hundreds of
// values (every chart tick), so each distinct format is built once.
const numberFormats = new Map<string, Intl.NumberFormat>();
function numberFormat(min: number, max: number) {
  const key = min + ":" + max;
  let f = numberFormats.get(key);
  if (!f) {
    f = new Intl.NumberFormat("zh-TW", {
      minimumFractionDigits: min,
      maximumFractionDigits: max,
    });
    numberFormats.set(key, f);
  }
  return f;
}
export const number = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v) ? "—" : numberFormat(0, digits).format(v);
/** Fixed decimals, so a polled value keeps the same number of characters. */
export const fixed = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : numberFormat(digits, digits).format(v);
/**
 * A numeric slot that keeps its width: tabular figures plus a reserved minimum
 * width in `ch`, so a value that changes length (or is briefly "—") never moves
 * what sits beside it. Use for every polled number that is not in a table cell.
 */
export function Slot({
  ch,
  align = "left",
  className = "",
  children,
}: {
  ch: number;
  align?: "left" | "right";
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-block whitespace-nowrap tabular-nums ${align === "right" ? "text-right" : "text-left"} ${className}`}
      style={{ minWidth: `${ch}ch` }}
    >
      {children}
    </span>
  );
}
const clockFormat = new Intl.DateTimeFormat("zh-TW", {
  hourCycle: "h23",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});
export const clock = (t: number) => clockFormat.format(t);
export const elapsed = (seconds: number | null | undefined) =>
  seconds == null
    ? "—"
    : seconds >= 3600
      ? `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
      : seconds >= 60
        ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
        : `${number(seconds)}s`;
export const modelLabel = (id: string) =>
  id.split("/").filter(Boolean).at(-1) ?? id;
export const isOnline = (engine: Engine) => engine.phase === "online";
export function ConnectionState({
  engine,
  configure,
}: {
  engine: Engine;
  configure: () => void;
}) {
  if (engine.phase === "online") return null;
  const cause = offlineCause(engine.phase, engine.errorStatus);
  const description = [
    cause.hint,
    engine.phase === "offline" ? "每 3 秒自動重試。" : "",
    engine.updatedAt
      ? `最後成功連線 ${clock(engine.updatedAt)}，下方保留上次資料。`
      : "",
  ]
    .filter(Boolean)
    .join("");
  return (
    <div role="status">
      <Banner
        tone={engine.phase === "connecting" ? "neutral" : "warning"}
        title={cause.title}
        description={description}
        actions={
          <>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void engine.refresh()}
            >
              <RefreshCw size={13} />
              重試
            </Button>
            <Button size="sm" variant="ghost" onClick={configure}>
              連線設定
            </Button>
          </>
        }
      />
    </div>
  );
}
/** A card with a Yunxin-style header: icon chip, title, description and a trailing action. */
export function SectionCard({
  icon: Icon,
  title,
  description,
  action,
  children,
  className,
  bodyClassName = "p-5",
  ...props
}: {
  icon: LucideIcon;
  bodyClassName?: string;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
} & Omit<HTMLAttributes<HTMLDivElement>, "title">) {
  return (
    <Card className={className} {...props}>
      <div className="flex flex-wrap items-center gap-3 px-5 pt-5">
        <span
          aria-hidden="true"
          className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-(--bg-elevated) text-muted-foreground"
        >
          <Icon size={17} strokeWidth={1.75} />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="yunui-section-title truncate text-base font-semibold">
            {title}
          </h2>
          {description && (
            <p className="mt-0.5 text-xs text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        {action}
      </div>
      <div className={bodyClassName}>{children}</div>
    </Card>
  );
}
export const supportsChat = (model: Model | undefined) =>
  !!model && /llm|vlm|batched|omni/i.test(model.type);
// YunUI's ModelIcon and its developer table are loaded only when a page shows
// a model icon, so the shell does not ship them.
const LazyModelIcon = lazy(() => import("./LocalModelIcon"));
export function LocalModelIcon({
  id,
  size = 28,
}: {
  id: string;
  size?: number;
}) {
  return (
    <Suspense
      fallback={
        <span
          aria-hidden="true"
          className="inline-block shrink-0"
          style={{ width: size, height: size }}
        />
      }
    >
      <LazyModelIcon id={id} size={size} />
    </Suspense>
  );
}

/** One mapping for request phases to status dots. Red is reserved for errors. */
export const phaseDot = (phase: string): "online" | "away" | "neutral" =>
  phase === "decode"
    ? "online"
    : phase === "prefill" || phase === "starting"
      ? "away"
      : "neutral";

/** A label + value pair with tabular numerals: the console's basic readout. */
export function Readout({
  label,
  value,
  unit,
  hint,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-base font-semibold tabular-nums">
        {value}
        {unit && (
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            {unit}
          </span>
        )}
      </p>
      <p className="mt-0.5 min-h-[1.125rem] truncate text-xs text-muted-foreground">
        {hint}
      </p>
    </div>
  );
}

/** Tracks a min-width media query (Tailwind `xl` is 1280px). */
export function useMinWidth(px: number) {
  const query = `(min-width: ${px}px)`;
  const [matches, setMatches] = useState(
    () => typeof matchMedia === "function" && matchMedia(query).matches,
  );
  useEffect(() => {
    const list = matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);
  return matches;
}

/** Model size in GB; the engine may not report it (null or 0), which is shown as unknown, never "0 GB". */
export function sizeGb(size: number | null | undefined): string {
  return size != null && Number.isFinite(size) && size > 0
    ? `${number(size)} GB`
    : "—";
}

/** A string kept in localStorage; storage failures fall back to the default. */
export function useStoredChoice<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return allowed.includes(raw as T) ? (raw as T) : fallback;
    } catch {
      return fallback;
    }
  });
  const set = (next: T) => {
    setValue(next);
    try {
      localStorage.setItem(key, next);
    } catch {
      /* storage may be unavailable */
    }
  };
  return [value, set] as const;
}

/** A read-only mono field with a copy button; the copy outcome is announced, never assumed. */
export function CopyField({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  const [state, setState] = useState<"idle" | "done" | "failed">("idle");
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setState("done");
    } catch {
      setState("failed");
    }
    setTimeout(() => setState("idle"), 2000);
  }
  return (
    <div className={className}>
      <label className="text-xs text-muted-foreground">
        {label}
        <div className="mt-1.5 flex items-center gap-2">
          <Input
            readOnly
            className="min-w-0 flex-1 font-mono text-sm"
            value={value}
            onFocus={(e) => e.currentTarget.select()}
          />
          <Button
            size="sm"
            variant="secondary"
            type="button"
            aria-label={`複製${label}`}
            onClick={() => void copy()}
          >
            {state === "done" ? <Check size={13} /> : <Copy size={13} />}
            {state === "done"
              ? "已複製"
              : state === "failed"
                ? "無法寫入剪貼簿"
                : "複製"}
          </Button>
        </div>
      </label>
    </div>
  );
}
