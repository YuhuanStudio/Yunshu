import { Button, Card, Input } from "@yuhuanowo/yunui";
import { Banner } from "@yuhuanowo/yunui/patterns";
import {
  ModelIcon,
  getDeveloperIconPath,
  getModelDeveloperId,
} from "@yuhuanowo/yunui/ai";
import { Check, Copy, RefreshCw, type LucideIcon } from "lucide-react";
import {
  useEffect,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import type { EngineStatus } from "./api";
import type { useEngine } from "./useEngine";
export type Engine = ReturnType<typeof useEngine>;
export type Model = EngineStatus["models"][number];
export const number = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : v.toLocaleString("zh-TW", { maximumFractionDigits: digits });
/** Fixed decimals, so a polled value keeps the same number of characters. */
export const fixed = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : v.toLocaleString("zh-TW", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
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
export const clock = (t: number) =>
  new Date(t).toLocaleTimeString("zh-TW", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
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
  const title =
    engine.phase === "connecting"
      ? "正在連接引擎"
      : engine.phase === "unauthorized"
        ? "需要有效的存取權杖"
        : "無法連接引擎";
  return (
    <div role="status">
      <Banner
        tone={engine.phase === "connecting" ? "neutral" : "warning"}
        title={title}
        description={`${engine.error ?? "正在取得服務狀態…"}${
          engine.phase === "offline" ? " · 每 3 秒自動重試" : ""
        }${
          engine.updatedAt
            ? ` · 最後成功：${clock(engine.updatedAt)}，下方保留上次資料。`
            : ""
        }`}
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
// YunUI's ModelIcon defaults to jsDelivr; the console is local-first, so
// resolve the icon files that ship in the package to bundled asset URLs.
const bundledIcons = import.meta.glob(
  "../node_modules/@yuhuanowo/yunui/icons/models/*.{webp,png,jpeg}",
  { eager: true, query: "?url", import: "default" },
) as Record<string, string>;
const iconByFile = new Map(
  Object.entries(bundledIcons).map(([path, url]) => [
    path.split("/").at(-1) ?? path,
    url,
  ]),
);
export function LocalModelIcon({
  id,
  size = 28,
}: {
  id: string;
  size?: number;
}) {
  const developer = getModelDeveloperId(id);
  const file = getDeveloperIconPath(developer)?.split("/").at(-1);
  const url = file ? iconByFile.get(file) : undefined;
  return url ? (
    <ModelIcon
      iconUrl={url}
      developer={developer}
      provider={developer}
      size={size}
      rounded
    />
  ) : null;
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
