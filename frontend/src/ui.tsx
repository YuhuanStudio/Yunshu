import {
  Alert,
  Button,
  Card,
  Input,
  PasswordInput,
  Switch,
} from "@yuhuanowo/yunui";
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
import { t } from "./i18n/index.ts";
import type { useEngine } from "./useEngine";
import { offlineCause } from "./errors";
export type Engine = ReturnType<typeof useEngine>;
export type Model = EngineStatus["models"][number];
export {
  number,
  fixed,
  clock,
  clockShort,
  elapsed,
  percent,
  gb,
  bytes,
  relative,
  dateTime,
} from "./i18n/format.ts";
import { clock, clockShort, elapsed, number } from "./i18n/format.ts";
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
export const modelLabel = (id: string) =>
  id.split("/").filter(Boolean).at(-1) ?? id;
export const isOnline = (engine: Engine) => engine.phase === "online";
/** Ticks once a second while `enabled`, otherwise holds still (no timer, no re-render). */
export function useNow(enabled: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [enabled]);
  return now;
}

/** The unlock form of a refused token: in place, so the page behind it stays. */
function TokenPrompt({
  onToken,
}: {
  onToken: (token: string, remember: boolean) => void;
}) {
  const [token, setToken] = useState(""),
    [remember, setRemember] = useState(false);
  const submit = () => token.trim() && onToken(token.trim(), remember);
  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <PasswordInput
        className="w-full sm:w-64"
        autoComplete="off"
        value={token}
        onChange={(e) => setToken(e.target.value)}
        placeholder={t("common.unlock.placeholder")}
        aria-label={t("common.unlock.label")}
        labels={{
          show: t("settings.connection.showToken"),
          hide: t("settings.connection.hideToken"),
        }}
      />
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <Switch
          label={t("common.unlock.remember")}
          checked={remember}
          onCheckedChange={setRemember}
        />
        {t("common.unlock.remember")}
      </label>
      <Button size="sm" type="submit" disabled={!token.trim()}>
        {t("common.unlock.submit")}
      </Button>
    </form>
  );
}

export function ConnectionState({
  engine,
  configure,
  onToken,
}: {
  engine: Engine;
  configure: () => void;
  /** Set when the console can take a token in place; the banner then shows the unlock form. */
  onToken?: (token: string, remember: boolean) => void;
}) {
  const offline = engine.phase === "offline";
  const now = useNow(offline && engine.updatedAt != null);
  if (engine.phase === "online") return null;
  const cause = offlineCause(engine.phase, engine.errorStatus);
  const description = [
    cause.hint,
    offline ? t("common.autoRetry") : "",
    offline && engine.updatedAt
      ? t("common.offlineFor", {
          t: elapsed(Math.max(0, (now - engine.updatedAt) / 1000)),
        })
      : "",
    engine.updatedAt
      ? t("common.lastOk", { time: clock(engine.updatedAt) })
      : "",
  ]
    .filter(Boolean)
    .join(t("common.sentenceGap"));
  const prompt = engine.phase === "unauthorized" && onToken;
  return (
    <div role="status">
      <Banner
        tone={engine.phase === "connecting" ? "neutral" : "warning"}
        title={cause.title}
        description={description}
        actions={
          <>
            {prompt && <TokenPrompt onToken={onToken} />}
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void engine.refresh()}
            >
              <RefreshCw size={13} />
              {t("common.retry")}
            </Button>
            <Button size="sm" variant="ghost" onClick={configure}>
              {t("common.configure")}
            </Button>
          </>
        }
      />
    </div>
  );
}

/** 「資料停在 HH:MM」: shown over panels whose numbers are no longer being refreshed. */
export function StaleStamp({ engine }: { engine: Engine }) {
  if (engine.phase !== "offline" || engine.updatedAt == null) return null;
  return (
    <p
      className="text-xs text-muted-foreground"
      data-testid="stale-stamp"
      role="status"
    >
      {t("common.staleAt", { time: clockShort(engine.updatedAt) })}
    </p>
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
            aria-label={t("common.copyLabel", { label })}
            onClick={() => void copy()}
          >
            {state === "done" ? <Check size={13} /> : <Copy size={13} />}
            {state === "done"
              ? t("common.copied")
              : state === "failed"
                ? t("common.copyFailed")
                : t("common.copy")}
          </Button>
        </div>
      </label>
    </div>
  );
}

/**
 * A feature the connected engine does not offer (older version, no access): one compact notice
 * at normal measure, never a giant empty container.
 */
export function UnavailableNotice({
  title,
  description,
  "data-testid": testId,
}: {
  title: string;
  description: string;
  "data-testid"?: string;
}) {
  return (
    <Alert
      variant="info"
      title={title}
      data-testid={testId}
      className="max-w-3xl"
    >
      {description}
    </Alert>
  );
}
