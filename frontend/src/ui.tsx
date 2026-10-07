import { Button, Card } from "@yuhuanowo/yunui";
import {
  ModelIcon,
  getDeveloperIconPath,
  getModelDeveloperId,
} from "@yuhuanowo/yunui/ai";
import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import type { EngineStatus } from "./api";
import type { useEngine } from "./useEngine";
export type Engine = ReturnType<typeof useEngine>;
export type Model = EngineStatus["models"][number];
export const number = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : v.toLocaleString("zh-TW", { maximumFractionDigits: digits });
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
  return (
    <Card
      className="flex flex-wrap items-center justify-between gap-3 border-warning-soft bg-warning-soft px-4 py-3"
      role="status"
    >
      <div>
        <p className="text-sm font-medium">
          {engine.phase === "connecting"
            ? "正在連接引擎"
            : engine.phase === "unauthorized"
              ? "需要有效的存取權杖"
              : "無法連接引擎"}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {engine.error ?? "正在取得服務狀態…"}
          {engine.updatedAt
            ? ` · 最後成功：${clock(engine.updatedAt)}，下方保留上次資料。`
            : ""}
        </p>
      </div>
      <div className="flex gap-2">
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
      </div>
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
      <p className="text-[11px] tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold tabular-nums">
        {value}
        {unit && (
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            {unit}
          </span>
        )}
      </p>
      {hint && (
        <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
          {hint}
        </p>
      )}
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
