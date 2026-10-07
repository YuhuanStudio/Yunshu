import { Badge, Button, Card, EmptyState, AreaChart } from "@yuhuanowo/yunui";
import { RefreshCw } from "lucide-react";
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
export function MetricChart({
  title,
  description,
  data,
  unit,
  tone = "accent",
  height = 170,
}: {
  title: string;
  description: string;
  data: { value: number; label: string }[];
  unit: string;
  tone?: "accent" | "info" | "success" | "warning";
  height?: number;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">{title}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        </div>
        <Badge variant="outline">{unit}</Badge>
      </div>
      <AreaChart
        data={data}
        height={height}
        tone={tone}
        ariaLabel={`${title}，單位 ${unit}`}
        formatValue={(v) => `${number(v)} ${unit}`}
        noDataLabel={
          data.length === 1
            ? "已取得第一筆資料，等待下一次採樣"
            : "目前沒有可用的採樣"
        }
        showTooltip
      />
      {data.length > 1 && (
        <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
          <span>{data[0].label}</span>
          <span>{data.at(-1)?.label}</span>
        </div>
      )}
    </div>
  );
}
