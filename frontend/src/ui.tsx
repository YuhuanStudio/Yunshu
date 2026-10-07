import { Button, Card } from "@yuhuanowo/yunui";
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
export const supportsChat = (model: Model | undefined) =>
  !!model && /llm|vlm|batched|omni/i.test(model.type);
