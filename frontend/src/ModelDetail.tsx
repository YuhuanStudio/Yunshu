import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Gauge,
  SegmentedBar,
  StatusIndicator,
} from "@yuhuanowo/yunui";
import { IDBadge } from "@yuhuanowo/yunui/ai";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import { DetailList, DetailRow, PageHeader } from "@yuhuanowo/yunui/patterns";
import { ArrowLeft, Box, FileJson, MemoryStick, Timer } from "lucide-react";
import { getModel, type Connection } from "./api";
import { ModelSize } from "./ModelSize";
import { MemoryLedgerView } from "./MemoryLedger";
import type { LedgerState } from "./memory-api";
import { ModelActions, type Perform } from "./ModelActions";
import {
  LocalModelIcon,
  SectionCard,
  elapsed,
  modelLabel,
  number,
  type Engine,
  type Model,
} from "./ui";

export const modelState = (model: Model) =>
  model.error
    ? ({ status: "offline", text: "載入失敗" } as const)
    : model.loading
      ? ({ status: "away", text: "載入中" } as const)
      : model.loaded
        ? ({ status: "online", text: "已載入" } as const)
        : ({ status: "neutral", text: "未載入" } as const);

export const retention = (model: Model) =>
  !model.loaded
    ? "—"
    : model.pinned
      ? "固定保留"
      : model.expires_in_s != null
        ? `${elapsed(model.expires_in_s)} 後釋放`
        : model.keep_alive_s != null
          ? `閒置 ${elapsed(model.keep_alive_s)} 後釋放`
          : "依服務預設";

/** Per-model page: only facts the status and model endpoints report. */
export function ModelDetail({
  id,
  engine,
  connection,
  online,
  busy,
  perform,
  test,
  requestUnload,
  back,
  ledger,
}: {
  id: string;
  engine: Engine;
  connection: Connection;
  online: boolean;
  busy: string | null;
  perform: Perform;
  test: (id: string) => void;
  requestUnload: (model: Model, button: HTMLButtonElement) => void;
  back: () => void;
  ledger?: LedgerState;
}) {
  const model = engine.status?.models.find((m) => m.id === id);
  const [card, setCard] = useState<unknown>(null),
    [cardError, setCardError] = useState("");
  useEffect(() => {
    if (!online) return;
    const controller = new AbortController();
    setCard(null);
    setCardError("");
    getModel(connection, id)
      .then((value) => {
        if (!controller.signal.aborted) setCard(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setCardError(e instanceof Error ? e.message : String(e));
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, id, online]);
  const backButton = (
    <Button size="sm" variant="secondary" onClick={back}>
      <ArrowLeft size={14} />
      返回模型庫
    </Button>
  );
  if (!model)
    return (
      <>
        <PageHeader title={modelLabel(id)} actions={backButton} />
        <Card>
          <EmptyState
            icon={<Box size={25} />}
            title={engine.status ? "服務沒有註冊這個模型" : "等待模型清單"}
            description={
              engine.status
                ? "模型可能已被刪除或重新命名；返回模型庫查看目前清單。"
                : "確認服務位址與存取權杖後重新整理。"
            }
          />
        </Card>
      </>
    );
  const state = modelState(model);
  const total = engine.status?.memory.total_gb,
    active = engine.status?.memory.active_gb;
  const share =
    model.size_gb && total
      ? Math.min(100, (model.size_gb / total) * 100)
      : null;
  const memorySegments =
    model.loaded && model.size_gb && total && active != null
      ? [
          {
            value: Math.min(model.size_gb, active),
            tone: "accent" as const,
            label: "此模型權重",
          },
          {
            value: Math.max(0, active - model.size_gb),
            tone: "neutral" as const,
            label: "其他活躍配置",
          },
        ]
      : null;
  return (
    <>
      <PageHeader
        title={modelLabel(model.id)}
        description={`${model.type} · ${state.text}`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <ModelActions
              model={model}
              connection={connection}
              online={online}
              busy={busy}
              perform={perform}
              test={test}
              requestUnload={requestUnload}
              freeGb={ledger ? (ledger.data?.free_gb ?? undefined) : undefined}
            />
            {backButton}
          </div>
        }
      />
      {model.error && (
        <p role="alert" className="text-sm text-error">
          {model.error}
        </p>
      )}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-6">
          <SectionCard
            icon={MemoryStick}
            title="記憶體"
            description={
              model.loaded
                ? "模型權重在統一記憶體中的佔比。"
                : "尚未載入；顯示的是載入後權重佔用的預估比例。"
            }
          >
            {share == null ? (
              <p className="text-sm text-muted-foreground">
                服務沒有回報這個模型的大小或統一記憶體總量。
              </p>
            ) : (
              <div className="flex flex-wrap items-center gap-6">
                <Gauge
                  value={share}
                  size={96}
                  thickness={8}
                  tone={share > 85 ? "warning" : "accent"}
                  label={`${number(share, 0)}%`}
                  ariaLabel={`模型權重佔統一記憶體 ${number(share, 0)}%`}
                />
                <div className="min-w-0 flex-1 space-y-3">
                  <p className="text-sm tabular-nums">
                    <ModelSize gb={model.size_gb} />
                    <span className="text-muted-foreground">
                      {" "}
                      / {number(total, 0)} GB 統一記憶體
                    </span>
                  </p>
                  {memorySegments && (
                    <SegmentedBar
                      segments={memorySegments}
                      total={total}
                      legend
                      height={8}
                      formatValue={(v) => `${number(v)} GB`}
                      label="MLX 活躍配置組成"
                    />
                  )}
                </div>
              </div>
            )}
          </SectionCard>
          <SectionCard
            icon={MemoryStick}
            title="記憶體持有者"
            description="統一記憶體目前由誰持有；估算值標示「估算」，未知不是 0。"
          >
            {ledger?.unsupported ? (
              <p
                className="text-sm text-muted-foreground"
                data-testid="memory-ledger-unsupported"
              >
                此引擎版本沒有提供記憶體持有者明細（需要較新的 Yunshu）。
              </p>
            ) : ledger?.data ? (
              <MemoryLedgerView data={ledger.data} />
            ) : (
              <p role="status" className="text-sm text-muted-foreground">
                {ledger?.error
                  ? `無法讀取記憶體明細：${ledger.error}`
                  : "讀取記憶體明細…"}
              </p>
            )}
          </SectionCard>
          <SectionCard
            icon={FileJson}
            title="模型卡"
            description="服務對此模型回傳的原始資料。"
          >
            {cardError ? (
              <p role="alert" className="text-sm text-error">
                {cardError}
              </p>
            ) : card ? (
              <CodeBlock language="json">
                {JSON.stringify(card, null, 2)}
              </CodeBlock>
            ) : (
              <p role="status" className="text-sm text-muted-foreground">
                {online ? "讀取模型資訊…" : "引擎未連線"}
              </p>
            )}
          </SectionCard>
        </div>
        <SectionCard
          icon={Timer}
          title="狀態與保留"
          className="h-fit"
          data-testid="model-facts"
        >
          <div className="mb-4 flex min-w-0 items-center gap-3">
            <LocalModelIcon id={model.id} size={32} />
            <IDBadge text={model.id} truncate />
          </div>
          <DetailList ariaLabel="模型事實">
            <DetailRow
              label="類型"
              value={<Badge>{model.type}</Badge>}
              mono={false}
            />
            <DetailRow
              label="狀態"
              mono={false}
              value={
                <StatusIndicator status={state.status}>
                  {state.text}
                </StatusIndicator>
              }
            />
            <DetailRow label="大小" value={<ModelSize gb={model.size_gb} />} />
            <DetailRow label="保留" value={retention(model)} mono={false} />
            <DetailRow
              label="固定保留"
              value={model.pinned ? "是" : "否"}
              mono={false}
            />
            <DetailRow label="閒置" value={elapsed(model.idle_s)} />
            <DetailRow
              label="保留時間"
              value={
                model.keep_alive_s != null ? elapsed(model.keep_alive_s) : "—"
              }
            />
          </DetailList>
        </SectionCard>
      </div>
    </>
  );
}
