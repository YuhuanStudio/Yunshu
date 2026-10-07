import { useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
  EmptyState,
  Progress,
  SearchInput,
  SegmentedSelect,
  StatusIndicator,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
} from "@yuhuanowo/yunui";
import {
  PageHeader,
  CodeBlock,
  CapabilityBadge,
} from "@yuhuanowo/yunui/patterns";
import { isKnownCapability } from "@yuhuanowo/yunui/ai";
import { Box, Play, RefreshCw, Square, Zap } from "lucide-react";
import {
  getModel,
  loadModel,
  unloadModel,
  warmupModel,
  type Connection,
} from "./api";
import {
  elapsed,
  LocalModelIcon,
  isOnline,
  supportsChat,
  modelLabel,
  number,
  type Engine,
  type Model,
} from "./ui";
import { ModelManagement } from "./ModelManagement";
/** Fit against unified memory; empty when the engine did not report the numbers. */
function fitHint(model: Model, free: number | undefined) {
  if (model.loaded || model.loading || free == null || model.size_gb == null)
    return null;
  return model.size_gb <= free
    ? { ok: true, text: `可載入 · 剩 ${number(free, 0)} GB` }
    : { ok: false, text: "記憶體不足" };
}
export type Perform = (
  key: string,
  action: () => Promise<unknown>,
) => Promise<void>;
export function Models({
  engine,
  connection,
  perform,
  busy,
  test,
}: {
  engine: Engine;
  connection: Connection;
  perform: Perform;
  busy: string | null;
  test: (id: string) => void;
}) {
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [details, setDetails] = useState<Model | null>(null),
    [metadata, setMetadata] = useState<unknown>(null),
    [detailError, setDetailError] = useState(""),
    [unloading, setUnloading] = useState<Model | null>(null);
  const opener = useRef<HTMLButtonElement | null>(null),
    detailSequence = useRef(0);
  const online = isOnline(engine);
  const rows = (engine.status?.models ?? []).filter(
    (m) =>
      (filter === "all" || m.loaded || m.loading) &&
      m.id.toLowerCase().includes(query.toLowerCase()),
  );
  const restore = (e: Event) => {
    if (opener.current?.isConnected) {
      e.preventDefault();
      opener.current.focus();
    }
  };
  async function show(model: Model, button: HTMLButtonElement) {
    opener.current = button;
    setDetails(model);
    setMetadata(null);
    setDetailError("");
    const seq = ++detailSequence.current;
    try {
      const value = await getModel(connection, model.id);
      if (seq === detailSequence.current) setMetadata(value);
    } catch (e) {
      if (seq === detailSequence.current)
        setDetailError(e instanceof Error ? e.message : String(e));
    }
  }
  const memTotal = engine.status?.memory.total_gb,
    memActive = engine.status?.memory.active_gb,
    memFree =
      memTotal != null && memActive != null ? memTotal - memActive : undefined;
  const loadedRows = rows.filter((m) => m.loaded || m.loading),
    availableRows = rows.filter((m) => !m.loaded && !m.loading);
  const group = (title: string, items: Model[]) => (
    <Card className="overflow-hidden">
      <h2 className="flex items-center gap-2 border-b border-border/60 px-4 py-3 text-sm font-medium sm:px-5">
        {title}
        <span className="text-muted-foreground tabular-nums">
          {items.length}
        </span>
      </h2>
      <Table scrollLabel={title} className="min-w-[640px]">
        <Thead>
          <Tr>
            <Th>模型</Th>
            <Th className="w-40">狀態</Th>
            <Th className="hidden w-24 md:table-cell">大小</Th>
            <Th className="hidden w-32 xl:table-cell">保留</Th>
            <Th>操作</Th>
          </Tr>
        </Thead>
        <Tbody>
          {items.map((model) => (
            <Tr key={model.id}>
              <Td>
                <div className="flex min-w-0 items-start gap-3">
                  <LocalModelIcon id={model.id} />
                  <div className="min-w-0">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto max-w-72 whitespace-normal break-all p-0 text-left"
                      onClick={(e) => void show(model, e.currentTarget)}
                    >
                      {modelLabel(model.id)}
                    </Button>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <Badge variant="secondary">{model.type}</Badge>
                      {/vlm|omni/i.test(model.type) &&
                        isKnownCapability("vision") && (
                          <CapabilityBadge capability="vision" short />
                        )}
                      {fitHint(model, memFree) && (
                        <span
                          className={`text-[11px] tabular-nums ${fitHint(model, memFree)?.ok ? "text-muted-foreground" : "text-error"}`}
                        >
                          {fitHint(model, memFree)?.text}
                        </span>
                      )}
                    </div>
                    {model.error && (
                      <p className="mt-1 max-w-xs break-words text-xs text-error">
                        {model.error}
                      </p>
                    )}
                  </div>
                </div>
              </Td>
              <Td>
                <div className="w-28">
                  <StatusIndicator
                    status={
                      model.error
                        ? "offline"
                        : model.loading
                          ? "busy"
                          : model.loaded
                            ? "online"
                            : "neutral"
                    }
                    pulse={model.loading}
                  >
                    <span className="whitespace-nowrap text-foreground">
                      {model.loading
                        ? "載入中"
                        : model.error
                          ? "載入失敗"
                          : model.loaded
                            ? "已載入"
                            : "未載入"}
                    </span>
                  </StatusIndicator>
                  {model.loading && (
                    <Progress
                      className="mt-2 h-1"
                      label={`${modelLabel(model.id)} 載入中`}
                    />
                  )}
                </div>
              </Td>
              <Td className="hidden text-sm tabular-nums md:table-cell">
                {number(model.size_gb)}
                <span className="ml-1 text-xs text-muted-foreground">GB</span>
              </Td>
              <Td className="hidden text-xs tabular-nums text-muted-foreground xl:table-cell">
                {model.loaded
                  ? model.pinned
                    ? "固定保留"
                    : `${elapsed(model.expires_in_s)} 後釋放`
                  : "—"}
              </Td>
              <Td>
                <div className="flex flex-wrap items-center gap-1">
                  {model.loaded ? (
                    <>
                      <Button
                        size="sm"
                        disabled={!online || !supportsChat(model)}
                        title={
                          supportsChat(model)
                            ? "文字或視覺推理測試"
                            : "此模型請使用 API 接入對應端點"
                        }
                        onClick={() => test(model.id)}
                      >
                        測試
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={!online || !!busy}
                        onClick={() =>
                          void perform(`warmup:${model.id}`, () =>
                            warmupModel(connection, {
                              model: model.id,
                              max_tokens: 1,
                            }),
                          )
                        }
                      >
                        <Zap size={12} />
                        預熱
                      </Button>
                    </>
                  ) : (
                    <Button
                      size="sm"
                      disabled={!online || !!busy || model.loading}
                      onClick={() =>
                        void perform(`load:${model.id}`, () =>
                          loadModel(connection, model.id),
                        )
                      }
                    >
                      <Play size={12} />
                      {busy?.endsWith(model.id) ? "處理中" : "載入"}
                    </Button>
                  )}
                  {model.loaded && (
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={
                        !online || !!busy || model.loading || model.pinned
                      }
                      onClick={(e) => {
                        opener.current = e.currentTarget;
                        setUnloading(model);
                      }}
                    >
                      <Square size={12} />
                      {busy?.endsWith(model.id) ? "處理中" : "卸載"}
                    </Button>
                  )}
                  <ModelManagement
                    connection={connection}
                    modelId={model.id}
                    disabled={!online || !!busy || model.loading}
                    perform={perform}
                  />
                </div>
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Card>
  );
  return (
    <section className="w-full max-w-7xl space-y-6" data-testid="models">
      <PageHeader
        title="模型庫"
        description="管理此服務註冊的模型，查看載入狀態、記憶體與保留時間。"
        actions={
          <div className="flex flex-wrap gap-2">
            <ModelManagement
              connection={connection}
              disabled={!online || !!busy}
              perform={perform}
            />
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void engine.refresh()}
            >
              <RefreshCw size={14} />
              重新整理
            </Button>
          </div>
        }
      />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SearchInput
          className="sm:max-w-xs"
          aria-label="搜尋模型"
          value={query}
          onChange={setQuery}
          placeholder="搜尋模型 ID"
        />
        <SegmentedSelect
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "全部模型" },
            { value: "loaded", label: "已載入" },
          ]}
        />
      </div>
      {memTotal != null && memTotal > 0 && (
        <Card className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4">
          <div className="min-w-0">
            <p className="text-[11px] tracking-wide text-muted-foreground">
              統一記憶體使用
            </p>
            <p className="mt-1 text-lg font-semibold tabular-nums">
              {number(memActive)}
              <span className="ml-1 text-xs font-normal text-muted-foreground">
                / {number(memTotal, 0)} GB
              </span>
            </p>
          </div>
          <Progress
            className="h-1.5 min-w-40 flex-1"
            value={Math.max(
              0,
              Math.min(100, ((memActive ?? 0) / memTotal) * 100),
            )}
            label="統一記憶體使用"
          />
          <p className="text-xs tabular-nums text-muted-foreground">
            已載入 {loadedRows.length} · 可用 {number(memFree)} GB
          </p>
        </Card>
      )}
      {!rows.length ? (
        <Card>
          <EmptyState
            icon={<Box size={25} />}
            title={engine.status ? "沒有符合條件的模型" : "等待模型清單"}
            description={
              engine.status
                ? "清除篩選條件；或匯入 Hugging Face 原生 MLX 模型。"
                : "確認服務位址與存取權杖後重新整理。"
            }
          />
        </Card>
      ) : (
        <>
          {loadedRows.length > 0 && group("已載入的模型", loadedRows)}
          {filter === "all" &&
            availableRows.length > 0 &&
            group("可用模型", availableRows)}
        </>
      )}
      <p className="text-xs text-muted-foreground">
        載入、卸載與預熱會呼叫此引擎。單模型模式固定保留；服務可能因正在執行請求而拒絕卸載。
      </p>
      <Dialog
        open={!!unloading}
        onOpenChange={(open) => {
          if (!open) setUnloading(null);
        }}
      >
        <DialogContent closeLabel="關閉卸載確認" onCloseAutoFocus={restore}>
          <DialogTitle>卸載模型？</DialogTitle>
          <DialogDescription>
            {unloading?.id} 將釋放記憶體，之後使用前需要重新載入。
          </DialogDescription>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setUnloading(null)}>
              保留模型
            </Button>
            <Button
              onClick={() => {
                const m = unloading;
                setUnloading(null);
                if (m)
                  void perform(`unload:${m.id}`, () =>
                    unloadModel(connection, m.id),
                  );
              }}
            >
              卸載
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!details}
        onOpenChange={(open) => {
          if (!open) {
            setDetails(null);
            detailSequence.current++;
          }
        }}
      >
        <DialogContent closeLabel="關閉模型資訊" onCloseAutoFocus={restore}>
          <DialogTitle>
            {details ? modelLabel(details.id) : "模型資訊"}
          </DialogTitle>
          <DialogDescription>服務回傳的模型卡與能力資訊</DialogDescription>
          {detailError ? (
            <p role="alert" className="text-sm text-error">
              {detailError}
            </p>
          ) : metadata ? (
            <CodeBlock
              code={JSON.stringify(metadata, null, 2)}
              language="json"
            />
          ) : (
            <p role="status" className="text-sm text-muted-foreground">
              讀取模型資訊…
            </p>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}
