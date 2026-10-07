import { useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogFooter,
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
  CapabilityBadge,
  DashboardPage,
  PageHeader,
  SectionRow,
} from "@yuhuanowo/yunui/patterns";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import { IDBadge, ModelCard, isKnownCapability } from "@yuhuanowo/yunui/ai";
import { Box, LayoutGrid, RefreshCw, Table2 } from "lucide-react";
import { getModel, unloadModel, type Connection } from "./api";
import {
  elapsed,
  fixed,
  LocalModelIcon,
  isOnline,
  modelLabel,
  number,
  sizeGb,
  Slot,
  useStoredChoice,
  type Engine,
  type Model,
} from "./ui";
import { ModelManagement } from "./ModelManagement";
import { ModelActions, type Perform } from "./ModelActions";
import { ModelDetail, modelState, retention } from "./ModelDetail";
export type { Perform } from "./ModelActions";
/** Fit against unified memory; empty when the engine did not report the numbers. */
function fitHint(model: Model, free: number | undefined) {
  if (model.loaded || model.loading || free == null || model.size_gb == null)
    return null;
  return model.size_gb <= free
    ? { ok: true, text: `可載入 · 剩 ${number(free, 0)} GB` }
    : { ok: false, text: "記憶體不足" };
}
export function Models({
  engine,
  connection,
  perform,
  busy,
  test,
  selected,
  open,
}: {
  engine: Engine;
  connection: Connection;
  perform: Perform;
  busy: string | null;
  test: (id: string) => void;
  /** Model id from the route (#/models/<id>), or null for the list. */
  selected: string | null;
  open: (id: string | null) => void;
}) {
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [kind, setKind] = useState("all"),
    [view, setView] = useStoredChoice(
      "yunshu.console.models.view",
      ["table", "cards"] as const,
      "table",
    ),
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
      (kind === "all" || m.type === kind) &&
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
  const requestUnload = (model: Model, button: HTMLButtonElement) => {
    opener.current = button;
    setUnloading(model);
  };
  const kinds = [...new Set((engine.status?.models ?? []).map((m) => m.type))];
  const group = (title: string, items: Model[]) =>
    view === "cards" ? (
      <div className="space-y-3">
        <SectionRow
          title={
            <span className="flex items-center gap-2">
              {title}
              <span className="text-muted-foreground tabular-nums">
                {items.length}
              </span>
            </span>
          }
        />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((model) => {
            const state = modelState(model);
            return (
              <ModelCard
                key={model.id}
                name={modelLabel(model.id)}
                icon={<LocalModelIcon id={model.id} />}
                ids={[model.id]}
                capabilities={
                  /vlm|omni/i.test(model.type) && isKnownCapability("vision")
                    ? ["vision"]
                    : []
                }
                description={
                  <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs tabular-nums">
                    <StatusIndicator status={state.status}>
                      {state.text}
                    </StatusIndicator>
                    <span>{model.type}</span>
                    <span>{sizeGb(model.size_gb)}</span>
                    <span>{retention(model)}</span>
                  </span>
                }
                onClick={() => open(model.id)}
              />
            );
          })}
        </div>
      </div>
    ) : (
      <div className="space-y-3">
        <SectionRow
          title={
            <span className="flex items-center gap-2">
              {title}
              <span className="text-muted-foreground tabular-nums">
                {items.length}
              </span>
            </span>
          }
        />
        <Card className="overflow-hidden">
          <Table scrollLabel={title} className="min-w-[640px] table-fixed">
            <Thead>
              <Tr>
                <Th>模型</Th>
                <Th className="w-40">狀態</Th>
                <Th className="hidden w-24 md:table-cell">大小</Th>
                <Th className="hidden w-32 xl:table-cell">保留</Th>
                <Th className="w-72">操作</Th>
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
                              className={`text-xs tabular-nums ${fitHint(model, memFree)?.ok ? "text-muted-foreground" : "text-error"}`}
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
                              ? "away"
                              : model.loaded
                                ? "online"
                                : "neutral"
                        }
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
                      <div className="mt-2 h-1">
                        {model.loading && (
                          <Progress
                            indeterminate
                            className="h-1"
                            label={`${modelLabel(model.id)} 載入中`}
                          />
                        )}
                      </div>
                    </div>
                  </Td>
                  <Td className="hidden tabular-nums md:table-cell">
                    {model.size_gb ? number(model.size_gb) : "—"}
                    {model.size_gb ? (
                      <span className="ml-1 text-xs text-muted-foreground">
                        GB
                      </span>
                    ) : null}
                  </Td>
                  <Td className="hidden tabular-nums text-muted-foreground xl:table-cell">
                    {model.loaded
                      ? model.pinned
                        ? "固定保留"
                        : `${elapsed(model.expires_in_s)} 後釋放`
                      : "—"}
                  </Td>
                  <Td>
                    <ModelActions
                      model={model}
                      connection={connection}
                      online={online}
                      busy={busy}
                      perform={perform}
                      test={test}
                      requestUnload={requestUnload}
                    />
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Card>
      </div>
    );
  return (
    <DashboardPage data-testid="models">
      {selected ? (
        <ModelDetail
          id={selected}
          engine={engine}
          connection={connection}
          online={online}
          busy={busy}
          perform={perform}
          test={test}
          requestUnload={requestUnload}
          back={() => open(null)}
        />
      ) : (
        <>
          <PageHeader
            title="模型庫"
            description="管理此服務註冊的模型，查看載入狀態、記憶體與保留時間。"
            actions={
              <div className="flex flex-wrap gap-2">
                <ModelManagement
                  connection={connection}
                  disabled={!online || !!busy}
                  disabledReason={
                    !online
                      ? "引擎未連線，連線後才能匯入模型"
                      : busy
                        ? "另一項操作進行中"
                        : undefined
                  }
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
          <Card className="flex flex-wrap items-center gap-3 p-4">
            <SearchInput
              className="w-full sm:max-w-xs"
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
            {kinds.length > 1 && (
              <SegmentedSelect
                value={kind}
                onChange={setKind}
                options={[
                  { value: "all", label: "所有類型" },
                  ...kinds.map((k) => ({ value: k, label: k })),
                ]}
              />
            )}
            <SegmentedSelect
              className="ml-auto"
              value={view}
              onChange={(v) => setView(v as "table" | "cards")}
              options={[
                { value: "table", label: "表格", icon: Table2 },
                { value: "cards", label: "卡片", icon: LayoutGrid },
              ]}
            />
          </Card>
          {memTotal != null && memTotal > 0 && (
            <Card className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4">
              <div className="min-w-0">
                <p className="text-xs text-muted-foreground">統一記憶體使用</p>
                <p className="mt-1 text-2xl font-semibold tabular-nums">
                  <Slot ch={5} align="right">
                    {fixed(memActive)}
                  </Slot>
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
              <Slot
                ch={22}
                align="right"
                className="text-xs text-muted-foreground"
              >
                已載入 {loadedRows.length} · 可用 {fixed(memFree)} GB
              </Slot>
            </Card>
          )}
          {!rows.length ? (
            <Card className="p-2">
              <EmptyState
                size="inline"
                icon={<Box size={22} />}
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
        </>
      )}
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
            <CodeBlock language="json">
              {JSON.stringify(metadata, null, 2)}
            </CodeBlock>
          ) : (
            <p role="status" className="text-sm text-muted-foreground">
              讀取模型資訊…
            </p>
          )}
          {details && (
            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  const id = details.id;
                  setDetails(null);
                  detailSequence.current++;
                  open(id);
                }}
              >
                開啟詳細頁
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
    </DashboardPage>
  );
}
