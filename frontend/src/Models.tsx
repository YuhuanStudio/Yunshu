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
  Input,
  SegmentedSelect,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
} from "@yuhuanowo/yunui";
import { PageHeader, CodeBlock } from "@yuhuanowo/yunui/patterns";
import { Box, Play, RefreshCw, Search, Square, Zap } from "lucide-react";
import {
  getModel,
  loadModel,
  unloadModel,
  warmupModel,
  type Connection,
} from "./api";
import {
  elapsed,
  isOnline,
  modelLabel,
  number,
  type Engine,
  type Model,
} from "./ui";
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
  return (
    <section
      className="mx-auto max-w-7xl space-y-5 p-4 sm:p-7"
      data-testid="models"
    >
      <PageHeader
        title="模型庫"
        description="管理此服務註冊的模型，查看載入狀態、記憶體與保留時間。"
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void engine.refresh()}
          >
            <RefreshCw size={14} />
            重新整理
          </Button>
        }
      />
      <div className="flex flex-wrap justify-between gap-3">
        <Input
          className="sm:max-w-xs"
          icon={<Search size={14} />}
          aria-label="搜尋模型"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
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
      <Card className="overflow-hidden">
        <Table scrollLabel="模型清單" className="min-w-[580px]">
          <Thead>
            <Tr>
              <Th>模型</Th>
              <Th>狀態</Th>
              <Th className="hidden md:table-cell">記憶體</Th>
              <Th className="hidden xl:table-cell">保留時間</Th>
              <Th>操作</Th>
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((model) => (
              <Tr key={model.id}>
                <Td>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto max-w-72 whitespace-normal break-all text-left"
                    onClick={(e) => void show(model, e.currentTarget)}
                  >
                    {modelLabel(model.id)}
                  </Button>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {model.type}
                    {model.pinned ? " · 固定保留" : ""}
                  </p>
                  {model.error && (
                    <p className="mt-1 max-w-xs break-words text-xs text-error">
                      {model.error}
                    </p>
                  )}
                </Td>
                <Td>
                  <Badge
                    className="whitespace-nowrap"
                    variant={
                      model.error
                        ? "error"
                        : model.loaded
                          ? "success"
                          : "secondary"
                    }
                  >
                    {model.loading
                      ? "載入中"
                      : model.error
                        ? "載入失敗"
                        : model.loaded
                          ? "已載入"
                          : "未載入"}
                  </Badge>
                </Td>
                <Td className="hidden font-mono text-xs md:table-cell">
                  {number(model.size_gb)} GB
                </Td>
                <Td className="hidden text-xs xl:table-cell">
                  {model.pinned ? "固定" : elapsed(model.expires_in_s)}
                </Td>
                <Td>
                  <div className="flex flex-wrap gap-1">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={
                        !online ||
                        !!busy ||
                        model.loading ||
                        (model.pinned && model.loaded)
                      }
                      onClick={(e) => {
                        if (model.loaded) {
                          opener.current = e.currentTarget;
                          setUnloading(model);
                        } else
                          void perform(`load:${model.id}`, () =>
                            loadModel(connection, model.id),
                          );
                      }}
                    >
                      {model.loaded ? <Square size={12} /> : <Play size={12} />}
                      {busy?.endsWith(model.id)
                        ? "處理中"
                        : model.loaded
                          ? "卸載"
                          : "載入"}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!online || !!busy || !model.loaded}
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
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!online || !model.loaded}
                      onClick={() => test(model.id)}
                    >
                      測試
                    </Button>
                  </div>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
        {!rows.length && (
          <EmptyState
            icon={<Box size={25} />}
            title={engine.status ? "沒有符合條件的模型" : "等待模型清單"}
            description={
              engine.status
                ? "清除篩選條件；新增模型請透過 Yunshu 的啟動設定註冊。"
                : "確認服務位址與存取權杖後重新整理。"
            }
          />
        )}
      </Card>
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
