import { useEffect, useMemo, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  EmptyState,
  Input,
  Progress,
  SegmentedSelect,
  Sheet,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { PageHeader } from "@yuhuanowo/yunui/patterns";
import { Download, Search } from "lucide-react";
import { ApiError, cancelRequest, requestJson, type Connection } from "./api";
import {
  clock,
  elapsed,
  isOnline,
  modelLabel,
  number,
  type Engine,
} from "./ui";
import type { Perform } from "./Models";
type Row = {
  id: string;
  phase: string;
  model?: string;
  elapsed_s?: number;
  prompt_tokens?: number;
  cached_tokens?: number;
  completion_tokens?: number;
  percent?: number;
  tokens_per_second?: number | null;
  ttft_ms?: number | null;
  decode_tps?: number | null;
  prefill_tps?: number | null;
  t?: number;
  path?: string;
};
const labels: Record<string, string> = {
  queued: "排隊",
  starting: "準備中",
  prefill: "Prefill",
  decode: "Decode",
  complete: "已結束",
};
function csv(rows: Row[]) {
  const keys: (keyof Row)[] = [
    "id",
    "model",
    "phase",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "ttft_ms",
    "elapsed_s",
  ];
  const quote = (v: unknown) =>
    '"' +
    String(v ?? "")
      .replace(/^[=+\-@]/, "'$&")
      .replaceAll('"', '""') +
    '"';
  const blob = new Blob(
    [
      "\uFEFF" +
        [
          keys.join(","),
          ...rows.map((row) => keys.map((k) => quote(row[k])).join(",")),
        ].join("\r\n"),
    ],
    { type: "text/csv;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "yunshu-observed-requests.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function Requests({
  engine,
  connection,
  perform,
  busy,
}: {
  engine: Engine;
  connection: Connection;
  perform: Perform;
  busy: string | null;
}) {
  const [filter, setFilter] = useState("active"),
    [query, setQuery] = useState(""),
    [detail, setDetail] = useState<Row | null>(null),
    [cancel, setCancel] = useState<Row | null>(null),
    opener = useRef<HTMLButtonElement | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null),
    [detailUpdated, setDetailUpdated] = useState<number | null>(null);
  useEffect(() => {
    setDetailError(null);
    setDetailUpdated(null);
    if (!detail || detail.phase === "complete") return;
    const id = detail.id,
      controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined,
      ended = false;
    const poll = async () => {
      try {
        const value = await requestJson<Record<string, unknown>>(
          connection,
          `/requests/${encodeURIComponent(id)}`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        if (value.request_id !== id) throw Error("服務回傳了不同的 request ID");
        setDetail((current) =>
          current?.id === id ? ({ ...current, ...value, id } as Row) : current,
        );
        setDetailError(null);
        setDetailUpdated(Date.now());
      } catch (e) {
        if (controller.signal.aborted) return;
        ended = e instanceof ApiError && e.status === 404;
        setDetailError(
          ended
            ? "此請求已結束或已不在活動清單；下方保留最近採樣。"
            : e instanceof Error
              ? e.message
              : "無法取得請求詳情",
        );
      } finally {
        if (!controller.signal.aborted && !ended)
          timer = setTimeout(() => void poll(), 1500);
      }
    };
    void poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [detail?.id, connection.baseUrl, connection.token]);
  const rows = useMemo(() => {
    const observed = new Map<string, Row>();
    for (const sample of engine.history) {
      const last = sample.status.last;
      if (last)
        observed.set(last.request_id, {
          ...last,
          id: last.request_id,
          phase: "complete",
        });
    }
    for (const row of engine.status?.requests.items ?? [])
      observed.set(row.request_id, { ...row, id: row.request_id });
    return [...observed.values()].reverse();
  }, [engine.history, engine.status]);
  const shown = rows.filter(
    (row) =>
      (filter === "all" ||
        (filter === "active"
          ? row.phase !== "complete"
          : row.phase === "complete")) &&
      `${row.id} ${row.model ?? ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const restore = (e: Event) => {
    if (opener.current?.isConnected) {
      e.preventDefault();
      opener.current.focus();
    }
  };
  return (
    <section className="w-full max-w-5xl space-y-6" data-testid="requests">
      <PageHeader
        title="請求與效能"
        description="查看正在處理的工作，以及本頁觀測到的最近已結束請求。"
        actions={
          <Button
            size="sm"
            variant="secondary"
            disabled={!shown.length}
            onClick={() => csv(shown)}
          >
            <Download size={14} />
            匯出 CSV
          </Button>
        }
      />
      <div className="flex flex-wrap justify-between gap-3">
        <Input
          className="sm:max-w-xs"
          aria-label="搜尋請求"
          icon={<Search size={14} />}
          placeholder="Request ID 或模型"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <SegmentedSelect
          value={filter}
          onChange={setFilter}
          options={[
            { value: "active", label: "進行中" },
            { value: "complete", label: "觀測到的已結束請求" },
            { value: "all", label: "全部" },
          ]}
        />
      </div>
      <Card className="overflow-hidden">
        <Table scrollLabel="引擎請求清單" className="min-w-[720px]">
          <Thead>
            <Tr>
              <Th>請求</Th>
              <Th>階段</Th>
              <Th>輸入 / 快取</Th>
              <Th>輸出</Th>
              <Th>進度 / 時間</Th>
              <Th>操作</Th>
            </Tr>
          </Thead>
          <Tbody>
            {shown.map((row) => (
              <Tr key={row.id}>
                <Td>
                  <p className="max-w-44 truncate font-mono text-xs">
                    {row.id}
                  </p>
                  <p className="mt-1 max-w-44 truncate text-[11px] text-muted-foreground">
                    {row.model
                      ? modelLabel(row.model)
                      : row.t
                        ? clock(row.t * 1000)
                        : "模型尚未回報"}
                  </p>
                </Td>
                <Td>
                  <Badge variant={"secondary"}>
                    {labels[row.phase] ?? row.phase}
                  </Badge>
                </Td>
                <Td className="font-mono text-xs">
                  {number(row.prompt_tokens, 0)} /{" "}
                  {number(row.cached_tokens, 0)}
                </Td>
                <Td className="font-mono text-xs">
                  {number(row.completion_tokens, 0)}
                </Td>
                <Td>
                  {row.phase === "prefill" && row.percent != null ? (
                    <div className="w-24">
                      <Progress value={row.percent} label="Prefill 進度" />
                      <span className="mt-1 block text-xs">
                        {number(row.percent)}%
                      </span>
                    </div>
                  ) : (
                    <span className="text-xs">
                      {row.phase === "complete"
                        ? `${number(row.ttft_ms, 0)} ms TTFT`
                        : elapsed(row.elapsed_s)}
                    </span>
                  )}
                </Td>
                <Td>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.currentTarget.focus();
                        setDetail(row);
                      }}
                    >
                      詳情
                    </Button>
                    {row.phase !== "complete" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={!isOnline(engine) || !!busy}
                        onClick={(e) => {
                          opener.current = e.currentTarget;
                          setCancel(row);
                        }}
                      >
                        取消
                      </Button>
                    )}
                  </div>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
        {!shown.length && (
          <EmptyState
            title={engine.status ? "目前沒有符合條件的請求" : "等待請求資料"}
            description="完成記錄只包含開啟本頁後採樣到的最近請求，並非完整歷史。"
          />
        )}
      </Card>
      <p className="text-xs text-muted-foreground">
        服務目前只提供最新完成記錄；高頻請求之間可能有未觀測到的完成資料。此頁不把消失的活動請求推測成成功。
      </p>
      <Sheet
        open={!!detail}
        onClose={() => setDetail(null)}
        title="請求詳情"
        closeLabel="關閉請求詳情"
      >
        {detail && (
          <div className="space-y-5">
            <p className="break-all font-mono text-xs">{detail.id}</p>
            {detailError && (
              <p role="status" className="text-xs text-warning">
                {detailError}
              </p>
            )}
            {detailUpdated && (
              <p className="text-xs text-muted-foreground">
                即時更新 {clock(detailUpdated)}
              </p>
            )}
            <Badge>{labels[detail.phase] ?? detail.phase}</Badge>
            <dl className="grid grid-cols-2 gap-5">
              {[
                ["模型", detail.model ?? "未回報"],
                ["經過時間", elapsed(detail.elapsed_s)],
                ["Prompt tokens", number(detail.prompt_tokens, 0)],
                ["Cached tokens", number(detail.cached_tokens, 0)],
                ["Output tokens", number(detail.completion_tokens, 0)],
                ["首 Token 延遲", `${number(detail.ttft_ms)} ms`],
                ["Decode", `${number(detail.decode_tps)} tok/s`],
                [
                  "Prefill",
                  `${number(detail.prefill_tps ?? detail.tokens_per_second)} tok/s`,
                ],
              ].map(([name, value]) => (
                <div key={name}>
                  <dt className="text-xs text-muted-foreground">{name}</dt>
                  <dd className="mt-1 break-all text-sm">{value}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </Sheet>
      <Dialog
        open={!!cancel}
        onOpenChange={(open) => {
          if (!open) setCancel(null);
        }}
      >
        <DialogContent closeLabel="關閉取消確認" onCloseAutoFocus={restore}>
          <DialogTitle>取消這個請求？</DialogTitle>
          <DialogDescription>
            只取消 {cancel?.id}，其他請求不受影響。
          </DialogDescription>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setCancel(null)}>
              繼續執行
            </Button>
            <Button
              onClick={() => {
                const r = cancel;
                setCancel(null);
                if (r)
                  void perform(`cancel:${r.id}`, () =>
                    cancelRequest(connection, r.id),
                  );
              }}
            >
              確認取消
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
}
