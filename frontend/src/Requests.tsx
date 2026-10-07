import { useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  EmptyState,
  Input,
  SegmentedSelect,
  Sheet,
  Sparkline,
  StatusIndicator,
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
  PrefillMeter,
  Readout,
  StageRail,
  TokenTrace,
  finishedFromHistory,
  isLive,
  median,
  phaseDot,
  phaseLabels as labels,
  prefillPercent,
  speculativeText,
  type Row,
} from "./RequestTrace";
import {
  clock,
  elapsed,
  isOnline,
  modelLabel,
  number,
  type Engine,
} from "./ui";
import type { Perform } from "./Models";
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
  const finished = useMemo(
    () => finishedFromHistory(engine.history),
    [engine.history],
  );
  const rows = useMemo(() => {
    const observed = new Map<string, Row>();
    for (const row of finished) observed.set(row.id, row);
    for (const row of engine.status?.requests.items ?? [])
      observed.set(row.request_id, { ...row, id: row.request_id } as Row);
    return [...observed.values()].reverse();
  }, [finished, engine.status]);
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
  const status = engine.status,
    activeNow = status?.requests.active ?? 0;
  const activeSeries = engine.history.map((p) => p.status.requests.active);
  const decodeSeries = engine.history.flatMap((p) => {
    const v =
      p.status.throughput.live_decode_tps ??
      p.status.throughput.mean_decode_tps;
    return v == null ? [] : [v];
  });
  const ttfts = finished.flatMap((r) => (r.ttft_ms == null ? [] : [r.ttft_ms]));
  const hits = finished.flatMap((r) =>
    (r.prompt_tokens ?? 0) > 0
      ? [((r.cached_tokens ?? 0) / (r.prompt_tokens ?? 1)) * 100]
      : [],
  );
  const promptSum = finished.reduce((n, r) => n + (r.prompt_tokens ?? 0), 0),
    cachedSum = finished.reduce((n, r) => n + (r.cached_tokens ?? 0), 0);
  const liveDecode =
    status?.throughput.live_decode_tps ?? status?.throughput.mean_decode_tps;
  const tiles = [
    {
      label: "進行中請求",
      value: status ? number(activeNow, 0) : "—",
      hint: status
        ? `排隊 ${number(status.requests.queued, 0)} · Prefill ${number(status.requests.prefill, 0)} · Decode ${number(status.requests.decode, 0)}`
        : undefined,
      data: activeSeries,
      tone: "accent" as const,
      name: "進行中請求數趨勢",
    },
    {
      label: "TTFT（已結束請求）",
      value: ttfts.length ? number(ttfts.at(-1), 0) : "—",
      unit: ttfts.length ? "ms" : undefined,
      hint: ttfts.length
        ? `中位 ${number(median(ttfts), 0)} ms · ${number(ttfts.length, 0)} 筆`
        : "尚未觀測到已結束請求",
      data: ttfts,
      tone: "info" as const,
      name: "已結束請求 TTFT 趨勢",
    },
    {
      label: "Decode 速度",
      value: number(liveDecode),
      unit: liveDecode == null ? undefined : "tok/s",
      hint: "引擎視窗平均",
      data: decodeSeries,
      tone: "success" as const,
      name: "Decode tok/s 趨勢",
    },
    {
      label: "前綴快取命中",
      value: promptSum > 0 ? number((cachedSum / promptSum) * 100, 0) : "—",
      unit: promptSum > 0 ? "%" : undefined,
      hint:
        promptSum > 0
          ? `已結束請求加權 · ${number(hits.length, 0)} 筆`
          : "尚未觀測到已結束請求",
      data: hits,
      tone: "accent" as const,
      name: "快取命中率趨勢",
    },
  ];
  return (
    <section
      className="mx-auto w-full max-w-6xl space-y-5"
      data-testid="requests"
    >
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
      <Card className="grid grid-cols-2 gap-x-6 gap-y-5 p-4 lg:grid-cols-4">
        {tiles.map((tile) => (
          <div key={tile.label} className="min-w-0 space-y-2">
            <Readout
              label={tile.label}
              value={tile.value}
              unit={tile.unit}
              hint={tile.hint}
            />
            {tile.data.length > 1 ? (
              <Sparkline
                data={tile.data.slice(-60)}
                tone={tile.tone}
                area
                height={28}
                className="h-7 w-full"
                label={tile.name}
              />
            ) : (
              <div className="h-7" />
            )}
          </div>
        ))}
      </Card>
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
        <Table scrollLabel="引擎請求清單" className="min-w-[860px]">
          <Thead>
            <Tr>
              <Th>請求</Th>
              <Th>輸入 / 快取</Th>
              <Th>輸出</Th>
              <Th>tok/s</Th>
              <Th>進度 / 時間</Th>
              <Th>推測解碼</Th>
              <Th>操作</Th>
            </Tr>
          </Thead>
          <Tbody>
            {shown.map((row) => {
              const percent =
                  row.phase === "prefill" ? prefillPercent(row) : null,
                speed =
                  row.phase === "complete"
                    ? row.decode_tps
                    : row.tokens_per_second;
              return (
                <Tr key={row.id}>
                  <Td>
                    <div className="flex items-center gap-2">
                      <StatusIndicator
                        status={phaseDot(row.phase)}
                        pulse={isLive(row.phase)}
                      />
                      <span className="text-xs font-medium">
                        {labels[row.phase] ?? row.phase}
                      </span>
                      <span className="max-w-40 truncate font-mono text-xs text-muted-foreground">
                        {row.id}
                      </span>
                    </div>
                    <p className="mt-1 max-w-72 truncate pl-4 text-[11px] text-muted-foreground">
                      {row.model
                        ? modelLabel(row.model)
                        : row.t
                          ? clock(row.t * 1000)
                          : "模型尚未回報"}
                    </p>
                  </Td>
                  <Td className="text-xs tabular-nums">
                    {number(row.prompt_tokens, 0)} /{" "}
                    <span className="text-muted-foreground">
                      {number(row.cached_tokens, 0)}
                    </span>
                  </Td>
                  <Td className="text-xs tabular-nums">
                    {number(row.completion_tokens, 0)}
                  </Td>
                  <Td className="text-xs tabular-nums">{number(speed)}</Td>
                  <Td className="text-xs tabular-nums">
                    {percent != null ? (
                      <span>
                        Prefill {number(percent, 0)}%
                        <span className="ml-2 text-muted-foreground">
                          {elapsed(row.elapsed_s)}
                        </span>
                      </span>
                    ) : row.phase === "complete" ? (
                      `${number(row.ttft_ms, 0)} ms TTFT`
                    ) : (
                      elapsed(row.elapsed_s)
                    )}
                  </Td>
                  <Td className="text-xs">
                    {speculativeText(row) ?? (
                      <span className="text-muted-foreground">—</span>
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
              );
            })}
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
        「已結束」為本頁開啟後觀測到的已完成請求（依 Request ID
        去重）。服務目前只提供最新一筆完成記錄，高頻請求之間可能有未觀測到的完成資料；此頁不把消失的活動請求推測成成功。
      </p>
      <Sheet
        open={!!detail}
        onClose={() => setDetail(null)}
        title="請求詳情"
        closeLabel="關閉請求詳情"
      >
        {detail && (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <StatusIndicator
                  status={phaseDot(detail.phase)}
                  pulse={isLive(detail.phase)}
                >
                  <span className="text-sm font-medium">
                    {labels[detail.phase] ?? detail.phase}
                  </span>
                </StatusIndicator>
              </div>
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
            </div>
            <section className="space-y-3" aria-label="階段與 token 組成">
              <StageRail phase={detail.phase} />
              {detail.phase === "queued" && (
                <div className="grid grid-cols-2 gap-5">
                  <Readout
                    label="佇列位置"
                    value={number(detail.queue_position, 0)}
                  />
                  <Readout
                    label="預估等待（服務端估計）"
                    value={number(detail.queue_est_wait_ms, 0)}
                    unit="ms"
                  />
                </div>
              )}
              <PrefillMeter row={detail} />
              <TokenTrace row={detail} />
            </section>
            <div className="grid grid-cols-2 gap-5">
              <Readout
                label="模型"
                value={detail.model ? modelLabel(detail.model) : "未回報"}
              />
              <Readout label="經過時間" value={elapsed(detail.elapsed_s)} />
              <Readout
                label="Prompt tokens"
                value={number(detail.prompt_tokens, 0)}
              />
              <Readout
                label="Cached tokens"
                value={number(detail.cached_tokens, 0)}
              />
              <Readout
                label="Output tokens"
                value={number(detail.completion_tokens, 0)}
              />
              <Readout
                label="首 Token 延遲"
                value={number(detail.ttft_ms)}
                unit="ms"
              />
              <Readout
                label="Decode"
                value={number(
                  detail.decode_tps ??
                    (detail.phase === "decode"
                      ? detail.tokens_per_second
                      : null),
                )}
                unit="tok/s"
              />
              <Readout
                label="Prefill"
                value={number(
                  detail.prefill_tps ??
                    (detail.phase === "prefill"
                      ? detail.tokens_per_second
                      : null),
                )}
                unit="tok/s"
              />
            </div>
            {detail.speculative && (
              <div className="grid grid-cols-3 gap-5 border-t border-border/60 pt-5">
                <Readout
                  label="推測解碼"
                  value={detail.speculative.mode ?? "—"}
                />
                <Readout
                  label="接受率"
                  value={number(
                    detail.speculative.acceptance_rate == null
                      ? null
                      : detail.speculative.acceptance_rate * 100,
                    0,
                  )}
                  unit="%"
                />
                <Readout
                  label="回合"
                  value={number(detail.speculative.rounds, 0)}
                />
              </div>
            )}
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
