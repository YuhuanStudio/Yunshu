import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  EmptyState,
  Input,
  ScrollFade,
  Switch,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { Search, SlidersHorizontal } from "lucide-react";
import { ApiError, requestJson, type Connection } from "./api";
import {
  SOURCE_LABEL,
  STABILITY_LABEL,
  filterConfig,
  formatConfigValue,
  isChanged,
  parseConfig,
  type ConfigPayload,
} from "./config-view";
import { ErrorNote } from "./error-note";
import { SectionCard } from "./ui";

/** Effective settings from GET /v1/yunshu/config: read-only, secrets masked. */
export function ConfigView({ connection }: { connection: Connection }) {
  const [data, setData] = useState<ConfigPayload | null>(null),
    [state, setState] = useState<"loading" | "ok" | "missing" | "error">(
      "loading",
    ),
    [failure, setFailure] = useState<unknown>(null),
    [query, setQuery] = useState(""),
    [changedOnly, setChangedOnly] = useState(false),
    [all, setAll] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setState("loading");
    void requestJson<unknown>(connection, "/yunshu/config", {
      signal: controller.signal,
      search: { include: all ? "all" : "stable" },
    })
      .then((payload) => {
        if (controller.signal.aborted) return;
        const parsed = parseConfig(payload);
        if (!parsed) throw new ApiError("引擎回傳的資料格式不符合預期。");
        setData(parsed);
        setState("ok");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setFailure(e);
        setState(
          e instanceof ApiError && e.status === 404 ? "missing" : "error",
        );
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, all]);
  const rows = useMemo(
    () => filterConfig(data?.rows ?? [], query, changedOnly),
    [data, query, changedOnly],
  );
  const changed = (data?.rows ?? []).filter(isChanged).length;
  return (
    <SectionCard
      icon={SlidersHorizontal}
      title="有效設定"
      description={
        state === "ok"
          ? `引擎實際生效的設定，共 ${data?.rows.length ?? 0} 項，${changed} 項與預設不同。唯讀；修改請用 yunshu config 或環境變數並重新啟動。`
          : "引擎實際生效的設定與來源。"
      }
      data-testid="config-view"
      className="min-w-0 overflow-hidden"
      bodyClassName="p-0"
    >
      {state === "missing" && (
        <EmptyState
          size="inline"
          title="這個引擎尚未提供有效設定"
          description="需要提供 /v1/yunshu/config 的引擎版本；請以命令列 yunshu config 查看。"
        />
      )}
      {state === "error" && (
        <div className="px-5 pb-5">
          <ErrorNote
            tone="warning"
            message={
              failure instanceof ApiError
                ? failure.publicMessage
                : "無法取得有效設定。"
            }
            error={failure}
          />
        </div>
      )}
      {state === "loading" && (
        <p role="status" className="px-5 pb-5 text-sm text-muted-foreground">
          正在讀取有效設定…
        </p>
      )}
      {state === "ok" && data && (
        <>
          <div className="flex flex-wrap items-center gap-4 px-5 pb-4 pt-3">
            <Input
              className="sm:max-w-sm"
              icon={<Search size={13} />}
              aria-label="搜尋設定"
              placeholder="搜尋名稱、值或類別"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Switch
                label="只看與預設不同"
                checked={changedOnly}
                onCheckedChange={setChangedOnly}
              />
              只看與預設不同
            </span>
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Switch
                label="含實驗與內部設定"
                checked={all}
                onCheckedChange={setAll}
              />
              含實驗與內部設定
            </span>
            <span className="text-xs text-muted-foreground">
              實驗旗標 {data.experimentalCount} / {data.experimentalMax}
            </span>
          </div>
          {data.warnings.length > 0 && (
            <div className="px-5 pb-3">
              <ErrorNote
                tone="warning"
                message={`有 ${data.warnings.length} 則設定警告。`}
                detail={data.warnings.join("\n")}
              />
            </div>
          )}
          <ScrollFade className="max-h-[36rem] overflow-auto">
            <Table scrollLabel="有效設定">
              <Thead>
                <Tr>
                  <Th>名稱</Th>
                  <Th>值</Th>
                  <Th>預設</Th>
                  <Th>來源</Th>
                  <Th>穩定度</Th>
                </Tr>
              </Thead>
              <Tbody>
                {rows.map((row) => (
                  <Tr
                    key={row.name}
                    data-changed={isChanged(row) ? "true" : undefined}
                    className={isChanged(row) ? "bg-muted/60" : undefined}
                  >
                    <Td>
                      <span
                        className="break-all font-mono text-xs"
                        title={row.description}
                      >
                        {row.name}
                      </span>
                    </Td>
                    <Td>
                      <span
                        className={`break-all font-mono text-xs ${isChanged(row) ? "font-semibold" : ""}`}
                      >
                        {formatConfigValue(row.value)}
                      </span>
                    </Td>
                    <Td>
                      <span className="break-all font-mono text-xs text-muted-foreground">
                        {formatConfigValue(row.default)}
                      </span>
                    </Td>
                    <Td>
                      <Badge
                        variant={row.source === "default" ? "outline" : "info"}
                      >
                        {SOURCE_LABEL[row.source] ?? row.source}
                      </Badge>
                    </Td>
                    <Td>
                      <Badge
                        variant={
                          row.stability === "stable" ? "outline" : "warning"
                        }
                      >
                        {STABILITY_LABEL[row.stability] ?? row.stability}
                      </Badge>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </ScrollFade>
          {!rows.length && <EmptyState size="inline" title="沒有符合的設定" />}
        </>
      )}
    </SectionCard>
  );
}
