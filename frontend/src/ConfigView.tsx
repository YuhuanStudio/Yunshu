import { useEffect, useMemo, useState } from "react";
import {
  StatusIndicator,
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
  sourceLabel,
  stabilityLabel,
  filterConfig,
  formatConfigValue,
  isChanged,
  parseConfig,
  type ConfigPayload,
} from "./config-view";
import { t } from "./i18n/index.ts";
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
        if (!parsed) throw new ApiError(t("diagnostics.config.badShape"));
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
      title={t("diagnostics.config.title")}
      description={
        state === "ok"
          ? t("diagnostics.config.description", {
              count: data?.rows.length ?? 0,
              changed,
            })
          : t("diagnostics.config.descriptionShort")
      }
      data-testid="config-view"
      className="min-w-0 overflow-hidden"
      bodyClassName="p-0"
    >
      {state === "missing" && (
        <EmptyState
          size="inline"
          title={t("diagnostics.config.missing.title")}
          description={t("diagnostics.config.missing.description")}
        />
      )}
      {state === "error" && (
        <div className="px-5 pb-5">
          <ErrorNote
            tone="warning"
            message={
              failure instanceof ApiError
                ? failure.publicMessage
                : t("diagnostics.config.error")
            }
            error={failure}
          />
        </div>
      )}
      {state === "loading" && (
        <p role="status" className="px-5 pb-5 text-sm text-muted-foreground">
          {t("diagnostics.config.loading")}
        </p>
      )}
      {state === "ok" && data && (
        <>
          <div className="flex flex-wrap items-center gap-4 px-5 pb-4 pt-3">
            <Input
              className="sm:max-w-sm"
              icon={<Search size={13} />}
              aria-label={t("diagnostics.config.search.aria")}
              placeholder={t("diagnostics.config.search.placeholder")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Switch
                label={t("diagnostics.config.changedOnly")}
                checked={changedOnly}
                onCheckedChange={setChangedOnly}
              />
              {t("diagnostics.config.changedOnly")}
            </span>
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Switch
                label={t("diagnostics.config.includeAll")}
                checked={all}
                onCheckedChange={setAll}
              />
              {t("diagnostics.config.includeAll")}
            </span>
            <span className="text-xs text-muted-foreground">
              {t("diagnostics.config.experimental", {
                count: data.experimentalCount,
                max: data.experimentalMax,
              })}
            </span>
          </div>
          {data.warnings.length > 0 && (
            <div className="px-5 pb-3">
              <ErrorNote
                tone="warning"
                message={t("diagnostics.config.warnings", {
                  count: data.warnings.length,
                })}
                detail={data.warnings.join("\n")}
              />
            </div>
          )}
          <ScrollFade className="max-h-[36rem] overflow-auto">
            <Table scrollLabel={t("diagnostics.config.table.aria")}>
              <Thead>
                <Tr>
                  <Th>{t("diagnostics.config.table.name")}</Th>
                  <Th>{t("diagnostics.config.table.value")}</Th>
                  <Th>{t("diagnostics.config.table.default")}</Th>
                  <Th>{t("diagnostics.config.table.source")}</Th>
                  <Th>{t("diagnostics.config.table.stability")}</Th>
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
                      <span
                        className={`text-xs ${row.source === "default" ? "text-muted-foreground" : "font-medium"}`}
                      >
                        {sourceLabel(row.source)}
                      </span>
                    </Td>
                    <Td>
                      <StatusIndicator
                        className="gap-1.5 text-xs text-muted-foreground"
                        status={row.stability === "stable" ? "neutral" : "away"}
                      >
                        {stabilityLabel(row.stability)}
                      </StatusIndicator>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </ScrollFade>
          {!rows.length && (
            <EmptyState size="inline" title={t("diagnostics.config.empty")} />
          )}
        </>
      )}
    </SectionCard>
  );
}
