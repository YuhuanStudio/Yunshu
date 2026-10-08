import { useEffect, useMemo, useState } from "react";
import { Banner } from "@yuhuanowo/yunui/patterns";
import {
  Button,
  ConfirmModal,
  EmptyState,
  Input,
  NumberInput,
  PasswordInput,
  ScrollFade,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusIndicator,
  Switch,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import {
  CheckCircle2,
  CircleAlert,
  RotateCcw,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import { ApiError, requestJson, type Connection } from "./api";
import {
  AdminError,
  openapiUrl,
  findReloadRoute,
  patchConfig,
  reloadModel,
  summarizePatch,
  validationErrors,
  type PatchResult,
} from "./admin-settings-api";
import {
  appliesLabel,
  canReset,
  draftIsValid,
  experimentalNames,
  fieldKind,
  filterConfig,
  formatConfigValue,
  isChanged,
  parseConfig,
  shownValue,
  sourceLabel,
  stabilityLabel,
  withDraft,
  type ConfigPayload,
  type ConfigRow,
  type Drafts,
} from "./config-view";
import { has, t, tr } from "./i18n/index.ts";
import { ErrorNote } from "./error-note";
import { RestartControl } from "./Service";
import { CopyField, SectionCard } from "./ui";

/** Effective settings from GET /v1/yunshu/config, editable through PATCH (admin). */
/** The localized description of a registry setting; a setting this console predates keeps the engine's own text. */
// i18n-keys: settingdesc.
const describe = (row: { name: string; description: string }) =>
  has(`settingdesc.${row.name}`)
    ? tr(`settingdesc.${row.name}`)
    : row.description;

export function ConfigView({
  connection,
  loadedModels = [],
}: {
  connection: Connection;
  /** Ids of the loaded models, for the 「重新載入模型」 action. */
  loadedModels?: string[];
}) {
  const [data, setData] = useState<ConfigPayload | null>(null),
    [state, setState] = useState<"loading" | "ok" | "missing" | "error">(
      "loading",
    ),
    [failure, setFailure] = useState<unknown>(null),
    [query, setQuery] = useState(""),
    [changedOnly, setChangedOnly] = useState(false),
    [all, setAll] = useState(false),
    [version, setVersion] = useState(0),
    [drafts, setDrafts] = useState<Drafts>({}),
    [errors, setErrors] = useState<Record<string, string>>({}),
    [busy, setBusy] = useState<"" | "preview" | "save">(""),
    [saveFailure, setSaveFailure] = useState<AdminError | null>(null),
    [result, setResult] = useState<PatchResult | null>(null),
    [confirmExp, setConfirmExp] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setState((s) => (s === "ok" ? s : "loading"));
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
  }, [connection.baseUrl, connection.token, all, version]);
  const rows = useMemo(
    () => filterConfig(data?.rows ?? [], query, changedOnly),
    [data, query, changedOnly],
  );
  const changed = (data?.rows ?? []).filter(isChanged).length;
  const byName = useMemo(
    () => new Map((data?.rows ?? []).map((r) => [r.name, r])),
    [data],
  );
  const dirty = Object.keys(drafts);
  const invalid = dirty.filter((n) => {
    const row = byName.get(n);
    return row ? !draftIsValid(row, drafts[n]) : false;
  });
  const experimental = experimentalNames(data?.rows ?? [], drafts);

  function edit(row: ConfigRow, value: unknown) {
    setDrafts((d) => withDraft(d, row, value));
    setErrors((e) => {
      if (!(row.name in e)) return e;
      const { [row.name]: _drop, ...rest } = e;
      void _drop;
      return rest;
    });
    setResult(null);
  }

  async function submit(dryRun: boolean, confirmed = false) {
    if (!dryRun && experimental.length && !confirmed) {
      setConfirmExp(true);
      return;
    }
    setConfirmExp(false);
    setBusy(dryRun ? "preview" : "save");
    setSaveFailure(null);
    setErrors({});
    try {
      const out = await patchConfig(connection, drafts, {
        dryRun,
        confirmExperimental: experimental.length > 0,
      });
      setResult(out);
      if (!dryRun) {
        setDrafts({});
        setVersion((v) => v + 1);
      }
    } catch (e) {
      const mapped = validationErrors(e);
      setErrors(mapped);
      setSaveFailure(
        e instanceof AdminError
          ? e
          : new AdminError(0, "network", t("settings.admin.error.network")),
      );
    } finally {
      setBusy("");
    }
  }

  const unsupported = saveFailure?.unsupported === true;
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
          {result && (
            <SaveSummary
              connection={connection}
              result={result}
              loadedModels={loadedModels}
              onDismiss={() => setResult(null)}
            />
          )}
          {saveFailure && (
            <div className="px-5 pb-3" data-testid="config-save-error">
              <Banner
                tone="warning"
                icon={<CircleAlert size={16} />}
                title={
                  Object.keys(errors).length
                    ? t("settings.config.invalidCount", {
                        count: Object.keys(errors).length,
                      })
                    : unsupported
                      ? t("settings.config.unsupported")
                      : saveFailure.message
                }
                description={
                  unsupported ? t("settings.config.unsupportedHelp") : undefined
                }
              />
            </div>
          )}
          <div
            className="flex min-h-12 flex-wrap items-center gap-3 border-t border-b border-border px-5 py-2"
            data-testid="config-actions"
          >
            <span className="mr-auto text-sm text-muted-foreground tabular-nums">
              {dirty.length
                ? t("settings.config.pending", { count: dirty.length })
                : t("settings.config.noChanges")}
            </span>
            <Button
              size="sm"
              variant="secondary"
              disabled={!dirty.length || busy !== ""}
              title={dirty.length ? undefined : t("settings.config.noChanges")}
              onClick={() => {
                setDrafts({});
                setErrors({});
                setSaveFailure(null);
              }}
            >
              {t("settings.config.discard")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={!dirty.length || invalid.length > 0 || busy !== ""}
              title={invalid.length ? t("settings.config.fixFirst") : undefined}
              onClick={() => void submit(true)}
            >
              {busy === "preview"
                ? t("settings.config.previewing")
                : t("settings.config.preview")}
            </Button>
            <Button
              size="sm"
              disabled={!dirty.length || invalid.length > 0 || busy !== ""}
              title={invalid.length ? t("settings.config.fixFirst") : undefined}
              onClick={() => void submit(false)}
            >
              {busy === "save"
                ? t("settings.config.saving")
                : t("settings.config.save")}
            </Button>
          </div>
          <ScrollFade className="max-h-[40rem] overflow-auto">
            <Table scrollLabel={t("diagnostics.config.table.aria")}>
              <Thead>
                <Tr>
                  <Th>{t("diagnostics.config.table.name")}</Th>
                  <Th>{t("diagnostics.config.table.value")}</Th>
                  <Th>{t("settings.config.applies")}</Th>
                  <Th>{t("diagnostics.config.table.source")}</Th>
                </Tr>
              </Thead>
              <Tbody>
                {rows.map((row) => (
                  <ConfigRowView
                    key={row.name}
                    row={row}
                    drafts={drafts}
                    error={errors[row.name]}
                    onEdit={(v) => edit(row, v)}
                  />
                ))}
              </Tbody>
            </Table>
          </ScrollFade>
          {!rows.length && (
            <EmptyState size="inline" title={t("diagnostics.config.empty")} />
          )}
        </>
      )}
      <ConfirmModal
        isOpen={confirmExp}
        variant="warning"
        title={t("settings.config.experimentalTitle")}
        message={
          <span>
            {t("settings.config.experimentalBody")}
            <span className="mt-2 block break-all font-mono text-xs">
              {experimental.join(", ")}
            </span>
          </span>
        }
        confirmText={t("settings.config.experimentalConfirm")}
        cancelText={t("settings.admin.cancel")}
        onConfirm={() => void submit(false, true)}
        onClose={() => setConfirmExp(false)}
      />
    </SectionCard>
  );
}

function ConfigRowView({
  row,
  drafts,
  error,
  onEdit,
}: {
  row: ConfigRow;
  drafts: Drafts;
  error?: string;
  onEdit: (value: unknown) => void;
}) {
  const edited = row.name in drafts;
  const forced = row.source === "env" || row.source === "cli";
  const applies = appliesLabel(row.applies);
  return (
    <Tr
      data-name={row.name}
      data-dirty={edited ? "true" : undefined}
      data-changed={isChanged(row) ? "true" : undefined}
      className={edited ? "bg-muted/60" : undefined}
    >
      <Td className="min-w-48 max-w-72 align-top">
        <span className="break-all font-mono text-xs">{row.name}</span>
        {describe(row) && (
          <span
            className="mt-0.5 line-clamp-2 block text-xs text-muted-foreground"
            title={describe(row)}
          >
            {describe(row)}
          </span>
        )}
        {row.stability !== "stable" && (
          <StatusIndicator
            className="mt-1 gap-1.5 text-xs text-muted-foreground"
            status="away"
            title={t("diagnostics.config.table.stability")}
          >
            {stabilityLabel(row.stability)}
          </StatusIndicator>
        )}
      </Td>
      <Td className="min-w-56 align-top">
        <div className="flex items-center gap-2">
          <div className="min-w-0 flex-1">
            <RowControl row={row} drafts={drafts} onEdit={onEdit} />
          </div>
          {(canReset(row) || edited) && (
            <Button
              size="sm"
              variant="ghost"
              type="button"
              aria-label={t("settings.config.resetAria", { name: row.name })}
              title={t("settings.config.reset")}
              onClick={() => onEdit(null)}
              disabled={drafts[row.name] === null}
            >
              <RotateCcw size={13} />
            </Button>
          )}
        </div>
        <p className="mt-1 flex flex-wrap gap-x-1.5 break-all text-xs text-muted-foreground">
          <span>{t("diagnostics.config.table.default")}</span>
          <span className="font-mono">{formatConfigValue(row.default)}</span>
        </p>
        {drafts[row.name] === null && (
          <p className="mt-1 text-xs">{t("settings.config.willReset")}</p>
        )}
        {error && (
          <p role="alert" className="mt-1 text-xs text-error">
            {error}
          </p>
        )}
        {forced && (
          <p className="mt-1 text-xs text-muted-foreground">
            {row.source === "env"
              ? t("settings.config.overriddenEnv", { name: row.name })
              : t("settings.config.overriddenCli")}
          </p>
        )}
      </Td>
      <Td className="align-top">
        {applies && (
          <StatusIndicator
            className="gap-1.5 whitespace-nowrap text-xs text-muted-foreground"
            status={
              row.applies === "live"
                ? "online"
                : row.applies === "reload"
                  ? "away"
                  : "busy"
            }
          >
            {applies}
          </StatusIndicator>
        )}
      </Td>
      <Td className="align-top">
        <span
          className={`text-xs ${row.source === "default" ? "text-muted-foreground" : "font-medium"}`}
        >
          {sourceLabel(row.source)}
        </span>
      </Td>
    </Tr>
  );
}

function RowControl({
  row,
  drafts,
  onEdit,
}: {
  row: ConfigRow;
  drafts: Drafts;
  onEdit: (value: unknown) => void;
}) {
  const value = shownValue(row, drafts);
  const kind = fieldKind(row);
  const label = row.name;
  if (kind === "bool")
    return (
      <Switch
        label={label}
        checked={value === true || value === "true"}
        onCheckedChange={onEdit}
      />
    );
  if (kind === "enum")
    return (
      <Select
        value={value == null ? undefined : String(value)}
        onValueChange={onEdit}
      >
        <SelectTrigger aria-label={label} className="w-full">
          <SelectValue placeholder={t("diagnostics.config.value.unset")} />
        </SelectTrigger>
        <SelectContent>
          {row.choices.map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  if (kind === "number")
    return (
      <NumberInput
        aria-label={label}
        value={typeof value === "number" ? value : undefined}
        min={row.minimum ?? undefined}
        step={row.type === "int" ? 1 : 0.1}
        placeholder={t("diagnostics.config.value.unset")}
        labels={{
          increment: t("settings.config.increment"),
          decrement: t("settings.config.decrement"),
        }}
        onChange={onEdit}
      />
    );
  if (kind === "secret")
    return (
      <div className="flex items-center gap-2">
        <PasswordInput
          aria-label={label}
          autoComplete="new-password"
          value={
            typeof drafts[row.name] === "string" ? String(drafts[row.name]) : ""
          }
          placeholder={
            row.value
              ? t("settings.config.secretSet")
              : t("settings.config.secretUnset")
          }
          labels={{
            show: t("settings.connection.showToken"),
            hide: t("settings.connection.hideToken"),
          }}
          onChange={(e) => onEdit(e.target.value)}
        />
      </div>
    );
  return (
    <Input
      aria-label={label}
      className="font-mono text-xs"
      value={
        value == null
          ? ""
          : typeof value === "string"
            ? value
            : JSON.stringify(value)
      }
      placeholder={t("diagnostics.config.value.unset")}
      onChange={(e) => onEdit(e.target.value)}
    />
  );
}

/** What a save (or a preview) did, and the one action still needed. */
function SaveSummary({
  connection,
  result,
  loadedModels,
  onDismiss,
}: {
  connection: Connection;
  result: PatchResult;
  loadedModels: string[];
  onDismiss: () => void;
}) {
  const sum = summarizePatch(result);
  const [canReload, setCanReload] = useState(false),
    [reloadState, setReloadState] = useState<
      "idle" | "busy" | "done" | "failed"
    >("idle");
  const needsReload = sum.needsReload.length > 0;
  useEffect(() => {
    if (!needsReload || result.dryRun) return;
    const controller = new AbortController();
    void fetch(openapiUrl(connection), {
      signal: controller.signal,
      headers: connection.token.trim()
        ? { Authorization: `Bearer ${connection.token.trim()}` }
        : undefined,
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((doc) => setCanReload(findReloadRoute(doc)))
      .catch(() => setCanReload(false));
    return () => controller.abort();
  }, [needsReload, result.dryRun, connection]);
  async function reload() {
    setReloadState("busy");
    try {
      for (const id of loadedModels) await reloadModel(connection, id);
      setReloadState("done");
    } catch {
      setReloadState("failed");
    }
  }
  const parts: string[] = [];
  if (sum.applied.length)
    parts.push(t("settings.config.sum.applied", { count: sum.applied.length }));
  if (sum.needsReload.length)
    parts.push(
      t("settings.config.sum.reload", { count: sum.needsReload.length }),
    );
  if (sum.needsRestart.length)
    parts.push(
      t("settings.config.sum.restart", { count: sum.needsRestart.length }),
    );
  if (sum.overridden.length)
    parts.push(
      t("settings.config.sum.overridden", { count: sum.overridden.length }),
    );
  const needsAction = sum.needsReload.length + sum.needsRestart.length > 0;
  return (
    <div className="space-y-3 px-5 pb-3" data-testid="config-summary">
      <Banner
        tone={needsAction || sum.overridden.length ? "warning" : "success"}
        icon={<CheckCircle2 size={16} />}
        title={
          result.dryRun
            ? t("settings.config.sum.previewTitle")
            : t("settings.config.sum.title")
        }
        description={parts.join(" · ")}
        dismissible
        onDismiss={onDismiss}
        dismissLabel={t("settings.admin.dismiss")}
      />
      {sum.overridden.length > 0 && (
        <ul className="space-y-1 text-xs text-muted-foreground">
          {sum.overridden.map((o) => (
            <li key={o.name} className="flex flex-wrap gap-x-2">
              <span className="font-mono">{o.name}</span>
              {o.source === "env"
                ? t("settings.config.overriddenEnv", { name: o.name })
                : t("settings.config.overriddenCli")}
            </li>
          ))}
        </ul>
      )}
      {!result.dryRun && needsReload && (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span className="text-muted-foreground">
            {t("settings.config.sum.reloadHelp")}
          </span>
          {canReload && loadedModels.length > 0 && (
            <Button
              size="sm"
              variant="secondary"
              disabled={reloadState === "busy"}
              onClick={() => void reload()}
            >
              {t("settings.config.reloadModel")}
            </Button>
          )}
          {reloadState === "done" && (
            <span role="status">{t("settings.config.reloaded")}</span>
          )}
          {reloadState === "failed" && (
            <span role="alert" className="text-error">
              {t("settings.config.reloadFailed")}
            </span>
          )}
        </div>
      )}
      {!result.dryRun && sum.needsRestart.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            {t("settings.config.sum.restartHelp")}
          </p>
          {result.restart?.available ? (
            <RestartControl connection={connection} />
          ) : (
            result.restart?.manual && (
              <CopyField
                label={t("settings.config.manualRestart")}
                value={result.restart.manual}
              />
            )
          )}
        </div>
      )}
    </div>
  );
}
