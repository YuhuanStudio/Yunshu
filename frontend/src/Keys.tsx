import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Button,
  Checkbox,
  ConfirmModal,
  EmptyState,
  Input,
  Modal,
  NumberInput,
  Progress,
  Select,
  SegmentedSelect,
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
  ScrollFade,
  Tag,
} from "@yuhuanowo/yunui";
import { Banner, DashboardPage, PageHeader } from "@yuhuanowo/yunui/patterns";
import { BarChart3, KeyRound, Plus, TriangleAlert } from "lucide-react";
import type { Connection } from "./api";
import { AdminError } from "./admin-settings-api";
import {
  QUOTA_FIELDS,
  SCOPES,
  createKey,
  dailySeries,
  deleteKey,
  epochToLocal,
  getUsage,
  listKeys,
  localToEpoch,
  patchKey,
  quotaFill,
  rotateKey,
  type ApiKey,
  type CreatedKey,
  type KeyDraft,
  type Quotas,
  type Scope,
  type UsageDay,
} from "./admin-keys-api";
import { t, useLocale } from "./i18n/index.ts";
import { CopyField, SectionCard, dateTime, number, relative } from "./ui";

type Phase = "loading" | "ok" | "unsupported" | "denied" | "error";

const emptyQuotas = (): Quotas => ({
  requests_per_day: null,
  tokens_per_day: null,
  max_concurrent: null,
});

/** Multiple API keys: create (secret shown once), rotate, disable, delete, edit quotas, per-key usage. */
export default function Keys({ connection }: { connection: Connection }) {
  useLocale();
  const [keys, setKeys] = useState<ApiKey[]>([]),
    [usage, setUsage] = useState<UsageDay[]>([]),
    [phase, setPhase] = useState<Phase>("loading"),
    [editing, setEditing] = useState<ApiKey | "new" | null>(null),
    [created, setCreated] = useState<{
      what: "create" | "rotate";
      made: CreatedKey;
    } | null>(null),
    [deleting, setDeleting] = useState<ApiKey | null>(null),
    [rotating, setRotating] = useState<ApiKey | null>(null),
    [notice, setNotice] = useState(""),
    [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      listKeys(connection, controller.signal),
      getUsage(connection, 30, controller.signal).catch(() => []),
    ])
      .then(([k, u]) => {
        if (controller.signal.aborted) return;
        setKeys(k);
        setUsage(u);
        setPhase("ok");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setPhase(
          e instanceof AdminError
            ? e.unsupported
              ? "unsupported"
              : e.denied
                ? "denied"
                : "error"
            : "error",
        );
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, tick]);

  async function act(run: () => Promise<void>) {
    setNotice("");
    try {
      await run();
      refresh();
    } catch (e) {
      setNotice(
        e instanceof AdminError ? e.message : t("settings.admin.error.network"),
      );
    }
  }

  return (
    <DashboardPage data-testid="keys">
      <PageHeader
        title={t("keys.title")}
        description={t("keys.description")}
        actions={
          phase === "ok" ? (
            <Button size="sm" onClick={() => setEditing("new")}>
              <Plus size={14} />
              {t("keys.create")}
            </Button>
          ) : undefined
        }
      />
      {phase === "loading" && (
        <p role="status" className="text-sm text-muted-foreground">
          {t("keys.loading")}
        </p>
      )}
      {(phase === "unsupported" || phase === "denied" || phase === "error") && (
        <EmptyState
          title={t(`keys.unavailable.${phase}.title`)}
          description={t(`keys.unavailable.${phase}.description`)}
        />
      )}
      {created && (
        <SecretCard
          what={created.what}
          made={created.made}
          onDone={() => setCreated(null)}
        />
      )}
      {notice && (
        <p role="alert" className="text-sm text-error">
          {notice}
        </p>
      )}
      {phase === "ok" && (
        <>
          <SectionCard
            icon={KeyRound}
            title={t("keys.list.title")}
            description={t("keys.list.description", { count: keys.length })}
            className="min-w-0 overflow-hidden"
            bodyClassName="p-0"
            data-testid="keys-list"
          >
            {keys.length === 0 ? (
              <EmptyState
                size="inline"
                title={t("keys.empty.title")}
                description={t("keys.empty.description")}
              />
            ) : (
              <ScrollFade className="overflow-auto">
                <Table scrollLabel={t("keys.table.aria")}>
                  <Thead>
                    <Tr>
                      <Th>{t("keys.table.name")}</Th>
                      <Th>{t("keys.table.scopes")}</Th>
                      <Th>{t("keys.table.requests")}</Th>
                      <Th>{t("keys.table.tokens")}</Th>
                      <Th>{t("keys.table.lastUsed")}</Th>
                      <Th>{t("keys.table.expires")}</Th>
                      <Th>{t("keys.table.enabled")}</Th>
                      <Th>
                        <span className="sr-only">
                          {t("keys.table.actions")}
                        </span>
                      </Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {keys.map((k) => (
                      <KeyRow
                        key={k.id}
                        k={k}
                        onToggle={(on) =>
                          void act(
                            async () =>
                              void (await patchKey(connection, k.id, {
                                enabled: on,
                              })),
                          )
                        }
                        onEdit={() => setEditing(k)}
                        onRotate={() => setRotating(k)}
                        onDelete={() => setDeleting(k)}
                      />
                    ))}
                  </Tbody>
                </Table>
              </ScrollFade>
            )}
          </SectionCard>
          <UsageChart keys={keys} usage={usage} />
        </>
      )}
      <KeyForm
        key={editing === "new" ? "new" : (editing?.id ?? "closed")}
        target={editing}
        onClose={() => setEditing(null)}
        onSubmit={async (draft) => {
          if (editing === "new") {
            const made = await createKey(connection, draft);
            setCreated({ what: "create", made });
          } else if (editing) {
            await patchKey(connection, editing.id, draft);
          }
          setEditing(null);
          refresh();
        }}
      />
      <ConfirmModal
        isOpen={rotating !== null}
        variant="warning"
        title={t("keys.rotate.title")}
        message={t("keys.rotate.body", { name: rotating?.name ?? "" })}
        confirmText={t("keys.rotate.confirm")}
        cancelText={t("settings.admin.cancel")}
        onClose={() => setRotating(null)}
        onConfirm={() => {
          const k = rotating;
          setRotating(null);
          if (k)
            void act(async () => {
              setCreated({
                what: "rotate",
                made: await rotateKey(connection, k.id),
              });
            });
        }}
      />
      <ConfirmModal
        isOpen={deleting !== null}
        variant="danger"
        title={t("keys.delete.title")}
        message={t("keys.delete.body", { name: deleting?.name ?? "" })}
        confirmText={t("keys.delete.confirm")}
        cancelText={t("settings.admin.cancel")}
        onClose={() => setDeleting(null)}
        onConfirm={() => {
          const k = deleting;
          setDeleting(null);
          if (k) void act(() => deleteKey(connection, k.id));
        }}
      />
    </DashboardPage>
  );
}

function SecretCard({
  what,
  made,
  onDone,
}: {
  what: "create" | "rotate";
  made: CreatedKey;
  onDone: () => void;
}) {
  return (
    <div className="space-y-3" data-testid="key-secret">
      <Banner
        tone="warning"
        icon={<TriangleAlert size={16} />}
        title={
          what === "create"
            ? t("keys.secret.titleCreate", { name: made.key.name })
            : t("keys.secret.titleRotate", { name: made.key.name })
        }
        description={t("keys.secret.warning")}
      />
      <CopyField label={t("keys.secret.label")} value={made.secret} />
      <p className="text-xs text-muted-foreground">
        {t("keys.secret.use", { env: "YUNSHU_AUTH_TOKEN" })}
      </p>
      <Button size="sm" variant="secondary" onClick={onDone}>
        {t("keys.secret.done")}
      </Button>
    </div>
  );
}

function QuotaCell({
  used,
  quota,
  label,
}: {
  used: number;
  quota: number | null;
  label: string;
}) {
  const fill = quotaFill(used, quota);
  return (
    <div className="min-w-32">
      <div className="flex items-baseline gap-1 text-sm tabular-nums">
        <span>{number(used, 0)}</span>
        <span className="text-xs text-muted-foreground">
          {quota ? `/ ${number(quota, 0)}` : t("keys.unlimited")}
        </span>
      </div>
      {fill != null && (
        <Progress
          className="mt-1 h-1"
          value={Math.min(100, fill * 100)}
          label={label}
        />
      )}
    </div>
  );
}

function KeyRow({
  k,
  onToggle,
  onEdit,
  onRotate,
  onDelete,
}: {
  k: ApiKey;
  onToggle: (on: boolean) => void;
  onEdit: () => void;
  onRotate: () => void;
  onDelete: () => void;
}) {
  return (
    <Tr data-key-id={k.id}>
      <Td className="min-w-40 align-top">
        <span className="block text-sm font-medium">{k.name}</span>
        <span className="font-mono text-xs text-muted-foreground">
          {k.prefix}…
        </span>
      </Td>
      <Td className="whitespace-nowrap align-top">
        <span className="flex gap-1">
          {k.scopes.map((s) => (
            <Tag key={s} className="whitespace-nowrap">
              {t(`keys.scope.${s}`)}
            </Tag>
          ))}
        </span>
      </Td>
      <Td className="align-top">
        <QuotaCell
          used={k.window.requests}
          quota={k.quotas.requests_per_day}
          label={t("keys.table.requests")}
        />
      </Td>
      <Td className="align-top">
        <QuotaCell
          used={k.window.tokens}
          quota={k.quotas.tokens_per_day}
          label={t("keys.table.tokens")}
        />
      </Td>
      <Td className="whitespace-nowrap align-top text-sm text-muted-foreground">
        {k.lastUsed == null
          ? "—"
          : relative(Math.max(0, Date.now() / 1000 - k.lastUsed))}
      </Td>
      <Td className="whitespace-nowrap align-top text-sm">
        {k.expires == null ? (
          <span className="text-muted-foreground">{t("keys.never")}</span>
        ) : k.expired ? (
          <StatusIndicator status="offline" className="gap-1.5 text-sm">
            {t("keys.expired")}
          </StatusIndicator>
        ) : (
          dateTime(k.expires * 1000)
        )}
      </Td>
      <Td className="align-top">
        <Switch
          label={t("keys.toggleAria", { name: k.name })}
          checked={k.enabled}
          onCheckedChange={onToggle}
        />
      </Td>
      <Td className="whitespace-nowrap align-top">
        <span className="flex gap-1">
          <Button size="sm" variant="ghost" onClick={onEdit}>
            {t("keys.edit")}
          </Button>
          <Button size="sm" variant="ghost" onClick={onRotate}>
            {t("keys.rotate.button")}
          </Button>
          <Button size="sm" variant="ghost" onClick={onDelete}>
            {t("keys.delete.button")}
          </Button>
        </span>
      </Td>
    </Tr>
  );
}

function KeyForm({
  target,
  onClose,
  onSubmit,
}: {
  target: ApiKey | "new" | null;
  onClose: () => void;
  onSubmit: (draft: KeyDraft) => Promise<void>;
}) {
  const key = target && target !== "new" ? target : null;
  const [name, setName] = useState(key?.name ?? ""),
    [scopes, setScopes] = useState<Scope[]>(key?.scopes ?? ["infer"]),
    [quotas, setQuotas] = useState<Quotas>(key?.quotas ?? emptyQuotas()),
    [expires, setExpires] = useState(epochToLocal(key?.expires ?? null)),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const toggle = (s: Scope, on: boolean) =>
    setScopes((cur) =>
      on ? [...new Set([...cur, s])] : cur.filter((x) => x !== s),
    );
  async function submit() {
    if (!name.trim()) return setError(t("keys.form.nameRequired"));
    if (!scopes.length) return setError(t("keys.form.scopeRequired"));
    setBusy(true);
    setError("");
    try {
      await onSubmit({ name, scopes, quotas, expires: localToEpoch(expires) });
    } catch (e) {
      setError(
        e instanceof AdminError ? e.message : t("settings.admin.error.network"),
      );
      setBusy(false);
    }
  }
  return (
    <Modal
      isOpen={target !== null}
      onClose={onClose}
      title={key ? t("keys.form.editTitle") : t("keys.form.createTitle")}
      footer={
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            {t("settings.admin.cancel")}
          </Button>
          <Button disabled={busy} onClick={() => void submit()}>
            {key ? t("keys.form.save") : t("keys.form.create")}
          </Button>
        </div>
      }
    >
      <div className="space-y-4" data-testid="key-form">
        <label className="block text-xs text-muted-foreground">
          {t("keys.form.name")}
          <Input
            className="mt-1.5 text-sm"
            value={name}
            maxLength={80}
            placeholder={t("keys.form.namePlaceholder")}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <fieldset className="space-y-2">
          <legend className="text-xs text-muted-foreground">
            {t("keys.form.scopes")}
          </legend>
          {SCOPES.map((s) => (
            <label key={s} className="flex items-start gap-2 text-sm">
              <Checkbox
                aria-label={t(`keys.scope.${s}`)}
                checked={scopes.includes(s)}
                onCheckedChange={(on) => toggle(s, on)}
              />
              <span>
                {t(`keys.scope.${s}`)}
                <span className="block text-xs text-muted-foreground">
                  {t(`keys.scope.${s}Help`)}
                </span>
              </span>
            </label>
          ))}
          {scopes.includes("admin") && (
            <p className="text-xs text-error">{t("keys.form.adminWarning")}</p>
          )}
        </fieldset>
        <div className="grid gap-3 sm:grid-cols-3">
          {QUOTA_FIELDS.map((f) => (
            <label key={f} className="block text-xs text-muted-foreground">
              {t(`keys.quota.${f}`)}
              <NumberInput
                className="mt-1.5"
                min={0}
                step={f === "max_concurrent" ? 1 : 100}
                value={quotas[f] ?? 0}
                labels={{
                  increment: t("settings.config.increment"),
                  decrement: t("settings.config.decrement"),
                }}
                onChange={(v) => setQuotas({ ...quotas, [f]: v || null })}
              />
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {t("keys.form.quotaHelp")}
        </p>
        <label className="block text-xs text-muted-foreground">
          {t("keys.form.expires")}
          <Input
            type="datetime-local"
            className="mt-1.5 text-sm sm:w-64"
            value={expires}
            onChange={(e) => setExpires(e.target.value)}
          />
        </label>
        {error && (
          <p role="alert" className="text-sm text-error">
            {error}
          </p>
        )}
      </div>
    </Modal>
  );
}

function UsageChart({ keys, usage }: { keys: ApiKey[]; usage: UsageDay[] }) {
  const [keyId, setKeyId] = useState("all"),
    [days, setDays] = useState<"7" | "14" | "30">("14");
  const series = useMemo(
    () => dailySeries(usage, keyId === "all" ? null : keyId, Number(days)),
    [usage, keyId, days],
  );
  const total = series.reduce((n, p) => n + p.requests, 0);
  const datum = (pick: (p: (typeof series)[number]) => number) =>
    series.map((p) => ({
      id: p.day,
      label: p.day.slice(5),
      value: pick(p),
    }));
  return (
    <SectionCard
      icon={BarChart3}
      title={t("keys.usage.title")}
      description={t("keys.usage.description")}
      data-testid="keys-usage"
      action={
        <div className="flex flex-wrap items-center gap-3">
          <Select value={keyId} onValueChange={setKeyId}>
            <SelectTrigger
              aria-label={t("keys.usage.keyAria")}
              className="w-44"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("keys.usage.allKeys")}</SelectItem>
              {keys.map((k) => (
                <SelectItem key={k.id} value={k.id}>
                  {k.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <SegmentedSelect
            options={[
              { value: "7", label: t("keys.usage.days", { count: 7 }) },
              { value: "14", label: t("keys.usage.days", { count: 14 }) },
              { value: "30", label: t("keys.usage.days", { count: 30 }) },
            ]}
            value={days}
            onChange={setDays}
          />
        </div>
      }
    >
      {total === 0 ? (
        <p className="pt-4 text-sm text-muted-foreground">
          {t("keys.usage.empty")}
        </p>
      ) : (
        <div className="grid gap-6 pt-4">
          <div className="min-w-0">
            <p className="mb-2 text-xs text-muted-foreground">
              {t("keys.usage.requests")}
            </p>
            <BarChart
              data={datum((p) => p.requests)}
              ariaLabel={t("keys.usage.requests")}
              formatValue={(v) => number(v, 0)}
              height={160}
            />
          </div>
          <div className="min-w-0">
            <p className="mb-2 text-xs text-muted-foreground">
              {t("keys.usage.tokens")}
            </p>
            <BarChart
              data={datum((p) => p.tokens)}
              ariaLabel={t("keys.usage.tokens")}
              formatValue={(v) => number(v, 0)}
              height={160}
            />
          </div>
        </div>
      )}
    </SectionCard>
  );
}
