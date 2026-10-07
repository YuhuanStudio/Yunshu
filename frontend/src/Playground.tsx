import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  CustomSelect,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  toast,
  StatusIndicator,
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  FileDropzone,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  Sheet,
  Slider,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
} from "@yuhuanowo/yunui";
import {
  ChatComposer,
  ChatHeader,
  ChatMessage,
  ChatMessageList,
  GenerationStats,
} from "@yuhuanowo/yunui/chat";
import {
  CapabilityIcon,
  getModelDeveloperId,
  getProviderName,
  ModelSelect,
  ThinkingBlock,
  type ModelSelectOption,
} from "@yuhuanowo/yunui/ai";
import { PageHeader } from "@yuhuanowo/yunui/patterns";
import {
  SlidersHorizontal,
  Plus,
  Sparkles,
  ImagePlus,
  X,
  Code2,
  MoreHorizontal,
} from "lucide-react";
import {
  describeStreamError,
  streamCompletion,
  type CompletionBody,
  type Dialect,
} from "./stream";
import { ErrorNote } from "./error-note";
import { UNDO_WINDOW_MS, thinkingOpen } from "./playground-ui-state";

const UNDO_TOAST_ID = "playground-undo";
import {
  buildSnippets,
  DIALECT_LABEL,
  type CodeLanguage,
} from "./playground-code";
import { loadModel, type Connection } from "./api";
import {
  LocalModelIcon,
  modelLabel,
  fixed,
  number,
  sizeGb,
  supportsChat,
  type Engine,
} from "./ui";
import { compareOutputs, runStats, type RunTiming } from "./playground-metrics";
import { SegmentedTray } from "./SegmentedTray";
import { t, useLocale } from "./i18n/index.ts";
const CodeBlock = lazy(() =>
  import("@yuhuanowo/yunui/content").then((m) => ({ default: m.CodeBlock })),
);
const Markdown = lazy(() =>
  import("@yuhuanowo/yunui/content").then((m) => ({
    default: m.MarkdownRenderer,
  })),
);
type Run = {
  content: string;
  reasoning?: string;
  model: string;
  temperature: number;
  incomplete?: boolean;
  finishReason?: string;
  timing?: RunTiming;
};
type Message = Run & {
  id: string;
  role: "user" | "assistant";
  image?: { name: string; url: string };
};
type Mode = "chat" | "compare";
const dialectOptions = (["chat", "responses", "messages"] as Dialect[]).map(
  (value) => ({ value, label: DIALECT_LABEL[value] }),
);
const codeTabs: { value: CodeLanguage; label: string; language: string }[] = [
  { value: "curl", label: "curl", language: "bash" },
  { value: "python", label: "Python", language: "python" },
  { value: "javascript", label: "JavaScript", language: "javascript" },
];
type Pair = readonly [Run | null, Run | null];
const statLabels = () => ({
  tokens: "tokens",
  speed: "tok/s",
  latency: "ms",
  ttft: t("playground.stat.ttft"),
  cached: t("playground.stat.cached"),
  prompt: t("playground.stat.prompt"),
});
const seconds = (ms: number) => `${number(ms / 1000, 2)} s`;
const signed = (v: number, digits: number) =>
  `${v > 0 ? "+" : v < 0 ? "−" : ""}${number(Math.abs(v), digits)}`;

/** Speculative decoding as reported by the engine: mode, tokens committed per
 *  verify round and draft acceptance. Only what x_yunshu.speculative carries. */
function specLabel(
  spec: { mode: string; rounds?: number; acceptanceRate?: number },
  tokens: number | undefined,
) {
  const mode = spec.mode.toUpperCase();
  const perRound =
    spec.rounds && tokens
      ? t("playground.stat.perRound", { n: number(tokens / spec.rounds, 2) })
      : "";
  const rate =
    spec.acceptanceRate != null
      ? t("playground.stat.acceptance", {
          n: number(spec.acceptanceRate * 100, 0),
        })
      : "";
  return `${mode}${perRound}${rate}`;
}

/** Per-reply engine stats; live while `now` ticks, final once `endAt` is set. */
function ReplyStats({ run, now }: { run: Run; now: number }) {
  if (!run.timing) return null;
  const s = runStats(run.timing, now),
    live = run.timing.endAt === undefined;
  return (
    <div
      className="flex flex-wrap items-center gap-1.5 tabular-nums"
      data-testid="reply-stats"
    >
      <GenerationStats
        tokens={s.tokens}
        tokensPerSecond={s.tokensPerSecond}
        latencyMs={Math.round(s.latencyMs)}
        ttftMs={s.ttftMs}
        cachedTokens={s.cachedTokens}
        promptTokens={s.promptTokens}
        labels={statLabels()}
      />
      {live && s.ttftMs === undefined && (
        <span className="text-xs text-muted-foreground">
          {t("playground.stat.waitingFirst")}
        </span>
      )}
      {run.timing.usage?.spec && (
        <span className="text-xs text-muted-foreground">
          {specLabel(run.timing.usage.spec, s.tokens)}
        </span>
      )}
      {s.estimated && !live && (
        <span className="text-xs text-muted-foreground">
          {t("playground.stat.estimated")}
        </span>
      )}
    </div>
  );
}

function ReplyBody({ run, streaming }: { run: Run; streaming: boolean }) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  return (
    <>
      {run.reasoning && (
        <ThinkingBlock
          labels={{
            title: t("playground.reply.thinkTitle"),
            active: t("playground.reply.thinkActive"),
            completed: t("playground.reply.thinkDone"),
            inProgress: t("playground.reply.thinkProgress"),
          }}
          content={run.reasoning}
          isStreaming={streaming}
          open={thinkingOpen(streaming, !!run.content, userOpen)}
          onOpenChange={setUserOpen}
        />
      )}
      <Suspense
        fallback={
          <div className="whitespace-pre-wrap text-sm leading-7">
            {run.content}
          </div>
        }
      >
        <Markdown
          content={
            run.content ||
            (streaming
              ? t("playground.reply.waiting")
              : t("playground.reply.noText"))
          }
        />
      </Suspense>
      {run.finishReason === "length" && (
        <p className="mt-2 text-xs text-warning">
          {t("playground.reply.lengthLimit")}
        </p>
      )}
      {run.incomplete && (
        <p className="mt-2 text-xs text-muted-foreground">
          {t("playground.reply.incomplete")}
        </p>
      )}
    </>
  );
}
export function Playground({
  connection,
  engine,
  initialModel,
}: {
  connection: Connection;
  engine: Engine;
  initialModel: string;
}) {
  useLocale();
  const [model, setModel] = useState(initialModel),
    [mode, setMode] = useState<Mode>("chat"),
    [dialect, setDialect] = useState<Dialect>("chat"),
    [codeOpen, setCodeOpen] = useState(false),
    [codeTab, setCodeTab] = useState<CodeLanguage>("curl"),
    [modelB, setModelB] = useState(""),
    [tempMode, setTempMode] = useState<readonly [string, string]>([
      "shared",
      "shared",
    ]),
    [pair, setPair] = useState<Pair>([null, null]),
    [pairPrompt, setPairPrompt] = useState<Message | null>(null),
    [now, setNow] = useState(() => performance.now()),
    [draft, setDraft] = useState(""),
    [messages, setMessages] = useState<Message[]>([]),
    [loading, setLoading] = useState(false),
    [error, setErrorState] = useState<{
      message: string;
      detail?: string;
    } | null>(null),
    [settings, setSettings] = useState(false),
    [temperature, setTemperature] = useState(0.7),
    [system, setSystem] = useState(""),
    [maxTokens, setMaxTokens] = useState(512),
    [thinking, setThinking] = useState("auto"),
    [jsonMode, setJsonMode] = useState("text"),
    [image, setImage] = useState<{ name: string; url: string } | null>(null),
    [attachmentOpen, setAttachmentOpen] = useState(false),
    [readingImage, setReadingImage] = useState(false);
  const setError = (message: string) =>
    setErrorState(message ? { message } : null);
  const controller = useRef<AbortController | null>(null),
    mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
    };
  }, []);
  useEffect(() => () => toast.dismiss(UNDO_TOAST_ID), []);
  // Esc stops a running generation (unless a dialog has the key); the composer
  // takes focus on arrival where there is a keyboard.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (
        e.key !== "Escape" ||
        !controller.current ||
        e.defaultPrevented ||
        document.querySelector('[role="dialog"]')
      )
        return;
      controller.current.abort();
    };
    addEventListener("keydown", onKey);
    if (matchMedia("(pointer: fine)").matches)
      document
        .querySelector<HTMLTextAreaElement>(
          '[data-testid="playground"] textarea',
        )
        ?.focus({ preventScroll: true });
    return () => removeEventListener("keydown", onKey);
  }, []);
  const [loadingModels, setLoadingModels] = useState<ReadonlySet<string>>(
    new Set(),
  );
  /** Load an unloaded model here, so a compare never needs a trip to Models. */
  async function loadHere(id: string) {
    setLoadingModels((s) => new Set(s).add(id));
    try {
      await loadModel(connection, id);
    } catch (e) {
      if (mounted.current) setErrorState(describeStreamError(e));
    } finally {
      await engine.refresh();
      if (mounted.current)
        setLoadingModels((s) => {
          const next = new Set(s);
          next.delete(id);
          return next;
        });
    }
  }
  useEffect(() => {
    if (!loading) return;
    const timer = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [loading]);
  const models = engine.status?.models ?? [],
    chosen = models.find((m) => m.id === model),
    chosenB = models.find((m) => m.id === modelB),
    compare = mode === "compare",
    runnable = (m: typeof chosen) =>
      engine.phase === "online" && !!m?.loaded && supportsChat(m),
    canSend = compare
      ? runnable(chosen) && runnable(chosenB)
      : runnable(chosen);
  useEffect(() => {
    if (!model && models.length)
      setModel(models.find((m) => m.loaded)?.id ?? models[0].id);
  }, [model, models]);
  useEffect(() => {
    if (compare && !modelB && models.length)
      setModelB(models.find((m) => m.loaded && m.id !== model)?.id ?? model);
  }, [compare, modelB, model, models]);
  async function attach(files: File[]) {
    const file = files[0];
    if (!file) return;
    if (
      !["image/png", "image/jpeg", "image/webp"].includes(file.type) ||
      file.size > 8 * 1024 * 1024
    ) {
      setError(t("playground.error.imageType"));
      return;
    }
    setReadingImage(true);
    setError("");
    try {
      const url = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () =>
          typeof reader.result === "string"
            ? resolve(reader.result)
            : reject(Error(t("playground.error.imageRead")));
        reader.onerror = () => reject(Error(t("playground.error.imageRead")));
        reader.readAsDataURL(file);
      });
      if (mounted.current) {
        setImage({ name: file.name, url });
        setAttachmentOpen(false);
      }
    } catch (e) {
      if (mounted.current)
        setError(
          e instanceof Error ? e.message : t("playground.error.imageRead"),
        );
    } finally {
      if (mounted.current) setReadingImage(false);
    }
  }
  const isVlm = (m: typeof chosen) =>
      m?.type.toLowerCase().includes("vlm") ?? false,
    supportsImage = compare ? isVlm(chosen) && isVlm(chosenB) : isVlm(chosen),
    hasImage = !!image || messages.some((message) => !!message.image),
    columnTemp = (i: 0 | 1) => (tempMode[i] === "greedy" ? 0 : temperature);
  /** The chat-shaped request for one reply; each dialect maps it to its wire format. */
  function requestBody(
    cfg: { model: string; temperature: number },
    history: Message[],
  ): CompletionBody {
    return {
      model: cfg.model,
      messages: [
        ...(system ? [{ role: "system", content: system }] : []),
        ...history.map((m) => ({
          role: m.role,
          content: m.image
            ? [
                { type: "text" as const, text: m.content },
                {
                  type: "image_url" as const,
                  image_url: { url: m.image.url },
                },
              ]
            : m.content,
        })),
      ],
      temperature: cfg.temperature,
      max_tokens: maxTokens,
      stream_options: { include_usage: true },
      ...(thinking !== "auto" ? { enable_thinking: thinking === "on" } : {}),
      ...(jsonMode === "json"
        ? { response_format: { type: "json_object" as const } }
        : {}),
    };
  }
  /** Stream one reply; `patch` updates the owning Run state. Never throws on abort. */
  async function execute(
    c: AbortController,
    cfg: { model: string; temperature: number },
    history: Message[],
    patch: (fn: (r: Run) => Run) => void,
  ) {
    const timing: RunTiming = { start: performance.now(), chunks: 0 };
    patch((r) => ({ ...r, timing: { ...timing } }));
    try {
      await streamCompletion(
        connection,
        requestBody(cfg, history),
        (delta) => {
          if (!mounted.current) return;
          const t = performance.now();
          if (delta.content || delta.reasoning) {
            timing.chunks += 1;
            timing.firstAt ??= t;
          }
          if (delta.usage) timing.usage = { ...timing.usage, ...delta.usage };
          patch((m) => ({
            ...m,
            content: m.content + (delta.content ?? ""),
            reasoning: (m.reasoning ?? "") + (delta.reasoning ?? ""),
            finishReason: delta.finishReason ?? m.finishReason,
            timing: { ...timing },
          }));
        },
        c.signal,
        dialect,
      );
      timing.endAt = performance.now();
      if (mounted.current) patch((m) => ({ ...m, timing: { ...timing } }));
    } catch (e) {
      timing.endAt = performance.now();
      if (mounted.current) {
        setErrorState(
          c.signal.aborted
            ? { message: t("playground.error.stopped") }
            : describeStreamError(e),
        );
        patch((m) => ({ ...m, incomplete: true, timing: { ...timing } }));
      }
    }
  }
  async function send() {
    if (
      !canSend ||
      !draft.trim() ||
      controller.current ||
      readingImage ||
      (hasImage && !supportsImage)
    )
      return;
    const content = draft.trim(),
      user: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        model,
        temperature,
        ...(image ? { image } : {}),
      };
    const c = new AbortController();
    controller.current = c;
    setDraft("");
    setImage(null);
    setLoading(true);
    setError("");
    toast.dismiss(UNDO_TOAST_ID);
    try {
      if (compare) {
        const cfgs = [
          { model, temperature: columnTemp(0) },
          { model: modelB, temperature: columnTemp(1) },
        ] as const;
        const blank = (i: 0 | 1): Run => ({
          content: "",
          model: cfgs[i].model,
          temperature: cfgs[i].temperature,
        });
        setPairPrompt(user);
        setPair([blank(0), null]);
        // Sequential on purpose: one GPU, and overlapping runs would skew timing.
        for (const i of [0, 1] as const) {
          if (c.signal.aborted) break;
          setPair((p) => (i === 1 ? [p[0], blank(1)] : p));
          await execute(c, cfgs[i], [user], (fn) =>
            setPair((p) => {
              const cur = p[i];
              if (!cur) return p;
              return i === 0 ? [fn(cur), p[1]] : [p[0], fn(cur)];
            }),
          );
        }
      } else {
        const id = crypto.randomUUID(),
          history = [...messages.filter((m) => !m.incomplete), user];
        setMessages([
          ...messages,
          user,
          { id, role: "assistant", content: "", model, temperature },
        ]);
        await execute(c, { model, temperature }, history, (fn) =>
          setMessages((rows) =>
            rows.map((m) => (m.id === id ? { ...m, ...fn(m) } : m)),
          ),
        );
      }
    } finally {
      if (controller.current === c) {
        controller.current = null;
        if (mounted.current) setLoading(false);
      }
      void engine.refresh();
    }
  }
  const lastId = messages.at(-1)?.id;
  const verdict = (() => {
    const [a, b] = pair;
    if (loading || !a?.timing?.endAt || !b?.timing?.endAt) return null;
    if (a.incomplete || b.incomplete || !a.content || !b.content) return null;
    return compareOutputs(
      { text: a.content, temperature: a.temperature },
      { text: b.content, temperature: b.temperature },
    );
  })();
  const deltas = (() => {
    const [a, b] = pair;
    if (loading || !a?.timing?.endAt || !b?.timing?.endAt) return null;
    const sa = runStats(a.timing, now),
      sb = runStats(b.timing, now);
    return {
      tps:
        sa.tokensPerSecond !== undefined && sb.tokensPerSecond !== undefined
          ? sb.tokensPerSecond - sa.tokensPerSecond
          : undefined,
      ttft:
        sa.ttftMs !== undefined && sb.ttftMs !== undefined
          ? sb.ttftMs - sa.ttftMs
          : undefined,
    };
  })();
  const modelOptions: ModelSelectOption[] = models.map((x) => {
    const developer = getModelDeveloperId(x.id),
      chat = supportsChat(x),
      reason = !chat ? t("playground.model.notChat") : undefined;
    return {
      id: x.id,
      label: modelLabel(x.id),
      group: developer,
      groupLabel: getProviderName(developer),
      searchText: `${x.id} ${x.type}`,
      icon: <LocalModelIcon id={x.id} size={20} />,
      badges: isVlm(x) ? (
        <CapabilityIcon capability="vision" size={13} />
      ) : undefined,
      detail:
        reason ??
        (x.loaded
          ? t("playground.model.loadedDetail", { type: x.type })
          : t("playground.model.loadHere")),
      meta: <span className="tabular-nums">{number(x.size_gb, 1)} GB</span>,
      disabled: !!reason,
    };
  });
  const modelFilters = [
    {
      key: "vision",
      node: (
        <span className="inline-flex items-center gap-1">
          <CapabilityIcon capability="vision" size={12} />
          {t("playground.model.vision")}
        </span>
      ),
      title: t("playground.model.visionTitle"),
      match: (o: ModelSelectOption) => isVlm(models.find((m) => m.id === o.id)),
    },
  ];
  const modelSelect = (
    value: string,
    onChange: (v: string) => void,
    label: string,
  ) => (
    <div
      className={`max-sm:w-full ${loading ? "pointer-events-none opacity-60" : ""}`}
      role="group"
      aria-label={label}
    >
      <ModelSelect
        className="w-60 max-sm:w-full"
        options={modelOptions}
        value={value}
        onChange={onChange}
        filters={modelFilters}
        labels={{
          placeholder: t("playground.model.placeholder"),
          search: t("playground.model.search"),
        }}
      />
    </div>
  );
  const codeRequest = (() => {
    const text = draft.trim() || "Hello";
    const past = messages.filter((m) => !m.incomplete);
    const pending: Message = {
      id: "code",
      role: "user",
      content: text,
      model,
      temperature,
      ...(image && supportsImage ? { image } : {}),
    };
    return {
      dialect,
      baseUrl: connection.baseUrl,
      body: requestBody(
        { model, temperature: compare ? columnTemp(0) : temperature },
        compare ? [pending] : [...past, pending],
      ),
    };
  })();
  function newTest() {
    if (messages.length || pairPrompt)
      toast.info(t("playground.header.cleared"), undefined, {
        id: UNDO_TOAST_ID,
        duration: UNDO_WINDOW_MS,
        action: {
          label: t("playground.header.undo"),
          onClick: () => {
            setMessages(messages);
            setPair(pair);
            setPairPrompt(pairPrompt);
          },
        },
      });
    setMessages([]);
    setPair([null, null]);
    setPairPrompt(null);
    setError("");
  }
  const empty = (
    <EmptyState
      icon={<Sparkles size={25} />}
      title={
        compare
          ? t("playground.empty.compareTitle")
          : t("playground.empty.chatTitle")
      }
      description={
        compare
          ? t("playground.empty.compareBody")
          : t("playground.empty.chatBody")
      }
    />
  );
  return (
    <section
      className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col px-4 pb-2 pt-4 lg:px-6 lg:pt-6"
      data-testid="playground"
    >
      <PageHeader
        title={t("shell.page.playground")}
        description={t("playground.page.desc")}
      />
      <ChatHeader
        className="flex-wrap gap-3 border-0 bg-transparent px-0 py-3 sm:h-auto sm:px-0 sm:py-3 max-sm:[&>*]:w-full"
        left={
          <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
            {modelSelect(model, setModel, t("playground.model.testModel"))}
            {compare &&
              modelSelect(
                modelB,
                setModelB,
                t("playground.model.compareModel"),
              )}
          </div>
        }
        status={
          <div className="flex flex-wrap items-center gap-2 max-sm:flex-nowrap">
            <div
              role="group"
              aria-label={t("playground.header.apiFormat")}
              className="max-sm:min-w-0 max-sm:flex-1"
            >
              <CustomSelect
                className="w-44 max-sm:w-full [&_button]:h-8 [&_button]:text-xs"
                value={dialect}
                disabled={loading}
                onChange={(v) => setDialect(v as Dialect)}
                options={dialectOptions}
              />
            </div>
            <SegmentedTray
              aria-label={t("playground.header.mode")}
              value={mode}
              onChange={(v) => !loading && setMode(v as Mode)}
              options={[
                { value: "chat", label: t("playground.header.modeChat") },
                { value: "compare", label: t("playground.header.modeCompare") },
              ]}
            />
          </div>
        }
        actions={
          <>
            <div className="hidden items-center gap-1 sm:flex">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setCodeOpen(true)}
              >
                <Code2 size={13} />
                {t("playground.header.viewCode")}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={loading}
                onClick={newTest}
              >
                <Plus size={13} />
                {t("playground.header.newTest")}
              </Button>
            </div>
            <div className="sm:hidden">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <IconButton
                    icon={<MoreHorizontal size={16} />}
                    label={t("playground.header.more")}
                  />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onSelect={() => setCodeOpen(true)}>
                    <Code2 size={14} />
                    {t("playground.header.viewCode")}
                  </DropdownMenuItem>
                  <DropdownMenuItem disabled={loading} onSelect={newTest}>
                    <Plus size={14} />
                    {t("playground.header.newTest")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </>
        }
      />
      <ChatMessageList
        className="px-4 sm:px-7"
        empty={<div className="m-auto max-w-lg py-16">{empty}</div>}
      >
        {compare ? (
          <div
            className="mx-auto w-full max-w-5xl space-y-4 py-6"
            data-testid="compare"
          >
            {!pairPrompt && empty}
            {pairPrompt && (
              <ChatMessage role="user" density="compact">
                {pairPrompt.image && (
                  <img
                    src={pairPrompt.image.url}
                    alt={pairPrompt.image.name}
                    className="mb-3 max-h-56 max-w-full rounded-lg object-contain"
                  />
                )}
                {pairPrompt.content}
              </ChatMessage>
            )}
            {pairPrompt && (
              <div className="grid gap-4 md:grid-cols-2">
                {([0, 1] as const).map((i) => {
                  const run = pair[i];
                  return (
                    <Card
                      key={i}
                      className="min-w-0 space-y-3 p-4"
                      data-testid={`compare-col-${i === 0 ? "a" : "b"}`}
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-muted-foreground">
                          {i === 0 ? "A" : "B"}
                        </span>
                        <span className="min-w-0 truncate text-sm font-medium">
                          {modelLabel(i === 0 ? model : modelB)}
                        </span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          T={number(run?.temperature ?? columnTemp(i), 1)}
                        </span>
                      </div>
                      {run ? (
                        <>
                          <ReplyBody
                            run={run}
                            streaming={loading && !run.timing?.endAt}
                          />
                          <ReplyStats run={run} now={now} />
                        </>
                      ) : (
                        <p className="text-xs text-muted-foreground">
                          {loading
                            ? t("playground.compare.queued")
                            : t("playground.compare.notRun")}
                        </p>
                      )}
                    </Card>
                  );
                })}
              </div>
            )}
            {(deltas || verdict) && (
              <Card
                className="flex flex-wrap items-center gap-2 p-3 tabular-nums"
                data-testid="compare-delta"
              >
                <span className="text-xs text-muted-foreground">
                  {t("playground.compare.bVsA")}
                </span>
                {deltas?.tps !== undefined && (
                  <span className="text-xs text-muted-foreground">
                    Δ tok/s {signed(deltas.tps, 1)}
                  </span>
                )}
                {deltas?.ttft !== undefined && (
                  <span className="text-xs text-muted-foreground">
                    {t("playground.compare.deltaTtft", {
                      value: signed(deltas.ttft, 0),
                    })}
                  </span>
                )}
                {verdict?.kind === "identical" &&
                  (verdict.greedy ? (
                    <StatusIndicator
                      status="online"
                      className="gap-1.5 text-xs text-muted-foreground"
                    >
                      {t("playground.compare.identical")}
                    </StatusIndicator>
                  ) : (
                    <span className="text-xs text-muted-foreground">
                      {t("playground.compare.sameText")}
                    </span>
                  ))}
                {verdict?.kind === "diverged" && (
                  <StatusIndicator
                    status="away"
                    className="gap-1.5 text-xs text-muted-foreground"
                  >
                    {t("playground.compare.diverged", {
                      offset: number(verdict.offset, 0),
                    })}
                  </StatusIndicator>
                )}
              </Card>
            )}
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-6 py-6">
            {!messages.length && empty}
            {messages.map((m) => (
              <ChatMessage
                key={m.id}
                role={m.role}
                density="compact"
                name={m.role === "assistant" ? modelLabel(m.model) : undefined}
                footer={
                  m.role === "assistant" ? (
                    <ReplyStats run={m} now={now} />
                  ) : undefined
                }
              >
                {m.image && (
                  <img
                    src={m.image.url}
                    alt={m.image.name}
                    className="mb-3 max-h-56 max-w-full rounded-lg object-contain"
                  />
                )}
                {m.role === "assistant" ? (
                  <ReplyBody run={m} streaming={loading && lastId === m.id} />
                ) : (
                  m.content
                )}
              </ChatMessage>
            ))}
          </div>
        )}
      </ChatMessageList>
      <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-5 pt-3">
        {error && (
          <ErrorNote
            className="mb-3"
            message={error.message}
            detail={error.detail}
          />
        )}
        {!supportsImage && hasImage && (
          <p className="mb-3 text-xs text-warning">
            {t("playground.composer.needVision")}
          </p>
        )}
        {compare && (
          <p
            className="mb-3 text-xs text-muted-foreground"
            data-testid="spec-note"
          >
            {t("playground.compare.specNote")}
          </p>
        )}
        {compare && (
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
            {([0, 1] as const).map((i) => (
              <div key={i} className="flex items-center gap-2">
                <span>
                  {t("playground.compare.sampling", {
                    side: i === 0 ? "A" : "B",
                  })}
                </span>
                <SegmentedTray
                  aria-label={t("playground.compare.samplingLabel", {
                    side: i === 0 ? "A" : "B",
                  })}
                  value={tempMode[i]}
                  onChange={(v) =>
                    !loading &&
                    setTempMode(i === 0 ? [v, tempMode[1]] : [tempMode[0], v])
                  }
                  options={[
                    {
                      value: "shared",
                      label: t("playground.compare.shared", {
                        t: number(temperature, 1),
                      }),
                    },
                    { value: "greedy", label: t("playground.compare.greedy") },
                  ]}
                />
              </div>
            ))}
          </div>
        )}
        {!canSend && (
          <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2 text-xs text-muted-foreground">
            <p>
              {engine.phase !== "online"
                ? t("playground.composer.offline")
                : !supportsChat(chosen) || (compare && !supportsChat(chosenB))
                  ? t("playground.composer.notChat")
                  : t("playground.composer.notLoaded")}
            </p>
            {engine.phase === "online" &&
              [chosen, ...(compare ? [chosenB] : [])]
                .filter(
                  (m, i, all) =>
                    m && !m.loaded && supportsChat(m) && all.indexOf(m) === i,
                )
                .map((m) => (
                  <Button
                    key={m!.id}
                    size="sm"
                    variant="secondary"
                    disabled={loadingModels.has(m!.id) || m!.loading}
                    data-testid="load-inline"
                    onClick={() => void loadHere(m!.id)}
                  >
                    {loadingModels.has(m!.id) || m!.loading
                      ? t("playground.model.loadingNow", {
                          name: modelLabel(m!.id),
                        })
                      : t("playground.model.loadAction", {
                          name: modelLabel(m!.id),
                          size: sizeGb(m!.size_gb),
                        })}
                  </Button>
                ))}
          </div>
        )}
        <ChatComposer
          value={draft}
          onChange={setDraft}
          onSend={() => void send()}
          onStop={() => controller.current?.abort()}
          loading={loading}
          sendDisabled={
            !canSend || readingImage || (hasImage && !supportsImage)
          }
          attachments={
            image ? (
              <div className="flex items-center gap-2 text-xs">
                <img
                  src={image.url}
                  alt={t("playground.composer.pendingImage")}
                  className="size-10 rounded object-cover"
                />
                <span className="min-w-0 truncate">{image.name}</span>
                <IconButton
                  icon={<X size={13} />}
                  label={t("playground.composer.removeImage")}
                  disabled={loading}
                  onClick={() => setImage(null)}
                />
              </div>
            ) : undefined
          }
          placeholder={t("playground.composer.placeholder")}
          labels={{
            send: t("playground.composer.send"),
            stop: t("playground.composer.stop"),
          }}
          toolbar={
            <div className="flex items-center gap-1">
              <IconButton
                icon={<ImagePlus size={15} />}
                label={t("playground.composer.addImage")}
                disabled={loading || !supportsImage}
                onClick={(e) => {
                  e.currentTarget.focus();
                  setAttachmentOpen(true);
                }}
              />
              <IconButton
                icon={<SlidersHorizontal size={15} />}
                label={t("playground.composer.params")}
                disabled={loading}
                onClick={(e) => {
                  e.currentTarget.focus();
                  setSettings(true);
                }}
              />
            </div>
          }
        />
        <p className="mt-2 text-center text-xs text-muted-foreground">
          {t("playground.composer.note")}
        </p>
      </div>
      <Sheet
        open={settings}
        onClose={() => setSettings(false)}
        title={t("playground.composer.params")}
        closeLabel={t("playground.params.close")}
      >
        <div className="space-y-6">
          <div>
            <p className="mb-3 text-sm">{t("playground.params.thinking")}</p>
            <SegmentedTray
              aria-label={t("playground.params.thinking")}
              value={thinking}
              onChange={setThinking}
              options={[
                { value: "auto", label: t("playground.params.thinkAuto") },
                { value: "on", label: t("playground.params.thinkOn") },
                { value: "off", label: t("playground.params.thinkOff") },
              ]}
            />
          </div>
          <div>
            <p className="mb-3 text-sm">{t("playground.params.format")}</p>
            <SegmentedTray
              aria-label={t("playground.params.format")}
              value={jsonMode}
              onChange={setJsonMode}
              options={[
                { value: "text", label: t("playground.params.formatText") },
                { value: "json", label: "JSON" },
              ]}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              {dialect === "messages"
                ? t("playground.params.noJson")
                : t("playground.params.jsonNote")}
            </p>
          </div>
          <div>
            <p className="mb-4 text-sm">
              Temperature · {fixed(temperature, 1)}
            </p>
            <Slider
              label="Temperature"
              value={[temperature]}
              onValueChange={(v) => setTemperature(v[0])}
              min={0}
              max={2}
              step={0.1}
            />
          </div>
          <div>
            <label htmlFor="max-tokens" className="text-sm">
              {t("playground.params.maxTokens")}
            </label>
            <Input
              id="max-tokens"
              type="number"
              min={1}
              max={32768}
              value={maxTokens}
              onChange={(e) =>
                setMaxTokens(
                  Math.max(1, Math.min(32768, Number(e.target.value) || 1)),
                )
              }
            />
          </div>
          <div>
            <label htmlFor="system" className="text-sm">
              {t("playground.params.system")}
            </label>
            <Textarea
              id="system"
              className="mt-2 min-h-40"
              value={system}
              onChange={(e) => setSystem(e.target.value)}
            />
          </div>
        </div>
      </Sheet>
      <Dialog open={codeOpen} onOpenChange={setCodeOpen}>
        <DialogContent
          closeLabel={t("playground.code.close")}
          className="max-w-3xl"
        >
          <DialogHeader>
            <DialogTitle>{t("playground.code.title")}</DialogTitle>
            <DialogDescription>
              {t("playground.code.desc", { dialect: DIALECT_LABEL[dialect] })}
              {compare && t("playground.code.compareNote")}
            </DialogDescription>
          </DialogHeader>
          {codeOpen &&
            (() => {
              const code = buildSnippets(codeRequest);
              return (
                <Tabs
                  value={codeTab}
                  onValueChange={(v) => setCodeTab(v as CodeLanguage)}
                >
                  <TabsList>
                    {codeTabs.map((t) => (
                      <TabsTrigger key={t.value} value={t.value}>
                        {t.label}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {codeTabs.map((t) => (
                    <TabsContent key={t.value} value={t.value}>
                      <div data-testid={`code-${t.value}`}>
                        <Suspense
                          fallback={
                            <pre className="overflow-auto text-xs">
                              {code.snippets[t.value]}
                            </pre>
                          }
                        >
                          <CodeBlock language={t.language} defaultExpanded>
                            {code.snippets[t.value]}
                          </CodeBlock>
                        </Suspense>
                      </div>
                    </TabsContent>
                  ))}
                  <p className="mt-3 text-xs text-muted-foreground">
                    {t("playground.code.keyNote")}
                    {code.shortened && t("playground.code.shortened")}
                    {dialect === "messages" &&
                      jsonMode === "json" &&
                      t("playground.code.noJson")}
                  </p>
                </Tabs>
              );
            })()}
        </DialogContent>
      </Dialog>
      <Sheet
        open={attachmentOpen}
        onClose={() => setAttachmentOpen(false)}
        title={t("playground.image.title")}
        closeLabel={t("playground.image.close")}
      >
        <FileDropzone
          accept="image/png,image/jpeg,image/webp"
          disabled={readingImage}
          label={
            readingImage
              ? t("playground.image.reading")
              : t("playground.image.pick")
          }
          hint={t("playground.image.hint")}
          onFiles={(files) => void attach(files)}
        />
      </Sheet>
    </section>
  );
}
