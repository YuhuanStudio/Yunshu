import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  IconButton,
  Input,
  FileDropzone,
  SegmentedSelect,
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
import {
  SlidersHorizontal,
  Plus,
  Sparkles,
  ImagePlus,
  X,
  Code2,
  MessageSquare,
  Reply,
  Bot,
} from "lucide-react";
import {
  describeStreamError,
  streamCompletion,
  type CompletionBody,
  type Dialect,
} from "./stream";
import { ErrorNote } from "./error-note";
import { UNDO_WINDOW_MS, thinkingOpen } from "./playground-ui-state";
import {
  buildSnippets,
  DIALECT_LABEL,
  type CodeLanguage,
} from "./playground-code";
import type { Connection } from "./api";
import {
  LocalModelIcon,
  modelLabel,
  number,
  supportsChat,
  type Engine,
} from "./ui";
import { compareOutputs, runStats, type RunTiming } from "./playground-metrics";
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
const dialectOptions = [
  { value: "chat" as Dialect, label: DIALECT_LABEL.chat, icon: MessageSquare },
  {
    value: "responses" as Dialect,
    label: DIALECT_LABEL.responses,
    icon: Reply,
  },
  { value: "messages" as Dialect, label: DIALECT_LABEL.messages, icon: Bot },
];
const codeTabs: { value: CodeLanguage; label: string; language: string }[] = [
  { value: "curl", label: "curl", language: "bash" },
  { value: "python", label: "Python", language: "python" },
  { value: "javascript", label: "JavaScript", language: "javascript" },
];
type Pair = readonly [Run | null, Run | null];
const statLabels = {
  tokens: "tokens",
  speed: "tok/s",
  latency: "ms",
  ttft: "首 token 延遲",
  cached: "前綴命中",
  prompt: "輸入",
};
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
      ? ` · 每輪 ${number(tokens / spec.rounds, 2)} tok`
      : "";
  const rate =
    spec.acceptanceRate != null
      ? ` · 接受率 ${number(spec.acceptanceRate * 100, 0)}%`
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
        labels={statLabels}
      />
      {live && s.ttftMs === undefined && (
        <Badge variant="outline">等待首個 token</Badge>
      )}
      {run.timing.usage?.spec && (
        <Badge variant="outline">
          {specLabel(run.timing.usage.spec, s.tokens)}
        </Badge>
      )}
      {s.estimated && !live && (
        <Badge variant="secondary">token 數為串流片段估算</Badge>
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
            title: "思考過程",
            active: "思考中",
            completed: "已完成",
            inProgress: "進行中",
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
            run.content || (streaming ? "等待模型輸出…" : "未收到文字內容")
          }
        />
      </Suspense>
      {run.finishReason === "length" && (
        <p className="mt-2 text-xs text-warning">
          已達輸出上限，可調高最大輸出 tokens 或關閉思考模式再測試。
        </p>
      )}
      {run.incomplete && (
        <p className="mt-2 text-xs text-muted-foreground">未完成的回覆</p>
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
    [undo, setUndo] = useState<{
      messages: Message[];
      pair: Pair;
      pairPrompt: Message | null;
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
  useEffect(() => {
    if (!undo) return;
    const timer = window.setTimeout(() => setUndo(null), UNDO_WINDOW_MS);
    return () => window.clearTimeout(timer);
  }, [undo]);
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
      setError("請選擇 8 MB 以下的 PNG、JPEG 或 WebP 圖片。");
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
            : reject(Error("無法讀取圖片"));
        reader.onerror = () => reject(Error("無法讀取圖片"));
        reader.readAsDataURL(file);
      });
      if (mounted.current) {
        setImage({ name: file.name, url });
        setAttachmentOpen(false);
      }
    } catch (e) {
      if (mounted.current)
        setError(e instanceof Error ? e.message : "無法讀取圖片");
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
            ? { message: "已停止生成；部分回覆保留於下方。" }
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
    setUndo(null);
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
      reason = !chat ? "非文字聊天模型" : !x.loaded ? "未載入" : undefined;
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
      detail: reason ?? `${x.type} · 已載入`,
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
          視覺
        </span>
      ),
      title: "可接受圖片輸入的模型",
      match: (o: ModelSelectOption) => isVlm(models.find((m) => m.id === o.id)),
    },
  ];
  const modelSelect = (
    value: string,
    onChange: (v: string) => void,
    label: string,
  ) => (
    <div
      className={loading ? "pointer-events-none opacity-60" : undefined}
      role="group"
      aria-label={label}
    >
      <ModelSelect
        className="w-60"
        options={modelOptions}
        value={value}
        onChange={onChange}
        filters={modelFilters}
        labels={{ placeholder: "選擇模型", search: "搜尋模型" }}
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
  const empty = (
    <EmptyState
      icon={<Sparkles size={25} />}
      title={compare ? "比較兩組設定" : "驗證模型回應"}
      description={
        compare
          ? "同一提示詞依序送往兩組設定（不會同時執行，以免干擾計時），並排比較輸出與速度。"
          : "向已載入的模型傳送提示詞，查看真實串流輸出。"
      }
    />
  );
  return (
    <section className="flex min-h-0 flex-1 flex-col" data-testid="playground">
      <ChatHeader
        className="flex-wrap gap-3 border-b border-border/60 p-4"
        left={
          <div className="flex flex-wrap items-center gap-2">
            {modelSelect(model, setModel, "測試模型")}
            {compare && modelSelect(modelB, setModelB, "比較模型")}
          </div>
        }
        status={
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedSelect
              aria-label="API 格式"
              value={dialect}
              onChange={(v) => !loading && setDialect(v)}
              options={dialectOptions}
              wrap
            />
            <SegmentedSelect
              aria-label="測試模式"
              value={mode}
              onChange={(v) => !loading && setMode(v as Mode)}
              options={[
                { value: "chat", label: "對話" },
                { value: "compare", label: "比較" },
              ]}
            />
          </div>
        }
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={() => setCodeOpen(true)}>
              <Code2 size={13} />
              檢視程式碼
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={loading}
              onClick={() => {
                if (messages.length || pairPrompt)
                  setUndo({ messages, pair, pairPrompt });
                setMessages([]);
                setPair([null, null]);
                setPairPrompt(null);
                setError("");
              }}
            >
              <Plus size={13} />
              新測試
            </Button>
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
                        <Badge variant="secondary">{i === 0 ? "A" : "B"}</Badge>
                        <span className="min-w-0 truncate text-sm font-medium">
                          {modelLabel(i === 0 ? model : modelB)}
                        </span>
                        <Badge variant="outline" className="tabular-nums">
                          T={number(run?.temperature ?? columnTemp(i), 1)}
                        </Badge>
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
                          {loading ? "排隊中，等待 A 完成後執行。" : "未執行"}
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
                <span className="text-xs text-muted-foreground">B 相對 A</span>
                {deltas?.tps !== undefined && (
                  <Badge variant="outline">
                    Δ tok/s {signed(deltas.tps, 1)}
                  </Badge>
                )}
                {deltas?.ttft !== undefined && (
                  <Badge variant="outline">
                    Δ 首 token 延遲 {signed(deltas.ttft, 0)} ms
                  </Badge>
                )}
                {verdict?.kind === "identical" &&
                  (verdict.greedy ? (
                    <Badge variant="success">輸出完全一致</Badge>
                  ) : (
                    <Badge variant="secondary">
                      文字相同（取樣非貪婪，不代表確定性）
                    </Badge>
                  ))}
                {verdict?.kind === "diverged" && (
                  <Badge variant="warning">
                    首次分歧於字元偏移 {number(verdict.offset, 0)}（從 0 起算）
                  </Badge>
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
      {undo && (
        <div
          role="status"
          className="mx-auto mb-2 flex w-full max-w-3xl items-center justify-between gap-3 rounded-lg border border-border bg-(--bg-card) px-3 py-2 text-xs shadow-sm"
        >
          <span>已清除這段測試。</span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setMessages(undo.messages);
              setPair(undo.pair);
              setPairPrompt(undo.pairPrompt);
              setUndo(null);
            }}
          >
            復原
          </Button>
        </div>
      )}
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
            此測試包含圖片，請選擇視覺模型，或建立新測試。
          </p>
        )}
        {compare && (
          <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
            {([0, 1] as const).map((i) => (
              <div key={i} className="flex items-center gap-2">
                <span>{i === 0 ? "A" : "B"} 取樣</span>
                <SegmentedSelect
                  aria-label={`${i === 0 ? "A" : "B"} 取樣方式`}
                  value={tempMode[i]}
                  onChange={(v) =>
                    !loading &&
                    setTempMode(i === 0 ? [v, tempMode[1]] : [tempMode[0], v])
                  }
                  options={[
                    {
                      value: "shared",
                      label: `沿用 T=${number(temperature, 1)}`,
                    },
                    { value: "greedy", label: "貪婪 T=0" },
                  ]}
                />
              </div>
            ))}
          </div>
        )}
        {!canSend && (
          <p className="mb-3 text-xs text-muted-foreground">
            {engine.phase !== "online"
              ? "連接引擎後即可開始測試。"
              : !supportsChat(chosen) || (compare && !supportsChat(chosenB))
                ? "此模型不適用文字聊天測試，請使用 API 接入對應端點。"
                : "請先在模型庫載入此模型。"}
          </p>
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
                  alt="待傳送圖片"
                  className="size-10 rounded object-cover"
                />
                <span className="min-w-0 truncate">{image.name}</span>
                <IconButton
                  icon={<X size={13} />}
                  label="移除圖片"
                  disabled={loading}
                  onClick={() => setImage(null)}
                />
              </div>
            ) : undefined
          }
          placeholder="輸入測試提示詞…"
          labels={{ send: "傳送測試", stop: "停止生成" }}
          toolbar={
            <div className="flex items-center gap-1">
              <IconButton
                icon={<ImagePlus size={15} />}
                label="加入圖片"
                disabled={loading || !supportsImage}
                onClick={(e) => {
                  e.currentTarget.focus();
                  setAttachmentOpen(true);
                }}
              />
              <IconButton
                icon={<SlidersHorizontal size={15} />}
                label="生成參數"
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
          內容只保留於此頁。離開測試頁會停止生成。
        </p>
      </div>
      <Sheet
        open={settings}
        onClose={() => setSettings(false)}
        title="生成參數"
        closeLabel="關閉生成參數"
      >
        <div className="space-y-6">
          <div>
            <p className="mb-3 text-sm">思考模式</p>
            <SegmentedSelect
              aria-label="思考模式"
              value={thinking}
              onChange={setThinking}
              options={[
                { value: "auto", label: "模型預設" },
                { value: "on", label: "開啟" },
                { value: "off", label: "關閉" },
              ]}
            />
          </div>
          <div>
            <p className="mb-3 text-sm">輸出格式</p>
            <SegmentedSelect
              aria-label="輸出格式"
              value={jsonMode}
              onChange={setJsonMode}
              options={[
                { value: "text", label: "文字" },
                { value: "json", label: "JSON" },
              ]}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              {dialect === "messages"
                ? "Anthropic Messages 沒有 JSON 模式，此設定不會送出。"
                : "JSON 模式會傳送 response_format，由引擎約束輸出格式。"}
            </p>
          </div>
          <div>
            <p className="mb-4 text-sm">
              Temperature · {temperature.toFixed(1)}
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
              最大輸出 tokens
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
              系統提示詞
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
        <DialogContent closeLabel="關閉程式碼" className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>檢視程式碼</DialogTitle>
            <DialogDescription>
              {DIALECT_LABEL[dialect]} · 與目前設定送出的請求相同。
              {compare && "比較模式顯示 A 組請求。"}
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
                          <CodeBlock language={t.language}>
                            {code.snippets[t.value]}
                          </CodeBlock>
                        </Suspense>
                      </div>
                    </TabsContent>
                  ))}
                  <p className="mt-3 text-xs text-muted-foreground">
                    以環境變數 YUNSHU_API_KEY 帶入權杖，不會寫入範例。
                    {code.shortened && "圖片內容已縮短顯示，請換成實際檔案。"}
                    {dialect === "messages" &&
                      jsonMode === "json" &&
                      "Messages 沒有 JSON 模式，未送出。"}
                  </p>
                </Tabs>
              );
            })()}
        </DialogContent>
      </Dialog>
      <Sheet
        open={attachmentOpen}
        onClose={() => setAttachmentOpen(false)}
        title="圖片輸入"
        closeLabel="關閉圖片輸入"
      >
        <FileDropzone
          accept="image/png,image/jpeg,image/webp"
          disabled={readingImage}
          label={readingImage ? "讀取中…" : "選擇或拖入圖片"}
          hint="PNG、JPEG、WebP，最多 8 MB。圖片會隨提示詞傳給所選 VLM。"
          onFiles={(files) => void attach(files)}
        />
      </Sheet>
    </section>
  );
}
