import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Input,
  FileDropzone,
  SegmentedSelect,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Sheet,
  Slider,
  Textarea,
} from "@yuhuanowo/yunui";
import {
  ChatComposer,
  ChatMessage,
  ChatMessageList,
} from "@yuhuanowo/yunui/chat";
import { ThinkingBlock } from "@yuhuanowo/yunui/ai";
import { SlidersHorizontal, Plus, Sparkles, ImagePlus, X } from "lucide-react";
import { streamCompletion } from "./stream";
import type { Connection } from "./api";
import { modelLabel, supportsChat, type Engine } from "./ui";
const Markdown = lazy(() =>
  import("@yuhuanowo/yunui/content").then((m) => ({
    default: m.MarkdownRenderer,
  })),
);
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model: string;
  incomplete?: boolean;
  image?: { name: string; url: string };
  finishReason?: string;
};
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
    [draft, setDraft] = useState(""),
    [messages, setMessages] = useState<Message[]>([]),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [settings, setSettings] = useState(false),
    [temperature, setTemperature] = useState(0.7),
    [system, setSystem] = useState(""),
    [maxTokens, setMaxTokens] = useState(512),
    [thinking, setThinking] = useState("auto"),
    [jsonMode, setJsonMode] = useState("text"),
    [image, setImage] = useState<{ name: string; url: string } | null>(null),
    [attachmentOpen, setAttachmentOpen] = useState(false),
    [readingImage, setReadingImage] = useState(false);
  const controller = useRef<AbortController | null>(null),
    mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
    };
  }, []);
  const models = engine.status?.models ?? [],
    chosen = models.find((m) => m.id === model),
    canSend =
      engine.phase === "online" && !!chosen?.loaded && supportsChat(chosen);
  useEffect(() => {
    if (!model && models.length)
      setModel(models.find((m) => m.loaded)?.id ?? models[0].id);
  }, [model, models]);
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
  const supportsImage = chosen?.type.toLowerCase().includes("vlm") ?? false;
  async function send() {
    if (
      !canSend ||
      !draft.trim() ||
      controller.current ||
      readingImage ||
      ((image || messages.some((message) => !!message.image)) && !supportsImage)
    )
      return;
    const content = draft.trim(),
      id = crypto.randomUUID(),
      user: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        model,
        ...(image ? { image } : {}),
      };
    const history = [...messages.filter((m) => !m.incomplete), user];
    const c = new AbortController();
    controller.current = c;
    setMessages([
      ...messages,
      user,
      { id, role: "assistant", content: "", model },
    ]);
    setDraft("");
    setImage(null);
    setLoading(true);
    setError("");
    try {
      await streamCompletion(
        connection,
        {
          model,
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
          temperature,
          max_tokens: maxTokens,
          ...(thinking !== "auto"
            ? { enable_thinking: thinking === "on" }
            : {}),
          ...(jsonMode === "json"
            ? { response_format: { type: "json_object" as const } }
            : {}),
        },
        (delta) => {
          if (mounted.current)
            setMessages((rows) =>
              rows.map((m) =>
                m.id === id
                  ? {
                      ...m,
                      content: m.content + (delta.content ?? ""),
                      reasoning: (m.reasoning ?? "") + (delta.reasoning ?? ""),
                      finishReason: delta.finishReason ?? m.finishReason,
                    }
                  : m,
              ),
            );
        },
        c.signal,
      );
    } catch (e) {
      if (mounted.current) {
        setError(
          c.signal.aborted
            ? "已停止生成；部分回覆保留於下方。"
            : e instanceof Error
              ? e.message
              : "生成失敗",
        );
        setMessages((rows) =>
          rows.map((m) => (m.id === id ? { ...m, incomplete: true } : m)),
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
  return (
    <section className="flex min-h-0 flex-1 flex-col" data-testid="playground">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 p-4">
        <div className="flex items-center gap-2">
          <Select
            value={model || undefined}
            onValueChange={setModel}
            disabled={loading}
          >
            <SelectTrigger aria-label="測試模型" className="w-52">
              <SelectValue placeholder="選擇模型" />
            </SelectTrigger>
            <SelectContent>
              {models.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {modelLabel(m.id)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Badge variant={chosen?.loaded ? "success" : "secondary"}>
            {chosen?.loaded ? "已載入" : "未載入"}
          </Badge>
        </div>
        <Button
          size="sm"
          variant="ghost"
          disabled={loading}
          onClick={() => {
            setMessages([]);
            setError("");
          }}
        >
          <Plus size={13} />
          新測試
        </Button>
      </div>
      <ChatMessageList
        className="px-4 sm:px-7"
        empty={
          <div className="m-auto max-w-lg py-16">
            <EmptyState
              icon={<Sparkles size={25} />}
              title="驗證模型回應"
              description="向已載入的模型傳送提示詞，查看真實串流輸出。"
            />
          </div>
        }
      >
        <div className="mx-auto max-w-3xl space-y-6 py-6">
          {!messages.length && (
            <EmptyState
              icon={<Sparkles size={25} />}
              title="驗證模型回應"
              description="向已載入的模型傳送提示詞，查看真實串流輸出。"
            />
          )}
          {messages.map((m) => (
            <ChatMessage
              key={m.id}
              role={m.role}
              density="compact"
              name={m.role === "assistant" ? modelLabel(m.model) : undefined}
            >
              {m.image && (
                <img
                  src={m.image.url}
                  alt={m.image.name}
                  className="mb-3 max-h-56 max-w-full rounded-lg object-contain"
                />
              )}
              {m.reasoning && (
                <ThinkingBlock
                  labels={{
                    title: "思考過程",
                    active: "思考中",
                    completed: "已完成",
                    inProgress: "進行中",
                  }}
                  content={m.reasoning}
                  isStreaming={loading && messages.at(-1)?.id === m.id}
                />
              )}
              <Suspense
                fallback={
                  <div className="whitespace-pre-wrap text-sm leading-7">
                    {m.content}
                  </div>
                }
              >
                <Markdown
                  content={
                    m.content ||
                    (loading && messages.at(-1)?.id === m.id
                      ? "等待模型輸出…"
                      : "未收到文字內容")
                  }
                />
              </Suspense>
              {m.finishReason === "length" && (
                <p className="mt-2 text-xs text-warning">
                  已達輸出上限，可調高最大輸出 tokens 或關閉思考模式再測試。
                </p>
              )}
              {m.incomplete && (
                <p className="mt-2 text-xs text-muted-foreground">
                  未完成的回覆
                </p>
              )}
            </ChatMessage>
          ))}
        </div>
      </ChatMessageList>
      <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-5 pt-3">
        {error && (
          <p role="status" className="mb-3 text-xs text-error">
            {error}
          </p>
        )}
        {!supportsImage &&
          (image || messages.some((message) => !!message.image)) && (
            <p className="mb-3 text-xs text-warning">
              此測試包含圖片，請選擇視覺模型，或建立新測試。
            </p>
          )}
        {!canSend && (
          <p className="mb-3 text-xs text-muted-foreground">
            {engine.phase !== "online"
              ? "連接引擎後即可開始測試。"
              : !supportsChat(chosen)
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
            !canSend ||
            readingImage ||
            ((!!image || messages.some((message) => !!message.image)) &&
              !supportsImage)
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
        <p className="mt-2 text-center text-[10px] text-muted-foreground">
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
              JSON 模式會傳送 response_format，由引擎約束輸出格式。
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
