"use client";

import { useMemo, useState } from "react";
import {
  Button,
  Card,
  Textarea,
  Input,
  Switch,
  Alert,
  Badge,
  Spinner,
  SegmentedBar,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  cn,
  toast,
} from "yunui";
import { Copy, Hash, Type, Calculator } from "lucide-react";
import { api, usePolling, ApiError } from "@/lib/api";
import { fmtNumber, fmtPct } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import type { Model } from "@/lib/types";

/**
 * Page-local mirrors of the backend tokenizer contract (kept here rather than in
 * lib/types so this page owns its own shapes):
 *   POST /v1/tokenize     → { tokens: number[] | number[][], count, model }
 *   POST /v1/detokenize   → { text, model }
 *   POST /v1/token_count  → { token_count, max_context_tokens, over_context_limit, model }
 */
interface TokenizeResult {
  tokens: number[] | number[][];
  count: number;
  model: string;
}
interface DetokenizeResult {
  text: string;
  model: string;
}
interface TokenCountResult {
  token_count: number;
  max_context_tokens: number;
  over_context_limit: boolean;
  model: string;
}

function errMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Request failed";
}

/** Normalize `number[] | number[][]` into a flat list of token-id rows. */
function toRows(tokens: number[] | number[][]): number[][] {
  if (tokens.length === 0) return [];
  return Array.isArray(tokens[0]) ? (tokens as number[][]) : [tokens as number[]];
}

export default function TokenizePage() {
  const models = usePolling<{ data: Model[] }>((s) => api.get("/v1/models", s), 15000);
  const modelList = useMemo(() => models.data?.data ?? [], [models.data]);

  const [model, setModel] = useState<string>("");
  const activeModel =
    model || modelList.find((m) => m.loaded)?.id || modelList[0]?.id || "";

  return (
    <PageShell
      title="Tokenize"
      description="Tokenize, detokenize and count tokens against a loaded model."
      width="narrow"
    >
      <Card className="p-5">
        <label className="mb-2 block text-sm font-medium">Model</label>
        <ModelPicker models={modelList} value={activeModel} onChange={setModel} className="w-full" />
        {models.error && (
          <p className="mt-2 text-xs text-error">
            Could not load models: {errMessage(models.error)}
          </p>
        )}
      </Card>

      <Tabs defaultValue="tokenize" className="mt-4">
        <TabsList>
          <TabsTrigger value="tokenize">
            <Hash className="h-4 w-4" /> Tokenize
          </TabsTrigger>
          <TabsTrigger value="detokenize">
            <Type className="h-4 w-4" /> Detokenize
          </TabsTrigger>
          <TabsTrigger value="count">
            <Calculator className="h-4 w-4" /> Count
          </TabsTrigger>
        </TabsList>

        <TabsContent value="tokenize">
          <TokenizeTab model={activeModel} />
        </TabsContent>
        <TabsContent value="detokenize">
          <DetokenizeTab model={activeModel} />
        </TabsContent>
        <TabsContent value="count">
          <CountTab model={activeModel} />
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}

// --- Tokenize -------------------------------------------------------------
function TokenizeTab({ model }: { model: string }) {
  const [text, setText] = useState("");
  const [addSpecial, setAddSpecial] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TokenizeResult | null>(null);

  const run = async () => {
    if (!model || !text.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<TokenizeResult>("/v1/tokenize", {
        model,
        text,
        add_special_tokens: addSpecial,
      });
      setResult(res);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const copyTokens = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result.tokens));
      toast.success("Copied token ids");
    } catch {
      toast.error("Copy failed");
    }
  };

  const rows = result ? toRows(result.tokens) : [];

  return (
    <Card className="mt-4 p-5">
      <label className="mb-2 block text-sm font-medium">Text</label>
      <Textarea
        rows={5}
        placeholder="Enter text to tokenize…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="mt-3 flex items-center justify-between gap-3">
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <Switch checked={addSpecial} onCheckedChange={setAddSpecial} />
          Add special tokens
        </label>
        <Button onClick={run} disabled={!model || !text.trim() || busy}>
          {busy ? <Spinner size="sm" /> : <Hash className="h-4 w-4" />} Tokenize
        </Button>
      </div>

      {error && (
        <Alert variant="error" className="mt-4">
          {error}
        </Alert>
      )}

      {result && (
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium">Tokens</span>
              <Badge variant="info">{fmtNumber(result.count)}</Badge>
            </div>
            <Button variant="secondary" size="sm" onClick={copyTokens}>
              <Copy className="h-4 w-4" /> Copy JSON
            </Button>
          </div>
          {rows.length === 0 ? (
            <p className="py-4 text-center text-sm text-muted-foreground">No tokens produced.</p>
          ) : (
            rows.map((row, r) => (
              <div key={r}>
                {rows.length > 1 && (
                  <div className="mb-1.5 text-xs font-medium text-muted-foreground">
                    Sequence {r + 1} · {fmtNumber(row.length)} tokens
                  </div>
                )}
                <div className="flex flex-wrap gap-1.5">
                  {row.map((t, i) => (
                    <span
                      key={`${r}-${i}-${t}`}
                      className="inline-flex rounded-md bg-muted px-2 py-0.5 text-xs tabular-nums text-foreground"
                      title={`token #${i}`}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </Card>
  );
}

// --- Detokenize -----------------------------------------------------------
function DetokenizeTab({ model }: { model: string }) {
  const [raw, setRaw] = useState("");
  const [skipSpecial, setSkipSpecial] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetokenizeResult | null>(null);

  const tokens = useMemo(
    () =>
      raw
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter(Boolean)
        .map(Number)
        .filter((n) => Number.isInteger(n)),
    [raw],
  );

  const run = async () => {
    if (!model || tokens.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<DetokenizeResult>("/v1/detokenize", {
        model,
        tokens,
        skip_special_tokens: skipSpecial,
      });
      setResult(res);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="mt-4 p-5">
      <label className="mb-2 block text-sm font-medium">Token ids</label>
      <Input
        placeholder="Comma or space separated ids, e.g. 1, 15043, 29892"
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") run();
        }}
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <span className="text-xs text-muted-foreground">
            {fmtNumber(tokens.length)} token{tokens.length === 1 ? "" : "s"} parsed
          </span>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Switch checked={skipSpecial} onCheckedChange={setSkipSpecial} />
            Skip special tokens
          </label>
        </div>
        <Button onClick={run} disabled={!model || tokens.length === 0 || busy}>
          {busy ? <Spinner size="sm" /> : <Type className="h-4 w-4" />} Detokenize
        </Button>
      </div>

      {error && (
        <Alert variant="error" className="mt-4">
          {error}
        </Alert>
      )}

      {result && (
        <div className="mt-4">
          <div className="mb-2 text-sm font-medium">Decoded text</div>
          <Card className="bg-muted p-4">
            <pre className="whitespace-pre-wrap break-words font-mono text-sm text-foreground">
              {result.text || " "}
            </pre>
          </Card>
        </div>
      )}
    </Card>
  );
}

// --- Count ----------------------------------------------------------------
function CountTab({ model }: { model: string }) {
  const [prompt, setPrompt] = useState("");
  const [addSpecial, setAddSpecial] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TokenCountResult | null>(null);

  const run = async () => {
    if (!model || !prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<TokenCountResult>("/v1/token_count", {
        model,
        prompt,
        add_special_tokens: addSpecial,
      });
      setResult(res);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const pct =
    result && result.max_context_tokens > 0
      ? (result.token_count / result.max_context_tokens) * 100
      : 0;
  const tone = result?.over_context_limit ? "error" : pct >= 80 ? "warning" : "success";

  return (
    <Card className="mt-4 p-5">
      <label className="mb-2 block text-sm font-medium">Prompt</label>
      <Textarea
        rows={5}
        placeholder="Enter a prompt to count tokens…"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <Switch checked={addSpecial} onCheckedChange={setAddSpecial} />
          Add special tokens
        </label>
        <Button onClick={run} disabled={!model || !prompt.trim() || busy}>
          {busy ? <Spinner size="sm" /> : <Calculator className="h-4 w-4" />} Count
        </Button>
      </div>

      {error && (
        <Alert variant="error" className="mt-4">
          {error}
        </Alert>
      )}

      {result && (
        <div className="mt-4 space-y-3">
          <Card className="p-5">
            <div className="mb-3 flex items-end justify-between gap-3">
              <div>
                <div className="text-sm text-muted-foreground">Context budget</div>
                <div
                  className={cn(
                    "mt-1 text-3xl font-semibold tabular-nums",
                    result.over_context_limit ? "text-error" : "text-foreground",
                  )}
                >
                  {fmtNumber(result.token_count)}
                  <span className="ml-1 text-base font-normal text-muted-foreground">
                    / {fmtNumber(result.max_context_tokens)}
                  </span>
                </div>
              </div>
              <Badge variant={result.over_context_limit ? "error" : "info"}>{fmtPct(pct)}</Badge>
            </div>
            <SegmentedBar
              height={10}
              total={result.max_context_tokens}
              segments={[{ value: result.token_count, tone, label: "Prompt tokens" }]}
              formatValue={fmtNumber}
            />
          </Card>
          {result.over_context_limit && (
            <Alert variant="error" title="Over context limit">
              The prompt ({fmtNumber(result.token_count)} tokens) exceeds this model&apos;s{" "}
              {fmtNumber(result.max_context_tokens)}-token context window.
            </Alert>
          )}
        </div>
      )}
    </Card>
  );
}
