"use client";

import { useMemo, useState } from "react";
import {
  Button,
  Card,
  Textarea,
  Checkbox,
  NumberInput,
  Alert,
  Badge,
  Spinner,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  cn,
  toast,
} from "yunui";
import { Copy, Hash, Type, Calculator } from "lucide-react";
import { api, usePolling, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import type {
  Model,
  TokenizeResult,
  DetokenizeResult,
  TokenCountResult,
} from "@/lib/types";

function errMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Request failed";
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
          <Checkbox checked={addSpecial} onCheckedChange={setAddSpecial} />
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
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium">Tokens</span>
              <Badge variant="info">{fmtNumber(result.tokens.length)}</Badge>
            </div>
            <Button variant="secondary" size="sm" onClick={copyTokens}>
              <Copy className="h-4 w-4" /> Copy JSON
            </Button>
          </div>
          {result.tokens.length === 0 ? (
            <p className="py-4 text-center text-sm text-muted-foreground">No tokens produced.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {result.tokens.map((t, i) => (
                <span
                  key={`${i}-${t}`}
                  className="inline-flex rounded-md bg-muted px-2 py-0.5 text-xs tabular-nums text-foreground"
                >
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// --- Detokenize -----------------------------------------------------------
function DetokenizeTab({ model }: { model: string }) {
  const [raw, setRaw] = useState("");
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
        .filter((n) => Number.isFinite(n)),
    [raw],
  );

  const run = async () => {
    if (!model || tokens.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<DetokenizeResult>("/v1/detokenize", { model, tokens });
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
      <Textarea
        rows={5}
        placeholder="Paste comma or space separated token ids, e.g. 1, 15043, 29892"
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
      />
      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">
          {fmtNumber(tokens.length)} token{tokens.length === 1 ? "" : "s"} parsed
        </span>
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
  const [maxTokens, setMaxTokens] = useState(4096);
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
        max_tokens: maxTokens,
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
      <label className="mb-2 block text-sm font-medium">Prompt</label>
      <Textarea
        rows={5}
        placeholder="Enter a prompt to count tokens…"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <label className="mb-1 block text-sm font-medium">Max tokens</label>
          <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} step={256} />
        </div>
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
          <Card className="p-5 text-center">
            <div className="text-sm text-muted-foreground">Token count</div>
            <div
              className={cn(
                "mt-1 text-4xl font-semibold tabular-nums",
                result.over_context_limit ? "text-error" : "text-foreground",
              )}
            >
              {fmtNumber(result.token_count)}
            </div>
          </Card>
          {result.over_context_limit && (
            <Alert variant="error" title="Over context limit">
              The prompt plus {fmtNumber(maxTokens)} max tokens exceeds this model&apos;s context
              window.
            </Alert>
          )}
        </div>
      )}
    </Card>
  );
}
