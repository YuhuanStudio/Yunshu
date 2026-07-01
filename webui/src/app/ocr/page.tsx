"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Button,
  Input,
  Card,
  Badge,
  Alert,
  IconButton,
  Spinner,
  SegmentedSelect,
  FileDropzone,
  cn,
  toast,
} from "yunui";
import { ScanText, Copy, Check, ImageIcon, X, Sigma, Table as TableIcon, Type } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import { fmtBytes, fmtNumber } from "@/lib/format";
import type { Model } from "@/lib/types";

/**
 * OCR playground — upload an image and extract its text, a formula, or a table
 * via `POST /v1/ocr` (multipart). The optional model picker narrows to OCR/VLM
 * models; language is a free-text hint. Results render selectable, preformatted
 * text with a copy button and a small usage meta row.
 */

type Task = "text" | "formula" | "table";

interface OcrResponse {
  text: string;
  model: string;
  confidence: number | null;
  language: string;
  task: string;
  prompt_tokens: number;
  completion_tokens: number;
  usage: { image_tokens?: number; [k: string]: unknown };
}

const TASKS = [
  { value: "text" as const, label: "Text", icon: Type },
  { value: "formula" as const, label: "Formula", icon: Sigma },
  { value: "table" as const, label: "Table", icon: TableIcon },
];

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPT = ".png,.jpg,.jpeg,.webp,.tiff,.bmp,image/*";

export default function OcrPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [task, setTask] = useState<Task>("text");
  const [language, setLanguage] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OcrResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const previewRef = useRef<string | null>(null);
  previewRef.current = preview;

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => setModels(res.data ?? []))
      .catch(() => {
        /* model list unavailable — the request will surface its own error */
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  const onFiles = useCallback((files: File[]) => {
    const f = files[0];
    if (!f) return;
    if (f.size > MAX_BYTES) {
      setError(`Image is too large (${fmtBytes(f.size)}). The limit is 10 MB.`);
      return;
    }
    setError(null);
    setResult(null);
    setFile(f);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }, []);

  const clearFile = useCallback(() => {
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setFile(null);
    setResult(null);
  }, []);

  const run = useCallback(async () => {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("task", task);
      if (language.trim()) form.append("language", language.trim());
      if (model) form.append("model", model);
      const res = await api.postForm<OcrResponse>("/v1/ocr", form);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : "OCR failed.");
    } finally {
      setLoading(false);
    }
  }, [file, task, language, model, loading]);

  const copyText = useCallback(async () => {
    if (!result?.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  }, [result]);

  return (
    <PageShell
      title="OCR"
      description="Extract text, formulas, or tables from an image."
      width="narrow"
    >
      <div className="space-y-6">
        {error && (
          <Alert
            variant="error"
            title="Request failed"
            icon={
              <IconButton
                icon={<X className="h-4 w-4" />}
                label="Dismiss"
                onClick={() => setError(null)}
                className="p-0 text-current opacity-70 transition-opacity hover:bg-transparent hover:opacity-100"
              />
            }
          >
            {error}
          </Alert>
        )}

        <Card className="space-y-4 p-5">
          <div className="space-y-2">
            <label className="text-sm font-medium">Image</label>
            {preview ? (
              <div className="flex items-center gap-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="OCR source"
                  className="h-24 w-24 shrink-0 rounded-lg border border-border object-cover"
                />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="truncate text-sm font-medium">{file?.name}</p>
                  {file && <Badge variant="info">{fmtBytes(file.size)}</Badge>}
                </div>
                <Button variant="ghost" size="sm" onClick={clearFile}>
                  <X className="h-4 w-4" /> Clear
                </Button>
              </div>
            ) : (
              <FileDropzone
                accept={ACCEPT}
                onFiles={onFiles}
                icon={<ImageIcon className="h-6 w-6" />}
                label="Drop an image or click to browse"
                hint="PNG, JPG, WEBP, TIFF or BMP · up to 10 MB"
              />
            )}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Task</label>
            <SegmentedSelect<Task> options={TASKS} value={task} onChange={setTask} />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Language <span className="text-muted-foreground">(optional hint)</span>
              </label>
              <Input
                placeholder="e.g. en, zh, ja"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Model <span className="text-muted-foreground">(optional)</span>
              </label>
              <ModelPicker models={models} value={model} onChange={setModel} />
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={run} disabled={!file || loading}>
              {loading ? <Spinner size="sm" /> : <ScanText className="h-4 w-4" />}
              {loading ? "Reading…" : "Extract"}
            </Button>
          </div>
        </Card>

        {result && (
          <Card className="space-y-3 p-5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">Extracted {result.task || task}</span>
              <Button variant="ghost" size="sm" onClick={copyText} disabled={!result.text}>
                {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>

            <pre
              className={cn(
                "max-h-112 select-text overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted/40 p-3 font-mono text-sm",
                result.text ? "" : "text-muted-foreground",
              )}
            >
              {result.text || "No text detected."}
            </pre>

            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              {result.model && <Badge variant="default">{result.model}</Badge>}
              {result.language && <Badge variant="default">{result.language}</Badge>}
              {result.confidence != null && (
                <Badge variant="info">confidence {(result.confidence * 100).toFixed(1)}%</Badge>
              )}
              <span>
                {fmtNumber(result.prompt_tokens)} prompt + {fmtNumber(result.completion_tokens)}{" "}
                completion tokens
                {result.usage?.image_tokens != null &&
                  ` · ${fmtNumber(result.usage.image_tokens)} image`}
              </span>
            </div>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
