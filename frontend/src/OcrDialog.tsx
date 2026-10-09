import { useEffect, useRef, useState } from "react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  FileDropzone,
  Spinner,
} from "@yuhuanowo/yunui";
import { MediaInspector } from "@yuhuanowo/yunui/patterns";
import { ApiError, type Connection } from "./api";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format";
import { runOcr, type OcrResult } from "./ocr-api";

const errText = (e: unknown) =>
  e instanceof ApiError ? e.publicMessage : e instanceof Error ? e.message : "";

/**
 * Image to text through `/v1/ocr`, with the picture beside what the model read. The engine
 * returns text only: no box positions and no real confidence, so the inspector says it cannot
 * link the two instead of pretending to.
 */
export function OcrDialog({
  open,
  onClose,
  connection,
  model,
}: {
  open: boolean;
  onClose: () => void;
  connection: Connection;
  /** A loaded OCR engine or VLM; the route falls back to any loaded VLM when empty. */
  model: string;
}) {
  useLocale();
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [dims, setDims] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<{ result: OcrResult; ms: number } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!file) return;
    const u = URL.createObjectURL(file);
    setUrl(u);
    setDims(null);
    const img = new Image();
    img.onload = () => setDims(`${img.naturalWidth} × ${img.naturalHeight}`);
    img.src = u;
    return () => URL.revokeObjectURL(u);
  }, [file]);

  async function run() {
    if (!file || busy) return;
    const c = new AbortController();
    abort.current = c;
    setBusy(true);
    setError(null);
    setOut(null);
    try {
      setOut(await runOcr(connection, file, model, c.signal));
    } catch (e) {
      if (!c.signal.aborted) setError(errText(e) || t("playground.ocr.failed"));
    } finally {
      setBusy(false);
    }
  }
  const close = () => {
    abort.current?.abort();
    onClose();
  };
  const caption = [
    file?.name,
    dims,
    out ? `${number(out.ms / 1000, 1)} s` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Dialog open={open} onOpenChange={(o) => !o && close()}>
      <DialogContent
        closeLabel={t("playground.params.close")}
        className="max-h-[90dvh] max-w-3xl overflow-y-auto"
        data-testid="ocr-dialog"
      >
        <DialogTitle>{t("playground.ocr.title")}</DialogTitle>
        <DialogDescription>
          {t("playground.ocr.desc", { model })}
        </DialogDescription>
        {!file ? (
          <FileDropzone
            accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp"
            onFiles={(files: File[]) => setFile(files[0] ?? null)}
            label={t("playground.ocr.drop")}
            hint={t("playground.ocr.hint")}
          />
        ) : (
          <MediaInspector
            src={url ?? ""}
            alt={file.name}
            caption={caption}
            labels={{
              preview: t("playground.ocr.preview"),
              result: t("playground.ocr.result"),
              noBoxes: t("playground.ocr.noBoxes"),
            }}
            result={
              busy ? (
                <span className="inline-flex items-center gap-2 text-muted-foreground">
                  <Spinner size="sm" />
                  {t("playground.ocr.running")}
                </span>
              ) : out ? (
                out.result.text ? (
                  <pre
                    className="m-0 whitespace-pre-wrap font-sans"
                    data-testid="ocr-text"
                  >
                    {out.result.text}
                  </pre>
                ) : (
                  <span className="text-muted-foreground">
                    {t("playground.ocr.empty")}
                  </span>
                )
              ) : (
                <span className="text-muted-foreground">
                  {t("playground.ocr.idle")}
                </span>
              )
            }
          />
        )}
        {error && (
          <p
            className="text-sm text-danger"
            role="alert"
            data-testid="ocr-error"
          >
            {error}
          </p>
        )}
        {out && (
          <p className="text-xs tabular-nums text-muted-foreground">
            {t("playground.ocr.usage", {
              model: out.result.model ?? model,
              prompt:
                out.result.promptTokens != null
                  ? number(out.result.promptTokens, 0)
                  : "—",
              completion:
                out.result.completionTokens != null
                  ? number(out.result.completionTokens, 0)
                  : "—",
            })}
          </p>
        )}
        <div className="flex flex-wrap justify-end gap-2">
          {file && (
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => {
                setFile(null);
                setOut(null);
                setError(null);
              }}
            >
              {t("playground.ocr.another")}
            </Button>
          )}
          <Button disabled={!file || busy} onClick={() => void run()}>
            {t("playground.ocr.run")}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
