import { useState } from "react";
import { Button } from "@yuhuanowo/yunui";
import { Check, Copy } from "lucide-react";
import { t } from "./i18n/index.ts";
import { detailText } from "./errors.ts";

/** One calm line with the raw backend text tucked behind the details. */
export function ErrorNote({
  message,
  detail,
  error,
  tone = "error",
  className,
}: {
  message: string;
  detail?: string;
  /** When given, the detail is derived from this error (status, backend text, cause). */
  error?: unknown;
  tone?: "error" | "warning" | "muted";
  className?: string;
}) {
  const raw = detail ?? (error === undefined ? undefined : detailText(error));
  const color =
    tone === "error"
      ? "text-error"
      : tone === "warning"
        ? "text-warning"
        : "text-muted-foreground";
  return (
    <div role="status" className={`text-xs ${className ?? ""}`}>
      <p className={color}>{message}</p>
      {raw && (
        <details className="mt-1 text-muted-foreground">
          <summary className="cursor-pointer select-none">
            {t("common.details")}
          </summary>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all font-mono">
            {raw}
          </pre>
        </details>
      )}
    </div>
  );
}

/** Icon-only copy button for a URL or path. */
export function CopyIconButton({
  value,
  label,
}: {
  value: string;
  label: string;
}) {
  const [done, setDone] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      type="button"
      aria-label={label}
      title={label}
      onClick={() => {
        void navigator.clipboard
          ?.writeText(value)
          .then(() => {
            setDone(true);
            setTimeout(() => setDone(false), 1800);
          })
          .catch(() => undefined);
      }}
    >
      {done ? <Check size={13} /> : <Copy size={13} />}
    </Button>
  );
}
