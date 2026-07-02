import type { ReactNode } from "react";
import { PageHeader } from "yunui/patterns";
import { ThemeToggle, cn } from "yunui";
import { ShellChrome } from "./app-shell";

/**
 * Standard page container for the Yunshu webui: consistent padding + max width,
 * a YunUI PageHeader, and a subtle enter animation. Every page wraps its body
 * in this so the whole app shares one rhythm.
 */
export function PageShell({
  title,
  description,
  actions,
  children,
  className,
  width = "wide",
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** `wide` (default) for dashboards/grids, `narrow` for form/playground pages. */
  width?: "wide" | "narrow";
}) {
  return (
    <div className="animate-enter px-6 py-6 sm:px-8">
      <div className={cn("mx-auto", width === "narrow" ? "max-w-3xl" : "max-w-7xl")}>
        {/* Shell nav controls + theme toggle live in THIS header row — the app
            shell has no bar of its own (mirrors Yunxin/YunUI). */}
        <div className="flex items-start gap-3">
          <ShellChrome className="h-9" />
          <PageHeader
            className="min-w-0 flex-1"
            title={title}
            description={description}
            actions={
              <>
                {actions}
                <ThemeToggle variant="pill" />
              </>
            }
          />
        </div>
        <div className={cn("mt-6", className)}>{children}</div>
      </div>
    </div>
  );
}
