import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "@yuhuanowo/yunui";
import { t } from "./i18n/index.ts";

/**
 * One page's error must not blank the app: the shell, the navigation and every other page keep
 * working. The fallback names the error, offers a retry (which remounts the page) and the docs.
 */
export class PageBoundary extends Component<
  { children: ReactNode; page: string; onDocs?: () => void },
  { error: Error | null; attempt: number }
> {
  state = { error: null as Error | null, attempt: 0 };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(
      "console page failed",
      this.props.page,
      error,
      info.componentStack,
    );
  }

  componentDidUpdate(prev: { page: string }) {
    // Moving to another page clears the failure; the failing page stays failed until retried.
    if (prev.page !== this.props.page && this.state.error)
      this.setState({ error: null });
  }

  render() {
    if (this.state.error)
      return (
        <div
          role="alert"
          data-testid="page-error"
          className="mx-auto w-full max-w-2xl space-y-3 p-6"
        >
          <h1 className="text-xl font-semibold">{t("shell.boundary.title")}</h1>
          <p className="text-sm text-muted-foreground">
            {t("shell.boundary.body")}
          </p>
          <pre className="max-h-40 overflow-auto rounded-lg border border-border bg-(--bg-elevated) p-3 font-mono text-xs whitespace-pre-wrap break-all">
            {this.state.error.message}
          </pre>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              onClick={() =>
                this.setState((s) => ({ error: null, attempt: s.attempt + 1 }))
              }
            >
              {t("shell.boundary.retry")}
            </Button>
            {this.props.onDocs && (
              <Button size="sm" variant="ghost" onClick={this.props.onDocs}>
                {t("shell.boundary.docs")}
              </Button>
            )}
          </div>
        </div>
      );
    return (
      <div key={this.state.attempt} className="contents">
        {this.props.children}
      </div>
    );
  }
}
