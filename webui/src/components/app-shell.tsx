"use client";

import { useState, type ReactNode } from "react";
import { ThemeToggle, IconButton, cn } from "yunui";
import { Menu, PanelLeftOpen } from "lucide-react";
import Sidebar from "./Sidebar";

/**
 * App shell: the fixed YunUI Sidebar (mobile drawer + desktop collapse) plus a
 * sticky header (mobile menu button, desktop re-open button, theme toggle),
 * with the main content offset by the sidebar width on large screens. Mirrors
 * the canonical YunUI / Yunxin dashboard shell.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex min-h-dvh overflow-hidden bg-background text-foreground">
      <Sidebar
        isOpen={open}
        onClose={() => setOpen(false)}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
      />

      <div
        className={cn(
          "flex min-w-0 flex-1 flex-col transition-[margin] duration-200 ease-in-out",
          collapsed ? "lg:ml-0" : "lg:ml-64",
        )}
      >
        <header className="sticky top-0 z-30 flex shrink-0 items-center gap-3 px-4 pt-4 lg:px-6">
          {/* Mobile: open drawer */}
          <IconButton
            icon={<Menu size={20} />}
            label="Open menu"
            onClick={() => setOpen(true)}
            className="-ml-2 lg:hidden"
          />
          {/* Desktop: re-open the collapsed sidebar */}
          <button
            onClick={() => setCollapsed(false)}
            className={cn(
              "hidden shrink-0 items-center justify-center rounded-lg p-2 text-muted-foreground transition-all duration-200 hover:bg-muted hover:text-foreground",
              collapsed ? "lg:flex" : "lg:hidden",
            )}
            aria-label="Open menu"
          >
            <PanelLeftOpen size={18} />
          </button>

          <div className="ml-auto flex items-center gap-1.5">
            <ThemeToggle variant="pill" />
          </div>
        </header>

        <main className="min-w-0 flex-1 overflow-y-auto overflow-x-hidden">{children}</main>
      </div>
    </div>
  );
}
