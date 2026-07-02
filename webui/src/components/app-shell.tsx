"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { IconButton, cn } from "yunui";
import { Menu, PanelLeftOpen } from "lucide-react";
import Sidebar from "./Sidebar";

interface AppShellState {
  /** Open the mobile nav drawer. */
  openNav: () => void;
  /** Whether the desktop sidebar is collapsed away. */
  collapsed: boolean;
  /** Re-expand the collapsed desktop sidebar. */
  expand: () => void;
}

const AppShellContext = createContext<AppShellState>({
  openNav: () => {},
  collapsed: false,
  expand: () => {},
});

export const useAppShell = () => useContext(AppShellContext);

/**
 * Shell controls for a page's own header row: the mobile nav hamburger and,
 * while the desktop sidebar is collapsed, a re-open button. Pages (PageShell,
 * chat, responses) put this at the START of their header — the shell renders
 * no bar of its own, matching Yunxin/YunUI where the theme toggle and nav
 * controls live in the page header instead of a dedicated strip.
 */
export function ShellChrome({ className }: { className?: string }) {
  const { openNav, collapsed, expand } = useAppShell();
  return (
    // When expanded on desktop both buttons are hidden — hide the container
    // too so it doesn't leave a phantom flex-gap indent before the title.
    <div className={cn("flex shrink-0 items-center gap-1.5", !collapsed && "lg:hidden", className)}>
      <IconButton icon={<Menu size={18} />} label="Open menu" onClick={openNav} className="lg:hidden" />
      {collapsed && (
        <IconButton
          icon={<PanelLeftOpen size={18} />}
          label="Open sidebar"
          onClick={expand}
          className="hidden lg:inline-flex"
        />
      )}
    </div>
  );
}

/**
 * App shell: the fixed YunUI Sidebar (mobile drawer + desktop collapse) with
 * <main> as the single scroll container. h-dvh (not min-h-dvh): the shell must
 * be the height bound so main's overflow-y-auto engages — with min-h the shell
 * grows with tall content, the document scrolls instead, full-height pages
 * (chat/responses) push their composer below the fold, and the fixed sidebar
 * desyncs from in-flow columns.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <AppShellContext.Provider
      value={{ openNav: () => setOpen(true), collapsed, expand: () => setCollapsed(false) }}
    >
      <div className="flex h-dvh overflow-hidden bg-background text-foreground">
        <Sidebar
          isOpen={open}
          onClose={() => setOpen(false)}
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed((c) => !c)}
        />
        <main
          className={cn(
            "min-w-0 flex-1 overflow-y-auto overflow-x-hidden transition-[margin] duration-200 ease-in-out",
            collapsed ? "lg:ml-0" : "lg:ml-64",
          )}
        >
          {children}
        </main>
      </div>
    </AppShellContext.Provider>
  );
}
