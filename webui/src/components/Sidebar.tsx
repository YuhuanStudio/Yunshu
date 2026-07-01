"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Sidebar as YunUISidebar, type SidebarSection } from "yunui/patterns";
import { StatusIndicator } from "yunui";
import { api } from "@/lib/api";
import type { SystemStats } from "@/lib/types";
import {
  LayoutDashboard,
  MessageSquare,
  MessagesSquare,
  AudioLines,
  Mic,
  ImageIcon,
  Video,
  ScanText,
  Box,
  Activity,
  Radar,
  Power,
  Zap,
  Radio,
  Settings2,
  Cpu,
  VectorSquare,
  ListOrdered,
  Database,
  FileText,
  Hash,
  Wrench,
  Layers,
} from "lucide-react";

// Grouped nav sections — the target IA (a lead item + Playground / Generate /
// Toolkit / Operate). More items land as later rewrite phases add their pages.
const SECTIONS: SidebarSection[] = [
  { items: [{ href: "/", label: "Dashboard", icon: LayoutDashboard }] },
  {
    title: "Playground",
    items: [
      { href: "/chat", label: "Chat", icon: MessageSquare },
      { href: "/responses", label: "Responses", icon: MessagesSquare },
      { href: "/completions", label: "Completions", icon: FileText },
      { href: "/realtime", label: "Realtime", icon: Radio },
      { href: "/omni", label: "Omni", icon: AudioLines },
    ],
  },
  {
    title: "Generate",
    items: [
      { href: "/images", label: "Images", icon: ImageIcon },
      { href: "/video", label: "Video", icon: Video },
      { href: "/audio", label: "Audio", icon: Mic },
      { href: "/ocr", label: "OCR", icon: ScanText },
    ],
  },
  {
    title: "Toolkit",
    items: [
      { href: "/embeddings", label: "Embeddings", icon: VectorSquare },
      { href: "/rerank", label: "Rerank", icon: ListOrdered },
      { href: "/tokenize", label: "Tokenize", icon: Hash },
      { href: "/batch", label: "Batch", icon: Layers },
      { href: "/cached", label: "Cached", icon: Database },
      { href: "/mcp", label: "MCP", icon: Wrench },
    ],
  },
  {
    title: "Operate",
    items: [
      { href: "/models", label: "Models", icon: Box },
      { href: "/monitoring", label: "Monitoring", icon: Activity },
      { href: "/activity", label: "Activity", icon: Radar },
      { href: "/benchmarks", label: "Benchmarks", icon: Zap },
      { href: "/profiler", label: "Profiler", icon: Cpu },
      { href: "/power", label: "Power", icon: Power },
      { href: "/settings", label: "Settings", icon: Settings2 },
    ],
  },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function Sidebar({ isOpen, onClose, collapsed, onToggleCollapse }: SidebarProps) {
  const pathname = usePathname();
  const [connected, setConnected] = useState(false);
  const [mlx, setMlx] = useState<string | undefined>();

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/health");
        setConnected(res.ok);
      } catch {
        setConnected(false);
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // Monitoring requires the auth token; use the api client (attaches Bearer)
    // and read the MLX version off the nested gpu object (real /gw/ shape).
    api
      .get<SystemStats>("/api/v1/gw/monitoring/system", controller.signal)
      .then((d) => setMlx(d.gpu?.mlx_version))
      .catch(() => {
        /* no token / not authed — MLX version is optional chrome */
      });
    return () => controller.abort();
  }, []);

  const footer = (
    <div className="card space-y-1.5 px-3 py-2.5">
      <StatusIndicator status={connected ? "online" : "offline"}>
        <span className={connected ? "text-success" : "text-muted-foreground"}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </StatusIndicator>
      {mlx && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Cpu className="h-3 w-3" /> MLX {mlx}
        </div>
      )}
    </div>
  );

  return (
    <YunUISidebar
      appName="Yunshu"
      homeHref="/"
      sections={SECTIONS}
      currentPath={pathname}
      isOpen={isOpen}
      onClose={onClose}
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      closeLabel="Close menu"
      footer={footer}
    />
  );
}
