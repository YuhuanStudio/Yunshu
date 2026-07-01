"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Sidebar as YunUISidebar, type SidebarSection } from "yunui/patterns";
import { StatusIndicator } from "yunui";
import {
  LayoutDashboard,
  MessageSquare,
  Mic,
  ImageIcon,
  Box,
  Activity,
  Zap,
  Radio,
  Settings2,
  Cpu,
  VectorSquare,
  FileText,
  Hash,
  Wrench,
  Layers,
} from "lucide-react";

// Grouped nav sections — mirrors the YunUI / Yunxin sidebar structure (a lead
// item, then titled groups) instead of one flat list.
const SECTIONS: SidebarSection[] = [
  { items: [{ href: "/", label: "Dashboard", icon: LayoutDashboard }] },
  {
    title: "Inference",
    items: [
      { href: "/chat", label: "Chat", icon: MessageSquare },
      { href: "/completions", label: "Completions", icon: FileText },
      { href: "/embeddings", label: "Embeddings", icon: VectorSquare },
      { href: "/tokenize", label: "Tokenize", icon: Hash },
      { href: "/audio", label: "Audio", icon: Mic },
      { href: "/images", label: "Images", icon: ImageIcon },
      { href: "/realtime", label: "Realtime", icon: Radio },
    ],
  },
  {
    title: "Manage",
    items: [
      { href: "/models", label: "Models", icon: Box },
      { href: "/monitoring", label: "Monitoring", icon: Activity },
      { href: "/mcp", label: "MCP", icon: Wrench },
      { href: "/batch", label: "Batch", icon: Layers },
      { href: "/benchmarks", label: "Benchmarks", icon: Zap },
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
    fetch("/api/v1/monitoring/system")
      .then((r) => r.json())
      .then((d) => setMlx(d.mlx_version))
      .catch(() => {});
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
