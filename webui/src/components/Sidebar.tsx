"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useYunUI } from "yunui/adapters";
import { StatusIndicator, ThemeToggle, cn } from "yunui";
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

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/completions", label: "Completions", icon: FileText },
  { href: "/embeddings", label: "Embeddings", icon: VectorSquare },
  { href: "/tokenize", label: "Tokenize", icon: Hash },
  { href: "/audio", label: "Audio", icon: Mic },
  { href: "/images", label: "Images", icon: ImageIcon },
  { href: "/models", label: "Models", icon: Box },
  { href: "/monitoring", label: "Monitoring", icon: Activity },
  { href: "/realtime", label: "Realtime", icon: Radio },
  { href: "/mcp", label: "MCP", icon: Wrench },
  { href: "/batch", label: "Batch", icon: Layers },
  { href: "/benchmarks", label: "Benchmarks", icon: Zap },
  { href: "/settings", label: "Settings", icon: Settings2 },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { Link } = useYunUI();
  const [connected, setConnected] = useState(false);
  const [info, setInfo] = useState<{ mlx?: string }>({});

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
      .then((d) => setInfo({ mlx: d.mlx_version }))
      .catch(() => {});
  }, []);

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-card">
      {/* Logo */}
      <div className="border-b border-border px-4 py-4">
        <h1 className="text-lg font-bold tracking-tight text-accent">Yunshu</h1>
        <p className="mt-0.5 text-[11px] text-muted-foreground">MLX Inference Platform</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-0.5 overflow-auto p-2">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-accent/10 font-medium text-accent"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-r bg-accent" />
              )}
              <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Status footer */}
      <div className="space-y-1.5 border-t border-border p-3">
        <div className="flex items-center gap-2">
          <StatusIndicator status={connected ? "online" : "offline"}>
            <span className={connected ? "text-success" : "text-muted-foreground"}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </StatusIndicator>
          <ThemeToggle className="ml-auto" />
        </div>
        {info.mlx && (
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <Cpu className="h-3 w-3" />
            MLX {info.mlx}
          </div>
        )}
      </div>
    </aside>
  );
}
