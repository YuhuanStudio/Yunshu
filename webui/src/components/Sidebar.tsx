"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
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
  Shield,
  Cpu,
  Wifi,
  WifiOff,
  Sun,
  Moon,
  VectorSquare,
  FileText,
  Hash,
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
  { href: "/admin", label: "Admin", icon: Shield },
  { href: "/benchmarks", label: "Benchmarks", icon: Zap },
  { href: "/settings", label: "Settings", icon: Settings2 },
];

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = localStorage.getItem("yunshu_theme") as "dark" | "light" | null;
    const initial = saved || "dark";
    setTheme(initial);
    document.documentElement.setAttribute("data-theme", initial);
    if (initial === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("yunshu_theme", next);
    document.documentElement.setAttribute("data-theme", next);
    if (next === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return { theme, toggle };
}

export default function Sidebar() {
  const pathname = usePathname();
  const [connected, setConnected] = useState(false);
  const [info, setInfo] = useState<{ mlx?: string; python?: string }>({});
  const { theme, toggle } = useTheme();

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
      .then((d) => setInfo({ mlx: d.mlx_version, python: d.python_version }))
      .catch(() => {});
  }, []);

  return (
    <aside className="w-60 border-r border-[var(--color-border)] bg-[var(--color-bg-secondary)] flex flex-col shrink-0">
      {/* Logo */}
      <div className="px-4 py-4 border-b border-[var(--color-border)]">
        <h1 className="text-lg font-bold tracking-tight">
          <span className="text-[var(--color-accent)]">Yunshu</span>
        </h1>
        <p className="text-[11px] text-[var(--color-text-secondary)] mt-0.5">
          MLX Inference Platform
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-0.5 overflow-auto">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <a
              key={href}
              href={href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors relative ${
                active
                  ? "bg-[var(--color-accent-muted)] text-[var(--color-accent)] font-medium"
                  : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 bg-[var(--color-accent)] rounded-r" />
              )}
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </a>
          );
        })}
      </nav>

      {/* Status footer */}
      <div className="p-3 border-t border-[var(--color-border)] space-y-1.5">
        <div className="flex items-center gap-2 text-xs">
          {connected ? (
            <Wifi className="w-3 h-3 text-[var(--color-success)]" />
          ) : (
            <WifiOff className="w-3 h-3 text-[var(--color-danger)]" />
          )}
          <span className={connected ? "text-[var(--color-success)]" : "text-[var(--color-danger)]"}>
            {connected ? "Connected" : "Disconnected"}
          </span>
          <button
            onClick={toggle}
            className="ml-auto p-1 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] transition-colors"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>
        </div>
        {info.mlx && (
          <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-text-secondary)]">
            <Cpu className="w-3 h-3" />
            MLX {info.mlx}
          </div>
        )}
      </div>
    </aside>
  );
}
