import type { Metadata } from "next";
import Sidebar from "@/components/Sidebar";
import { Providers } from "@/components/providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Yunshu",
  description: "Production-grade MLX inference platform for Apple Silicon",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // `data-brand` / `data-accent` pick YunUI's palette; next-themes owns the
    // runtime `class` / `data-theme` (light / dark / true-black).
    <html
      lang="en"
      data-brand="blue"
      data-accent="blue"
      suppressHydrationWarning
    >
      <body className="antialiased">
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-auto">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
