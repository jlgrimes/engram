import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Conch — Biological Memory for AI Agents",
  description: "Semantic search + decay, no API keys needed. Replace memory.md with something that actually scales.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#0a0a0a] text-gray-100 antialiased font-sans">{children}</body>
    </html>
  );
}
