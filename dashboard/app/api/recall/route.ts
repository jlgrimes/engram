import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function conchBin() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_BIN || path.join(home, ".cargo", "bin", "conch");
}

function detectTimeWindow(query: string): { kind: "today" | "yesterday" | "last24h"; fromIso: string } | null {
  const q = query.toLowerCase();
  const now = new Date();

  if (q.includes("last 24h") || q.includes("last 24 hours")) {
    return { kind: "last24h", fromIso: new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString() };
  }

  if (q.includes("today")) {
    const from = new Date(now);
    from.setHours(0, 0, 0, 0);
    return { kind: "today", fromIso: from.toISOString() };
  }

  if (q.includes("yesterday")) {
    const from = new Date(now);
    from.setDate(from.getDate() - 1);
    from.setHours(0, 0, 0, 0);
    return { kind: "yesterday", fromIso: from.toISOString() };
  }

  return null;
}

export async function GET(req: NextRequest) {
  try {
    const query = req.nextUrl.searchParams.get("q")?.trim() || "";
    const namespace = req.nextUrl.searchParams.get("namespace") || "default";
    const limit = Math.min(50, Math.max(1, Number(req.nextUrl.searchParams.get("limit") || 10)));
    const tag = req.nextUrl.searchParams.get("tag")?.trim();

    if (!query) {
      return NextResponse.json({ error: "query parameter q is required" }, { status: 400 });
    }

    const args = ["--db", dbPath(), "--namespace", namespace, "--json", "recall", query, "--limit", String(limit * 3)];
    if (tag) args.push("--tag", tag);

    const out = execFileSync(conchBin(), args, { encoding: "utf8" });
    const rawResults = out.trim() ? JSON.parse(out) : [];

    const window = detectTimeWindow(query);
    const results = window
      ? rawResults.filter((r: any) => {
          const created = r?.memory?.created_at;
          if (!created) return false;
          return new Date(created).toISOString() >= window.fromIso;
        }).slice(0, limit)
      : rawResults.slice(0, limit);

    return NextResponse.json({
      db: dbPath(),
      namespace,
      query,
      limit,
      tag: tag || null,
      time_window: window,
      results,
    });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
