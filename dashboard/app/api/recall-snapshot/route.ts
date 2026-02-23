import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function conchBin() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_BIN || path.join(home, ".cargo", "bin", "conch");
}

function runRecall(namespace: string, query: string, limit: number, tag?: string) {
  const args = ["--db", dbPath(), "--namespace", namespace, "--json", "recall", query, "--limit", String(limit)];
  if (tag) args.push("--tag", tag);
  const out = execFileSync(conchBin(), args, { encoding: "utf8" });
  return out.trim() ? JSON.parse(out) : [];
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const query = String(body?.q || "").trim();
    const namespace = String(body?.namespace || "default");
    const tag = body?.tag ? String(body.tag).trim() : undefined;
    const limit = Math.min(50, Math.max(1, Number(body?.limit || 10)));
    const outDir = String(body?.outDir || path.join(process.env.HOME || ".", ".openclaw", "workspace", "cron-logs"));

    if (!query) {
      return NextResponse.json({ error: "q is required" }, { status: 400 });
    }

    const results = runRecall(namespace, query, limit, tag);
    const now = new Date();
    const ts = now.toISOString().replace(/[:]/g, "-").replace(/\.\d{3}Z$/, "Z");

    const payload = {
      generated_at: now.toISOString(),
      db: dbPath(),
      namespace,
      query,
      tag: tag || null,
      limit,
      result_count: results.length,
      results,
    };

    mkdirSync(outDir, { recursive: true });
    const filePath = path.join(outDir, `conch-recall-snapshot-${ts}.json`);
    writeFileSync(filePath, JSON.stringify(payload, null, 2) + "\n", "utf8");

    return NextResponse.json({ ok: true, filePath, ...payload });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
