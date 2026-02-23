import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";

type RecallRow = {
  score?: number;
  memory?: {
    id?: number;
    kind?: { Fact?: { subject: string; relation: string; object: string }; Episode?: { text: string } };
    namespace?: string;
    access_count?: number;
    strength?: number;
  };
  explain?: {
    final_score?: number;
    base_score?: number;
    rrf_rank?: number;
    bm25_rank?: number | null;
    vector_rank?: number | null;
    modality_agreement?: boolean;
    matched_modalities?: number;
    score_margin_to_next?: number | null;
  };
};

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function conchBin() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_BIN || path.join(home, ".cargo", "bin", "conch");
}

function traceLabel(row: RecallRow) {
  const explain = row.explain || {};
  const agreement = explain.modality_agreement ? "dual" : "single";
  const confidence = (explain.score_margin_to_next || 0) > 0.05 ? "stable" : "tight";
  return `${agreement}/${confidence}/rrf#${explain.rrf_rank ?? "-"}`;
}

function toCsv(rows: RecallRow[]) {
  const header = [
    "rank",
    "memory_id",
    "kind",
    "label",
    "trace_label",
    "final_score",
    "base_score",
    "rrf_rank",
    "bm25_rank",
    "vector_rank",
    "matched_modalities",
    "modality_agreement",
    "score_margin_to_next",
  ];

  const esc = (v: unknown) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const lines = [header.join(",")];
  rows.forEach((r, idx) => {
    const fact = r.memory?.kind?.Fact;
    const episode = r.memory?.kind?.Episode;
    const label = fact ? `${fact.subject} ${fact.relation} ${fact.object}` : episode?.text || "";
    lines.push([
      idx + 1,
      r.memory?.id ?? "",
      fact ? "fact" : "episode",
      label,
      traceLabel(r),
      r.explain?.final_score ?? "",
      r.explain?.base_score ?? "",
      r.explain?.rrf_rank ?? "",
      r.explain?.bm25_rank ?? "",
      r.explain?.vector_rank ?? "",
      r.explain?.matched_modalities ?? "",
      r.explain?.modality_agreement ?? "",
      r.explain?.score_margin_to_next ?? "",
    ].map(esc).join(","));
  });

  return lines.join("\n") + "\n";
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const query = String(body?.q || "").trim();
    const namespace = String(body?.namespace || "default");
    const limit = Math.min(50, Math.max(1, Number(body?.limit || 12)));
    const outDir = String(body?.outDir || path.join(process.env.HOME || ".", ".openclaw", "workspace", "cron-logs"));

    if (!query) {
      return NextResponse.json({ error: "q is required" }, { status: 400 });
    }

    const args = ["--db", dbPath(), "--namespace", namespace, "--json", "recall", query, "--limit", String(limit)];
    const out = execFileSync(conchBin(), args, { encoding: "utf8" });
    const results = (out.trim() ? JSON.parse(out) : []) as RecallRow[];

    const now = new Date();
    const ts = now.toISOString().replace(/[:]/g, "-").replace(/\.\d{3}Z$/, "Z");
    mkdirSync(outDir, { recursive: true });

    const jsonPath = path.join(outDir, `conch-recall-explainability-${ts}.json`);
    const csvPath = path.join(outDir, `conch-recall-explainability-${ts}.csv`);

    writeFileSync(jsonPath, JSON.stringify({ generated_at: now.toISOString(), namespace, query, limit, result_count: results.length, results }, null, 2) + "\n", "utf8");
    writeFileSync(csvPath, toCsv(results), "utf8");

    return NextResponse.json({ ok: true, jsonPath, csvPath, result_count: results.length, generated_at: now.toISOString() });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
