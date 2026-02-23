import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";

type RollupPayload = {
  project?: string;
  date?: string; // YYYY-MM-DD
  shipped?: string[];
  in_progress?: string[];
  blockers?: string[];
  next?: string[];
};

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function conchBin() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_BIN || path.join(home, ".cargo", "bin", "conch");
}

function todayDate() {
  return new Date().toISOString().slice(0, 10);
}

function normalizeList(items?: string[]) {
  return (items || []).map((x) => x.trim()).filter(Boolean);
}

function buildRollupText(input: RollupPayload) {
  const project = (input.project || "Conch").trim();
  const date = (input.date || todayDate()).trim();
  const shipped = normalizeList(input.shipped);
  const inProgress = normalizeList(input.in_progress);
  const blockers = normalizeList(input.blockers);
  const next = normalizeList(input.next);

  if (!shipped.length && !inProgress.length && !blockers.length && !next.length) {
    throw new Error("at least one section must have items");
  }

  const toLines = (title: string, lines: string[]) =>
    lines.length ? [`${title}:`, ...lines.map((l) => `- ${l}`)] : [`${title}:`, "- none"];

  const text = [
    `[daily_rollup] ${project} ${date}`,
    ...toLines("shipped", shipped),
    ...toLines("in_progress", inProgress),
    ...toLines("blockers", blockers),
    ...toLines("next", next),
  ].join("\n");

  return { project, date, text, shipped, inProgress, blockers, next };
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as RollupPayload;
    const rollup = buildRollupText(body);

    execFileSync(conchBin(), ["--db", dbPath(), "remember-episode", rollup.text], { encoding: "utf8" });

    return NextResponse.json({ ok: true, rollup });
  } catch (err: any) {
    return NextResponse.json({ ok: false, error: err.message }, { status: 400 });
  }
}
