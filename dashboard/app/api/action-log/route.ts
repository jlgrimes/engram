import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";

type ActionPayload = {
  actor?: string;
  project?: string;
  repo?: string;
  action_type?: string;
  outcome?: string;
  summary?: string;
  artifact?: {
    commit?: string;
    pr?: string;
    issue?: string;
    file?: string;
  };
  timestamp?: string;
};

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function conchBin() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_BIN || path.join(home, ".cargo", "bin", "conch");
}

function isoNow() {
  return new Date().toISOString();
}

function normalize(input: ActionPayload) {
  const timestamp = input.timestamp || isoNow();
  const actor = (input.actor || "agent").trim();
  const project = (input.project || "unknown-project").trim();
  const repo = (input.repo || "unknown-repo").trim();
  const actionType = (input.action_type || "unspecified").trim();
  const outcome = (input.outcome || "unknown").trim();
  const summary = (input.summary || "").trim();
  const artifact = input.artifact || {};

  if (!summary) throw new Error("summary is required");

  const refParts = [
    artifact.commit ? `commit:${artifact.commit}` : null,
    artifact.pr ? `pr:${artifact.pr}` : null,
    artifact.issue ? `issue:${artifact.issue}` : null,
    artifact.file ? `file:${artifact.file}` : null,
  ].filter(Boolean);

  const episodeText = [
    `[action_event]`,
    `ts=${timestamp}`,
    `actor=${actor}`,
    `project=${project}`,
    `repo=${repo}`,
    `type=${actionType}`,
    `outcome=${outcome}`,
    refParts.length ? `refs=${refParts.join(",")}` : null,
    `summary=${summary}`,
  ]
    .filter(Boolean)
    .join(" | ");

  return {
    timestamp,
    actor,
    project,
    repo,
    actionType,
    outcome,
    summary,
    refs: refParts,
    episodeText,
  };
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as ActionPayload;
    const normalized = normalize(body);

    const args = ["--db", dbPath(), "remember-episode", normalized.episodeText];
    execFileSync(conchBin(), args, { encoding: "utf8" });

    return NextResponse.json({ ok: true, event: normalized });
  } catch (err: any) {
    return NextResponse.json({ ok: false, error: err.message }, { status: 400 });
  }
}
