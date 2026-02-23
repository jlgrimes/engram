import { NextRequest, NextResponse } from "next/server";
import path from "node:path";
import { readdirSync, readFileSync, statSync } from "node:fs";

type SnapshotResult = {
  score?: number;
  memory?: { id?: number };
  explain?: {
    final_score?: number;
    rrf_rank?: number;
    bm25_rank?: number | null;
    vector_rank?: number | null;
  };
};

type SnapshotFile = {
  generated_at?: string;
  namespace?: string;
  query?: string;
  tag?: string | null;
  results?: SnapshotResult[];
};

type SnapshotEntry = {
  filePath: string;
  payload: SnapshotFile;
};

type CacheEntry = { mtimeMs: number; payload: SnapshotFile | null };

const snapshotCache = new Map<string, CacheEntry>();

function snapshotsDir() {
  const home = process.env.HOME || ".";
  return path.join(home, ".openclaw", "workspace", "cron-logs");
}

function parseSnapshotCached(filePath: string): SnapshotFile | null {
  try {
    const mtimeMs = statSync(filePath).mtimeMs;
    const cached = snapshotCache.get(filePath);
    if (cached && cached.mtimeMs === mtimeMs) {
      return cached.payload;
    }
    const payload = JSON.parse(readFileSync(filePath, "utf8"));
    snapshotCache.set(filePath, { mtimeMs, payload });
    return payload;
  } catch {
    snapshotCache.set(filePath, { mtimeMs: 0, payload: null });
    return null;
  }
}

function resolvePrevious(
  matching: SnapshotEntry[],
  compareTo: string,
  nth: number,
  filePath: string | null,
) {
  if (compareTo === "file") {
    if (!filePath) return { error: "compareTo=file requires filePath" } as const;
    const normalized = path.resolve(filePath);
    const found = matching.find((m) => path.resolve(m.filePath) === normalized);
    if (!found) return { error: "Requested filePath was not found in filtered snapshots" } as const;
    return { previous: found } as const;
  }

  if (compareTo === "nth") {
    const idx = Math.max(2, nth) - 1;
    const previous = matching[idx];
    if (!previous) {
      return { error: `Requested nth baseline (${nth}) not available. matchCount=${matching.length}` } as const;
    }
    return { previous } as const;
  }

  // default: previous (latest-1)
  const previous = matching[1];
  if (!previous) return { error: "Need at least 2 matching snapshots to compare." } as const;
  return { previous } as const;
}

export async function GET(req: NextRequest) {
  try {
    const url = new URL(req.url);
    const namespace = String(url.searchParams.get("namespace") || "default");
    const query = String(url.searchParams.get("q") || "").trim();
    const tag = (url.searchParams.get("tag") || "").trim() || null;
    const compareTo = (url.searchParams.get("compareTo") || "previous").trim();
    const nth = Number(url.searchParams.get("nth") || "2");
    const baselineFilePath = (url.searchParams.get("filePath") || "").trim() || null;

    if (!query) {
      return NextResponse.json({ error: "q is required" }, { status: 400 });
    }

    const dir = snapshotsDir();
    const files = readdirSync(dir)
      .filter((name) => name.startsWith("conch-recall-snapshot-") && name.endsWith(".json"))
      .map((name) => path.join(dir, name));

    const matching = files
      .map((filePath) => ({ filePath, payload: parseSnapshotCached(filePath) }))
      .filter((it) => !!it.payload)
      .filter((it) => it.payload?.namespace === namespace)
      .filter((it) => (it.payload?.query || "") === query)
      .filter((it) => (it.payload?.tag || null) === tag)
      .sort((a, b) => {
        const ta = Date.parse(a.payload?.generated_at || "") || 0;
        const tb = Date.parse(b.payload?.generated_at || "") || 0;
        return tb - ta;
      }) as SnapshotEntry[];

    if (matching.length < 2) {
      return NextResponse.json({
        ok: true,
        hasComparison: false,
        message: "Need at least 2 matching snapshots to compare.",
        matchCount: matching.length,
      });
    }

    const latest = matching[0];
    const resolved = resolvePrevious(matching, compareTo, nth, baselineFilePath);
    if ("error" in resolved) {
      return NextResponse.json({ error: resolved.error }, { status: 400 });
    }

    const previous = resolved.previous;
    const latestResults = latest.payload?.results || [];
    const prevResults = previous.payload?.results || [];

    const prevById = new Map<number, { idx: number; item: SnapshotResult }>();
    prevResults.forEach((item, idx) => {
      const id = item.memory?.id;
      if (typeof id === "number") prevById.set(id, { idx, item });
    });

    const compared = latestResults
      .map((item, idx) => {
        const id = item.memory?.id;
        if (typeof id !== "number") return null;
        const prev = prevById.get(id);
        const latestRank = idx + 1;
        const prevRank = prev ? prev.idx + 1 : null;
        const rankDelta = prevRank == null ? null : prevRank - latestRank;
        const latestFinal = item.explain?.final_score ?? null;
        const prevFinal = prev?.item.explain?.final_score ?? null;
        const finalDelta = latestFinal == null || prevFinal == null ? null : latestFinal - prevFinal;

        return {
          memory_id: id,
          latest_rank: latestRank,
          previous_rank: prevRank,
          rank_delta: rankDelta,
          latest_final_score: latestFinal,
          previous_final_score: prevFinal,
          final_score_delta: finalDelta,
          latest_rrf_rank: item.explain?.rrf_rank ?? null,
          previous_rrf_rank: prev?.item.explain?.rrf_rank ?? null,
          latest_bm25_rank: item.explain?.bm25_rank ?? null,
          previous_bm25_rank: prev?.item.explain?.bm25_rank ?? null,
          latest_vector_rank: item.explain?.vector_rank ?? null,
          previous_vector_rank: prev?.item.explain?.vector_rank ?? null,
        };
      })
      .filter((v): v is NonNullable<typeof v> => !!v)
      .sort((a, b) => Math.abs(b.rank_delta ?? 0) - Math.abs(a.rank_delta ?? 0));

    const entered_count = compared.filter((row) => row.previous_rank == null).length;
    const moved_count = compared.filter((row) => (row.rank_delta ?? 0) !== 0 && row.previous_rank != null).length;

    const avgAbsRankDelta =
      compared.filter((row) => row.rank_delta != null).reduce((acc, row) => acc + Math.abs(row.rank_delta || 0), 0) /
      Math.max(1, compared.filter((row) => row.rank_delta != null).length);

    const avgAbsScoreDelta =
      compared.filter((row) => row.final_score_delta != null).reduce((acc, row) => acc + Math.abs(row.final_score_delta || 0), 0) /
      Math.max(1, compared.filter((row) => row.final_score_delta != null).length);

    const severity_score = Number((avgAbsRankDelta * 0.6 + avgAbsScoreDelta * 0.4 + entered_count * 0.15).toFixed(4));

    return NextResponse.json({
      ok: true,
      hasComparison: true,
      namespace,
      query,
      tag,
      compareTo,
      nth,
      latest: {
        filePath: latest.filePath,
        generated_at: latest.payload?.generated_at || null,
        result_count: latestResults.length,
      },
      previous: {
        filePath: previous.filePath,
        generated_at: previous.payload?.generated_at || null,
        result_count: prevResults.length,
      },
      metrics: {
        compared_count: compared.length,
        entered_count,
        moved_count,
        avg_abs_rank_delta: Number(avgAbsRankDelta.toFixed(4)),
        avg_abs_score_delta: Number(avgAbsScoreDelta.toFixed(4)),
        severity_score,
      },
      rows: compared,
    });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
