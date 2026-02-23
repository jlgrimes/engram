import { NextRequest, NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function q(sql: string) {
  const py = [
    "import sqlite3, json, sys",
    "db, sql = sys.argv[1], sys.argv[2]",
    "con = sqlite3.connect(db)",
    "con.row_factory = sqlite3.Row",
    "rows = [dict(r) for r in con.execute(sql).fetchall()]",
    "print(json.dumps(rows))",
  ].join("; ");

  const out = execFileSync("python3", ["-c", py, dbPath(), sql], { encoding: "utf8" });
  return out.trim() ? JSON.parse(out) : [];
}

export async function GET(req: NextRequest) {
  try {
    const search = req.nextUrl.searchParams.get("q")?.trim() || "";
    const namespace = req.nextUrl.searchParams.get("namespace") || "default";
    const limit = Math.min(200, Math.max(1, Number(req.nextUrl.searchParams.get("limit") || 50)));
    const offset = Math.max(0, Number(req.nextUrl.searchParams.get("offset") || 0));

    const where = [`namespace = '${namespace.replace(/'/g, "''")}'`];
    if (search) {
      const s = search.replace(/'/g, "''");
      where.push(`(lower(coalesce(subject,'')) like lower('%${s}%') or lower(coalesce(relation,'')) like lower('%${s}%') or lower(coalesce(object,'')) like lower('%${s}%') or lower(coalesce(episode_text,'')) like lower('%${s}%') or lower(coalesce(tags,'')) like lower('%${s}%'))`);
    }
    const whereSql = where.join(" AND ");

    const stats = q(`SELECT
      count(*) as total,
      sum(case when kind='fact' then 1 else 0 end) as facts,
      sum(case when kind='episode' then 1 else 0 end) as episodes,
      round(avg(strength), 4) as avg_strength
      FROM memories WHERE namespace='${namespace.replace(/'/g, "''")}';`)[0] || { total: 0, facts: 0, episodes: 0, avg_strength: 0 };

    const memories = q(`SELECT id, kind, subject, relation, object, episode_text, strength, access_count, tags, source, session_id, channel, importance, namespace, checksum, created_at, last_accessed_at
      FROM memories WHERE ${whereSql}
      ORDER BY last_accessed_at DESC, id DESC
      LIMIT ${limit} OFFSET ${offset};`);

    return NextResponse.json({ db: dbPath(), namespace, query: search, limit, offset, stats, memories });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
