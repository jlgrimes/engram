import { NextResponse } from "next/server";
import { execFileSync } from "node:child_process";
import path from "node:path";

type RetryTrendBucket = {
  bucket_start: string;
  retrying: number;
  recovered: number;
  failed: number;
  events: number;
};

type SystemStats = {
  db: string;
  window_hours: number;
  memories: number;
  namespaces: number;
  duplicate_rows: number;
  write_retry: {
    events_total: number;
    retrying: number;
    recovered: number;
    failed: number;
    severity: "healthy" | "watch" | "critical";
    trend: RetryTrendBucket[];
  };
  generated_at?: string;
  cache_age_ms?: number;
  cached?: boolean;
};

let lastStats: SystemStats | null = null;
let lastStatsAt = 0;
const CACHE_TTL_MS = 30_000;

function dbPath() {
  const home = process.env.HOME || ".";
  return process.env.CONCH_DB || path.join(home, ".conch", "default.db");
}

function fetchStats(db: string): SystemStats {
  const py = [
    "import sqlite3, json, sys, datetime",
    "db = sys.argv[1]",
    "con = sqlite3.connect(db)",
    "con.row_factory = sqlite3.Row",
    "now = datetime.datetime.utcnow()",
    "window_hours = 24",
    "bucket_hours = 2",
    "bucket_count = int(window_hours / bucket_hours)",
    "since = (now - datetime.timedelta(hours=window_hours)).isoformat() + 'Z'",
    "counts = dict(con.execute(\"SELECT count(*) as memories, count(distinct namespace) as namespaces FROM memories\").fetchone())",
    "dupes = dict(con.execute(\"SELECT coalesce(sum(c-1),0) as duplicate_rows FROM (SELECT count(*) as c FROM memories WHERE checksum is not null and checksum != '' GROUP BY namespace, checksum HAVING c > 1)\").fetchone())",
    "retry_rows = [dict(r) for r in con.execute(\"SELECT timestamp, details_json FROM audit_log WHERE action='write_retry' AND timestamp >= ? ORDER BY id DESC\", (since,)).fetchall()]",
    "retrying = 0",
    "recovered = 0",
    "failed = 0",
    "buckets = []",
    "for i in range(bucket_count):",
    "  start = now - datetime.timedelta(hours=window_hours - (i * bucket_hours))",
    "  end = start + datetime.timedelta(hours=bucket_hours)",
    "  buckets.append({'start': start, 'end': end, 'retrying': 0, 'recovered': 0, 'failed': 0, 'events': 0})",
    "for r in retry_rows:",
    "  d = r.get('details_json')",
    "  if not d: continue",
    "  try:",
    "    j = json.loads(d)",
    "  except Exception:",
    "    continue",
    "  s = j.get('status')",
    "  ts_raw = r.get('timestamp')",
    "  ts = None",
    "  if ts_raw:",
    "    try:",
    "      ts = datetime.datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00')).replace(tzinfo=None)",
    "    except Exception:",
    "      ts = None",
    "  if s == 'retrying': retrying += 1",
    "  elif s == 'recovered': recovered += 1",
    "  elif s == 'failed': failed += 1",
    "  if ts is None: continue",
    "  for b in buckets:",
    "    if b['start'] <= ts < b['end']:",
    "      b['events'] += 1",
    "      if s == 'retrying': b['retrying'] += 1",
    "      elif s == 'recovered': b['recovered'] += 1",
    "      elif s == 'failed': b['failed'] += 1",
    "      break",
    "severity = 'healthy'",
    "if failed > 0:",
    "  severity = 'critical'",
    "elif retrying > 2:",
    "  severity = 'watch'",
    "trend = [{'bucket_start': b['start'].isoformat() + 'Z', 'retrying': b['retrying'], 'recovered': b['recovered'], 'failed': b['failed'], 'events': b['events']} for b in buckets]",
    "print(json.dumps({",
    "  'db': db,",
    "  'window_hours': window_hours,",
    "  'memories': counts.get('memories', 0),",
    "  'namespaces': counts.get('namespaces', 0),",
    "  'duplicate_rows': dupes.get('duplicate_rows', 0),",
    "  'write_retry': {",
    "    'events_total': len(retry_rows),",
    "    'retrying': retrying,",
    "    'recovered': recovered,",
    "    'failed': failed,",
    "    'severity': severity,",
    "    'trend': trend",
    "  }",
    "}))",
  ].join("; ");

  const out = execFileSync("python3", ["-c", py, db], { encoding: "utf8", timeout: 7000 });
  return JSON.parse(out);
}

export async function GET() {
  try {
    const now = Date.now();
    if (lastStats && now - lastStatsAt <= CACHE_TTL_MS) {
      return NextResponse.json({
        ...lastStats,
        cached: true,
        cache_age_ms: now - lastStatsAt,
      });
    }

    const stats = fetchStats(dbPath());
    stats.generated_at = new Date().toISOString();
    stats.cached = false;
    stats.cache_age_ms = 0;
    lastStats = stats;
    lastStatsAt = now;
    return NextResponse.json(stats);
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
