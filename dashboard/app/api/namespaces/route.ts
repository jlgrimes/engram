import { NextResponse } from "next/server";
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

export async function GET() {
  try {
    const rows = q("SELECT namespace, count(*) as count FROM memories GROUP BY namespace ORDER BY count DESC, namespace ASC;");
    return NextResponse.json({ db: dbPath(), namespaces: rows });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
