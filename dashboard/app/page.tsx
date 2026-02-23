"use client";

import { useEffect, useMemo, useState } from "react";

type Memory = {
  id: number;
  kind: "fact" | "episode";
  subject?: string;
  relation?: string;
  object?: string;
  episode_text?: string;
  strength: number;
  access_count: number;
  tags?: string;
  source?: string;
  namespace: string;
  importance: number;
  created_at: string;
  last_accessed_at: string;
};

type RecallResult = {
  score: number;
  memory: {
    id: number;
    kind: { Fact?: { subject: string; relation: string; object: string }; Episode?: { text: string } };
    strength: number;
    access_count: number;
    tags: string[];
    source?: string | null;
    namespace: string;
    created_at: string;
    last_accessed_at: string;
  };
  explain: {
    rrf_score: number;
    rrf_rank: number;
    rrf_rank_percentile?: number;
    bm25_rank?: number | null;
    bm25_score?: number | null;
    vector_rank?: number | null;
    vector_similarity?: number | null;
    modality_agreement?: boolean;
    matched_modalities?: number;
    decayed_strength: number;
    recency_boost: number;
    access_weight: number;
    base_score: number;
    spread_boost: number;
    temporal_boost: number;
    score_margin_to_next?: number | null;
    final_score: number;
  };
};

type RecallCompare = {
  hasComparison: boolean;
  message?: string;
  latest?: { generated_at?: string; result_count?: number; filePath?: string };
  previous?: { generated_at?: string; result_count?: number; filePath?: string };
  metrics?: {
    compared_count: number;
    entered_count: number;
    moved_count: number;
    avg_abs_rank_delta?: number;
    avg_abs_score_delta?: number;
    severity_score?: number;
  };
  rows?: Array<{
    memory_id: number;
    latest_rank: number;
    previous_rank: number | null;
    rank_delta: number | null;
    latest_final_score: number | null;
    previous_final_score: number | null;
    final_score_delta: number | null;
    latest_rrf_rank?: number | null;
    previous_rrf_rank?: number | null;
    latest_bm25_rank?: number | null;
    previous_bm25_rank?: number | null;
    latest_vector_rank?: number | null;
    previous_vector_rank?: number | null;
  }>;
};

export default function Home() {
  const [q, setQ] = useState("");
  const [namespace, setNamespace] = useState("default");
  const [kindFilter, setKindFilter] = useState<"all" | "fact" | "episode">("all");
  const [namespaceOptions, setNamespaceOptions] = useState<Array<{ namespace: string; count: number }>>([]);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Memory | null>(null);
  const [recallLoading, setRecallLoading] = useState(false);
  const [recallError, setRecallError] = useState("");
  const [recallResults, setRecallResults] = useState<RecallResult[]>([]);
  const [snapshotPath, setSnapshotPath] = useState("");
  const [snapshotStatus, setSnapshotStatus] = useState("");
  const [exportStatus, setExportStatus] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");
  const [compareData, setCompareData] = useState<RecallCompare | null>(null);
  const [compareTo, setCompareTo] = useState<"previous" | "nth">("previous");
  const [compareNth, setCompareNth] = useState(2);
  const [systemStats, setSystemStats] = useState<any>(null);
  const [actionSummary, setActionSummary] = useState("");
  const [actionStatus, setActionStatus] = useState("");
  const [rollupStatus, setRollupStatus] = useState("");
  const pageSize = 120;

  async function load() {
    setLoading(true);
    setErr("");
    try {
      const url = `/api/memories?namespace=${encodeURIComponent(namespace)}&q=${encodeURIComponent(q)}&limit=${pageSize}&offset=${offset}`;
      const res = await fetch(url);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "request failed");
      setData(j);
      if (!selected || !j.memories?.some((m: Memory) => m.id === selected.id)) {
        setSelected(j.memories?.[0] || null);
      }
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runRecallExplain() {
    if (!q.trim()) {
      setRecallError("Enter a search query first.");
      setRecallResults([]);
      return;
    }
    setRecallLoading(true);
    setRecallError("");
    try {
      const res = await fetch(`/api/recall?namespace=${encodeURIComponent(namespace)}&q=${encodeURIComponent(q)}&limit=8`);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "recall request failed");
      setRecallResults(j.results || []);
    } catch (e: any) {
      setRecallError(e.message);
      setRecallResults([]);
    } finally {
      setRecallLoading(false);
    }
  }

  function recallTraceLabel(r: RecallResult) {
    const explain = r.explain;
    const agreement = explain.modality_agreement ? "dual" : "single";
    const confidence = explain.score_margin_to_next != null && explain.score_margin_to_next > 0.05 ? "stable" : "tight";
    return `${agreement}/${confidence}/rrf#${explain.rrf_rank}`;
  }

  function retrySeverityColor(severity?: string) {
    if (severity === "critical") return "#fca5a5";
    if (severity === "watch") return "#fde68a";
    return "#86efac";
  }

  function retryTrendBars() {
    const trend = systemStats?.write_retry?.trend || [];
    if (!trend.length) return "-";
    const values = trend.map((b: any) => (b.failed || 0) * 3 + (b.retrying || 0) * 2 + (b.recovered || 0));
    const max = Math.max(1, ...values);
    const glyphs = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    return values.map((v: number) => glyphs[Math.min(glyphs.length - 1, Math.floor((v / max) * (glyphs.length - 1)))]).join("");
  }

  async function exportRecallSnapshot() {
    if (!q.trim()) {
      setSnapshotStatus("Enter a search query first.");
      return;
    }
    setSnapshotStatus("Saving snapshot…");
    setSnapshotPath("");
    try {
      const res = await fetch("/api/recall-snapshot", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ namespace, q, limit: 8 }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "snapshot request failed");
      setSnapshotPath(j.filePath || "");
      setSnapshotStatus(`Saved ${j.result_count ?? 0} results.`);
      await compareWithPrevious();
    } catch (e: any) {
      setSnapshotStatus(`Snapshot failed: ${e.message}`);
    }
  }

  async function exportRecallExplainability() {
    if (!q.trim()) {
      setExportStatus("Enter a search query first.");
      return;
    }
    setExportStatus("Exporting explainability…");
    try {
      const res = await fetch("/api/recall-export", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ namespace, q, limit: 12 }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "recall export failed");
      setExportStatus(`Exported ${j.result_count ?? 0} rows to ${j.csvPath}`);
    } catch (e: any) {
      setExportStatus(`Export failed: ${e.message}`);
    }
  }

  async function compareWithPrevious() {
    if (!q.trim()) {
      setCompareError("Enter a search query first.");
      setCompareData(null);
      return;
    }
    setCompareLoading(true);
    setCompareError("");
    try {
      const params = new URLSearchParams({ namespace, q });
      params.set("compareTo", compareTo);
      if (compareTo === "nth") params.set("nth", String(compareNth));
      const res = await fetch(`/api/recall-compare?${params.toString()}`);
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "compare request failed");
      setCompareData(j);
    } catch (e: any) {
      setCompareError(e.message);
      setCompareData(null);
    } finally {
      setCompareLoading(false);
    }
  }

  async function loadNamespaces() {
    try {
      const res = await fetch("/api/namespaces");
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "namespace fetch failed");
      setNamespaceOptions(j.namespaces || []);
    } catch {
      // non-fatal
    }
  }

  async function loadSystemStats() {
    try {
      const res = await fetch("/api/system-stats");
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "system stats fetch failed");
      setSystemStats(j);
    } catch {
      // non-fatal
    }
  }

  async function logActionEvent() {
    if (!actionSummary.trim()) {
      setActionStatus("Add a summary first.");
      return;
    }

    setActionStatus("Logging action event…");
    try {
      const res = await fetch("/api/action-log", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          actor: "claw",
          project: "conch",
          repo: "jlgrimes/conch",
          action_type: "manual-log",
          outcome: "success",
          summary: actionSummary,
        }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "action log failed");
      setActionStatus(`Logged at ${j.event?.timestamp || "now"}`);
      setActionSummary("");
      setOffset(0);
      await load();
    } catch (e: any) {
      setActionStatus(`Log failed: ${e.message}`);
    }
  }

  async function writeDailyRollup() {
    setRollupStatus("Writing daily rollup…");
    try {
      const shipped: string[] = ["Structured action event logging endpoint added", "Time-window aware recall filtering added"];
      const inProgress: string[] = ["Automating action journaling hooks across build/commit/push workflows"];
      const blockers: string[] = [];
      const next: string[] = ["Attach artifact provenance (commit/file/issue) consistently", "Add deterministic today/yesterday rollup query shortcuts"];

      const res = await fetch("/api/daily-rollup", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          project: "Conch",
          shipped,
          in_progress: inProgress,
          blockers,
          next,
        }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "daily rollup failed");
      setRollupStatus(`Saved rollup for ${j.rollup?.date || "today"}`);
      setOffset(0);
      await load();
    } catch (e: any) {
      setRollupStatus(`Rollup failed: ${e.message}`);
    }
  }

  useEffect(() => {
    loadNamespaces();
    loadSystemStats();
  }, []);

  useEffect(() => {
    load();
  }, [offset]);

  const grouped = useMemo(() => {
    const memories: Memory[] = data?.memories || [];
    return {
      facts: memories.filter((m) => m.kind === "fact"),
      episodes: memories.filter((m) => m.kind === "episode"),
    };
  }, [data]);

  const rows: Memory[] = useMemo(() => {
    const source = kindFilter === "all" ? [...grouped.facts, ...grouped.episodes] : grouped[`${kindFilter}s` as "facts" | "episodes"];
    return [...source].sort((a, b) => b.last_accessed_at.localeCompare(a.last_accessed_at));
  }, [grouped, kindFilter]);

  return (
    <main style={{ maxWidth: 1250, margin: "0 auto", padding: 16, fontFamily: "Inter, ui-sans-serif, system-ui", color: "#e5e7eb" }}>
      <h1 style={{ fontSize: 28, marginBottom: 6 }}>🐚 Conch Memory Browser</h1>
      <p style={{ color: "#9ca3af", marginBottom: 16 }}>Local, self-hosted memory inspection + search (Plane-style memory workspace).</p>
      {!!systemStats && (
        <div style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, marginBottom: 14, fontSize: 12, color: "#d1d5db" }}>
          <div>
            ops: memories={systemStats.memories ?? 0} • namespaces={systemStats.namespaces ?? 0} • duplicate_rows={systemStats.duplicate_rows ?? 0} • write_retry(24h): events={systemStats.write_retry?.events_total ?? 0}, retrying={systemStats.write_retry?.retrying ?? 0}, recovered={systemStats.write_retry?.recovered ?? 0}, failed={systemStats.write_retry?.failed ?? 0} • severity=<span style={{ color: retrySeverityColor(systemStats.write_retry?.severity), fontWeight: 700 }}>{systemStats.write_retry?.severity ?? "healthy"}</span>
          </div>
          <div style={{ marginTop: 4, color: "#93c5fd" }}>
            retry trend (2h buckets): <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>{retryTrendBars()}</span>
          </div>
        </div>
      )}

      <section style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, marginBottom: 14 }}>
        <h2 style={{ marginTop: 0, marginBottom: 8, fontSize: 16 }}>Action memory pipeline (prototype)</h2>
        <p style={{ marginTop: 0, color: "#9ca3af", fontSize: 12 }}>Log timestamped action events + write deterministic daily rollups into Conch memory.</p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <input
            value={actionSummary}
            onChange={(e) => setActionSummary(e.target.value)}
            placeholder="e.g., Shipped recall compare caching endpoint"
            style={{ padding: 8, flex: 1, minWidth: 300, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}
          />
          <button onClick={logActionEvent} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Log action event</button>
          <button onClick={writeDailyRollup} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Write daily rollup</button>
        </div>
        {!!actionStatus && <div style={{ marginTop: 8, fontSize: 12, color: actionStatus.startsWith("Log failed") ? "#fca5a5" : "#86efac" }}>{actionStatus}</div>}
        {!!rollupStatus && <div style={{ marginTop: 4, fontSize: 12, color: rollupStatus.startsWith("Rollup failed") ? "#fca5a5" : "#86efac" }}>{rollupStatus}</div>}
      </section>

      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <select value={namespace} onChange={(e) => setNamespace(e.target.value)} style={{ padding: 8, width: 180, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>
          {[namespace, ...namespaceOptions.map((n) => n.namespace)]
            .filter((v, i, arr) => v && arr.indexOf(v) === i)
            .map((ns) => (
              <option key={ns} value={ns}>{ns}</option>
            ))}
        </select>
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as any)} style={{ padding: 8, width: 120, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>
          <option value="all">all</option>
          <option value="fact">facts</option>
          <option value="episode">episodes</option>
        </select>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search text/tags" style={{ padding: 8, flex: 1, minWidth: 220, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }} />
        <button onClick={() => { setOffset(0); load(); }} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Search</button>
        <button onClick={runRecallExplain} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Explain ranking</button>
        <button onClick={exportRecallSnapshot} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Export snapshot</button>
        <button onClick={exportRecallExplainability} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Export explainability CSV</button>
        <select value={compareTo} onChange={(e) => setCompareTo(e.target.value as any)} style={{ padding: 8, width: 150, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>
          <option value="previous">baseline: previous</option>
          <option value="nth">baseline: nth newest</option>
        </select>
        {compareTo === "nth" && (
          <input
            type="number"
            min={2}
            value={compareNth}
            onChange={(e) => setCompareNth(Math.max(2, Number(e.target.value || 2)))}
            style={{ padding: 8, width: 110, background: "#0b1220", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}
          />
        )}
        <button onClick={compareWithPrevious} style={{ padding: "8px 12px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Compare baseline</button>
      </div>

      {err && <div style={{ color: "crimson", marginBottom: 12 }}>Error: {err}</div>}
      {loading && <div>Loading…</div>}

      {!!recallError && <div style={{ color: "crimson", marginBottom: 12 }}>Recall: {recallError}</div>}
      {!!snapshotStatus && <div style={{ color: snapshotStatus.startsWith("Snapshot failed") ? "crimson" : "#444", marginBottom: 6 }}>{snapshotStatus}</div>}
      {!!snapshotPath && <div style={{ color: "#d1d5db", marginBottom: 12, fontSize: 12 }}>Snapshot file: <code>{snapshotPath}</code></div>}
      {!!compareError && <div style={{ color: "crimson", marginBottom: 12 }}>Compare: {compareError}</div>}
      {!!exportStatus && <div style={{ color: exportStatus.startsWith("Export failed") ? "crimson" : "#d1d5db", marginBottom: 8 }}>{exportStatus}</div>}
      {compareLoading && <div style={{ marginBottom: 10 }}>Comparing with previous snapshot…</div>}
      {recallLoading && <div style={{ marginBottom: 10 }}>Computing recall explainability…</div>}
      {recallResults.length > 0 && (
        <section style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, marginBottom: 12 }}>
          <h2 style={{ marginTop: 0, marginBottom: 8 }}>Recall ranking explainability</h2>
          <div style={{ display: "grid", gap: 8 }}>
            {recallResults.map((r) => {
              const fact = (r.memory.kind as any).Fact;
              const episode = (r.memory.kind as any).Episode;
              const label = fact ? `${fact.subject} ${fact.relation} ${fact.object}` : episode?.text || "(unknown)";
              return (
                <article key={r.memory.id} style={{ border: "1px solid #374151", borderRadius: 8, padding: 8, background: "#1f2937" }}>
                  <div style={{ fontSize: 12, color: "#9ca3af" }}>
                    id={r.memory.id} • final={r.explain.final_score.toFixed(3)} • base={r.explain.base_score.toFixed(3)} • margin→next={r.explain.score_margin_to_next?.toFixed(3) ?? "-"} • agree={r.explain.modality_agreement ? "yes" : "no"} • modes={r.explain.matched_modalities ?? (r.explain.modality_agreement ? 2 : 1)} • rrf#{r.explain.rrf_rank} (p{(((r.explain.rrf_rank_percentile ?? 1) * 100).toFixed(1))}) • bm25#{r.explain.bm25_rank ?? "-"} ({r.explain.bm25_score?.toFixed(3) ?? "-"}) • vec#{r.explain.vector_rank ?? "-"} ({r.explain.vector_similarity?.toFixed(3) ?? "-"})
                  </div>
                  <div style={{ marginTop: 4 }}><b>{label}</b></div>
                  <div style={{ marginTop: 4, fontSize: 12, color: "#d1d5db" }}>
                    decay={r.explain.decayed_strength.toFixed(3)} • recency={r.explain.recency_boost.toFixed(3)} • access={r.explain.access_weight.toFixed(3)} • spread={r.explain.spread_boost.toFixed(3)} • temporal={r.explain.temporal_boost.toFixed(3)}
                  </div>
                  <div style={{ marginTop: 2, fontSize: 12, color: "#93c5fd" }}>trace={recallTraceLabel(r)}</div>
                </article>
              );
            })}
          </div>
        </section>
      )}

      {!!compareData && (
        <section style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, marginBottom: 12 }}>
          <h2 style={{ marginTop: 0, marginBottom: 8 }}>Snapshot drift vs previous</h2>
          {!compareData.hasComparison && (
            <div style={{ color: "#9ca3af" }}>{compareData.message || "Need at least 2 snapshots."}</div>
          )}
          {compareData.hasComparison && (
            <>
              <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 8 }}>
                latest={compareData.latest?.generated_at || "-"} • previous={compareData.previous?.generated_at || "-"} • compared={compareData.metrics?.compared_count ?? 0} • moved={compareData.metrics?.moved_count ?? 0} • entered={compareData.metrics?.entered_count ?? 0} • avg|Δrank|={compareData.metrics?.avg_abs_rank_delta ?? "-"} • avg|Δscore|={compareData.metrics?.avg_abs_score_delta ?? "-"} • severity={compareData.metrics?.severity_score ?? "-"}
              </div>
              <div style={{ display: "grid", gap: 6 }}>
                {(compareData.rows || []).slice(0, 12).map((row) => (
                  <article key={row.memory_id} style={{ border: "1px solid #374151", borderRadius: 8, padding: 8, background: "#1f2937", fontSize: 12 }}>
                    id={row.memory_id} • rank {row.previous_rank ?? "new"} → {row.latest_rank} ({row.rank_delta == null ? "new" : row.rank_delta >= 0 ? `+${row.rank_delta}` : row.rank_delta}) • final Δ {row.final_score_delta == null ? "-" : row.final_score_delta.toFixed(3)} • rrf {row.previous_rrf_rank ?? "-"}→{row.latest_rrf_rank ?? "-"} • bm25 {row.previous_bm25_rank ?? "-"}→{row.latest_bm25_rank ?? "-"} • vec {row.previous_vector_rank ?? "-"}→{row.latest_vector_rank ?? "-"}
                  </article>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      {data && (
        <>
          <div style={{ display: "flex", gap: 12, fontSize: 13, color: "#d1d5db", marginBottom: 12, flexWrap: "wrap" }}>
            <span><b>Total</b> {data.stats.total}</span>
            <span><b>Facts</b> {data.stats.facts}</span>
            <span><b>Episodes</b> {data.stats.episodes}</span>
            <span><b>Avg strength</b> {data.stats.avg_strength}</span>
            <span><b>DB</b> {data.db}</span>
            <span><b>Offset</b> {offset}</span>
            <button disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - pageSize))} style={{ padding: "6px 10px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Prev</button>
            <button disabled={loading || (data.memories?.length || 0) < pageSize} onClick={() => setOffset(offset + pageSize)} style={{ padding: "6px 10px", background: "#111827", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8 }}>Next</button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
            <section style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, maxHeight: "70vh", overflow: "auto" }}>
              <h2 style={{ marginTop: 0 }}>Memories ({rows.length})</h2>
              <div style={{ display: "grid", gap: 8 }}>
                {rows.map((m) => (
                  <article
                    key={m.id}
                    onClick={() => setSelected(m)}
                    style={{ border: selected?.id === m.id ? "2px solid #60a5fa" : "1px solid #374151", borderRadius: 8, padding: 10, cursor: "pointer", background: selected?.id === m.id ? "#1e293b" : "#0b1220", color: "#e5e7eb" }}
                  >
                    <div style={{ fontSize: 12, color: "#9ca3af" }}>{m.kind.toUpperCase()} • id={m.id}</div>
                    <div style={{ marginTop: 4 }}>
                      {m.kind === "fact" ? <><b>{m.subject}</b> {m.relation} <b>{m.object}</b></> : m.episode_text}
                    </div>
                    <small style={{ color: "#9ca3af" }}>strength={m.strength.toFixed(3)} • access={m.access_count} • importance={m.importance.toFixed(2)}</small>
                  </article>
                ))}
              </div>
            </section>

            <section style={{ border: "1px solid #374151", borderRadius: 8, padding: 10, maxHeight: "70vh", overflow: "auto" }}>
              <h2 style={{ marginTop: 0 }}>Details</h2>
              {!selected && <div style={{ color: "#9ca3af" }}>Select a memory to inspect.</div>}
              {selected && (
                <>
                  <div style={{ marginBottom: 8 }}>
                    <b>{selected.kind.toUpperCase()}</b> #{selected.id}
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, background: "#111827", border: "1px solid #374151", borderRadius: 8, padding: 10 }}>
{JSON.stringify(selected, null, 2)}
                  </pre>
                </>
              )}
            </section>
          </div>
        </>
      )}
    </main>
  );
}
