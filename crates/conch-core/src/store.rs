use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection, OptionalExtension, Result as SqlResult};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::memory::{AuditEntry, AuditIntegrityResult, CorruptedMemory, Episode, Fact, MemoryKind, MemoryRecord, MemoryStats, TamperedAuditEntry, VerifyResult};

pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn open<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    pub fn open_in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                kind            TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                subject         TEXT,
                relation        TEXT,
                object          TEXT,
                episode_text    TEXT,
                strength        REAL NOT NULL DEFAULT 1.0,
                embedding       BLOB,
                created_at      TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                access_count    INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);",
        )?;
        // Migration: add tags column if missing
        let has_tags: bool = self.conn
            .prepare("SELECT tags FROM memories LIMIT 0")
            .is_ok();
        if !has_tags {
            self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '';"
            )?;
        }
        // Migration: add source tracking columns if missing
        let has_source: bool = self.conn
            .prepare("SELECT source FROM memories LIMIT 0")
            .is_ok();
        if !has_source {
            self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN source TEXT;
                 ALTER TABLE memories ADD COLUMN session_id TEXT;
                 ALTER TABLE memories ADD COLUMN channel TEXT;"
            )?;
        }
        // Migration: add importance column if missing
        let has_importance: bool = self.conn
            .prepare("SELECT importance FROM memories LIMIT 0")
            .is_ok();
        if !has_importance {
            self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5;"
            )?;
        }
        // Migration: add namespace column if missing
        let has_namespace: bool = self.conn
            .prepare("SELECT namespace FROM memories LIMIT 0")
            .is_ok();
        if !has_namespace {
            self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';"
            )?;
        }
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);"
        )?;
        // Migration: add checksum column if missing
        let has_checksum: bool = self.conn
            .prepare("SELECT checksum FROM memories LIMIT 0")
            .is_ok();
        if !has_checksum {
            self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN checksum TEXT;"
            )?;
        }
        // Audit log table
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                action      TEXT NOT NULL,
                memory_id   INTEGER,
                actor       TEXT NOT NULL DEFAULT 'system',
                details_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_log_memory_id ON audit_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log(actor);
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);"
        )?;
        // Migration: add entry_hash column to audit_log if missing
        let has_entry_hash: bool = self.conn
            .prepare("SELECT entry_hash FROM audit_log LIMIT 0")
            .is_ok();
        if !has_entry_hash {
            self.conn.execute_batch(
                "ALTER TABLE audit_log ADD COLUMN entry_hash TEXT;"
            )?;
        }
        Ok(())
    }

    // ── Remember ─────────────────────────────────────────────

    pub fn remember_fact(&self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>) -> SqlResult<i64> {
        self.remember_fact_with_tags(subject, relation, object, embedding, &[])
    }

    pub fn remember_fact_with_tags(&self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>, tags: &[String]) -> SqlResult<i64> {
        self.remember_fact_full(subject, relation, object, embedding, tags, None, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn remember_fact_full(
        &self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        self.remember_fact_ns(subject, relation, object, embedding, tags, source, session_id, channel, "default")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn remember_fact_ns(
        &self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
        namespace: &str,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        let content = format!("{subject} {relation} {object}");
        let checksum = compute_checksum(&content);
        self.conn.execute(
            "INSERT INTO memories (kind, subject, relation, object, embedding, created_at, last_accessed_at, tags, source, session_id, channel, namespace, checksum)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![subject, relation, object, emb_blob, now, tags_str, source, session_id, channel, namespace, checksum],
        )?;
        let id = self.conn.last_insert_rowid();
        self.log_audit("remember", Some(id), "system", Some(&format!("{{\"kind\":\"fact\",\"subject\":{},\"relation\":{},\"object\":{},\"namespace\":{}}}", serde_json::json!(subject), serde_json::json!(relation), serde_json::json!(object), serde_json::json!(namespace))))?;
        Ok(id)
    }

    // ── Upsert ────────────────────────────────────────────────

    /// Upsert a fact: if a fact with the same subject+relation exists in the same namespace,
    /// update its object (and embedding/tags). Otherwise insert a new fact.
    /// Returns `(id, was_updated)`.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_fact(
        &self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<(i64, bool)> {
        self.upsert_fact_ns(subject, relation, object, embedding, tags, source, session_id, channel, "default")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_fact_ns(
        &self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
        namespace: &str,
    ) -> SqlResult<(i64, bool)> {
        // Check for existing fact with same subject+relation in the same namespace
        let existing_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM memories WHERE kind = 'fact' AND subject = ?1 AND relation = ?2 AND namespace = ?3 LIMIT 1",
            params![subject, relation, namespace],
            |row| row.get(0),
        ).optional()?;

        if let Some(id) = existing_id {
            // Update existing fact
            let now = Utc::now().to_rfc3339();
            let emb_blob = embedding.map(embedding_to_blob);
            let tags_str = tags.join(",");
            let content = format!("{subject} {relation} {object}");
            let checksum = compute_checksum(&content);
            self.conn.execute(
                "UPDATE memories SET object = ?1, embedding = COALESCE(?2, embedding), \
                 last_accessed_at = ?3, access_count = access_count + 1, \
                 tags = ?4, source = COALESCE(?5, source), \
                 session_id = COALESCE(?6, session_id), channel = COALESCE(?7, channel), \
                 checksum = ?8 \
                 WHERE id = ?9",
                params![object, emb_blob, now, tags_str, source, session_id, channel, checksum, id],
            )?;
            self.log_audit("update", Some(id), "system", Some(&format!("{{\"kind\":\"fact\",\"subject\":{},\"relation\":{},\"object\":{},\"namespace\":{}}}", serde_json::json!(subject), serde_json::json!(relation), serde_json::json!(object), serde_json::json!(namespace))))?;
            Ok((id, true))
        } else {
            let id = self.remember_fact_ns(subject, relation, object, embedding, tags, source, session_id, channel, namespace)?;
            Ok((id, false))
        }
    }

    pub fn remember_episode(&self, text: &str, embedding: Option<&[f32]>) -> SqlResult<i64> {
        self.remember_episode_with_tags(text, embedding, &[])
    }

    pub fn remember_episode_with_tags(&self, text: &str, embedding: Option<&[f32]>, tags: &[String]) -> SqlResult<i64> {
        self.remember_episode_full(text, embedding, tags, None, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn remember_episode_full(
        &self, text: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        self.remember_episode_ns(text, embedding, tags, source, session_id, channel, "default")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn remember_episode_ns(
        &self, text: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
        namespace: &str,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        let checksum = compute_checksum(text);
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, embedding, created_at, last_accessed_at, tags, source, session_id, channel, namespace, checksum)
             VALUES ('episode', ?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![text, emb_blob, now, tags_str, source, session_id, channel, namespace, checksum],
        )?;
        let id = self.conn.last_insert_rowid();
        self.log_audit("remember", Some(id), "system", Some(&format!("{{\"kind\":\"episode\",\"namespace\":{}}}", serde_json::json!(namespace))))?;
        Ok(id)
    }

    // ── Recall ───────────────────────────────────────────────

    pub fn all_memories_with_text(&self) -> SqlResult<Vec<(MemoryRecord, String)>> {
        self.all_memories_with_text_ns("default")
    }

    pub fn all_memories_with_text_ns(&self, namespace: &str) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE strength > 0.01 AND namespace = ?1",
        )?;
        let rows = stmt.query_map(params![namespace], |row| {
            let mem = row_to_memory(row)?;
            let text = mem.text_for_embedding();
            Ok((mem, text))
        })?;
        rows.collect()
    }

    pub fn all_memories_with_text_filtered_by_tag(&self, tag: &str) -> SqlResult<Vec<(MemoryRecord, String)>> {
        self.all_memories_with_text_filtered_by_tag_ns(tag, "default")
    }

    pub fn all_memories_with_text_filtered_by_tag_ns(&self, tag: &str, namespace: &str) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let pattern = format!("%{}%", tag);
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE strength > 0.01 AND tags LIKE ?1 AND namespace = ?2",
        )?;
        let rows = stmt.query_map(params![pattern, namespace], |row| {
            let mem = row_to_memory(row)?;
            let text = mem.text_for_embedding();
            Ok((mem, text))
        })?;
        // Post-filter for exact tag match (LIKE is approximate)
        let all: Vec<(MemoryRecord, String)> = rows.collect::<SqlResult<Vec<_>>>()?;
        Ok(all.into_iter().filter(|(m, _)| m.tags.iter().any(|t| t == tag)).collect())
    }

    pub fn touch_memory_with_strength(&self, id: i64, strength: f64, now: DateTime<Utc>) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE memories SET last_accessed_at = ?1, access_count = access_count + 1, strength = ?2 WHERE id = ?3",
            params![now.to_rfc3339(), strength.clamp(0.0, 1.0), id],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: i64) -> SqlResult<Option<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], row_to_memory)?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    // ── Decay ────────────────────────────────────────────────

    pub fn decay_all(&self, decay_factor: f64, half_life_hours: f64) -> SqlResult<usize> {
        self.decay_all_ns(decay_factor, half_life_hours, "default")
    }

    pub fn decay_all_ns(&self, decay_factor: f64, half_life_hours: f64, namespace: &str) -> SqlResult<usize> {
        let now = Utc::now();
        let mut stmt = self.conn.prepare(
            "SELECT id, last_accessed_at, strength, importance FROM memories WHERE strength > 0.01 AND namespace = ?1",
        )?;
        let rows: Vec<(i64, String, f64, f64)> = stmt
            .query_map(params![namespace], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get::<_, Option<f64>>(3)?.unwrap_or(0.5))))?
            .collect::<SqlResult<Vec<_>>>()?;

        let mut count = 0;
        for (id, last_accessed, strength, importance) in &rows {
            let last = DateTime::parse_from_rfc3339(last_accessed)
                .unwrap_or_else(|_| now.into())
                .with_timezone(&Utc);
            let hours = (now - last).num_seconds() as f64 / 3600.0;
            // Importance slows decay: effective_half_life = base_half_life * (1 + importance)
            // importance=0 → half_life*1, importance=1 → half_life*2
            let effective_half_life = half_life_hours * (1.0 + importance);
            let new_strength = (strength * decay_factor.powf(hours / effective_half_life)).max(0.0);
            if (new_strength - strength).abs() > 1e-6 {
                self.conn.execute("UPDATE memories SET strength = ?1 WHERE id = ?2", params![new_strength, id])?;
                count += 1;
            }
        }
        if count > 0 {
            self.log_audit("decay", None, "system", Some(&format!("{{\"decayed\":{count},\"namespace\":{}}}", serde_json::json!(namespace))))?;
        }
        Ok(count)
    }

    // ── Forget ───────────────────────────────────────────────

    pub fn forget_by_subject(&self, subject: &str) -> SqlResult<usize> {
        self.forget_by_subject_ns(subject, "default")
    }

    pub fn forget_by_subject_ns(&self, subject: &str, namespace: &str) -> SqlResult<usize> {
        let count = self.conn.execute("DELETE FROM memories WHERE subject = ?1 AND namespace = ?2", params![subject, namespace])?;
        if count > 0 {
            self.log_audit("forget", None, "system", Some(&format!("{{\"by\":\"subject\",\"subject\":{},\"count\":{},\"namespace\":{}}}", serde_json::json!(subject), count, serde_json::json!(namespace))))?;
        }
        Ok(count)
    }

    pub fn forget_by_id(&self, id: &str) -> SqlResult<usize> {
        let changed = self.conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        if changed > 0 {
            self.log_audit("forget", Some(id.parse::<i64>().unwrap_or(0)), "system", Some(&format!("{{\"by\":\"id\",\"id\":{}}}", serde_json::json!(id))))?;
        }
        Ok(changed)
    }

    pub fn forget_older_than(&self, duration: Duration) -> SqlResult<usize> {
        self.forget_older_than_ns(duration, "default")
    }

    pub fn forget_older_than_ns(&self, duration: Duration, namespace: &str) -> SqlResult<usize> {
        let cutoff = (Utc::now() - duration).to_rfc3339();
        let count = self.conn.execute("DELETE FROM memories WHERE created_at < ?1 AND namespace = ?2", params![cutoff, namespace])?;
        if count > 0 {
            self.log_audit("forget", None, "system", Some(&format!("{{\"by\":\"older_than\",\"count\":{},\"namespace\":{}}}", count, serde_json::json!(namespace))))?;
        }
        Ok(count)
    }

    // ── Embed ────────────────────────────────────────────────

    pub fn memories_missing_embeddings(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE embedding IS NULL",
        )?;
        let rows = stmt.query_map([], row_to_memory)?;
        rows.collect()
    }

    pub fn update_embedding(&self, id: i64, embedding: &[f32]) -> SqlResult<()> {
        self.conn.execute("UPDATE memories SET embedding = ?1 WHERE id = ?2", params![embedding_to_blob(embedding), id])?;
        Ok(())
    }

    // ── Export / Import ──────────────────────────────────────

    pub fn all_memories(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories",
        )?;
        let rows = stmt.query_map([], row_to_memory)?;
        rows.collect()
    }

    pub fn all_memories_ns(&self, namespace: &str) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE namespace = ?1",
        )?;
        let rows = stmt.query_map(params![namespace], row_to_memory)?;
        rows.collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_fact(
        &self, subject: &str, relation: &str, object: &str, strength: f64,
        embedding: Option<&[f32]>, created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        self.import_fact_ns(subject, relation, object, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel, "default")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_fact_ns(
        &self, subject: &str, relation: &str, object: &str, strength: f64,
        embedding: Option<&[f32]>, created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
        namespace: &str,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        let content = format!("{subject} {relation} {object}");
        let checksum = compute_checksum(&content);
        self.conn.execute(
            "INSERT INTO memories (kind, subject, relation, object, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel, namespace, checksum)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![subject, relation, object, strength, emb_blob, created_at, last_accessed_at, access_count, tags_str, source, session_id, channel, namespace, checksum],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_episode(
        &self, text: &str, strength: f64, embedding: Option<&[f32]>,
        created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        self.import_episode_ns(text, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel, "default")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_episode_ns(
        &self, text: &str, strength: f64, embedding: Option<&[f32]>,
        created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
        namespace: &str,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        let checksum = compute_checksum(text);
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel, namespace, checksum)
             VALUES ('episode', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![text, strength, emb_blob, created_at, last_accessed_at, access_count, tags_str, source, session_id, channel, namespace, checksum],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update tags on an existing memory.
    pub fn update_tags(&self, id: i64, tags: &[String]) -> SqlResult<()> {
        let tags_str = tags.join(",");
        self.conn.execute(
            "UPDATE memories SET tags = ?1 WHERE id = ?2",
            params![tags_str, id],
        )?;
        Ok(())
    }

    /// Update the importance score of a memory.
    pub fn update_importance(&self, id: i64, importance: f64) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE memories SET importance = ?1 WHERE id = ?2",
            params![importance.clamp(0.0, 1.0), id],
        )?;
        Ok(())
    }

    /// Set strength to zero (archive) for a memory.
    pub fn archive_memory(&self, id: i64) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE memories SET strength = 0.0 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Delete a memory by numeric ID.
    pub fn delete_memory(&self, id: i64) -> SqlResult<()> {
        self.conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    // ── Dedup ─────────────────────────────────────────────────

    /// Fetch all memory IDs with their embeddings for dedup similarity checks.
    pub fn all_embeddings(&self) -> SqlResult<Vec<(i64, Vec<f32>)>> {
        self.all_embeddings_ns("default")
    }

    pub fn all_embeddings_ns(&self, namespace: &str) -> SqlResult<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND namespace = ?1",
        )?;
        let rows = stmt.query_map(params![namespace], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob_to_embedding(&blob)))
        })?;
        rows.collect()
    }

    /// Reinforce an existing memory's strength (clamped to 1.0) and bump access count.
    pub fn reinforce_memory(&self, id: i64, boost: f64) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE memories SET strength = MIN(strength + ?1, 1.0), \
             last_accessed_at = ?2, access_count = access_count + 1 WHERE id = ?3",
            params![boost, now, id],
        )?;
        self.log_audit("reinforce", Some(id), "system", Some(&format!("{{\"boost\":{boost}}}")))
    }

    // ── Graph traversal ────────────────────────────────────────

    /// Find all facts where the given entity appears as subject OR object.
    pub fn facts_involving(&self, entity: &str) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance, namespace, checksum
             FROM memories WHERE kind = 'fact' AND (subject = ?1 OR object = ?1)",
        )?;
        let rows = stmt.query_map(params![entity], row_to_memory)?;
        rows.collect()
    }

    // ── Stats ────────────────────────────────────────────────

    pub fn stats(&self) -> SqlResult<MemoryStats> {
        self.stats_ns("default")
    }

    pub fn stats_ns(&self, namespace: &str) -> SqlResult<MemoryStats> {
        let total_memories: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE namespace = ?1", params![namespace], |r| r.get(0))?;
        let total_facts: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE kind = 'fact' AND namespace = ?1", params![namespace], |r| r.get(0))?;
        let total_episodes: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE kind = 'episode' AND namespace = ?1", params![namespace], |r| r.get(0))?;
        let avg_strength: f64 = self.conn.query_row("SELECT COALESCE(AVG(strength), 0.0) FROM memories WHERE namespace = ?1", params![namespace], |r| r.get(0))?;
        Ok(MemoryStats { total_memories, total_facts, total_episodes, avg_strength })
    }
    // ── Audit Log ─────────────────────────────────────────────

    pub fn log_audit(&self, action: &str, memory_id: Option<i64>, actor: &str, details_json: Option<&str>) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();

        // Fetch the previous entry's hash for chaining (or use "genesis" for first)
        let prev_hash: String = self.conn.query_row(
            "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get::<_, Option<String>>(0),
        ).optional()?.flatten().unwrap_or_else(|| "genesis".to_string());

        // Compute entry_hash = hex(sha256(prev_hash | timestamp | action | memory_id | actor | details))
        let mid_str = memory_id.unwrap_or(0).to_string();
        let details_str = details_json.unwrap_or("");
        let chain_input = format!("{prev_hash}|{now}|{action}|{mid_str}|{actor}|{details_str}");
        let entry_hash = compute_audit_hash(&chain_input);

        self.conn.execute(
            "INSERT INTO audit_log (timestamp, action, memory_id, actor, details_json, entry_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![now, action, memory_id, actor, details_json, entry_hash],
        )?;
        Ok(())
    }

    /// Verify the tamper-evident audit log chain by recomputing each entry's expected hash.
    pub fn verify_audit_integrity(&self) -> SqlResult<AuditIntegrityResult> {
        // Walk all entries in insertion order
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, action, memory_id, actor, details_json, entry_hash
             FROM audit_log ORDER BY id ASC",
        )?;
        let rows: Vec<(i64, String, String, Option<i64>, String, Option<String>, Option<String>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        let mut total = 0usize;
        let mut valid = 0usize;
        let mut tampered = Vec::new();
        let mut prev_hash = "genesis".to_string();

        for (id, timestamp, action, memory_id, actor, details_json, stored_hash) in &rows {
            total += 1;
            let mid_str = memory_id.unwrap_or(0).to_string();
            let details_str = details_json.as_deref().unwrap_or("");
            let chain_input = format!("{prev_hash}|{timestamp}|{action}|{mid_str}|{actor}|{details_str}");
            let expected_hash = compute_audit_hash(&chain_input);

            match stored_hash {
                Some(actual) if *actual == expected_hash => {
                    valid += 1;
                    prev_hash = actual.clone();
                }
                Some(actual) => {
                    tampered.push(TamperedAuditEntry {
                        id: *id,
                        expected: expected_hash.clone(),
                        actual: actual.clone(),
                    });
                    // Still advance prev_hash using the stored value so chain continues
                    prev_hash = actual.clone();
                }
                None => {
                    // Entry without hash — treat as tampered
                    tampered.push(TamperedAuditEntry {
                        id: *id,
                        expected: expected_hash.clone(),
                        actual: String::new(),
                    });
                    prev_hash = expected_hash;
                }
            }
        }

        Ok(AuditIntegrityResult { total, valid, tampered })
    }

    pub fn get_audit_log(&self, limit: usize, memory_id: Option<i64>, actor: Option<&str>) -> SqlResult<Vec<AuditEntry>> {
        let (sql, param_values) = build_audit_query(limit, memory_id, actor);
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(param_values.iter()), |row| {
            Ok(AuditEntry {
                id: row.get(0)?,
                timestamp: parse_datetime(&row.get::<_, String>(1)?),
                action: row.get(2)?,
                memory_id: row.get(3)?,
                actor: row.get(4)?,
                details_json: row.get(5)?,
                entry_hash: row.get(6)?,
            })
        })?;
        rows.collect()
    }

    // ── Verify ──────────────────────────────────────────────

    pub fn verify_integrity(&self) -> SqlResult<VerifyResult> {
        self.verify_integrity_ns("default")
    }

    pub fn verify_integrity_ns(&self, namespace: &str) -> SqlResult<VerifyResult> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text, checksum
             FROM memories WHERE namespace = ?1",
        )?;
        let rows: Vec<(i64, String, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)> = stmt
            .query_map(params![namespace], |row| {
                Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?,
                    row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?,
                ))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        let mut total_checked = 0;
        let mut valid = 0;
        let mut corrupted = Vec::new();
        let mut missing_checksum = 0;

        for (id, kind, subject, relation, object, episode_text, stored_checksum) in rows {
            total_checked += 1;
            let content = match kind.as_str() {
                "fact" => format!("{} {} {}",
                    subject.as_deref().unwrap_or(""),
                    relation.as_deref().unwrap_or(""),
                    object.as_deref().unwrap_or("")),
                _ => episode_text.unwrap_or_default(),
            };
            let actual_checksum = compute_checksum(&content);
            match stored_checksum {
                Some(expected) => {
                    if expected == actual_checksum {
                        valid += 1;
                    } else {
                        corrupted.push(CorruptedMemory {
                            id,
                            expected,
                            actual: actual_checksum,
                        });
                    }
                }
                None => {
                    missing_checksum += 1;
                }
            }
        }

        Ok(VerifyResult { total_checked, valid, corrupted, missing_checksum })
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn compute_audit_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn compute_checksum(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Public helper to compute checksum for content (used in lib.rs)
pub fn content_checksum(content: &str) -> String {
    compute_checksum(content)
}

fn build_audit_query(limit: usize, memory_id: Option<i64>, actor: Option<&str>) -> (String, Vec<String>) {
    let mut conditions = Vec::new();
    let mut param_values: Vec<String> = Vec::new();

    if let Some(mid) = memory_id {
        param_values.push(mid.to_string());
        conditions.push(format!("memory_id = ?{}", param_values.len()));
    }
    if let Some(a) = actor {
        param_values.push(a.to_string());
        conditions.push(format!("actor = ?{}", param_values.len()));
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    param_values.push(limit.to_string());
    let sql = format!(
        "SELECT id, timestamp, action, memory_id, actor, details_json, entry_hash FROM audit_log{} ORDER BY id DESC LIMIT ?{}",
        where_clause, param_values.len()
    );
    (sql, param_values)
}

fn embedding_to_blob(emb: &[f32]) -> Vec<u8> {
    emb.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_fact_with_tags_stores_and_retrieves() {
        let store = MemoryStore::open_in_memory().unwrap();
        let tags = vec!["preference".to_string(), "technical".to_string()];
        let id = store.remember_fact_with_tags("Rust", "is", "fast", None, &tags).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.tags, vec!["preference", "technical"]);
    }

    #[test]
    fn remember_episode_with_tags_stores_and_retrieves() {
        let store = MemoryStore::open_in_memory().unwrap();
        let tags = vec!["decision".to_string()];
        let id = store.remember_episode_with_tags("chose Rust over Go", None, &tags).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.tags, vec!["decision"]);
    }

    #[test]
    fn remember_without_tags_returns_empty_vec() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("Jared", "likes", "pizza", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.tags.is_empty());
    }

    #[test]
    fn update_tags_modifies_existing_memory() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("Jared", "uses", "Linux", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.tags.is_empty());

        store.update_tags(id, &["technical".to_string(), "preference".to_string()]).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.tags, vec!["technical", "preference"]);
    }

    #[test]
    fn tags_survive_export_import_roundtrip() {
        let store = MemoryStore::open_in_memory().unwrap();
        let tags = vec!["project".to_string(), "meta".to_string()];
        store.remember_fact_with_tags("Conch", "is_a", "memory system", None, &tags).unwrap();

        let all = store.all_memories().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].tags, vec!["project", "meta"]);

        // Import into a new store
        let store2 = MemoryStore::open_in_memory().unwrap();
        let mem = &all[0];
        let created = mem.created_at.to_rfc3339();
        let accessed = mem.last_accessed_at.to_rfc3339();
        if let MemoryKind::Fact(f) = &mem.kind {
            store2.import_fact(&f.subject, &f.relation, &f.object, mem.strength, None, &created, &accessed, mem.access_count, &mem.tags, None, None, None).unwrap();
        }
        let imported = store2.all_memories().unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].tags, vec!["project", "meta"]);
    }

    #[test]
    fn tags_appear_in_all_memories_with_text() {
        let store = MemoryStore::open_in_memory().unwrap();
        let tags = vec!["person".to_string()];
        store.remember_fact_with_tags("Jared", "is", "developer", None, &tags).unwrap();
        let results = store.all_memories_with_text().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.tags, vec!["person"]);
    }

    #[test]
    fn tag_filtered_recall_returns_only_matching() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_with_tags("A", "is", "tagged", None, &["technical".to_string()]).unwrap();
        store.remember_fact("B", "is", "untagged", None).unwrap();
        store.remember_fact_with_tags("C", "is", "also-tagged", None, &["technical".to_string(), "project".to_string()]).unwrap();

        let results = store.all_memories_with_text_filtered_by_tag("technical").unwrap();
        assert_eq!(results.len(), 2);
        for (m, _) in &results {
            assert!(m.tags.contains(&"technical".to_string()));
        }
    }

    #[test]
    fn tag_filter_exact_match_not_substring() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_with_tags("A", "is", "tech", None, &["technical".to_string()]).unwrap();
        store.remember_fact_with_tags("B", "is", "meta", None, &["meta".to_string()]).unwrap();

        // Filtering by "tech" should NOT match "technical"
        let results = store.all_memories_with_text_filtered_by_tag("tech").unwrap();
        assert_eq!(results.len(), 0);

        // Filtering by "meta" should match exactly
        let results = store.all_memories_with_text_filtered_by_tag("meta").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn tag_filter_returns_empty_when_no_match() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_with_tags("A", "is", "B", None, &["technical".to_string()]).unwrap();

        let results = store.all_memories_with_text_filtered_by_tag("nonexistent").unwrap();
        assert!(results.is_empty());
    }

    // ── Graph traversal tests ─────────────────────────────────

    #[test]
    fn facts_involving_finds_subject_and_object() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Alice", "knows", "Bob", None).unwrap();
        store.remember_fact("Bob", "works_at", "Acme", None).unwrap();
        store.remember_fact("Charlie", "knows", "Alice", None).unwrap();

        // Alice appears as subject in fact 1, and object in fact 3
        let results = store.facts_involving("Alice").unwrap();
        assert_eq!(results.len(), 2, "Alice should appear in 2 facts");

        // Bob appears as object in fact 1, and subject in fact 2
        let results = store.facts_involving("Bob").unwrap();
        assert_eq!(results.len(), 2, "Bob should appear in 2 facts");

        // Acme only appears as object in fact 2
        let results = store.facts_involving("Acme").unwrap();
        assert_eq!(results.len(), 1, "Acme should appear in 1 fact");
    }

    #[test]
    fn facts_involving_ignores_episodes() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Alice", "knows", "Bob", None).unwrap();
        store.remember_episode("Alice had a meeting", None).unwrap();

        let results = store.facts_involving("Alice").unwrap();
        assert_eq!(results.len(), 1, "should only return facts, not episodes");
    }

    #[test]
    fn facts_involving_returns_empty_for_unknown() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Alice", "knows", "Bob", None).unwrap();

        let results = store.facts_involving("Unknown").unwrap();
        assert!(results.is_empty());
    }

    // ── Source tracking tests ────────────────────────────────

    #[test]
    fn remember_fact_full_stores_source_fields() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact_full(
            "Jared", "uses", "conch", None, &[],
            Some("cli"), Some("session-123"), Some("#general"),
        ).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.source.as_deref(), Some("cli"));
        assert_eq!(mem.session_id.as_deref(), Some("session-123"));
        assert_eq!(mem.channel.as_deref(), Some("#general"));
    }

    #[test]
    fn remember_episode_full_stores_source_fields() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_episode_full(
            "had a meeting", None, &[],
            Some("discord"), Some("sess-abc"), Some("#dev"),
        ).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.source.as_deref(), Some("discord"));
        assert_eq!(mem.session_id.as_deref(), Some("sess-abc"));
        assert_eq!(mem.channel.as_deref(), Some("#dev"));
    }

    #[test]
    fn remember_without_source_returns_none() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("X", "Y", "Z", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.source.is_none());
        assert!(mem.session_id.is_none());
        assert!(mem.channel.is_none());
    }

    #[test]
    fn source_fields_survive_export_import_roundtrip() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_full(
            "Conch", "source_test", "value", None, &["meta".to_string()],
            Some("cron"), Some("daily-job"), None,
        ).unwrap();

        let all = store.all_memories().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].source.as_deref(), Some("cron"));
        assert_eq!(all[0].session_id.as_deref(), Some("daily-job"));
        assert!(all[0].channel.is_none());

        // Import into a new store
        let store2 = MemoryStore::open_in_memory().unwrap();
        let mem = &all[0];
        let created = mem.created_at.to_rfc3339();
        let accessed = mem.last_accessed_at.to_rfc3339();
        if let MemoryKind::Fact(f) = &mem.kind {
            store2.import_fact(
                &f.subject, &f.relation, &f.object, mem.strength, None,
                &created, &accessed, mem.access_count, &mem.tags,
                mem.source.as_deref(), mem.session_id.as_deref(), mem.channel.as_deref(),
            ).unwrap();
        }
        let imported = store2.all_memories().unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].source.as_deref(), Some("cron"));
        assert_eq!(imported[0].session_id.as_deref(), Some("daily-job"));
        assert!(imported[0].channel.is_none());
    }

    #[test]
    fn source_fields_appear_in_all_memories_with_text() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_full(
            "test", "source", "recall", None, &[],
            Some("mcp"), None, Some("#test-channel"),
        ).unwrap();
        let results = store.all_memories_with_text().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.source.as_deref(), Some("mcp"));
        assert!(results[0].0.session_id.is_none());
        assert_eq!(results[0].0.channel.as_deref(), Some("#test-channel"));
    }

    // ── Upsert tests ────────────────────────────────────────

    #[test]
    fn upsert_fact_inserts_when_no_match() {
        let store = MemoryStore::open_in_memory().unwrap();
        let (id, was_updated) = store.upsert_fact("Jared", "favorite_color", "blue", None, &[], None, None, None).unwrap();
        assert!(!was_updated);
        let mem = store.get_memory(id).unwrap().unwrap();
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.object, "blue");
        } else { panic!("expected fact"); }
    }

    #[test]
    fn upsert_fact_updates_existing_object() {
        let store = MemoryStore::open_in_memory().unwrap();
        let (id1, _) = store.upsert_fact("Jared", "favorite_color", "blue", None, &[], None, None, None).unwrap();
        let (id2, was_updated) = store.upsert_fact("Jared", "favorite_color", "green", None, &[], None, None, None).unwrap();
        assert!(was_updated);
        assert_eq!(id1, id2, "should update the same row");

        let mem = store.get_memory(id2).unwrap().unwrap();
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.object, "green", "object should be updated to green");
        } else { panic!("expected fact"); }

        // Should only have 1 memory total
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }

    #[test]
    fn upsert_fact_bumps_access_count() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.upsert_fact("Jared", "age", "30", None, &[], None, None, None).unwrap();
        let (id, _) = store.upsert_fact("Jared", "age", "31", None, &[], None, None, None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.access_count, 1, "access count should be bumped on upsert");
    }

    #[test]
    fn upsert_fact_different_relation_creates_new() {
        let store = MemoryStore::open_in_memory().unwrap();
        let (id1, _) = store.upsert_fact("Jared", "likes", "Rust", None, &[], None, None, None).unwrap();
        let (id2, was_updated) = store.upsert_fact("Jared", "uses", "Rust", None, &[], None, None, None).unwrap();
        assert!(!was_updated, "different relation should insert new fact");
        assert_ne!(id1, id2);
        assert_eq!(store.stats().unwrap().total_memories, 2);
    }

    #[test]
    fn upsert_fact_preserves_tags() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.upsert_fact("Jared", "color", "blue", None, &["preference".to_string()], None, None, None).unwrap();
        let (id, _) = store.upsert_fact("Jared", "color", "green", None, &["preference".to_string(), "updated".to_string()], None, None, None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.tags, vec!["preference", "updated"]);
    }

    // ════════════════════════════════════════════════════════════
    // Security Feature Tests
    // ════════════════════════════════════════════════════════════

    // ── Audit log tests ─────────────────────────────────────

    #[test]
    fn audit_log_records_remember_fact() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Jared", "likes", "Rust", None).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(!log.is_empty());
        let remember_entries: Vec<_> = log.iter().filter(|e| e.action == "remember").collect();
        assert_eq!(remember_entries.len(), 1);
        assert!(remember_entries[0].memory_id.is_some());
        assert_eq!(remember_entries[0].actor, "system");
    }

    #[test]
    fn audit_log_records_remember_episode() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_episode("had coffee", None).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        let remember_entries: Vec<_> = log.iter().filter(|e| e.action == "remember").collect();
        assert_eq!(remember_entries.len(), 1);
    }

    #[test]
    fn audit_log_records_forget_by_id() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("X", "Y", "Z", None).unwrap();
        store.forget_by_id(&id.to_string()).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "forget"));
    }

    #[test]
    fn audit_log_records_forget_by_subject() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Jared", "likes", "Rust", None).unwrap();
        store.forget_by_subject("Jared").unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "forget"));
    }

    #[test]
    fn audit_log_records_upsert_update() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.upsert_fact("Jared", "color", "blue", None, &[], None, None, None).unwrap();
        store.upsert_fact("Jared", "color", "green", None, &[], None, None, None).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "update"));
    }

    #[test]
    fn audit_log_records_reinforce() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "B", "C", None).unwrap();
        store.reinforce_memory(id, 0.1).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "reinforce"));
    }

    #[test]
    fn audit_log_records_decay() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();

        // Make memory old so decay actually happens
        let old_time = (Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
        store.conn().execute("UPDATE memories SET last_accessed_at = ?1", params![old_time]).unwrap();

        store.decay_all(0.5, 24.0).unwrap();
        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "decay"), "should log decay action");
    }

    #[test]
    fn audit_log_filter_by_memory_id() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id1 = store.remember_fact("A", "B", "C", None).unwrap();
        store.remember_fact("D", "E", "F", None).unwrap();

        let log = store.get_audit_log(10, Some(id1), None).unwrap();
        for entry in &log {
            assert_eq!(entry.memory_id, Some(id1));
        }
    }

    #[test]
    fn audit_log_filter_by_actor() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();
        store.log_audit("custom_action", None, "agent-x", None).unwrap();

        let log = store.get_audit_log(10, None, Some("agent-x")).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].actor, "agent-x");
    }

    #[test]
    fn audit_log_limit_works() {
        let store = MemoryStore::open_in_memory().unwrap();
        for i in 0..10 {
            store.remember_fact(&format!("S{i}"), "R", "O", None).unwrap();
        }
        let log = store.get_audit_log(3, None, None).unwrap();
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn audit_log_has_details_json() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Jared", "likes", "Rust", None).unwrap();

        let log = store.get_audit_log(1, None, None).unwrap();
        assert!(log[0].details_json.is_some());
        let details = log[0].details_json.as_ref().unwrap();
        assert!(details.contains("\"kind\":\"fact\""));
        assert!(details.contains("\"subject\":\"Jared\""));
    }

    // ── Checksum tests ──────────────────────────────────────

    #[test]
    fn fact_gets_checksum_on_insert() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("Jared", "likes", "Rust", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.checksum.is_some());
        assert!(!mem.checksum.as_ref().unwrap().is_empty());
    }

    #[test]
    fn episode_gets_checksum_on_insert() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_episode("had coffee", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.checksum.is_some());
    }

    #[test]
    fn checksum_is_consistent_for_same_content() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id1 = store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns1").unwrap();
        let id2 = store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns2").unwrap();
        let mem1 = store.get_memory(id1).unwrap().unwrap();
        let mem2 = store.get_memory(id2).unwrap().unwrap();
        assert_eq!(mem1.checksum, mem2.checksum, "same content should produce same checksum");
    }

    #[test]
    fn checksum_differs_for_different_content() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id1 = store.remember_fact("A", "B", "C", None).unwrap();
        let id2 = store.remember_fact("X", "Y", "Z", None).unwrap();
        let mem1 = store.get_memory(id1).unwrap().unwrap();
        let mem2 = store.get_memory(id2).unwrap().unwrap();
        assert_ne!(mem1.checksum, mem2.checksum);
    }

    #[test]
    fn upsert_updates_checksum() {
        let store = MemoryStore::open_in_memory().unwrap();
        let (id, _) = store.upsert_fact("Jared", "color", "blue", None, &[], None, None, None).unwrap();
        let mem1 = store.get_memory(id).unwrap().unwrap();
        let checksum1 = mem1.checksum.clone().unwrap();

        store.upsert_fact("Jared", "color", "green", None, &[], None, None, None).unwrap();
        let mem2 = store.get_memory(id).unwrap().unwrap();
        let checksum2 = mem2.checksum.clone().unwrap();
        assert_ne!(checksum1, checksum2, "upsert with new object should change checksum");
    }

    // ── Verify integrity tests ──────────────────────────────

    #[test]
    fn verify_all_valid() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();
        store.remember_episode("hello", None).unwrap();

        let result = store.verify_integrity().unwrap();
        assert_eq!(result.total_checked, 2);
        assert_eq!(result.valid, 2);
        assert!(result.corrupted.is_empty());
        assert_eq!(result.missing_checksum, 0);
    }

    #[test]
    fn verify_detects_corrupted_fact() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("Jared", "likes", "Rust", None).unwrap();

        // Corrupt the data
        store.conn().execute("UPDATE memories SET object = 'Go' WHERE id = ?1", params![id]).unwrap();

        let result = store.verify_integrity().unwrap();
        assert_eq!(result.corrupted.len(), 1);
        assert_eq!(result.corrupted[0].id, id);
    }

    #[test]
    fn verify_detects_corrupted_episode() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_episode("original text", None).unwrap();

        // Corrupt the data
        store.conn().execute("UPDATE memories SET episode_text = 'modified text' WHERE id = ?1", params![id]).unwrap();

        let result = store.verify_integrity().unwrap();
        assert_eq!(result.corrupted.len(), 1);
        assert_eq!(result.corrupted[0].id, id);
    }

    #[test]
    fn verify_reports_missing_checksums() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();

        // Null out checksum
        store.conn().execute("UPDATE memories SET checksum = NULL", []).unwrap();

        let result = store.verify_integrity().unwrap();
        assert_eq!(result.missing_checksum, 1);
        assert_eq!(result.valid, 0);
        assert!(result.corrupted.is_empty());
    }

    #[test]
    fn verify_namespace_scoped() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns-a").unwrap();
        store.remember_fact_ns("X", "Y", "Z", None, &[], None, None, None, "ns-b").unwrap();

        // Corrupt ns-b data
        store.conn().execute("UPDATE memories SET object = 'CORRUPTED' WHERE namespace = 'ns-b'", []).unwrap();

        let result_a = store.verify_integrity_ns("ns-a").unwrap();
        let result_b = store.verify_integrity_ns("ns-b").unwrap();

        assert_eq!(result_a.valid, 1);
        assert!(result_a.corrupted.is_empty(), "ns-a should be clean");
        assert_eq!(result_b.corrupted.len(), 1, "ns-b should have corruption");
    }

    // ── Namespace isolation tests ───────────────────────────

    #[test]
    fn namespace_default_on_remember() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "B", "C", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.namespace, "default");
    }

    #[test]
    fn namespace_set_on_remember_ns() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "project-x").unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.namespace, "project-x");
    }

    #[test]
    fn namespace_isolates_all_memories_with_text() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("X", "Y", "Z", None, &[], None, None, None, "ns2").unwrap();

        let ns1 = store.all_memories_with_text_ns("ns1").unwrap();
        let ns2 = store.all_memories_with_text_ns("ns2").unwrap();
        assert_eq!(ns1.len(), 1);
        assert_eq!(ns2.len(), 1);
        assert_ne!(ns1[0].0.id, ns2[0].0.id);
    }

    #[test]
    fn namespace_isolates_stats() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("D", "E", "F", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("X", "Y", "Z", None, &[], None, None, None, "ns2").unwrap();

        let stats1 = store.stats_ns("ns1").unwrap();
        let stats2 = store.stats_ns("ns2").unwrap();
        assert_eq!(stats1.total_memories, 2);
        assert_eq!(stats2.total_memories, 1);
    }

    #[test]
    fn namespace_isolates_upsert() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.upsert_fact_ns("Jared", "color", "blue", None, &[], None, None, None, "ns1").unwrap();
        store.upsert_fact_ns("Jared", "color", "red", None, &[], None, None, None, "ns2").unwrap();

        // Upsert in ns1 should only update ns1
        let (_, updated) = store.upsert_fact_ns("Jared", "color", "green", None, &[], None, None, None, "ns1").unwrap();
        assert!(updated);

        let ns2_mems = store.all_memories_ns("ns2").unwrap();
        if let MemoryKind::Fact(f) = &ns2_mems[0].kind {
            assert_eq!(f.object, "red", "ns2 should be unaffected");
        } else { panic!("expected fact"); }
    }

    #[test]
    fn namespace_isolates_forget_by_subject() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("Jared", "likes", "A", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("Jared", "likes", "B", None, &[], None, None, None, "ns2").unwrap();

        let deleted = store.forget_by_subject_ns("Jared", "ns1").unwrap();
        assert_eq!(deleted, 1);

        // ns2 should be untouched
        assert_eq!(store.stats_ns("ns2").unwrap().total_memories, 1);
        assert_eq!(store.stats_ns("ns1").unwrap().total_memories, 0);
    }

    #[test]
    fn namespace_isolates_decay() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("X", "Y", "Z", None, &[], None, None, None, "ns2").unwrap();

        // Make all memories old
        let old_time = (Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
        store.conn().execute("UPDATE memories SET last_accessed_at = ?1", params![old_time]).unwrap();

        let decayed = store.decay_all_ns(0.5, 24.0, "ns1").unwrap();
        assert_eq!(decayed, 1, "should only decay ns1 memories");

        // ns2 should be untouched (strength still 1.0)
        let ns2_mems = store.all_memories_ns("ns2").unwrap();
        assert!((ns2_mems[0].strength - 1.0).abs() < 0.01, "ns2 should not be decayed");
    }

    #[test]
    fn namespace_isolates_embeddings() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", Some(&[1.0, 0.0]), &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("X", "Y", "Z", Some(&[0.0, 1.0]), &[], None, None, None, "ns2").unwrap();

        let emb1 = store.all_embeddings_ns("ns1").unwrap();
        let emb2 = store.all_embeddings_ns("ns2").unwrap();
        assert_eq!(emb1.len(), 1);
        assert_eq!(emb2.len(), 1);
    }

    #[test]
    fn namespace_episode_isolation() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_episode_ns("event in ns1", None, &[], None, None, None, "ns1").unwrap();
        store.remember_episode_ns("event in ns2", None, &[], None, None, None, "ns2").unwrap();

        let ns1 = store.all_memories_with_text_ns("ns1").unwrap();
        let ns2 = store.all_memories_with_text_ns("ns2").unwrap();
        assert_eq!(ns1.len(), 1);
        assert_eq!(ns2.len(), 1);
        assert!(ns1[0].1.contains("ns1"));
        assert!(ns2[0].1.contains("ns2"));
    }

    #[test]
    fn namespace_import_export_scoped() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("A", "B", "C", None, &[], None, None, None, "ns1").unwrap();
        store.remember_fact_ns("X", "Y", "Z", None, &[], None, None, None, "ns2").unwrap();

        let ns1_mems = store.all_memories_ns("ns1").unwrap();
        assert_eq!(ns1_mems.len(), 1);
        assert_eq!(ns1_mems[0].namespace, "ns1");
    }

    // ════════════════════════════════════════════════════════════
    // Audit Log Tamper Evidence Tests (QRT-63 / QRT-70)
    // ════════════════════════════════════════════════════════════

    /// 1. Genesis entry works: first audit entry has a hash derived from "genesis".
    #[test]
    fn audit_hash_genesis_entry_works() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();

        let log = store.get_audit_log(1, None, None).unwrap();
        assert_eq!(log.len(), 1);
        let entry = &log[0];
        assert!(entry.entry_hash.is_some(), "first entry should have an entry_hash");
        let hash = entry.entry_hash.as_ref().unwrap();
        assert!(!hash.is_empty(), "entry_hash must not be empty");
        // Verify the hash is 64 hex characters (sha256)
        assert_eq!(hash.len(), 64, "SHA-256 hash should be 64 hex chars");
    }

    /// 2. Chain builds correctly: sequential entries reference each other's hashes.
    #[test]
    fn audit_hash_chain_builds_correctly() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();
        store.remember_fact("D", "E", "F", None).unwrap();

        let result = store.verify_audit_integrity().unwrap();
        assert_eq!(result.total, 2, "should have 2 audit entries");
        assert_eq!(result.valid, 2, "all entries should be valid");
        assert!(result.tampered.is_empty(), "no tampering should be detected");
    }

    /// 3. Multi-entry chain valid: store several entries, verify all pass.
    #[test]
    fn audit_hash_multi_entry_chain_valid() {
        let store = MemoryStore::open_in_memory().unwrap();
        for i in 0..5 {
            store.remember_fact(&format!("S{i}"), "R", "O", None).unwrap();
        }
        let result = store.verify_audit_integrity().unwrap();
        assert!(result.total >= 5, "should have at least 5 entries");
        assert_eq!(result.valid, result.total, "all entries should be valid");
        assert!(result.tampered.is_empty());
    }

    /// 4. Tampering detected: modifying an audit entry breaks the chain.
    #[test]
    fn audit_hash_tampering_detected() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();
        store.remember_fact("D", "E", "F", None).unwrap();
        store.remember_fact("G", "H", "I", None).unwrap();

        // Tamper with the first audit entry's action
        store.conn().execute(
            "UPDATE audit_log SET action = 'tampered_action' WHERE id = (SELECT MIN(id) FROM audit_log)",
            [],
        ).unwrap();

        let result = store.verify_audit_integrity().unwrap();
        assert!(!result.tampered.is_empty(), "tampering should be detected");
    }

    /// 5. Export includes entry_hash: get_audit_log returns entry_hash field.
    #[test]
    fn audit_log_export_includes_entry_hash() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", None).unwrap();
        store.remember_episode("test episode", None).unwrap();

        let log = store.get_audit_log(10, None, None).unwrap();
        assert!(!log.is_empty());
        for entry in &log {
            assert!(
                entry.entry_hash.is_some(),
                "all returned audit entries should have entry_hash"
            );
        }

        // Verify the hash can be serialized to JSON
        let json = serde_json::to_string_pretty(&log).unwrap();
        assert!(json.contains("entry_hash"), "JSON export should contain entry_hash");
    }
}

fn row_to_memory(row: &rusqlite::Row) -> SqlResult<MemoryRecord> {
    let kind_str: String = row.get(1)?;
    let kind = match kind_str.as_str() {
        "fact" => MemoryKind::Fact(Fact {
            subject: row.get(2)?,
            relation: row.get(3)?,
            object: row.get(4)?,
        }),
        _ => MemoryKind::Episode(Episode {
            text: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
        }),
    };
    let embedding: Option<Vec<u8>> = row.get(7)?;
    let tags_str: String = row.get::<_, Option<String>>(11)?.unwrap_or_default();
    let tags: Vec<String> = if tags_str.is_empty() {
        vec![]
    } else {
        tags_str.split(',').map(|s| s.trim().to_string()).collect()
    };
    Ok(MemoryRecord {
        id: row.get(0)?,
        kind,
        strength: row.get(6)?,
        embedding: embedding.map(|b| blob_to_embedding(&b)),
        created_at: parse_datetime(&row.get::<_, String>(8)?),
        last_accessed_at: parse_datetime(&row.get::<_, String>(9)?),
        access_count: row.get(10)?,
        tags,
        source: row.get::<_, Option<String>>(12)?,
        session_id: row.get::<_, Option<String>>(13)?,
        channel: row.get::<_, Option<String>>(14)?,
        importance: row.get::<_, Option<f64>>(15)?.unwrap_or(0.5),
        namespace: row.get::<_, Option<String>>(16)?.unwrap_or_else(|| "default".to_string()),
        checksum: row.get::<_, Option<String>>(17)?,
    })
}
