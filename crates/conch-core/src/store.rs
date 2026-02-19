use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection, OptionalExtension, Result as SqlResult};
use std::path::Path;

use crate::memory::{Episode, Fact, MemoryKind, MemoryRecord, MemoryStats};

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
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        self.conn.execute(
            "INSERT INTO memories (kind, subject, relation, object, embedding, created_at, last_accessed_at, tags, source, session_id, channel)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7, ?8, ?9)",
            params![subject, relation, object, emb_blob, now, tags_str, source, session_id, channel],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Upsert ────────────────────────────────────────────────

    /// Upsert a fact: if a fact with the same subject+relation exists, update
    /// its object (and embedding/tags). Otherwise insert a new fact.
    /// Returns `(id, was_updated)`.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_fact(
        &self, subject: &str, relation: &str, object: &str, embedding: Option<&[f32]>,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<(i64, bool)> {
        // Check for existing fact with same subject+relation
        let existing_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM memories WHERE kind = 'fact' AND subject = ?1 AND relation = ?2 LIMIT 1",
            params![subject, relation],
            |row| row.get(0),
        ).optional()?;

        if let Some(id) = existing_id {
            // Update existing fact
            let now = Utc::now().to_rfc3339();
            let emb_blob = embedding.map(embedding_to_blob);
            let tags_str = tags.join(",");
            self.conn.execute(
                "UPDATE memories SET object = ?1, embedding = COALESCE(?2, embedding), \
                 last_accessed_at = ?3, access_count = access_count + 1, \
                 tags = ?4, source = COALESCE(?5, source), \
                 session_id = COALESCE(?6, session_id), channel = COALESCE(?7, channel) \
                 WHERE id = ?8",
                params![object, emb_blob, now, tags_str, source, session_id, channel, id],
            )?;
            Ok((id, true))
        } else {
            let id = self.remember_fact_full(subject, relation, object, embedding, tags, source, session_id, channel)?;
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
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, embedding, created_at, last_accessed_at, tags, source, session_id, channel)
             VALUES ('episode', ?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7)",
            params![text, emb_blob, now, tags_str, source, session_id, channel],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Recall ───────────────────────────────────────────────

    pub fn all_memories_with_text(&self) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance
             FROM memories WHERE strength > 0.01",
        )?;
        let rows = stmt.query_map([], |row| {
            let mem = row_to_memory(row)?;
            let text = mem.text_for_embedding();
            Ok((mem, text))
        })?;
        rows.collect()
    }

    pub fn all_memories_with_text_filtered_by_tag(&self, tag: &str) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let pattern = format!("%{}%", tag);
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance
             FROM memories WHERE strength > 0.01 AND tags LIKE ?1",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
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
                    tags, source, session_id, channel, importance
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
        let now = Utc::now();
        let mut stmt = self.conn.prepare(
            "SELECT id, last_accessed_at, strength, importance FROM memories WHERE strength > 0.01",
        )?;
        let rows: Vec<(i64, String, f64, f64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get::<_, Option<f64>>(3)?.unwrap_or(0.5))))?
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
        Ok(count)
    }

    // ── Forget ───────────────────────────────────────────────

    pub fn forget_by_subject(&self, subject: &str) -> SqlResult<usize> {
        self.conn.execute("DELETE FROM memories WHERE subject = ?1", params![subject])
    }

    pub fn forget_by_id(&self, id: &str) -> SqlResult<usize> {
        let changed = self.conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(changed)
    }

    pub fn forget_older_than(&self, duration: Duration) -> SqlResult<usize> {
        let cutoff = (Utc::now() - duration).to_rfc3339();
        self.conn.execute("DELETE FROM memories WHERE created_at < ?1", params![cutoff])
    }

    // ── Embed ────────────────────────────────────────────────

    pub fn memories_missing_embeddings(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count,
                    tags, source, session_id, channel, importance
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
                    tags, source, session_id, channel, importance
             FROM memories",
        )?;
        let rows = stmt.query_map([], row_to_memory)?;
        rows.collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_fact(
        &self, subject: &str, relation: &str, object: &str, strength: f64,
        embedding: Option<&[f32]>, created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        self.conn.execute(
            "INSERT INTO memories (kind, subject, relation, object, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![subject, relation, object, strength, emb_blob, created_at, last_accessed_at, access_count, tags_str, source, session_id, channel],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_episode(
        &self, text: &str, strength: f64, embedding: Option<&[f32]>,
        created_at: &str, last_accessed_at: &str, access_count: i64,
        tags: &[String], source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        let tags_str = tags.join(",");
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, strength, embedding, created_at, last_accessed_at, access_count, tags, source, session_id, channel)
             VALUES ('episode', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![text, strength, emb_blob, created_at, last_accessed_at, access_count, tags_str, source, session_id, channel],
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
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
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
        Ok(())
    }

    // ── Stats ────────────────────────────────────────────────

    pub fn stats(&self) -> SqlResult<MemoryStats> {
        let total_memories: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
        let total_facts: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE kind = 'fact'", [], |r| r.get(0))?;
        let total_episodes: i64 = self.conn.query_row("SELECT COUNT(*) FROM memories WHERE kind = 'episode'", [], |r| r.get(0))?;
        let avg_strength: f64 = self.conn.query_row("SELECT COALESCE(AVG(strength), 0.0) FROM memories", [], |r| r.get(0))?;
        Ok(MemoryStats { total_memories, total_facts, total_episodes, avg_strength })
    }
}

// ── Helpers ──────────────────────────────────────────────────

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
    })
}
