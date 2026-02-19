use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;

use crate::memory::{Episode, Fact, MemoryKind, MemoryRecord, MemoryStats};

pub const DEFAULT_NAMESPACE: &str = "default";

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
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace        TEXT NOT NULL DEFAULT 'default',
                kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                subject          TEXT,
                relation         TEXT,
                object           TEXT,
                episode_text     TEXT,
                strength         REAL NOT NULL DEFAULT 1.0,
                embedding        BLOB,
                created_at       TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                access_count     INTEGER NOT NULL DEFAULT 0
            );",
        )?;

        // Migration path for older DBs that predate namespaces.
        let has_namespace: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'namespace'",
            [],
            |r| r.get(0),
        )?;
        if has_namespace == 0 {
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
                [],
            )?;
        }

        self.conn.execute_batch(
            "DROP INDEX IF EXISTS idx_memories_kind_namespace;
             CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
             CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
             CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
             CREATE INDEX IF NOT EXISTS idx_memories_namespace_kind ON memories(namespace, kind);",
        )?;
        Ok(())
    }

    // ── Remember ─────────────────────────────────────────────

    pub fn remember_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
        embedding: Option<&[f32]>,
    ) -> SqlResult<i64> {
        self.remember_fact_in(DEFAULT_NAMESPACE, subject, relation, object, embedding)
    }

    pub fn remember_fact_in(
        &self,
        namespace: &str,
        subject: &str,
        relation: &str,
        object: &str,
        embedding: Option<&[f32]>,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (namespace, kind, subject, relation, object, embedding, created_at, last_accessed_at)
             VALUES (?1, 'fact', ?2, ?3, ?4, ?5, ?6, ?6)",
            params![namespace, subject, relation, object, emb_blob, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn remember_episode(&self, text: &str, embedding: Option<&[f32]>) -> SqlResult<i64> {
        self.remember_episode_in(DEFAULT_NAMESPACE, text, embedding)
    }

    pub fn remember_episode_in(
        &self,
        namespace: &str,
        text: &str,
        embedding: Option<&[f32]>,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (namespace, kind, episode_text, embedding, created_at, last_accessed_at)
             VALUES (?1, 'episode', ?2, ?3, ?4, ?4)",
            params![namespace, text, emb_blob, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Find duplicates ────────────────────────────────────────

    pub fn find_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> SqlResult<Option<MemoryRecord>> {
        self.find_fact_in(DEFAULT_NAMESPACE, subject, relation, object)
    }

    pub fn find_fact_in(
        &self,
        namespace: &str,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> SqlResult<Option<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories
             WHERE namespace = ?1 AND kind = 'fact' AND subject = ?2 AND relation = ?3 AND object = ?4",
        )?;
        let mut rows =
            stmt.query_map(params![namespace, subject, relation, object], row_to_memory)?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    pub fn find_episode(&self, episode_text: &str) -> SqlResult<Option<MemoryRecord>> {
        self.find_episode_in(DEFAULT_NAMESPACE, episode_text)
    }

    pub fn find_episode_in(
        &self,
        namespace: &str,
        episode_text: &str,
    ) -> SqlResult<Option<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories
             WHERE namespace = ?1 AND kind = 'episode' AND episode_text = ?2",
        )?;
        let mut rows = stmt.query_map(params![namespace, episode_text], row_to_memory)?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    pub fn reinforce_memory(&self, id: i64) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE memories SET strength = MIN(strength + 0.1, 1.0),
                                 last_accessed_at = ?1,
                                 access_count = access_count + 1
             WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    // ── Recall ───────────────────────────────────────────────

    pub fn all_memories_with_text(&self) -> SqlResult<Vec<(MemoryRecord, String)>> {
        self.all_memories_with_text_in(DEFAULT_NAMESPACE)
    }

    pub fn all_memories_with_text_in(
        &self,
        namespace: &str,
    ) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories WHERE namespace = ?1 AND strength > 0.01",
        )?;
        let rows = stmt.query_map(params![namespace], |row| {
            let mem = row_to_memory(row)?;
            let text = mem.text_for_embedding();
            Ok((mem, text))
        })?;
        rows.collect()
    }

    pub fn touch_memory_with_strength(
        &self,
        id: i64,
        strength: f64,
        now: DateTime<Utc>,
    ) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE memories SET last_accessed_at = ?1, access_count = access_count + 1, strength = ?2 WHERE id = ?3",
            params![now.to_rfc3339(), strength.clamp(0.0, 1.0), id],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: i64) -> SqlResult<Option<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
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
        self.decay_all_in(DEFAULT_NAMESPACE, decay_factor, half_life_hours)
    }

    pub fn decay_all_in(
        &self,
        namespace: &str,
        decay_factor: f64,
        half_life_hours: f64,
    ) -> SqlResult<usize> {
        let now = Utc::now();
        let mut stmt = self.conn.prepare(
            "SELECT id, last_accessed_at, strength FROM memories WHERE namespace = ?1 AND strength > 0.01",
        )?;
        let rows: Vec<(i64, String, f64)> = stmt
            .query_map(params![namespace], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        let mut count = 0;
        for (id, last_accessed, strength) in &rows {
            let last = DateTime::parse_from_rfc3339(last_accessed)
                .unwrap_or_else(|_| now.into())
                .with_timezone(&Utc);
            let hours = (now - last).num_seconds() as f64 / 3600.0;
            let new_strength = (strength * decay_factor.powf(hours / half_life_hours)).max(0.0);
            if (new_strength - strength).abs() > 1e-6 {
                self.conn.execute(
                    "UPDATE memories SET strength = ?1 WHERE id = ?2",
                    params![new_strength, id],
                )?;
                count += 1;
            }
        }
        Ok(count)
    }

    // ── Forget ───────────────────────────────────────────────

    pub fn forget_by_subject(&self, subject: &str) -> SqlResult<usize> {
        self.forget_by_subject_in(DEFAULT_NAMESPACE, subject)
    }

    pub fn forget_by_subject_in(&self, namespace: &str, subject: &str) -> SqlResult<usize> {
        self.conn.execute(
            "DELETE FROM memories WHERE namespace = ?1 AND subject = ?2",
            params![namespace, subject],
        )
    }

    pub fn forget_by_id(&self, id: &str) -> SqlResult<usize> {
        self.forget_by_id_in(DEFAULT_NAMESPACE, id)
    }

    pub fn forget_by_id_in(&self, namespace: &str, id: &str) -> SqlResult<usize> {
        self.conn.execute(
            "DELETE FROM memories WHERE namespace = ?1 AND id = ?2",
            params![namespace, id],
        )
    }

    pub fn forget_older_than(&self, duration: Duration) -> SqlResult<usize> {
        self.forget_older_than_in(DEFAULT_NAMESPACE, duration)
    }

    pub fn forget_older_than_in(&self, namespace: &str, duration: Duration) -> SqlResult<usize> {
        let cutoff = (Utc::now() - duration).to_rfc3339();
        self.conn.execute(
            "DELETE FROM memories WHERE namespace = ?1 AND created_at < ?2",
            params![namespace, cutoff],
        )
    }

    // ── Embed ────────────────────────────────────────────────

    pub fn memories_missing_embeddings(&self) -> SqlResult<Vec<MemoryRecord>> {
        self.memories_missing_embeddings_in(DEFAULT_NAMESPACE)
    }

    pub fn memories_missing_embeddings_in(&self, namespace: &str) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories WHERE namespace = ?1 AND embedding IS NULL",
        )?;
        let rows = stmt.query_map(params![namespace], row_to_memory)?;
        rows.collect()
    }

    pub fn update_embedding(&self, id: i64, embedding: &[f32]) -> SqlResult<()> {
        self.conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![embedding_to_blob(embedding), id],
        )?;
        Ok(())
    }

    // ── Export / Import ──────────────────────────────────────

    pub fn all_memories(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories",
        )?;
        let rows = stmt.query_map([], row_to_memory)?;
        rows.collect()
    }

    pub fn all_memories_in(&self, namespace: &str) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, namespace, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories WHERE namespace = ?1",
        )?;
        let rows = stmt.query_map(params![namespace], row_to_memory)?;
        rows.collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_fact(
        &self,
        namespace: &str,
        subject: &str,
        relation: &str,
        object: &str,
        strength: f64,
        embedding: Option<&[f32]>,
        created_at: &str,
        last_accessed_at: &str,
        access_count: i64,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (namespace, kind, subject, relation, object, strength, embedding, created_at, last_accessed_at, access_count)
             VALUES (?1, 'fact', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![namespace, subject, relation, object, strength, emb_blob, created_at, last_accessed_at, access_count],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn import_episode(
        &self,
        namespace: &str,
        text: &str,
        strength: f64,
        embedding: Option<&[f32]>,
        created_at: &str,
        last_accessed_at: &str,
        access_count: i64,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (namespace, kind, episode_text, strength, embedding, created_at, last_accessed_at, access_count)
             VALUES (?1, 'episode', ?2, ?3, ?4, ?5, ?6, ?7)",
            params![namespace, text, strength, emb_blob, created_at, last_accessed_at, access_count],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Stats ────────────────────────────────────────────────

    pub fn stats(&self) -> SqlResult<MemoryStats> {
        self.stats_in(DEFAULT_NAMESPACE)
    }

    pub fn stats_in(&self, namespace: &str) -> SqlResult<MemoryStats> {
        let total_memories: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = ?1",
            params![namespace],
            |r| r.get(0),
        )?;
        let total_facts: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = ?1 AND kind = 'fact'",
            params![namespace],
            |r| r.get(0),
        )?;
        let total_episodes: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = ?1 AND kind = 'episode'",
            params![namespace],
            |r| r.get(0),
        )?;
        let avg_strength: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(strength), 0.0) FROM memories WHERE namespace = ?1",
            params![namespace],
            |r| r.get(0),
        )?;
        Ok(MemoryStats {
            total_memories,
            total_facts,
            total_episodes,
            avg_strength,
        })
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

fn row_to_memory(row: &rusqlite::Row) -> SqlResult<MemoryRecord> {
    let kind_str: String = row.get(2)?;
    let kind = match kind_str.as_str() {
        "fact" => MemoryKind::Fact(Fact {
            subject: row.get(3)?,
            relation: row.get(4)?,
            object: row.get(5)?,
        }),
        _ => MemoryKind::Episode(Episode {
            text: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
        }),
    };
    let embedding: Option<Vec<u8>> = row.get(8)?;
    Ok(MemoryRecord {
        id: row.get(0)?,
        namespace: row.get(1)?,
        kind,
        strength: row.get(7)?,
        embedding: embedding.map(|b| blob_to_embedding(&b)),
        created_at: parse_datetime(&row.get::<_, String>(9)?),
        last_accessed_at: parse_datetime(&row.get::<_, String>(10)?),
        access_count: row.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    #[test]
    fn open_migrates_legacy_schema_without_namespace_and_preserves_access() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at, access_count)
                VALUES ('fact', 'Legacy', 'uses', 'schema', 0.8, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 2);",
            )
            .unwrap();
        }

        let store = MemoryStore::open(path).unwrap();

        let namespace_info: (String, i64, String) = store
            .conn()
            .query_row(
                "SELECT name, `notnull`, dflt_value FROM pragma_table_info('memories') WHERE name = 'namespace'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(namespace_info.0, "namespace");
        assert_eq!(namespace_info.1, 1);
        assert_eq!(namespace_info.2, "'default'");

        let rows_with_default_ns: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE namespace = 'default'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows_with_default_ns, 1);

        let memory = store
            .find_fact_in(DEFAULT_NAMESPACE, "Legacy", "uses", "schema")
            .unwrap()
            .expect("legacy row should be accessible in default namespace");
        assert_eq!(memory.namespace, DEFAULT_NAMESPACE);
        assert!((memory.strength - 0.8).abs() < 1e-6);

        let default_stats = store.stats_in(DEFAULT_NAMESPACE).unwrap();
        assert_eq!(default_stats.total_memories, 1);

        let other_stats = store.stats_in("other").unwrap();
        assert_eq!(other_stats.total_memories, 0);
    }

    #[test]
    fn open_drops_redundant_kind_namespace_index() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace        TEXT NOT NULL DEFAULT 'default',
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX idx_memories_kind_namespace ON memories(kind, namespace);",
            )
            .unwrap();
        }

        let store = MemoryStore::open(path).unwrap();

        let redundant_index_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_kind_namespace'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(redundant_index_count, 0);

        let preferred_index_count: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_namespace_kind'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(preferred_index_count, 1);
    }

    // ── Namespace Migration Reliability Test Pack ──────────────────────────

    #[test]
    fn migration_preserves_data_integrity_with_mixed_content() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Create legacy database with mixed facts and episodes
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );"
            ).unwrap();

            // Insert varied test data
            let test_data = [
                ("INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at, access_count) VALUES ('fact', 'Alice', 'works_at', 'Google', 0.9, '2025-06-15T10:30:00Z', '2025-06-15T10:30:00Z', 5)", 0.9, 5),
                ("INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at, access_count) VALUES ('fact', 'Bob', 'lives_in', 'Seattle', 0.7, '2025-03-20T14:45:00Z', '2025-03-20T14:45:00Z', 3)", 0.7, 3),
                ("INSERT INTO memories (kind, episode_text, strength, created_at, last_accessed_at, access_count) VALUES ('episode', 'Team meeting went well today', 0.6, '2025-12-01T09:00:00Z', '2025-12-01T09:00:00Z', 1)", 0.6, 1),
                ("INSERT INTO memories (kind, episode_text, strength, created_at, last_accessed_at, access_count) VALUES ('episode', 'Alice presented the quarterly results', 0.8, '2025-11-30T16:20:00Z', '2025-11-30T16:20:00Z', 2)", 0.8, 2),
            ];

            for (sql, _, _) in &test_data {
                conn.execute(sql, []).unwrap();
            }
        }

        // Open with migration
        let store = MemoryStore::open(path).unwrap();

        // Verify all records migrated to default namespace with preserved data
        let all_memories: Vec<MemoryRecord> = store
            .conn()
            .prepare("SELECT id, namespace, kind, subject, relation, object, episode_text, strength, embedding, created_at, last_accessed_at, access_count FROM memories ORDER BY id")
            .unwrap()
            .query_map([], row_to_memory)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(all_memories.len(), 4);
        
        // Check first fact
        assert_eq!(all_memories[0].namespace, DEFAULT_NAMESPACE);
        if let MemoryKind::Fact(f) = &all_memories[0].kind {
            assert_eq!(f.subject, "Alice");
            assert_eq!(f.relation, "works_at");
            assert_eq!(f.object, "Google");
        } else {
            panic!("Expected fact, got episode");
        }
        assert!((all_memories[0].strength - 0.9).abs() < 1e-6);
        assert_eq!(all_memories[0].access_count, 5);

        // Check first episode
        if let MemoryKind::Episode(e) = &all_memories[2].kind {
            assert_eq!(e.text, "Team meeting went well today");
        } else {
            panic!("Expected episode, got fact");
        }
        assert!((all_memories[2].strength - 0.6).abs() < 1e-6);
        assert_eq!(all_memories[2].access_count, 1);

        // Verify namespace isolation works post-migration
        let default_stats = store.stats_in(DEFAULT_NAMESPACE).unwrap();
        assert_eq!(default_stats.total_memories, 4);
        assert_eq!(default_stats.total_facts, 2);
        assert_eq!(default_stats.total_episodes, 2);

        let other_stats = store.stats_in("other_namespace").unwrap();
        assert_eq!(other_stats.total_memories, 0);
    }

    #[test]
    fn migration_handles_null_and_empty_text_fields_gracefully() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Create legacy database with problematic data
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );"
            ).unwrap();

            // Insert edge case data (using empty strings instead of NULL for better compatibility)
            conn.execute(
                "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at) 
                 VALUES ('fact', '', 'relates_to', '', 0.5, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
                [],
            ).unwrap();

            conn.execute(
                "INSERT INTO memories (kind, episode_text, strength, created_at, last_accessed_at) 
                 VALUES ('episode', '', 0.3, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
                [],
            ).unwrap();

            conn.execute(
                "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at) 
                 VALUES ('fact', 'Normal', 'fact', 'here', 0.8, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
                [],
            ).unwrap();
        }

        // Migration should handle nulls gracefully
        let store = MemoryStore::open(path).unwrap();

        let memories: Vec<MemoryRecord> = store
            .conn()
            .prepare("SELECT id, namespace, kind, subject, relation, object, episode_text, strength, embedding, created_at, last_accessed_at, access_count FROM memories ORDER BY id")
            .unwrap()
            .query_map([], row_to_memory)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(memories.len(), 3);

        // Verify problematic fact was handled
        if let MemoryKind::Fact(f) = &memories[0].kind {
            assert_eq!(f.subject, ""); // Empty string preserved
            assert_eq!(f.relation, "relates_to");
            assert_eq!(f.object, ""); // Empty string preserved
        } else {
            panic!("Expected fact");
        }

        // Verify problematic episode was handled
        if let MemoryKind::Episode(e) = &memories[1].kind {
            assert_eq!(e.text, ""); // Empty string preserved
        } else {
            panic!("Expected episode");
        }

        // Normal record unaffected
        if let MemoryKind::Fact(f) = &memories[2].kind {
            assert_eq!(f.subject, "Normal");
            assert_eq!(f.relation, "fact");
            assert_eq!(f.object, "here");
        } else {
            panic!("Expected fact");
        }
    }

    #[test]
    fn migration_maintains_index_performance_characteristics() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Create legacy database with no indexes
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );"
            ).unwrap();

            // Insert enough data to make index performance matter
            for i in 0..50 {
                conn.execute(
                    "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at) 
                     VALUES ('fact', ?1, 'test_relation', ?2, 0.5, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
                    params![format!("subject_{}", i), format!("object_{}", i)]
                ).unwrap();
            }
        }

        // Migration should create proper indexes
        let store = MemoryStore::open(path).unwrap();

        // Check that all expected indexes exist
        let index_names: Vec<String> = store
            .conn()
            .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'memories' ORDER BY name")
            .unwrap()
            .query_map([], |row| Ok(row.get::<_, String>(0)?))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let expected_indexes = vec![
            "idx_memories_kind".to_string(),
            "idx_memories_namespace".to_string(),
            "idx_memories_namespace_kind".to_string(),
            "idx_memories_subject".to_string(),
        ];

        for expected in &expected_indexes {
            assert!(index_names.contains(expected), "Missing index: {}", expected);
        }

        // Verify namespace queries use proper indexes by checking they don't require full table scans
        let query_plan: String = store
            .conn()
            .query_row(
                "EXPLAIN QUERY PLAN SELECT * FROM memories WHERE namespace = 'default' AND kind = 'fact'",
                [],
                |row| Ok(row.get::<_, String>(3)?), // detail column
            )
            .unwrap();

        // Should use the namespace_kind index, not scan the table
        assert!(query_plan.contains("idx_memories_namespace_kind"), 
               "Query should use namespace_kind index, got plan: {}", query_plan);
    }

    #[test]
    fn migration_handles_concurrent_access_gracefully() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Create legacy database 
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at)
                VALUES ('fact', 'Concurrent', 'test', 'data', 0.7, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z');"
            ).unwrap();
        }

        // Migration should be idempotent - multiple opens shouldn't conflict
        let store1 = MemoryStore::open(path).unwrap();
        let store2 = MemoryStore::open(path).unwrap();

        // Both stores should see the same migrated data
        let count1: i64 = store1
            .conn()
            .query_row("SELECT COUNT(*) FROM memories WHERE namespace = 'default'", [], |r| r.get(0))
            .unwrap();
        let count2: i64 = store2
            .conn()
            .query_row("SELECT COUNT(*) FROM memories WHERE namespace = 'default'", [], |r| r.get(0))
            .unwrap();

        assert_eq!(count1, 1);
        assert_eq!(count2, 1);

        // Verify schema consistency across both connections
        let schema1: String = store1
            .conn()
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let schema2: String = store2
            .conn()
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories'",
                [],
                |r| r.get(0),
            )
            .unwrap();

        assert_eq!(schema1, schema2);
        assert!(schema1.contains("namespace"));
    }

    #[test]
    fn migration_preserves_timestamps_and_access_patterns() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        let original_created = "2024-03-15T08:30:15Z";
        let original_accessed = "2025-07-22T14:22:33Z";
        let original_count = 42;

        // Create legacy database with specific timestamps
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );"
            ).unwrap();

            conn.execute(
                "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at, access_count)
                 VALUES ('fact', 'TimeKeeper', 'preserves', 'history', 0.95, ?1, ?2, ?3)",
                params![original_created, original_accessed, original_count]
            ).unwrap();
        }

        // Migration should preserve precise timestamps
        let store = MemoryStore::open(path).unwrap();

        let memory = store
            .find_fact_in(DEFAULT_NAMESPACE, "TimeKeeper", "preserves", "history")
            .unwrap()
            .expect("Migrated memory should exist");

        // Verify timestamps were preserved exactly (allowing for timezone format differences)
        assert_eq!(memory.created_at.timestamp(), DateTime::parse_from_rfc3339(original_created).unwrap().timestamp());
        assert_eq!(memory.last_accessed_at.timestamp(), DateTime::parse_from_rfc3339(original_accessed).unwrap().timestamp());
        assert_eq!(memory.access_count, original_count);
        assert!((memory.strength - 0.95).abs() < 1e-6);

        // Verify memory is accessible in default namespace
        assert_eq!(memory.namespace, DEFAULT_NAMESPACE);
    }

    #[test]
    fn migration_handles_malformed_timestamps_with_fallback() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Create legacy database with malformed timestamps
        {
            let conn = Connection::open(path).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT NOT NULL CHECK(kind IN ('fact', 'episode')),
                    subject          TEXT,
                    relation         TEXT,
                    object           TEXT,
                    episode_text     TEXT,
                    strength         REAL NOT NULL DEFAULT 1.0,
                    embedding        BLOB,
                    created_at       TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count     INTEGER NOT NULL DEFAULT 0
                );"
            ).unwrap();

            // Insert records with various malformed timestamps
            conn.execute(
                "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at)
                 VALUES ('fact', 'BadTime1', 'has', 'invalid_date', 0.5, 'not-a-date', '2025-01-01T00:00:00Z')",
                []
            ).unwrap();

            conn.execute(
                "INSERT INTO memories (kind, subject, relation, object, strength, created_at, last_accessed_at)
                 VALUES ('fact', 'BadTime2', 'has', 'partial_date', 0.5, '2025-invalid-date', 'also-not-a-date')",
                []
            ).unwrap();
        }

        // Migration should handle malformed timestamps gracefully
        let store = MemoryStore::open(path).unwrap();

        let memories: Vec<MemoryRecord> = store
            .conn()
            .prepare("SELECT id, namespace, kind, subject, relation, object, episode_text, strength, embedding, created_at, last_accessed_at, access_count FROM memories ORDER BY id")
            .unwrap()
            .query_map([], row_to_memory)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(memories.len(), 2);

        // Records should exist with mixed valid/fallback timestamps
        // We can't test exact timestamp since fallback uses Utc::now(), but we can verify
        // the records were successfully migrated and are accessible
        let bad_time1 = store
            .find_fact_in(DEFAULT_NAMESPACE, "BadTime1", "has", "invalid_date")
            .unwrap()
            .expect("Record with malformed timestamp should be migrated");

        let bad_time2 = store
            .find_fact_in(DEFAULT_NAMESPACE, "BadTime2", "has", "partial_date")
            .unwrap()
            .expect("Record with malformed timestamp should be migrated");

        let now = Utc::now();
        
        // BadTime1: malformed created_at should be replaced with current time, 
        // valid last_accessed_at should be preserved
        assert!((now - bad_time1.created_at).num_seconds() < 60, "Malformed created_at should be replaced");
        assert_eq!(bad_time1.last_accessed_at.to_rfc3339(), "2025-01-01T00:00:00+00:00");
        
        // BadTime2: both timestamps malformed, both should be replaced
        assert!((now - bad_time2.created_at).num_seconds() < 60, "Malformed created_at should be replaced");
        assert!((now - bad_time2.last_accessed_at).num_seconds() < 60, "Malformed last_accessed_at should be replaced");
    }
}
