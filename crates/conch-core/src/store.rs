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
            "CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
             CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
             CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
             CREATE INDEX IF NOT EXISTS idx_memories_kind_namespace ON memories(kind, namespace);
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
