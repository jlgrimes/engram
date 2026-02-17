use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;

use crate::memory::{
    Association, Episode, Fact, MemoryKind, MemoryRecord, MemoryStats,
};

/// SQLite-backed memory store.
pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    /// Access the underlying SQLite connection (for modules like decay that operate directly).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

impl MemoryStore {
    /// Open (or create) a memory database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Open an in-memory database (useful for tests).
    pub fn open_in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> SqlResult<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
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
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

            CREATE TABLE IF NOT EXISTS associations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a    TEXT NOT NULL,
                relation    TEXT NOT NULL,
                entity_b    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE(entity_a, relation, entity_b)
            );

            CREATE INDEX IF NOT EXISTS idx_assoc_entity_a ON associations(entity_a);
            CREATE INDEX IF NOT EXISTS idx_assoc_entity_b ON associations(entity_b);
            ",
        )?;
        Ok(())
    }

    // ── Remember ─────────────────────────────────────────────

    /// Store a fact (subject-relation-object triple).
    pub fn remember_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
        embedding: Option<&[f32]>,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (kind, subject, relation, object, embedding, created_at, last_accessed_at)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?5)",
            params![subject, relation, object, emb_blob, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Store an episode (free-text event).
    pub fn remember_episode(
        &self,
        text: &str,
        embedding: Option<&[f32]>,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, embedding, created_at, last_accessed_at)
             VALUES ('episode', ?1, ?2, ?3, ?3)",
            params![text, emb_blob, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Recall ───────────────────────────────────────────────

    /// Retrieve all memories with their embeddings for semantic search.
    /// The caller (recall.rs) handles ranking.
    pub fn all_memories_with_embeddings(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories
             WHERE strength > 0.01",
        )?;
        let rows = stmt.query_map([], |row| {
            row_to_memory(row)
        })?;
        rows.collect()
    }

    /// Touch a memory: update last_accessed_at, bump access_count, reinforce strength.
    pub fn touch_memory(&self, id: i64) -> SqlResult<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE memories
             SET last_accessed_at = ?1,
                 access_count = access_count + 1,
                 strength = MIN(strength + 0.2, 1.0)
             WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    /// Get a single memory by ID.
    pub fn get_memory(&self, id: i64) -> SqlResult<Option<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| row_to_memory(row))?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    // ── Decay ────────────────────────────────────────────────

    /// Run a decay pass: reduce strength of all memories based on time since last access.
    /// Returns the number of memories that were decayed.
    pub fn decay_all(&self, decay_factor: f64, half_life_hours: f64) -> SqlResult<usize> {
        let now = Utc::now();
        let mut stmt = self.conn.prepare(
            "SELECT id, last_accessed_at, strength FROM memories WHERE strength > 0.01",
        )?;
        let rows: Vec<(i64, String, f64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })?
            .collect::<SqlResult<Vec<_>>>()?;

        let mut count = 0;
        for (id, last_accessed, strength) in &rows {
            let last = DateTime::parse_from_rfc3339(last_accessed)
                .unwrap_or_else(|_| now.into())
                .with_timezone(&Utc);
            let hours_elapsed = (now - last).num_seconds() as f64 / 3600.0;
            let factor = decay_factor.powf(hours_elapsed / half_life_hours);
            let new_strength = (strength * factor).max(0.0);
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

    /// Delete memories by subject.
    pub fn forget_by_subject(&self, subject: &str) -> SqlResult<usize> {
        let changed = self.conn.execute(
            "DELETE FROM memories WHERE subject = ?1",
            params![subject],
        )?;
        Ok(changed)
    }

    /// Delete memories older than a given duration.
    pub fn forget_older_than(&self, duration: Duration) -> SqlResult<usize> {
        let cutoff = (Utc::now() - duration).to_rfc3339();
        let changed = self.conn.execute(
            "DELETE FROM memories WHERE created_at < ?1",
            params![cutoff],
        )?;
        Ok(changed)
    }

    // ── Associations ─────────────────────────────────────────

    /// Create an association between two entities.
    pub fn relate(
        &self,
        entity_a: &str,
        relation: &str,
        entity_b: &str,
    ) -> SqlResult<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR IGNORE INTO associations (entity_a, relation, entity_b, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![entity_a, relation, entity_b, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all associations for an entity (as either side).
    pub fn get_associations(&self, entity: &str) -> SqlResult<Vec<Association>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_a, relation, entity_b, created_at
             FROM associations
             WHERE entity_a = ?1 OR entity_b = ?1",
        )?;
        let rows = stmt.query_map(params![entity], |row| {
            Ok(Association {
                id: row.get(0)?,
                entity_a: row.get(1)?,
                relation: row.get(2)?,
                entity_b: row.get(3)?,
                created_at: parse_datetime(&row.get::<_, String>(4)?),
            })
        })?;
        rows.collect()
    }

    // ── Content loading (for BM25 search) ─────────────────────

    /// Load all non-forgotten memories with their text content.
    /// Returns (MemoryRecord, text_for_bm25) pairs.
    pub fn all_memories_with_text(&self) -> SqlResult<Vec<(MemoryRecord, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories
             WHERE strength > 0.01",
        )?;
        let rows = stmt.query_map([], |row| {
            let mem = row_to_memory(row)?;
            let text = mem.text_for_embedding();
            Ok((mem, text))
        })?;
        rows.collect()
    }

    /// Load all associations as (Association, text) pairs for BM25 scoring.
    pub fn all_associations_with_text(&self) -> SqlResult<Vec<(Association, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_a, relation, entity_b, created_at
             FROM associations",
        )?;
        let rows = stmt.query_map([], |row| {
            let assoc = Association {
                id: row.get(0)?,
                entity_a: row.get(1)?,
                relation: row.get(2)?,
                entity_b: row.get(3)?,
                created_at: parse_datetime(&row.get::<_, String>(4)?),
            };
            let text = format!("{} {} {}", assoc.entity_a, assoc.relation, assoc.entity_b);
            Ok((assoc, text))
        })?;
        rows.collect()
    }

    // ── Batch embed ────────────────────────────────────────

    /// Get all memories that are missing embeddings.
    pub fn memories_missing_embeddings(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories
             WHERE embedding IS NULL",
        )?;
        let rows = stmt.query_map([], |row| row_to_memory(row))?;
        rows.collect()
    }

    /// Update the embedding for a specific memory.
    pub fn update_embedding(&self, id: i64, embedding: &[f32]) -> SqlResult<()> {
        let blob = embedding_to_blob(embedding);
        self.conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![blob, id],
        )?;
        Ok(())
    }

    // ── Export / Import ───────────────────────────────────────

    /// Get all memories (including weak ones), for export.
    pub fn all_memories(&self) -> SqlResult<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, subject, relation, object, episode_text,
                    strength, embedding, created_at, last_accessed_at, access_count
             FROM memories",
        )?;
        let rows = stmt.query_map([], |row| row_to_memory(row))?;
        rows.collect()
    }

    /// Get all associations, for export.
    pub fn all_associations(&self) -> SqlResult<Vec<Association>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_a, relation, entity_b, created_at
             FROM associations",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Association {
                id: row.get(0)?,
                entity_a: row.get(1)?,
                relation: row.get(2)?,
                entity_b: row.get(3)?,
                created_at: parse_datetime(&row.get::<_, String>(4)?),
            })
        })?;
        rows.collect()
    }

    /// Import a fact with full metadata (for import from JSON).
    pub fn import_fact(
        &self,
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
            "INSERT INTO memories (kind, subject, relation, object, strength, embedding, created_at, last_accessed_at, access_count)
             VALUES ('fact', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![subject, relation, object, strength, emb_blob, created_at, last_accessed_at, access_count],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Import an episode with full metadata (for import from JSON).
    pub fn import_episode(
        &self,
        text: &str,
        strength: f64,
        embedding: Option<&[f32]>,
        created_at: &str,
        last_accessed_at: &str,
        access_count: i64,
    ) -> SqlResult<i64> {
        let emb_blob = embedding.map(embedding_to_blob);
        self.conn.execute(
            "INSERT INTO memories (kind, episode_text, strength, embedding, created_at, last_accessed_at, access_count)
             VALUES ('episode', ?1, ?2, ?3, ?4, ?5, ?6)",
            params![text, strength, emb_blob, created_at, last_accessed_at, access_count],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ── Stats ────────────────────────────────────────────────

    /// Get database statistics.
    pub fn stats(&self) -> SqlResult<MemoryStats> {
        let total_memories: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
        let total_facts: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE kind = 'fact'",
            [],
            |r| r.get(0),
        )?;
        let total_episodes: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE kind = 'episode'",
            [],
            |r| r.get(0),
        )?;
        let total_associations: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM associations", [], |r| r.get(0))?;
        let avg_strength: f64 = self
            .conn
            .query_row(
                "SELECT COALESCE(AVG(strength), 0.0) FROM memories",
                [],
                |r| r.get(0),
            )?;

        Ok(MemoryStats {
            total_memories,
            total_facts,
            total_episodes,
            total_associations,
            avg_strength,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn embedding_to_blob(emb: &[f32]) -> Vec<u8> {
    emb.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn row_to_memory(row: &rusqlite::Row) -> SqlResult<MemoryRecord> {
    let kind_str: String = row.get(1)?;
    let kind = match kind_str.as_str() {
        "fact" => MemoryKind::Fact(Fact {
            subject: row.get(2)?,
            relation: row.get(3)?,
            object: row.get(4)?,
        }),
        "episode" => MemoryKind::Episode(Episode {
            text: row.get(5)?,
        }),
        _ => MemoryKind::Episode(Episode {
            text: String::from("unknown"),
        }),
    };

    let embedding: Option<Vec<u8>> = row.get(7)?;

    Ok(MemoryRecord {
        id: row.get(0)?,
        kind,
        strength: row.get(6)?,
        embedding: embedding.map(|b| blob_to_embedding(&b)),
        created_at: parse_datetime(&row.get::<_, String>(8)?),
        last_accessed_at: parse_datetime(&row.get::<_, String>(9)?),
        access_count: row.get(10)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remember_and_get_fact() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .remember_fact("Jared", "works at", "Microsoft", None)
            .unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(matches!(mem.kind, MemoryKind::Fact(_)));
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.subject, "Jared");
            assert_eq!(f.relation, "works at");
            assert_eq!(f.object, "Microsoft");
        }
        assert!((mem.strength - 1.0).abs() < f64::EPSILON);
        assert_eq!(mem.access_count, 0);
    }

    #[test]
    fn test_remember_and_get_episode() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .remember_episode("Had coffee with Alice at 3pm", None)
            .unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        if let MemoryKind::Episode(e) = &mem.kind {
            assert_eq!(e.text, "Had coffee with Alice at 3pm");
        } else {
            panic!("Expected episode");
        }
    }

    #[test]
    fn test_touch_memory() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "is", "B", None).unwrap();

        // Decay it first so we can see reinforcement
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.5 WHERE id = ?1",
                params![id],
            )
            .unwrap();

        store.touch_memory(id).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(mem.access_count, 1);
        assert!((mem.strength - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_forget_by_subject() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("X", "is", "Y", None).unwrap();
        store.remember_fact("X", "likes", "Z", None).unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();

        let deleted = store.forget_by_subject("X").unwrap();
        assert_eq!(deleted, 2);

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }

    #[test]
    fn test_relate_and_get_associations() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.relate("Alice", "knows", "Bob").unwrap();
        store.relate("Alice", "works with", "Charlie").unwrap();

        let assocs = store.get_associations("Alice").unwrap();
        assert_eq!(assocs.len(), 2);

        let assocs_bob = store.get_associations("Bob").unwrap();
        assert_eq!(assocs_bob.len(), 1);
    }

    #[test]
    fn test_stats() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();
        store.remember_fact("C", "is", "D", None).unwrap();
        store.remember_episode("something happened", None).unwrap();
        store.relate("A", "knows", "C").unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_memories, 3);
        assert_eq!(stats.total_facts, 2);
        assert_eq!(stats.total_episodes, 1);
        assert_eq!(stats.total_associations, 1);
        assert!((stats.avg_strength - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_memories_with_embeddings() {
        let store = MemoryStore::open_in_memory().unwrap();
        let emb = vec![0.1f32, 0.2, 0.3];
        store.remember_fact("A", "is", "B", Some(&emb)).unwrap();
        store.remember_episode("test", Some(&emb)).unwrap();

        let mems = store.all_memories_with_embeddings().unwrap();
        assert_eq!(mems.len(), 2);
        assert!(mems[0].embedding.is_some());
        let recovered = mems[0].embedding.as_ref().unwrap();
        assert!((recovered[0] - 0.1).abs() < 1e-6);
        assert!((recovered[1] - 0.2).abs() < 1e-6);
        assert!((recovered[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14];
        let blob = embedding_to_blob(&original);
        let recovered = blob_to_embedding(&blob);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_memories_missing_embeddings() {
        let store = MemoryStore::open_in_memory().unwrap();
        let emb = vec![0.1f32, 0.2, 0.3];
        store.remember_fact("A", "is", "B", Some(&emb)).unwrap();
        store.remember_fact("C", "is", "D", None).unwrap();
        store.remember_episode("no embedding", None).unwrap();

        let missing = store.memories_missing_embeddings().unwrap();
        assert_eq!(missing.len(), 2);

        // All should have no embedding
        for m in &missing {
            assert!(m.embedding.is_none());
        }
    }

    #[test]
    fn test_update_embedding() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "is", "B", None).unwrap();

        // Verify no embedding
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.embedding.is_none());

        // Update embedding
        let emb = vec![0.5f32, 0.6, 0.7];
        store.update_embedding(id, &emb).unwrap();

        // Verify embedding is set
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.embedding.is_some());
        let recovered = mem.embedding.unwrap();
        assert!((recovered[0] - 0.5).abs() < 1e-6);
        assert!((recovered[1] - 0.6).abs() < 1e-6);
        assert!((recovered[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_all_memories_includes_weak() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();
        store.remember_fact("C", "is", "D", None).unwrap();

        // Weaken one memory below the recall threshold
        store
            .conn
            .execute(
                "UPDATE memories SET strength = 0.005 WHERE subject = 'C'",
                [],
            )
            .unwrap();

        // all_memories_with_embeddings filters out weak ones
        let strong = store.all_memories_with_embeddings().unwrap();
        assert_eq!(strong.len(), 1);

        // all_memories includes everything
        let all = store.all_memories().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_all_associations() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.relate("A", "knows", "B").unwrap();
        store.relate("C", "works with", "D").unwrap();

        let assocs = store.all_associations().unwrap();
        assert_eq!(assocs.len(), 2);
    }

    #[test]
    fn test_import_fact() {
        let store = MemoryStore::open_in_memory().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let emb = vec![0.1f32, 0.2, 0.3];
        let id = store
            .import_fact("X", "is", "Y", 0.75, Some(&emb), &now, &now, 5)
            .unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        assert!((mem.strength - 0.75).abs() < 1e-6);
        assert_eq!(mem.access_count, 5);
        assert!(mem.embedding.is_some());
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.subject, "X");
            assert_eq!(f.relation, "is");
            assert_eq!(f.object, "Y");
        } else {
            panic!("Expected fact");
        }
    }

    #[test]
    fn test_import_episode() {
        let store = MemoryStore::open_in_memory().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let id = store
            .import_episode("test episode", 0.5, None, &now, &now, 3)
            .unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        assert!((mem.strength - 0.5).abs() < 1e-6);
        assert_eq!(mem.access_count, 3);
        assert!(mem.embedding.is_none());
        if let MemoryKind::Episode(e) = &mem.kind {
            assert_eq!(e.text, "test episode");
        } else {
            panic!("Expected episode");
        }
    }
}
