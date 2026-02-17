pub mod memory;
pub mod store;
pub mod embed;
pub mod decay;
pub mod recall;
pub mod relate;

// Re-export key types for convenience.
pub use memory::{Association, Episode, ExportData, Fact, MemoryKind, MemoryRecord, MemoryStats};
pub use store::MemoryStore;
pub use embed::{Embedder, EmbedError, FastEmbedder, SharedEmbedder, cosine_similarity};
pub use decay::{run_decay, DecayResult};
pub use recall::{recall, RecallResult, RecallError};
pub use relate::{relate, find_associations};

use chrono::Duration;

/// High-level API wrapping storage + embeddings.
pub struct ConchDB {
    store: MemoryStore,
    embedder: Box<dyn Embedder>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConchError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("embedding error: {0}")]
    Embed(#[from] EmbedError),
}

impl ConchDB {
    /// Open a database at the given path with the default (fastembed) embedder.
    pub fn open(path: &str) -> Result<Self, ConchError> {
        let store = MemoryStore::open(path)?;
        let embedder = embed::FastEmbedder::new()?;
        Ok(Self {
            store,
            embedder: Box::new(embedder),
        })
    }

    /// Create a ConchDB with an in-memory store and a custom embedder.
    pub fn open_in_memory_with(embedder: Box<dyn Embedder>) -> Result<Self, ConchError> {
        let store = MemoryStore::open_in_memory()?;
        Ok(Self { store, embedder })
    }

    /// Store a fact (subject-relation-object triple).
    pub fn remember_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> Result<MemoryRecord, ConchError> {
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;
        let id = self.store.remember_fact(subject, relation, object, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    /// Store an episode (free-text event).
    pub fn remember_episode(&self, text: &str) -> Result<MemoryRecord, ConchError> {
        let embedding = self.embedder.embed_one(text)?;
        let id = self.store.remember_episode(text, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    /// Semantic search: embed the query, rank by relevance × recency × strength.
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall(&self.store, query, self.embedder.as_ref(), limit)
            .map_err(|e| match e {
                RecallError::Db(e) => ConchError::Db(e),
                RecallError::Embedding(msg) => ConchError::Embed(EmbedError::Other(msg)),
            })
    }

    /// Create an association between two entities.
    pub fn relate(
        &self,
        entity_a: &str,
        relation: &str,
        entity_b: &str,
    ) -> Result<i64, ConchError> {
        Ok(relate::relate(&self.store, entity_a, relation, entity_b)?)
    }

    /// Get all associations for an entity.
    pub fn get_associations(&self, entity: &str) -> Result<Vec<Association>, ConchError> {
        Ok(relate::find_associations(&self.store, entity)?)
    }

    /// Delete memories by subject.
    pub fn forget_by_subject(&self, subject: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_subject(subject)?)
    }

    /// Delete memories older than `secs` seconds.
    pub fn forget_older_than(&self, secs: i64) -> Result<usize, ConchError> {
        let duration = Duration::seconds(secs);
        Ok(self.store.forget_older_than(duration)?)
    }

    /// Run the decay pass. Returns decay statistics.
    pub fn decay(&self) -> Result<DecayResult, ConchError> {
        Ok(decay::run_decay(&self.store, None, None)?)
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<MemoryStats, ConchError> {
        Ok(self.store.stats()?)
    }

    /// Generate embeddings for all memories that are missing them.
    /// Returns the number of memories that were embedded.
    pub fn embed_all(&self) -> Result<usize, ConchError> {
        let missing = self.store.memories_missing_embeddings()?;
        if missing.is_empty() {
            return Ok(0);
        }

        // Batch embed for efficiency
        let texts: Vec<String> = missing.iter().map(|m| m.text_for_embedding()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embedder.embed(&text_refs)?;

        for (mem, emb) in missing.iter().zip(embeddings.iter()) {
            self.store.update_embedding(mem.id, emb)?;
        }

        Ok(missing.len())
    }

    /// Export all memories and associations as an ExportData struct.
    pub fn export(&self) -> Result<ExportData, ConchError> {
        let memories = self.store.all_memories()?;
        let associations = self.store.all_associations()?;
        Ok(ExportData {
            memories,
            associations,
        })
    }

    /// Import memories and associations from an ExportData struct.
    /// Returns (memories_imported, associations_imported).
    pub fn import(&self, data: &ExportData) -> Result<(usize, usize), ConchError> {
        let mut mem_count = 0;
        for mem in &data.memories {
            let created = mem.created_at.to_rfc3339();
            let accessed = mem.last_accessed_at.to_rfc3339();
            match &mem.kind {
                MemoryKind::Fact(f) => {
                    self.store.import_fact(
                        &f.subject,
                        &f.relation,
                        &f.object,
                        mem.strength,
                        mem.embedding.as_deref(),
                        &created,
                        &accessed,
                        mem.access_count,
                    )?;
                }
                MemoryKind::Episode(e) => {
                    self.store.import_episode(
                        &e.text,
                        mem.strength,
                        mem.embedding.as_deref(),
                        &created,
                        &accessed,
                        mem.access_count,
                    )?;
                }
            }
            mem_count += 1;
        }

        let mut assoc_count = 0;
        for assoc in &data.associations {
            self.store.relate(&assoc.entity_a, &assoc.relation, &assoc.entity_b)?;
            assoc_count += 1;
        }

        Ok((mem_count, assoc_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embed::Embedding;

    struct MockEmbedder;
    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts.iter().map(|_| vec![0.5f32, 0.5, 0.5]).collect())
        }
        fn dimension(&self) -> usize {
            3
        }
    }

    fn test_db() -> ConchDB {
        ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap()
    }

    #[test]
    fn test_embed_all() {
        let db = test_db();

        // Insert memories without embeddings via the store directly
        db.store.remember_fact("A", "is", "B", None).unwrap();
        db.store.remember_episode("test event", None).unwrap();

        // Both should be missing embeddings
        let missing = db.store.memories_missing_embeddings().unwrap();
        assert_eq!(missing.len(), 2);

        // Run embed_all
        let count = db.embed_all().unwrap();
        assert_eq!(count, 2);

        // No more missing
        let missing = db.store.memories_missing_embeddings().unwrap();
        assert_eq!(missing.len(), 0);

        // Running again should return 0
        let count = db.embed_all().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let db1 = test_db();

        // Add some data
        db1.remember_fact("Jared", "works at", "Microsoft").unwrap();
        db1.remember_episode("Had a meeting").unwrap();
        db1.relate("Jared", "knows", "Alice").unwrap();

        // Export
        let data = db1.export().unwrap();
        assert_eq!(data.memories.len(), 2);
        assert_eq!(data.associations.len(), 1);

        // Serialize to JSON and back (like the CLI would)
        let json = serde_json::to_string(&data).unwrap();
        let imported_data: ExportData = serde_json::from_str(&json).unwrap();

        // Import into a fresh database
        let db2 = test_db();
        let (mem_count, assoc_count) = db2.import(&imported_data).unwrap();
        assert_eq!(mem_count, 2);
        assert_eq!(assoc_count, 1);

        // Verify data matches
        let stats = db2.stats().unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.total_facts, 1);
        assert_eq!(stats.total_episodes, 1);
        assert_eq!(stats.total_associations, 1);
    }

    #[test]
    fn test_export_preserves_metadata() {
        let db = test_db();
        db.remember_fact("X", "is", "Y").unwrap();

        let data = db.export().unwrap();
        let mem = &data.memories[0];

        // Verify metadata is preserved
        assert!((mem.strength - 1.0).abs() < f64::EPSILON);
        assert!(mem.embedding.is_some());
        assert_eq!(mem.access_count, 0);
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.subject, "X");
            assert_eq!(f.relation, "is");
            assert_eq!(f.object, "Y");
        } else {
            panic!("Expected fact");
        }
    }

    #[test]
    fn test_import_preserves_strength_and_access_count() {
        let db1 = test_db();
        let record = db1.remember_fact("A", "is", "B").unwrap();

        // Touch the memory to change metadata
        db1.store.touch_memory(record.id).unwrap();
        db1.store.touch_memory(record.id).unwrap();

        // Export
        let data = db1.export().unwrap();
        let exported_mem = &data.memories[0];
        assert_eq!(exported_mem.access_count, 2);

        // Import into new db
        let db2 = test_db();
        db2.import(&data).unwrap();

        // Verify metadata preserved
        let all = db2.store.all_memories().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].access_count, 2);
    }
}
