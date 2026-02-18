pub mod memory;
pub mod store;
pub mod embed;
pub mod decay;
pub mod recall;

pub use memory::{Episode, ExportData, Fact, MemoryKind, MemoryRecord, MemoryStats};
pub use store::MemoryStore;
pub use embed::{Embedder, EmbedError, FastEmbedder, SharedEmbedder, cosine_similarity};
pub use decay::{run_decay, DecayResult};
pub use recall::{recall, RecallResult, RecallError};

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
    pub fn open(path: &str) -> Result<Self, ConchError> {
        let store = MemoryStore::open(path)?;
        let embedder = embed::FastEmbedder::new()?;
        Ok(Self { store, embedder: Box::new(embedder) })
    }

    pub fn open_in_memory_with(embedder: Box<dyn Embedder>) -> Result<Self, ConchError> {
        let store = MemoryStore::open_in_memory()?;
        Ok(Self { store, embedder })
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn remember_fact(&self, subject: &str, relation: &str, object: &str) -> Result<MemoryRecord, ConchError> {
        // Check for existing duplicate
        if let Some(existing) = self.store.find_fact(subject, relation, object)? {
            self.store.reinforce_memory(existing.id)?;
            return Ok(self.store.get_memory(existing.id)?.expect("just reinforced"));
        }
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;
        let id = self.store.remember_fact(subject, relation, object, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    pub fn remember_episode(&self, text: &str) -> Result<MemoryRecord, ConchError> {
        // Check for existing duplicate
        if let Some(existing) = self.store.find_episode(text)? {
            self.store.reinforce_memory(existing.id)?;
            return Ok(self.store.get_memory(existing.id)?.expect("just reinforced"));
        }
        let embedding = self.embedder.embed_one(text)?;
        let id = self.store.remember_episode(text, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall(&self.store, query, self.embedder.as_ref(), limit)
            .map_err(|e| match e {
                RecallError::Db(e) => ConchError::Db(e),
                RecallError::Embedding(msg) => ConchError::Embed(EmbedError::Other(msg)),
            })
    }

    pub fn forget_by_subject(&self, subject: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_subject(subject)?)
    }

    pub fn forget_by_id(&self, id: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_id(id)?)
    }

    pub fn forget_older_than(&self, secs: i64) -> Result<usize, ConchError> {
        Ok(self.store.forget_older_than(Duration::seconds(secs))?)
    }

    pub fn decay(&self) -> Result<DecayResult, ConchError> {
        Ok(decay::run_decay(&self.store, None, None)?)
    }

    pub fn stats(&self) -> Result<MemoryStats, ConchError> {
        Ok(self.store.stats()?)
    }

    pub fn embed_all(&self) -> Result<usize, ConchError> {
        let missing = self.store.memories_missing_embeddings()?;
        if missing.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = missing.iter().map(|m| m.text_for_embedding()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embedder.embed(&text_refs)?;
        for (mem, emb) in missing.iter().zip(embeddings.iter()) {
            self.store.update_embedding(mem.id, emb)?;
        }
        Ok(missing.len())
    }

    pub fn export(&self) -> Result<ExportData, ConchError> {
        let memories = self.store.all_memories()?;
        Ok(ExportData { memories })
    }

    pub fn import(&self, data: &ExportData) -> Result<usize, ConchError> {
        let mut count = 0;
        for mem in &data.memories {
            let created = mem.created_at.to_rfc3339();
            let accessed = mem.last_accessed_at.to_rfc3339();
            match &mem.kind {
                MemoryKind::Fact(f) => {
                    self.store.import_fact(
                        &f.subject, &f.relation, &f.object,
                        mem.strength, mem.embedding.as_deref(),
                        &created, &accessed, mem.access_count,
                    )?;
                }
                MemoryKind::Episode(e) => {
                    self.store.import_episode(
                        &e.text, mem.strength, mem.embedding.as_deref(),
                        &created, &accessed, mem.access_count,
                    )?;
                }
            }
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embed::Embedding;

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
        fn dimension(&self) -> usize { 2 }
    }

    #[test]
    fn duplicate_fact_reinforces_instead_of_inserting() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let first = db.remember_fact("Rust", "is", "great").unwrap();
        assert_eq!(first.access_count, 0);
        assert!((first.strength - 1.0).abs() < f64::EPSILON);

        let second = db.remember_fact("Rust", "is", "great").unwrap();
        // Should be the same record, reinforced
        assert_eq!(second.id, first.id);
        assert_eq!(second.access_count, 1);
        assert!((second.strength - 1.0).abs() < f64::EPSILON); // capped at 1.0

        // Only one memory should exist
        let stats = db.stats().unwrap();
        assert_eq!(stats.total_facts, 1);
    }

    #[test]
    fn duplicate_fact_boosts_strength_when_decayed() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let first = db.remember_fact("Rust", "is", "great").unwrap();
        // Manually decay the strength
        db.store().conn().execute(
            "UPDATE memories SET strength = 0.5 WHERE id = ?1",
            rusqlite::params![first.id],
        ).unwrap();

        let second = db.remember_fact("Rust", "is", "great").unwrap();
        assert_eq!(second.id, first.id);
        assert!((second.strength - 0.6).abs() < 1e-6); // 0.5 + 0.1
    }

    #[test]
    fn different_facts_are_not_duplicates() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let first = db.remember_fact("Rust", "is", "great").unwrap();
        let second = db.remember_fact("Rust", "is", "fast").unwrap();
        assert_ne!(first.id, second.id);

        let stats = db.stats().unwrap();
        assert_eq!(stats.total_facts, 2);
    }

    #[test]
    fn duplicate_episode_reinforces_instead_of_inserting() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let first = db.remember_episode("had coffee this morning").unwrap();
        assert_eq!(first.access_count, 0);

        let second = db.remember_episode("had coffee this morning").unwrap();
        assert_eq!(second.id, first.id);
        assert_eq!(second.access_count, 1);

        let stats = db.stats().unwrap();
        assert_eq!(stats.total_episodes, 1);
    }

    #[test]
    fn different_episodes_are_not_duplicates() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let first = db.remember_episode("had coffee").unwrap();
        let second = db.remember_episode("had tea").unwrap();
        assert_ne!(first.id, second.id);

        let stats = db.stats().unwrap();
        assert_eq!(stats.total_episodes, 2);
    }
}
