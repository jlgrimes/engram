pub mod decay;
pub mod embed;
pub mod memory;
pub mod recall;
mod recall_scoring;
pub mod store;

pub use decay::{run_decay, run_decay_in, DecayResult};
pub use embed::{cosine_similarity, EmbedError, Embedder, FastEmbedder, SharedEmbedder};
pub use memory::{Episode, ExportData, Fact, MemoryKind, MemoryRecord, MemoryStats};
pub use recall::{
    recall, recall_with_filter, recall_with_filter_in, RecallError, RecallKindFilter, RecallResult,
};
pub use store::{MemoryStore, DEFAULT_NAMESPACE};

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
        Ok(Self {
            store,
            embedder: Box::new(embedder),
        })
    }

    pub fn open_in_memory_with(embedder: Box<dyn Embedder>) -> Result<Self, ConchError> {
        let store = MemoryStore::open_in_memory()?;
        Ok(Self { store, embedder })
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn remember_fact(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> Result<MemoryRecord, ConchError> {
        self.remember_fact_in(DEFAULT_NAMESPACE, subject, relation, object)
    }

    pub fn remember_fact_in(
        &self,
        namespace: &str,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> Result<MemoryRecord, ConchError> {
        // Check for existing duplicate in namespace only
        if let Some(existing) = self
            .store
            .find_fact_in(namespace, subject, relation, object)?
        {
            self.store.reinforce_memory(existing.id)?;
            return Ok(self
                .store
                .get_memory(existing.id)?
                .expect("just reinforced"));
        }
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;
        let id =
            self.store
                .remember_fact_in(namespace, subject, relation, object, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    pub fn remember_episode(&self, text: &str) -> Result<MemoryRecord, ConchError> {
        self.remember_episode_in(DEFAULT_NAMESPACE, text)
    }

    pub fn remember_episode_in(
        &self,
        namespace: &str,
        text: &str,
    ) -> Result<MemoryRecord, ConchError> {
        // Check for existing duplicate in namespace only
        if let Some(existing) = self.store.find_episode_in(namespace, text)? {
            self.store.reinforce_memory(existing.id)?;
            return Ok(self
                .store
                .get_memory(existing.id)?
                .expect("just reinforced"));
        }
        let embedding = self.embedder.embed_one(text)?;
        let id = self
            .store
            .remember_episode_in(namespace, text, Some(&embedding))?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallResult>, ConchError> {
        self.recall_in(DEFAULT_NAMESPACE, query, limit)
    }

    pub fn recall_in(
        &self,
        namespace: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall_with_filter_in(
            &self.store,
            namespace,
            query,
            self.embedder.as_ref(),
            limit,
            RecallKindFilter::All,
        )
        .map_err(|e| match e {
            RecallError::Db(e) => ConchError::Db(e),
            RecallError::Embedding(msg) => ConchError::Embed(EmbedError::Other(msg)),
        })
    }

    pub fn recall_filtered(
        &self,
        query: &str,
        limit: usize,
        kind: RecallKindFilter,
    ) -> Result<Vec<RecallResult>, ConchError> {
        self.recall_filtered_in(DEFAULT_NAMESPACE, query, limit, kind)
    }

    pub fn recall_filtered_in(
        &self,
        namespace: &str,
        query: &str,
        limit: usize,
        kind: RecallKindFilter,
    ) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall_with_filter_in(
            &self.store,
            namespace,
            query,
            self.embedder.as_ref(),
            limit,
            kind,
        )
        .map_err(|e| match e {
            RecallError::Db(e) => ConchError::Db(e),
            RecallError::Embedding(msg) => ConchError::Embed(EmbedError::Other(msg)),
        })
    }

    pub fn forget_by_subject(&self, subject: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_subject(subject)?)
    }

    pub fn forget_by_subject_in(
        &self,
        namespace: &str,
        subject: &str,
    ) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_subject_in(namespace, subject)?)
    }

    pub fn forget_by_id(&self, id: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_id(id)?)
    }

    pub fn forget_older_than(&self, secs: i64) -> Result<usize, ConchError> {
        Ok(self.store.forget_older_than(Duration::seconds(secs))?)
    }

    pub fn forget_older_than_in(&self, namespace: &str, secs: i64) -> Result<usize, ConchError> {
        Ok(self
            .store
            .forget_older_than_in(namespace, Duration::seconds(secs))?)
    }

    pub fn decay(&self) -> Result<DecayResult, ConchError> {
        self.decay_in(DEFAULT_NAMESPACE)
    }

    pub fn decay_in(&self, namespace: &str) -> Result<DecayResult, ConchError> {
        Ok(decay::run_decay_in(&self.store, namespace, None, None)?)
    }

    pub fn stats(&self) -> Result<MemoryStats, ConchError> {
        Ok(self.store.stats()?)
    }

    pub fn stats_in(&self, namespace: &str) -> Result<MemoryStats, ConchError> {
        Ok(self.store.stats_in(namespace)?)
    }

    pub fn embed_all(&self) -> Result<usize, ConchError> {
        self.embed_all_in(DEFAULT_NAMESPACE)
    }

    pub fn embed_all_in(&self, namespace: &str) -> Result<usize, ConchError> {
        let missing = self.store.memories_missing_embeddings_in(namespace)?;
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
                        &mem.namespace,
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
                        &mem.namespace,
                        &e.text,
                        mem.strength,
                        mem.embedding.as_deref(),
                        &created,
                        &accessed,
                        mem.access_count,
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
        fn dimension(&self) -> usize {
            2
        }
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
        db.store()
            .conn()
            .execute(
                "UPDATE memories SET strength = 0.5 WHERE id = ?1",
                rusqlite::params![first.id],
            )
            .unwrap();

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

    #[test]
    fn same_fact_in_different_namespaces_does_not_dedupe() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let a = db
            .remember_fact_in("agent-a", "Rust", "is", "great")
            .unwrap();
        let b = db
            .remember_fact_in("agent-b", "Rust", "is", "great")
            .unwrap();

        assert_ne!(a.id, b.id);
        assert_eq!(db.stats_in("agent-a").unwrap().total_facts, 1);
        assert_eq!(db.stats_in("agent-b").unwrap().total_facts, 1);
    }

    #[test]
    fn recall_is_namespace_isolated() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        db.remember_episode_in("team-a", "alpha context").unwrap();
        db.remember_episode_in("team-b", "bravo context").unwrap();

        let a = db.recall_in("team-a", "alpha", 10).unwrap();
        assert!(!a.is_empty());
        assert!(a.iter().all(|r| r.memory.namespace == "team-a"));

        let b = db.recall_in("team-b", "bravo", 10).unwrap();
        assert!(!b.is_empty());
        assert!(b.iter().all(|r| r.memory.namespace == "team-b"));
    }

    #[test]
    fn default_namespace_wrappers_still_work() {
        let db = ConchDB::open_in_memory_with(Box::new(MockEmbedder)).unwrap();

        let mem = db.remember_fact("Jared", "builds", "conch").unwrap();
        assert_eq!(mem.namespace, DEFAULT_NAMESPACE);

        let recalled = db.recall("Jared", 5).unwrap();
        assert!(!recalled.is_empty());
        assert!(recalled
            .iter()
            .all(|r| r.memory.namespace == DEFAULT_NAMESPACE));
    }
}
