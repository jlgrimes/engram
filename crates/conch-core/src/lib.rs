pub mod memory;
pub mod store;
pub mod embed;
pub mod decay;
pub mod recall;
pub mod consolidate;
pub mod importance;

pub use memory::{Episode, ExportData, Fact, MemoryKind, MemoryRecord, MemoryStats, RememberResult};
pub use store::MemoryStore;
pub use embed::{Embedder, EmbedError, FastEmbedder, SharedEmbedder, cosine_similarity};
pub use decay::{run_decay, DecayResult};
pub use recall::{recall, recall_with_tag_filter, RecallResult, RecallError};
pub use consolidate::{consolidate, find_clusters, ConsolidateResult, ConsolidateCluster};
pub use importance::{compute_importance, score_all as score_importance, list_importance, ImportanceInfo};

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
        self.remember_fact_with_tags(subject, relation, object, &[])
    }

    pub fn remember_fact_with_tags(&self, subject: &str, relation: &str, object: &str, tags: &[String]) -> Result<MemoryRecord, ConchError> {
        self.remember_fact_full(subject, relation, object, tags, None, None, None)
    }

    pub fn remember_fact_full(
        &self, subject: &str, relation: &str, object: &str, tags: &[String],
        source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> Result<MemoryRecord, ConchError> {
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;
        let id = self.store.remember_fact_full(subject, relation, object, Some(&embedding), tags, source, session_id, channel)?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    /// Upsert a fact: if a fact with the same subject+relation exists, update
    /// its object. Otherwise insert a new fact.
    /// Returns `(record, was_updated)`.
    pub fn upsert_fact(&self, subject: &str, relation: &str, object: &str) -> Result<(MemoryRecord, bool), ConchError> {
        self.upsert_fact_with_tags(subject, relation, object, &[])
    }

    pub fn upsert_fact_with_tags(&self, subject: &str, relation: &str, object: &str, tags: &[String]) -> Result<(MemoryRecord, bool), ConchError> {
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;
        let (id, was_updated) = self.store.upsert_fact(subject, relation, object, Some(&embedding), tags, None, None, None)?;
        Ok((self.store.get_memory(id)?.expect("just upserted"), was_updated))
    }

    pub fn remember_episode(&self, text: &str) -> Result<MemoryRecord, ConchError> {
        self.remember_episode_with_tags(text, &[])
    }

    pub fn remember_episode_with_tags(&self, text: &str, tags: &[String]) -> Result<MemoryRecord, ConchError> {
        self.remember_episode_full(text, tags, None, None, None)
    }

    pub fn remember_episode_full(
        &self, text: &str, tags: &[String],
        source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> Result<MemoryRecord, ConchError> {
        let embedding = self.embedder.embed_one(text)?;
        let id = self.store.remember_episode_full(text, Some(&embedding), tags, source, session_id, channel)?;
        Ok(self.store.get_memory(id)?.expect("just inserted"))
    }

    // ── Dedup-aware remember ──────────────────────────────────

    /// Cosine similarity threshold for dedup. Memories with similarity > this
    /// value are considered duplicates and merged instead of inserted.
    const DEDUP_SIMILARITY_THRESHOLD: f32 = 0.95;

    /// Strength boost applied when reinforcing a duplicate memory.
    const DEDUP_REINFORCE_BOOST: f64 = 0.10;

    /// Check if a new embedding is a duplicate of any existing memory.
    /// Returns the (id, similarity) of the best match above threshold, if any.
    fn find_duplicate(&self, embedding: &[f32]) -> Result<Option<(i64, f32)>, ConchError> {
        self.find_duplicate_excluding(embedding, -1)
    }

    fn find_duplicate_excluding(&self, embedding: &[f32], exclude_id: i64) -> Result<Option<(i64, f32)>, ConchError> {
        let all = self.store.all_embeddings()?;
        let mut best: Option<(i64, f32)> = None;
        for (id, existing_emb) in &all {
            if *id == exclude_id {
                continue;
            }
            let sim = cosine_similarity(embedding, existing_emb);
            if sim > Self::DEDUP_SIMILARITY_THRESHOLD {
                if best.is_none() || sim > best.unwrap().1 {
                    best = Some((*id, sim));
                }
            }
        }
        Ok(best)
    }

    /// Store a fact with dedup check. If a near-duplicate exists (cosine sim > 0.95),
    /// the existing memory is reinforced instead of creating a new one.
    pub fn remember_fact_dedup(&self, subject: &str, relation: &str, object: &str) -> Result<RememberResult, ConchError> {
        self.remember_fact_dedup_with_tags(subject, relation, object, &[])
    }

    /// Store a fact with dedup check and tags.
    pub fn remember_fact_dedup_with_tags(&self, subject: &str, relation: &str, object: &str, tags: &[String]) -> Result<RememberResult, ConchError> {
        self.remember_fact_dedup_full(subject, relation, object, tags, None, None, None)
    }

    /// Store a fact with upsert + dedup check, tags, and source tracking.
    ///
    /// Pipeline:
    /// 1. If a fact with the same subject+relation exists, update its object (upsert).
    /// 2. Otherwise, check for near-duplicate embeddings (cosine sim > 0.95).
    /// 3. If neither, create a new fact.
    pub fn remember_fact_dedup_full(
        &self, subject: &str, relation: &str, object: &str, tags: &[String],
        source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> Result<RememberResult, ConchError> {
        let text = format!("{subject} {relation} {object}");
        let embedding = self.embedder.embed_one(&text)?;

        // Step 1: Upsert — check for existing fact with same subject+relation
        let (id, was_updated) = self.store.upsert_fact(
            subject, relation, object, Some(&embedding), tags, source, session_id, channel,
        )?;
        if was_updated {
            let record = self.store.get_memory(id)?.expect("just upserted");
            return Ok(RememberResult::Updated(record));
        }
        // upsert_fact inserted a new row — but we should still check for dedup
        // against other memories. If we find a near-duplicate, delete the just-inserted
        // row and reinforce the duplicate instead.
        if let Some((existing_id, similarity)) = self.find_duplicate_excluding(&embedding, id)? {
            // Remove the just-inserted row and reinforce the duplicate
            self.store.forget_by_id(&id.to_string())?;
            self.store.reinforce_memory(existing_id, Self::DEDUP_REINFORCE_BOOST)?;
            let existing = self.store.get_memory(existing_id)?.expect("just reinforced");
            return Ok(RememberResult::Duplicate { existing, similarity });
        }

        let record = self.store.get_memory(id)?.expect("just inserted");
        Ok(RememberResult::Created(record))
    }

    /// Store an episode with dedup check. If a near-duplicate exists (cosine sim > 0.95),
    /// the existing memory is reinforced instead of creating a new one.
    pub fn remember_episode_dedup(&self, text: &str) -> Result<RememberResult, ConchError> {
        self.remember_episode_dedup_with_tags(text, &[])
    }

    /// Store an episode with dedup check and tags.
    pub fn remember_episode_dedup_with_tags(&self, text: &str, tags: &[String]) -> Result<RememberResult, ConchError> {
        self.remember_episode_dedup_full(text, tags, None, None, None)
    }

    /// Store an episode with dedup check, tags, and source tracking.
    pub fn remember_episode_dedup_full(
        &self, text: &str, tags: &[String],
        source: Option<&str>, session_id: Option<&str>, channel: Option<&str>,
    ) -> Result<RememberResult, ConchError> {
        let embedding = self.embedder.embed_one(text)?;

        if let Some((existing_id, similarity)) = self.find_duplicate(&embedding)? {
            self.store.reinforce_memory(existing_id, Self::DEDUP_REINFORCE_BOOST)?;
            let existing = self.store.get_memory(existing_id)?.expect("just reinforced");
            return Ok(RememberResult::Duplicate { existing, similarity });
        }

        let id = self.store.remember_episode_full(text, Some(&embedding), tags, source, session_id, channel)?;
        let record = self.store.get_memory(id)?.expect("just inserted");
        Ok(RememberResult::Created(record))
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallResult>, ConchError> {
        self.recall_with_tag(query, limit, None)
    }

    pub fn recall_with_tag(&self, query: &str, limit: usize, tag: Option<&str>) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall_with_tag_filter(&self.store, query, self.embedder.as_ref(), limit, tag)
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

    pub fn consolidate(&self, dry_run: bool) -> Result<ConsolidateResult, ConchError> {
        if dry_run {
            let clusters = find_clusters(&self.store, None)?;
            Ok(ConsolidateResult {
                clusters: clusters.len(),
                archived: clusters.iter().map(|c| c.duplicates.len()).sum(),
                boosted: clusters.len(),
            })
        } else {
            Ok(consolidate::consolidate(&self.store, None)?)
        }
    }

    pub fn consolidate_clusters(&self) -> Result<Vec<ConsolidateCluster>, ConchError> {
        Ok(find_clusters(&self.store, None)?)
    }

    pub fn score_importance(&self) -> Result<usize, ConchError> {
        Ok(importance::score_all(&self.store)?)
    }

    pub fn list_importance(&self) -> Result<Vec<ImportanceInfo>, ConchError> {
        Ok(importance::list_importance(&self.store)?)
    }

    pub fn set_importance(&self, id: i64, importance: f64) -> Result<(), ConchError> {
        self.store.update_importance(id, importance)?;
        Ok(())
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
                        &mem.tags,
                        mem.source.as_deref(),
                        mem.session_id.as_deref(),
                        mem.channel.as_deref(),
                    )?;
                }
                MemoryKind::Episode(e) => {
                    self.store.import_episode(
                        &e.text, mem.strength, mem.embedding.as_deref(),
                        &created, &accessed, mem.access_count,
                        &mem.tags,
                        mem.source.as_deref(),
                        mem.session_id.as_deref(),
                        mem.channel.as_deref(),
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
    use crate::embed::{EmbedError, Embedding};


    /// Mock embedder where all texts produce the exact same embedding.
    /// This guarantees cosine similarity = 1.0 for any pair of texts.
    struct IdenticalEmbedder;

    impl Embedder for IdenticalEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }

        fn dimension(&self) -> usize { 4 }
    }

    /// Mock embedder that produces orthogonal embeddings for each call.
    /// This guarantees cosine similarity = 0.0 between different texts.
    struct OrthogonalEmbedder {
        counter: std::sync::atomic::AtomicUsize,
    }

    impl OrthogonalEmbedder {
        fn new() -> Self {
            Self { counter: std::sync::atomic::AtomicUsize::new(0) }
        }
    }

    impl Embedder for OrthogonalEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts.iter().map(|_| {
                let i = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut emb = vec![0.0; 8];
                emb[i % 8] = 1.0;
                emb
            }).collect())
        }

        fn dimension(&self) -> usize { 8 }
    }

    #[test]
    fn dedup_detects_identical_embedding() {
        // All texts produce the same embedding. For facts with same subject+relation,
        // upsert takes priority over dedup.
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_fact_dedup("Jared", "likes", "Rust").unwrap();
        assert!(!r1.is_duplicate(), "first insert should not be duplicate");
        assert!(!r1.is_updated(), "first insert should not be updated");

        // Same subject+relation → upsert (not dedup)
        let r2 = db.remember_fact_dedup("Jared", "likes", "Rust").unwrap();
        assert!(r2.is_updated(), "second identical fact should be upserted");

        // Only 1 memory should exist in the database
        let stats = db.stats().unwrap();
        assert_eq!(stats.total_memories, 1, "should have 1 memory, not 2");
    }

    #[test]
    fn dedup_detects_identical_episode_embedding() {
        // Episodes don't have upsert, so dedup should still work.
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_episode_dedup("Meeting notes from standup").unwrap();
        assert!(!r1.is_duplicate(), "first insert should not be duplicate");

        let r2 = db.remember_episode_dedup("Meeting notes from standup").unwrap();
        assert!(r2.is_duplicate(), "second identical episode should be duplicate");

        if let RememberResult::Duplicate { similarity, .. } = r2 {
            assert!(similarity > 0.95, "similarity should be > 0.95, got {similarity}");
        }

        let stats = db.stats().unwrap();
        assert_eq!(stats.total_memories, 1, "should have 1 memory, not 2");
    }

    #[test]
    fn dedup_allows_different_memories() {
        // Orthogonal embeddings → cosine sim = 0.0 → both should be inserted.
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();

        let r1 = db.remember_fact_dedup("Jared", "likes", "Rust").unwrap();
        assert!(!r1.is_duplicate());

        let r2 = db.remember_episode_dedup("Had coffee this morning").unwrap();
        assert!(!r2.is_duplicate());

        let stats = db.stats().unwrap();
        assert_eq!(stats.total_memories, 2, "both memories should be stored");
    }

    #[test]
    fn dedup_reinforces_strength_and_bumps_access_count() {
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        // First insert: strength = 1.0, access_count = 0
        let r1 = db.remember_episode_dedup("Meeting notes from standup").unwrap();
        let initial = r1.memory().clone();
        assert_eq!(initial.access_count, 0);

        // Second insert: duplicate detected → reinforced
        let r2 = db.remember_episode_dedup("Meeting notes from standup").unwrap();
        assert!(r2.is_duplicate());
        let reinforced = r2.memory();
        assert_eq!(reinforced.id, initial.id, "should reinforce same memory");
        assert_eq!(reinforced.access_count, initial.access_count + 1);
        // strength should still be 1.0 (was 1.0 + 0.10 clamped to 1.0)
        assert!((reinforced.strength - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn dedup_reinforces_decayed_memory() {
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_episode_dedup("Important project context").unwrap();
        let id = r1.memory().id;

        // Manually decay the memory's strength
        db.store().conn().execute(
            "UPDATE memories SET strength = 0.5 WHERE id = ?1",
            rusqlite::params![id],
        ).unwrap();

        // Second insert should reinforce (0.5 + 0.10 = 0.6)
        let r2 = db.remember_episode_dedup("Important project context").unwrap();
        assert!(r2.is_duplicate());
        let reinforced = r2.memory();
        assert!((reinforced.strength - 0.6).abs() < 0.01,
            "strength should be ~0.6 after reinforcement, got {}", reinforced.strength);
    }

    #[test]
    fn dedup_episode_detected_as_duplicate_of_fact() {
        // With IdenticalEmbedder, even a fact and episode will have same embedding
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_fact_dedup("Jared", "prefers", "Rust").unwrap();
        assert!(!r1.is_duplicate());

        // Episode with same embedding should be detected as duplicate
        let r2 = db.remember_episode_dedup("Jared prefers Rust").unwrap();
        assert!(r2.is_duplicate(), "episode matching a fact should be detected as duplicate");
        assert_eq!(r2.memory().id, r1.memory().id);
    }

    #[test]
    fn dedup_with_empty_db_always_creates() {
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_fact_dedup("first", "memory", "ever").unwrap();
        assert!(!r1.is_duplicate(), "first memory in empty DB should always be created");
    }

    #[test]
    fn remember_result_memory_accessor() {
        let db = ConchDB::open_in_memory_with(Box::new(IdenticalEmbedder)).unwrap();

        let r1 = db.remember_fact_dedup("A", "B", "C").unwrap();
        assert!(r1.memory().id > 0);

        let r2 = db.remember_fact_dedup("A", "B", "C").unwrap();
        assert!(r2.memory().id > 0);
        assert_eq!(r1.memory().id, r2.memory().id);
    }

    #[test]
    fn store_all_embeddings_returns_correct_count() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "B", "C", Some(&[1.0, 0.0])).unwrap();
        store.remember_episode("test", Some(&[0.0, 1.0])).unwrap();
        store.remember_episode("no embedding", None).unwrap();

        let embeddings = store.all_embeddings().unwrap();
        assert_eq!(embeddings.len(), 2, "should only return memories with embeddings");
    }

    #[test]
    fn store_reinforce_memory_boosts_strength() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "B", "C", Some(&[1.0, 0.0])).unwrap();

        // Manually set low strength
        store.conn().execute(
            "UPDATE memories SET strength = 0.3 WHERE id = ?1",
            rusqlite::params![id],
        ).unwrap();

        store.reinforce_memory(id, 0.10).unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        assert!((mem.strength - 0.4).abs() < 0.01, "strength should be ~0.4, got {}", mem.strength);
        assert_eq!(mem.access_count, 1);
    }

    #[test]
    fn store_reinforce_memory_clamps_to_1() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "B", "C", Some(&[1.0, 0.0])).unwrap();

        // strength starts at 1.0, boost by 0.5 should still be 1.0
        store.reinforce_memory(id, 0.5).unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        assert!((mem.strength - 1.0).abs() < f64::EPSILON, "strength should be clamped to 1.0");
    }

    // ── Upsert integration tests ────────────────────────────

    #[test]
    fn upsert_via_dedup_updates_existing_fact() {
        // Orthogonal embedder ensures dedup doesn't fire; only upsert should trigger
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();

        let r1 = db.remember_fact_dedup_full("Jared", "favorite_color", "blue", &[], None, None, None).unwrap();
        assert!(!r1.is_duplicate());
        assert!(!r1.is_updated());

        let r2 = db.remember_fact_dedup_full("Jared", "favorite_color", "green", &[], None, None, None).unwrap();
        assert!(r2.is_updated(), "same subject+relation should trigger upsert");

        let mem = r2.memory();
        if let MemoryKind::Fact(f) = &mem.kind {
            assert_eq!(f.object, "green", "object should be updated to green");
        } else { panic!("expected fact"); }

        // Should still only have 1 memory
        let stats = db.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }

    #[test]
    fn upsert_different_subject_creates_new() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();

        db.remember_fact_dedup_full("Jared", "likes", "Rust", &[], None, None, None).unwrap();
        let r2 = db.remember_fact_dedup_full("Alice", "likes", "Python", &[], None, None, None).unwrap();
        assert!(!r2.is_updated(), "different subject should not trigger upsert");
        assert!(!r2.is_duplicate(), "orthogonal embeddings should not trigger dedup");

        assert_eq!(db.stats().unwrap().total_memories, 2);
    }
}
