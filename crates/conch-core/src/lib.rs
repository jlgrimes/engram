pub mod memory;
pub mod store;
pub mod embed;
pub mod decay;
pub mod recall;

pub use memory::{Episode, ExportData, Fact, MemoryKind, MemoryRecord, MemoryStats, RememberResult, AuditEntry, VerifyResult, CorruptedMemory};
pub use store::MemoryStore;
pub use embed::{Embedder, EmbedError, FastEmbedder, SharedEmbedder, cosine_similarity};
pub use decay::{run_decay, DecayResult};
pub use recall::{recall, recall_with_tag_filter, RecallResult, RecallError};

use chrono::Duration;

/// High-level API wrapping storage + embeddings.
pub struct ConchDB {
    store: MemoryStore,
    embedder: Box<dyn Embedder>,
    namespace: String,
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
        Self::open_with_namespace(path, "default")
    }

    pub fn open_with_namespace(path: &str, namespace: &str) -> Result<Self, ConchError> {
        let store = MemoryStore::open(path)?;
        let embedder = embed::FastEmbedder::new()?;
        Ok(Self { store, embedder: Box::new(embedder), namespace: namespace.to_string() })
    }

    pub fn open_in_memory_with(embedder: Box<dyn Embedder>) -> Result<Self, ConchError> {
        Self::open_in_memory_with_namespace(embedder, "default")
    }

    pub fn open_in_memory_with_namespace(embedder: Box<dyn Embedder>, namespace: &str) -> Result<Self, ConchError> {
        let store = MemoryStore::open_in_memory()?;
        Ok(Self { store, embedder, namespace: namespace.to_string() })
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
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
        let id = self.store.remember_fact_ns(subject, relation, object, Some(&embedding), tags, source, session_id, channel, &self.namespace)?;
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
        let (id, was_updated) = self.store.upsert_fact_ns(subject, relation, object, Some(&embedding), tags, None, None, None, &self.namespace)?;
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
        let id = self.store.remember_episode_ns(text, Some(&embedding), tags, source, session_id, channel, &self.namespace)?;
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
        let all = self.store.all_embeddings_ns(&self.namespace)?;
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
        let (id, was_updated) = self.store.upsert_fact_ns(
            subject, relation, object, Some(&embedding), tags, source, session_id, channel, &self.namespace,
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

        let id = self.store.remember_episode_ns(text, Some(&embedding), tags, source, session_id, channel, &self.namespace)?;
        let record = self.store.get_memory(id)?.expect("just inserted");
        Ok(RememberResult::Created(record))
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallResult>, ConchError> {
        self.recall_with_tag(query, limit, None)
    }

    pub fn recall_with_tag(&self, query: &str, limit: usize, tag: Option<&str>) -> Result<Vec<RecallResult>, ConchError> {
        recall::recall_with_tag_filter_ns(&self.store, query, self.embedder.as_ref(), limit, tag, &self.namespace)
            .map_err(|e| match e {
                RecallError::Db(e) => ConchError::Db(e),
                RecallError::Embedding(msg) => ConchError::Embed(EmbedError::Other(msg)),
            })
    }

    pub fn forget_by_subject(&self, subject: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_subject_ns(subject, &self.namespace)?)
    }

    pub fn forget_by_id(&self, id: &str) -> Result<usize, ConchError> {
        Ok(self.store.forget_by_id(id)?)
    }

    pub fn forget_older_than(&self, secs: i64) -> Result<usize, ConchError> {
        Ok(self.store.forget_older_than_ns(Duration::seconds(secs), &self.namespace)?)
    }

    pub fn decay(&self) -> Result<DecayResult, ConchError> {
        Ok(decay::run_decay_ns(&self.store, None, None, &self.namespace)?)
    }

    pub fn stats(&self) -> Result<MemoryStats, ConchError> {
        Ok(self.store.stats_ns(&self.namespace)?)
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
        let memories = self.store.all_memories_ns(&self.namespace)?;
        Ok(ExportData { memories })
    }

    pub fn import(&self, data: &ExportData) -> Result<usize, ConchError> {
        let mut count = 0;
        for mem in &data.memories {
            let created = mem.created_at.to_rfc3339();
            let accessed = mem.last_accessed_at.to_rfc3339();
            match &mem.kind {
                MemoryKind::Fact(f) => {
                    self.store.import_fact_ns(
                        &f.subject, &f.relation, &f.object,
                        mem.strength, mem.embedding.as_deref(),
                        &created, &accessed, mem.access_count,
                        &mem.tags,
                        mem.source.as_deref(),
                        mem.session_id.as_deref(),
                        mem.channel.as_deref(),
                        &self.namespace,
                    )?;
                }
                MemoryKind::Episode(e) => {
                    self.store.import_episode_ns(
                        &e.text, mem.strength, mem.embedding.as_deref(),
                        &created, &accessed, mem.access_count,
                        &mem.tags,
                        mem.source.as_deref(),
                        mem.session_id.as_deref(),
                        mem.channel.as_deref(),
                        &self.namespace,
                    )?;
                }
            }
            count += 1;
        }
        Ok(count)
    }

    // ── Security: Audit Log ─────────────────────────────────

    pub fn audit_log(&self, limit: usize, memory_id: Option<i64>, actor: Option<&str>) -> Result<Vec<AuditEntry>, ConchError> {
        Ok(self.store.get_audit_log(limit, memory_id, actor)?)
    }

    // ── Security: Verify ────────────────────────────────────

    pub fn verify(&self) -> Result<VerifyResult, ConchError> {
        Ok(self.store.verify_integrity_ns(&self.namespace)?)
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

    // ── Security: Namespace isolation tests ──────────────────

    #[test]
    fn namespace_isolation_facts() {
        // Test at the store level since we can't share store between ConchDB instances
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("X", "is", "A", None, &[], None, None, None, "ns-a").unwrap();
        store.remember_fact_ns("Y", "is", "B", None, &[], None, None, None, "ns-b").unwrap();

        let stats_a = store.stats_ns("ns-a").unwrap();
        let stats_b = store.stats_ns("ns-b").unwrap();
        assert_eq!(stats_a.total_memories, 1);
        assert_eq!(stats_b.total_memories, 1);

        // Default namespace should be empty
        let stats_default = store.stats_ns("default").unwrap();
        assert_eq!(stats_default.total_memories, 0, "default namespace should be empty");

        // Namespace-scoped queries only return their own memories
        let ns_a_mems = store.all_memories_ns("ns-a").unwrap();
        let ns_b_mems = store.all_memories_ns("ns-b").unwrap();
        assert_eq!(ns_a_mems.len(), 1);
        assert_eq!(ns_b_mems.len(), 1);
        assert_ne!(ns_a_mems[0].id, ns_b_mems[0].id);
    }

    #[test]
    fn namespace_isolation_recall() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact_ns("Jared", "likes", "Rust", Some(&[1.0, 0.0]), &[], None, None, None, "ns-a").unwrap();
        store.remember_fact_ns("Alice", "likes", "Python", Some(&[0.0, 1.0]), &[], None, None, None, "ns-b").unwrap();

        let memories_a = store.all_memories_with_text_ns("ns-a").unwrap();
        let memories_b = store.all_memories_with_text_ns("ns-b").unwrap();
        assert_eq!(memories_a.len(), 1);
        assert_eq!(memories_b.len(), 1);
        assert_ne!(memories_a[0].0.id, memories_b[0].0.id);
    }

    #[test]
    fn namespace_upsert_scoped() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.upsert_fact_ns("Jared", "color", "blue", None, &[], None, None, None, "ns-a").unwrap();
        store.upsert_fact_ns("Jared", "color", "red", None, &[], None, None, None, "ns-b").unwrap();

        // Both should exist (different namespaces)
        let all_a = store.all_memories_ns("ns-a").unwrap();
        let all_b = store.all_memories_ns("ns-b").unwrap();
        assert_eq!(all_a.len(), 1);
        assert_eq!(all_b.len(), 1);
        if let MemoryKind::Fact(f) = &all_a[0].kind { assert_eq!(f.object, "blue"); } else { panic!(); }
        if let MemoryKind::Fact(f) = &all_b[0].kind { assert_eq!(f.object, "red"); } else { panic!(); }

        // Upsert within ns-a should update only ns-a
        store.upsert_fact_ns("Jared", "color", "green", None, &[], None, None, None, "ns-a").unwrap();
        let all_a = store.all_memories_ns("ns-a").unwrap();
        assert_eq!(all_a.len(), 1);
        if let MemoryKind::Fact(f) = &all_a[0].kind { assert_eq!(f.object, "green"); } else { panic!(); }
        // ns-b unchanged
        let all_b = store.all_memories_ns("ns-b").unwrap();
        if let MemoryKind::Fact(f) = &all_b[0].kind { assert_eq!(f.object, "red"); } else { panic!(); }
    }

    // ── Security: Audit log tests ───────────────────────────

    #[test]
    fn audit_log_records_remember() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();
        db.remember_fact("Jared", "likes", "Rust").unwrap();

        let log = db.audit_log(10, None, None).unwrap();
        assert!(!log.is_empty(), "audit log should have entries");
        assert!(log.iter().any(|e| e.action == "remember"), "should have a remember action");
    }

    #[test]
    fn audit_log_records_forget() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();
        let mem = db.remember_fact("Jared", "likes", "Rust").unwrap();
        db.forget_by_id(&mem.id.to_string()).unwrap();

        let log = db.audit_log(10, None, None).unwrap();
        assert!(log.iter().any(|e| e.action == "forget"), "should have a forget action");
    }

    #[test]
    fn audit_log_filter_by_memory_id() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();
        let m1 = db.remember_fact("A", "B", "C").unwrap();
        db.remember_fact("D", "E", "F").unwrap();

        let log = db.audit_log(10, Some(m1.id), None).unwrap();
        for entry in &log {
            assert_eq!(entry.memory_id, Some(m1.id));
        }
    }

    // ── Security: Checksum & verify tests ───────────────────

    #[test]
    fn checksum_stored_on_remember() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("Jared", "likes", "Rust", None).unwrap();
        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.checksum.is_some(), "checksum should be set on remember");
    }

    #[test]
    fn verify_passes_for_clean_data() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();
        db.remember_fact("Jared", "likes", "Rust").unwrap();
        db.remember_episode("had coffee").unwrap();

        let result = db.verify().unwrap();
        assert_eq!(result.total_checked, 2);
        assert_eq!(result.valid, 2);
        assert!(result.corrupted.is_empty());
        assert_eq!(result.missing_checksum, 0);
    }

    #[test]
    fn verify_detects_corruption() {
        let db = ConchDB::open_in_memory_with(Box::new(OrthogonalEmbedder::new())).unwrap();
        let mem = db.remember_fact("Jared", "likes", "Rust").unwrap();

        // Corrupt the data by changing the object directly in SQL
        db.store().conn().execute(
            "UPDATE memories SET object = 'Python' WHERE id = ?1",
            rusqlite::params![mem.id],
        ).unwrap();

        let result = db.verify().unwrap();
        assert_eq!(result.corrupted.len(), 1);
        assert_eq!(result.corrupted[0].id, mem.id);
    }

    #[test]
    fn verify_reports_missing_checksums() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("Jared", "likes", "Rust", None).unwrap();
        // Null out the checksum directly
        store.conn().execute("UPDATE memories SET checksum = NULL", []).unwrap();

        let result = store.verify_integrity().unwrap();
        assert_eq!(result.missing_checksum, 1);
    }
}
