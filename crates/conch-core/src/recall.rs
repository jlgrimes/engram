use std::collections::HashMap;

use chrono::Utc;

use crate::embed::{cosine_similarity, Embedder};
use crate::memory::{MemoryKind, MemoryRecord};
use crate::store::MemoryStore;

/// Minimum cosine similarity threshold for vector search results.
const VECTOR_SIMILARITY_THRESHOLD: f32 = 0.3;

/// RRF constant k — standard value used by Elasticsearch, Qdrant, etc.
const RRF_K: f64 = 60.0;

/// A recalled memory with its relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallResult {
    pub memory: MemoryRecord,
    pub score: f64,
}

/// Global decay constants (lambda/day) by memory kind.
///
/// These are intentionally code-level policy constants (not stored per-memory),
/// so tuning affects all memories immediately.
const FACT_DECAY_LAMBDA_PER_DAY: f64 = 0.02;
const EPISODE_DECAY_LAMBDA_PER_DAY: f64 = 0.06;

/// Reinforcement boost applied when a memory is touched.
const FACT_TOUCH_BOOST: f64 = 0.10;
const EPISODE_TOUCH_BOOST: f64 = 0.20;

/// Overfetch multiplier for candidate reranking.
const CANDIDATE_MULTIPLIER: usize = 10;
const MIN_CANDIDATES: usize = 50;

/// Hybrid recall: BM25 + vector search fused via Reciprocal Rank Fusion.
///
/// 1. BM25 search (keyword relevance)
/// 2. Vector search (semantic relevance, cosine sim > 0.3)
/// 3. RRF fusion of both rankings
/// 4. Final score = RRF × decayed_strength
///
/// Recalled memories are "touched" (decay is applied, then reinforced, and access count bumped).
pub fn recall(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
) -> Result<Vec<RecallResult>, RecallError> {
    let all_memories = store.all_memories_with_text().map_err(RecallError::Db)?;

    if all_memories.is_empty() {
        return Ok(vec![]);
    }

    let now = Utc::now();

    // BM25
    let bm25_ranked = bm25_search(query, &all_memories);

    // Vector
    let query_embedding = embedder
        .embed_one(query)
        .map_err(|e| RecallError::Embedding(e.to_string()))?;
    let vector_ranked = vector_search(&query_embedding, &all_memories);

    // RRF fusion
    let fused = rrf(&bm25_ranked, &vector_ranked);

    // Overfetch candidates, then rerank with full score (including decay)
    // to avoid top-K cutoff errors when decay changes ordering.
    let candidate_count = (limit.saturating_mul(CANDIDATE_MULTIPLIER)).max(MIN_CANDIDATES);
    let candidates = fused.into_iter().take(candidate_count);

    // Score = RRF × decayed_strength
    let mut results: Vec<RecallResult> = candidates
        .map(|(idx, rrf_score)| {
            let mem = &all_memories[idx].0;
            let decayed_strength = effective_strength(mem, now);
            RecallResult {
                memory: mem.clone(),
                score: rrf_score * decayed_strength,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    // Touch recalled memories: apply decay first, then reinforce.
    for result in &results {
        let mem = &result.memory;
        let decayed = effective_strength(mem, now);
        let boosted = (decayed + touch_boost(mem)).min(1.0);
        store
            .touch_memory_with_strength(mem.id, boosted, now)
            .map_err(RecallError::Db)?;
    }

    Ok(results)
}

fn kind_decay_lambda_per_day(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_DECAY_LAMBDA_PER_DAY,
        MemoryKind::Episode(_) => EPISODE_DECAY_LAMBDA_PER_DAY,
    }
}

fn touch_boost(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_TOUCH_BOOST,
        MemoryKind::Episode(_) => EPISODE_TOUCH_BOOST,
    }
}

fn effective_strength(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    let elapsed_secs = (now - mem.last_accessed_at).num_seconds().max(0) as f64;
    let elapsed_days = elapsed_secs / 86_400.0;
    let lambda = kind_decay_lambda_per_day(mem);
    (mem.strength * (-lambda * elapsed_days).exp()).clamp(0.0, 1.0)
}

fn bm25_search(query: &str, memories: &[(MemoryRecord, String)]) -> Vec<(usize, f32)> {
    use bm25::{Document, Language, SearchEngineBuilder};

    let documents: Vec<Document<usize>> = memories
        .iter()
        .enumerate()
        .map(|(i, (_, text))| Document {
            id: i,
            contents: text.clone(),
        })
        .collect();

    let engine: bm25::SearchEngine<usize> =
        SearchEngineBuilder::with_documents(Language::English, documents)
            .b(0.5)
            .build();

    engine
        .search(query, memories.len())
        .into_iter()
        .map(|r| (r.document.id, r.score))
        .collect()
}

fn vector_search(query_emb: &[f32], memories: &[(MemoryRecord, String)]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = memories
        .iter()
        .enumerate()
        .filter_map(|(i, (mem, _))| {
            let emb = mem.embedding.as_ref()?;
            let sim = cosine_similarity(query_emb, emb);
            if sim > VECTOR_SIMILARITY_THRESHOLD {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

fn rrf(list_a: &[(usize, f32)], list_b: &[(usize, f32)]) -> Vec<(usize, f64)> {
    let mut scores: HashMap<usize, f64> = HashMap::new();

    for (rank, &(idx, _)) in list_a.iter().enumerate() {
        *scores.entry(idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }
    for (rank, &(idx, _)) in list_b.iter().enumerate() {
        *scores.entry(idx).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    let mut results: Vec<(usize, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("database error: {0}")]
    Db(rusqlite::Error),
    #[error("embedding error: {0}")]
    Embedding(String),
}

impl From<rusqlite::Error> for RecallError {
    fn from(e: rusqlite::Error) -> Self {
        RecallError::Db(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{EmbedError, Embedding};

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.contains("alpha") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[test]
    fn effective_strength_decays_by_kind() {
        let store = MemoryStore::open_in_memory().unwrap();
        let fact_id = store.remember_fact("Jared", "builds", "Gen", Some(&[1.0, 0.0])).unwrap();
        let ep_id = store
            .remember_episode("alpha project context", Some(&[1.0, 0.0]))
            .unwrap();

        let old_time = (Utc::now() - chrono::Duration::days(10)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET last_accessed_at = ?1 WHERE id IN (?2, ?3)",
                rusqlite::params![old_time, fact_id, ep_id],
            )
            .unwrap();

        let fact = store.get_memory(fact_id).unwrap().unwrap();
        let episode = store.get_memory(ep_id).unwrap().unwrap();

        let sf = effective_strength(&fact, Utc::now());
        let se = effective_strength(&episode, Utc::now());
        assert!(sf > se, "facts should decay slower than episodes");
    }

    #[test]
    fn recall_touch_applies_decay_then_reinforcement() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .remember_episode("alpha memory to recall", Some(&[1.0, 0.0]))
            .unwrap();

        let old_time = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET strength = 1.0, last_accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![old_time, id],
            )
            .unwrap();

        let results = recall(&store, "alpha", &MockEmbedder, 1).unwrap();
        assert_eq!(results.len(), 1);

        let after = store.get_memory(id).unwrap().unwrap();
        assert!(after.access_count >= 1);
        // Should have decayed meaningfully from 1.0 before reinforcement.
        assert!(after.strength < 1.0);
        // But reinforcement should keep it above a tiny decayed floor.
        assert!(after.strength > 0.2);
    }
}
