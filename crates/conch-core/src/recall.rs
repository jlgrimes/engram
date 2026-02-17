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

/// Hybrid recall: BM25 + vector search fused via Reciprocal Rank Fusion.
///
/// 1. BM25 search runs on all memories (proper IDF weighting)
/// 2. Vector search runs if embedder is available (cosine sim > 0.3 threshold)
/// 3. Rankings are fused via RRF — a memory must rank well in both to score high
/// 4. Relations are scored separately via BM25 (not hardcoded 0.5)
/// 5. Final results are sorted by fused score, weighted by strength and recency
///
/// Recalled memories are "touched" (strength reinforced, access count bumped).
pub fn recall(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
) -> Result<Vec<RecallResult>, RecallError> {
    // Load all memories with text content for BM25
    let all_memories = store.all_memories_with_text().map_err(RecallError::Db)?;

    if all_memories.is_empty() {
        return Ok(vec![]);
    }

    let now = Utc::now();

    // ── BM25 search ──────────────────────────────────────────
    let bm25_ranked = bm25_search_memories(query, &all_memories);

    // ── Vector search ────────────────────────────────────────
    let query_embedding = embedder
        .embed_one(query)
        .map_err(|e| RecallError::Embedding(e.to_string()))?;

    let vector_ranked = vector_search_memories(&query_embedding, &all_memories);

    // ── RRF fusion ───────────────────────────────────────────
    let fused = reciprocal_rank_fusion(&bm25_ranked, &vector_ranked);

    // Apply strength and recency weighting to fused scores
    let mut results: Vec<RecallResult> = fused
        .into_iter()
        .map(|(idx, rrf_score)| {
            let mem = &all_memories[idx].0;
            let hours_since_access =
                (now - mem.last_accessed_at).num_minutes() as f64 / 60.0;
            let recency_factor = (-0.01 * hours_since_access.max(0.0)).exp();
            let score = rrf_score * mem.strength * recency_factor;
            RecallResult {
                memory: mem.clone(),
                score,
            }
        })
        .collect();

    // ── Relation search (BM25-scored) ────────────────────────
    let relation_results = search_relations_scored(store, query)?;
    results.extend(relation_results);

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    // Touch recalled memories (reinforce strength, bump access count)
    // Skip synthetic relation results (negative IDs)
    for result in &results {
        if result.memory.id > 0 {
            store
                .touch_memory(result.memory.id)
                .map_err(RecallError::Db)?;
        }
    }

    Ok(results)
}

/// BM25 search over all memories, returning ranked (index, score) pairs.
fn bm25_search_memories(
    query: &str,
    memories: &[(MemoryRecord, String)],
) -> Vec<(usize, f32)> {
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
            .b(0.5) // Lower b for short documents (facts are typically 3-6 words)
            .build();
    let results = engine.search(query, memories.len());

    results
        .into_iter()
        .map(|r| (r.document.id, r.score))
        .collect()
}

/// Vector search over all memories, returning ranked (index, score) pairs.
/// Only includes results above the cosine similarity threshold.
fn vector_search_memories(
    query_embedding: &[f32],
    memories: &[(MemoryRecord, String)],
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = memories
        .iter()
        .enumerate()
        .filter_map(|(i, (mem, _))| {
            let embedding = mem.embedding.as_ref()?;
            let sim = cosine_similarity(query_embedding, embedding);
            if sim > VECTOR_SIMILARITY_THRESHOLD {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect();

    // Sort by similarity descending (for rank-based fusion)
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Reciprocal Rank Fusion: combines two ranked lists into a single fused ranking.
///
/// RRF_score(d) = Σ 1 / (k + rank_i(d))
///
/// This avoids needing to normalize BM25 and cosine similarity scores
/// (which are on completely different scales) by using ranks instead.
fn reciprocal_rank_fusion(
    list_a: &[(usize, f32)],
    list_b: &[(usize, f32)],
) -> Vec<(usize, f64)> {
    let mut scores: HashMap<usize, f64> = HashMap::new();

    for (rank, &(idx, _)) in list_a.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
        *scores.entry(idx).or_insert(0.0) += rrf_score;
    }

    for (rank, &(idx, _)) in list_b.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
        *scores.entry(idx).or_insert(0.0) += rrf_score;
    }

    let mut results: Vec<(usize, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Search associations with BM25 scoring instead of hardcoded relevance.
fn search_relations_scored(
    store: &MemoryStore,
    query: &str,
) -> Result<Vec<RecallResult>, RecallError> {
    use bm25::{Document, Language, SearchEngineBuilder};

    let associations = store
        .all_associations_with_text()
        .map_err(RecallError::Db)?;

    if associations.is_empty() {
        return Ok(vec![]);
    }

    let documents: Vec<Document<usize>> = associations
        .iter()
        .enumerate()
        .map(|(i, (_, text))| Document {
            id: i,
            contents: text.clone(),
        })
        .collect();

    let engine: bm25::SearchEngine<usize> =
        SearchEngineBuilder::with_documents(Language::English, documents)
            .b(0.5) // Lower b for short relation texts
            .build();
    let results = engine.search(query, 5);

    Ok(results
        .into_iter()
        .map(|r| {
            let assoc = &associations[r.document.id].0;
            // Create a synthetic memory record for the association
            let mem = MemoryRecord {
                id: -(assoc.id), // Negative ID to distinguish from real memories
                kind: MemoryKind::Fact(crate::memory::Fact {
                    subject: assoc.entity_a.clone(),
                    relation: assoc.relation.clone(),
                    object: assoc.entity_b.clone(),
                }),
                strength: 1.0,
                embedding: None,
                created_at: assoc.created_at,
                last_accessed_at: assoc.created_at,
                access_count: 0,
            };
            RecallResult {
                memory: mem,
                score: r.score as f64 * 0.8, // slight discount vs facts/episodes
            }
        })
        .collect())
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
    use crate::embed::{cosine_similarity, Embedding};

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall_with_store() {
        // Use a mock embedder for testing
        struct MockEmbedder;
        impl crate::embed::Embedder for MockEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, crate::embed::EmbedError> {
                Ok(texts.iter().map(|_| vec![1.0f32, 0.0, 0.0]).collect())
            }
            fn dimension(&self) -> usize {
                3
            }
        }

        let store = MemoryStore::open_in_memory().unwrap();
        let emb = vec![1.0f32, 0.0, 0.0];
        store
            .remember_fact("Jared", "works at", "Microsoft", Some(&emb))
            .unwrap();
        store
            .remember_fact("Alice", "lives in", "Seattle", Some(&vec![0.0f32, 1.0, 0.0]))
            .unwrap();

        let embedder = MockEmbedder;
        let results = recall(&store, "where does Jared work", &embedder, 10).unwrap();

        assert!(!results.is_empty());
        // The first result should be the one with matching embedding direction
        assert!(results[0].score > results.get(1).map_or(0.0, |r| r.score));
    }

    #[test]
    fn test_bm25_search_ranks_by_term_relevance() {
        let store = MemoryStore::open_in_memory().unwrap();
        // "Jared" appears in many facts — BM25 should downweight it via IDF
        store.remember_fact("Jared", "prefers", "dark mode", None).unwrap();
        store.remember_fact("Jared", "works on", "engram project", None).unwrap();
        store.remember_fact("Jared", "likes", "coffee", None).unwrap();
        store.remember_fact("Alice", "builds", "side projects", None).unwrap();

        let all = store.all_memories_with_text().unwrap();
        let results = bm25_search_memories("side projects", &all);

        // "Alice builds side projects" should rank highest for "side projects"
        assert!(!results.is_empty());
        let top_idx = results[0].0;
        let top_text = &all[top_idx].1;
        assert!(
            top_text.contains("side projects"),
            "Expected 'side projects' in top result, got: {top_text}"
        );
    }

    #[test]
    fn test_vector_search_threshold() {
        let emb_high = vec![1.0f32, 0.0, 0.0]; // cos_sim = 1.0 with query
        let emb_low = vec![0.1f32, 0.99, 0.0]; // cos_sim ≈ 0.1 with query (below 0.3)
        let query_emb = vec![1.0f32, 0.0, 0.0];

        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", Some(&emb_high)).unwrap();
        store.remember_fact("C", "is", "D", Some(&emb_low)).unwrap();

        let all = store.all_memories_with_text().unwrap();

        // Attach embeddings back for the search (all_memories_with_text loads them)
        let results = vector_search_memories(&query_emb, &all);

        // Only the high-similarity result should pass the 0.3 threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // first memory (A is B)
    }

    #[test]
    fn test_rrf_fusion() {
        // List A ranks: doc0 first, doc1 second
        let list_a = vec![(0, 1.0f32), (1, 0.5)];
        // List B ranks: doc1 first, doc0 second
        let list_b = vec![(1, 0.9f32), (0, 0.3)];

        let fused = reciprocal_rank_fusion(&list_a, &list_b);

        // Both docs appear in both lists; doc0 is rank 0+1, doc1 is rank 1+0
        // They should have similar RRF scores (symmetric ranking)
        assert_eq!(fused.len(), 2);
        let score_diff = (fused[0].1 - fused[1].1).abs();
        assert!(score_diff < 0.001, "RRF scores should be very close for symmetrically ranked items");
    }

    #[test]
    fn test_rrf_favors_items_in_both_lists() {
        // doc0 only in list A (rank 0), doc1 in both lists
        let list_a = vec![(0, 1.0f32), (1, 0.5)];
        let list_b = vec![(1, 0.9f32)]; // doc0 missing from list B

        let fused = reciprocal_rank_fusion(&list_a, &list_b);

        // doc1 should score higher because it appears in both lists
        let doc1_score = fused.iter().find(|(idx, _)| *idx == 1).unwrap().1;
        let doc0_score = fused.iter().find(|(idx, _)| *idx == 0).unwrap().1;
        assert!(
            doc1_score > doc0_score,
            "Item in both lists should score higher: doc1={doc1_score} vs doc0={doc0_score}"
        );
    }

    #[test]
    fn test_relation_scoring_not_hardcoded() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.relate("Jared", "works_on", "engram").unwrap();
        store.relate("Jared", "likes", "pizza").unwrap();
        store.relate("Alice", "contributes_to", "open source projects").unwrap();

        let results = search_relations_scored(&store, "projects").unwrap();

        // "projects" should match the open source relation better than pizza/engram
        assert!(!results.is_empty());
        let top = &results[0];
        if let MemoryKind::Fact(f) = &top.memory.kind {
            assert!(
                f.object.contains("projects") || f.relation.contains("projects"),
                "Top relation result should match 'projects', got: {} {} {}",
                f.subject, f.relation, f.object
            );
        }
        // Score should NOT be hardcoded 0.5
        assert!(
            (top.score - 0.5).abs() > 0.01,
            "Score should not be hardcoded 0.5, got: {}",
            top.score
        );
    }

    #[test]
    fn test_hybrid_recall_combines_signals() {
        struct MockEmbedder;
        impl crate::embed::Embedder for MockEmbedder {
            fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, crate::embed::EmbedError> {
                // Return embeddings that distinguish between texts
                Ok(texts
                    .iter()
                    .map(|t| {
                        if t.contains("project") || t.contains("work") {
                            vec![0.9f32, 0.1, 0.0]
                        } else {
                            vec![0.1f32, 0.9, 0.0]
                        }
                    })
                    .collect())
            }
            fn dimension(&self) -> usize {
                3
            }
        }

        let store = MemoryStore::open_in_memory().unwrap();
        let emb_project = vec![0.9f32, 0.1, 0.0];
        let emb_other = vec![0.1f32, 0.9, 0.0];

        store.remember_fact("Jared", "works on", "engram project", Some(&emb_project)).unwrap();
        store.remember_fact("Jared", "prefers", "dark mode", Some(&emb_other)).unwrap();
        store.remember_fact("Jared", "likes", "coffee", Some(&emb_other)).unwrap();

        let embedder = MockEmbedder;
        let results = recall(&store, "what projects does Jared work on", &embedder, 10).unwrap();

        // "engram project" should rank first — it matches both BM25 and vector
        assert!(!results.is_empty());
        if let MemoryKind::Fact(f) = &results[0].memory.kind {
            assert!(
                f.object.contains("engram"),
                "Top result should be engram project, got: {} {} {}",
                f.subject, f.relation, f.object
            );
        }
    }
}
