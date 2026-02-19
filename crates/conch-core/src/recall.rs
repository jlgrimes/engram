use std::collections::HashMap;

use chrono::Utc;

use crate::embed::{cosine_similarity, Embedder};
use crate::memory::{MemoryKind, MemoryRecord};
use crate::recall_scoring::{
    access_weight, apply_boosts, effective_strength, recency_boost, spread_activation,
    spread_activation_boosts, temporal_cooccurrence_boost, temporal_cooccurrence_boosts,
    touch_boost, SPREAD_FACTOR,
};
use crate::store::{MemoryStore, DEFAULT_NAMESPACE};

/// Minimum cosine similarity threshold for vector search results.
const VECTOR_SIMILARITY_THRESHOLD: f32 = 0.3;

/// RRF constant k — standard value used by Elasticsearch, Qdrant, etc.
const RRF_K: f64 = 60.0;

/// A recalled memory with its relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallResult {
    pub memory: MemoryRecord,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<RecallScoreExplanation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<RecallDiagnostics>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallDiagnostics {
    pub bm25_hits: usize,
    pub vector_hits: usize,
    pub fused_candidates: usize,
    pub filtered_memories: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallScoreExplanation {
    pub rrf_score: f64,
    pub decayed_strength: f64,
    pub recency_boost: f64,
    pub access_weight: f64,
    pub activation_boost: f64,
    pub temporal_boost: f64,
    pub final_score: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecallOptions {
    pub explain: bool,
    pub diagnostics: bool,
}

/// Overfetch multiplier for candidate reranking.
const CANDIDATE_MULTIPLIER: usize = 10;
const MIN_CANDIDATES: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecallKindFilter {
    All,
    Facts,
    Episodes,
}

impl RecallKindFilter {
    fn matches(self, kind: &MemoryKind) -> bool {
        match self {
            Self::All => true,
            Self::Facts => matches!(kind, MemoryKind::Fact(_)),
            Self::Episodes => matches!(kind, MemoryKind::Episode(_)),
        }
    }
}

/// Hybrid recall: BM25 + vector search fused via Reciprocal Rank Fusion,
/// enhanced with brain-inspired scoring heuristics.
///
/// Pipeline:
/// 1. Filter candidates by memory kind (if requested)
/// 2. BM25 search (keyword relevance)
/// 3. Vector search (semantic relevance, cosine sim > threshold)
/// 4. RRF fusion of both rankings
/// 5. Base score = RRF × decayed_strength × recency_boost × access_weight
/// 6. 1-hop spreading activation through the knowledge graph
/// 7. Temporal co-occurrence boost for memories created near top results
///
/// Recalled memories are "touched" (decay is applied, then reinforced, and
/// access count bumped).
pub fn recall(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
) -> Result<Vec<RecallResult>, RecallError> {
    recall_with_filter_in(
        store,
        DEFAULT_NAMESPACE,
        query,
        embedder,
        limit,
        RecallKindFilter::All,
    )
}

pub fn recall_with_filter(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    filter: RecallKindFilter,
) -> Result<Vec<RecallResult>, RecallError> {
    recall_with_filter_in(store, DEFAULT_NAMESPACE, query, embedder, limit, filter)
}

pub fn recall_with_filter_in(
    store: &MemoryStore,
    namespace: &str,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    filter: RecallKindFilter,
) -> Result<Vec<RecallResult>, RecallError> {
    recall_with_filter_in_options(
        store,
        namespace,
        query,
        embedder,
        limit,
        filter,
        RecallOptions::default(),
    )
}

pub fn recall_with_filter_in_options(
    store: &MemoryStore,
    namespace: &str,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    filter: RecallKindFilter,
    options: RecallOptions,
) -> Result<Vec<RecallResult>, RecallError> {
    let all_memories = store
        .all_memories_with_text_in(namespace)
        .map_err(RecallError::Db)?;
    let filtered_memories: Vec<(MemoryRecord, String)> = all_memories
        .into_iter()
        .filter(|(m, _)| filter.matches(&m.kind))
        .collect();

    if filtered_memories.is_empty() {
        return Ok(vec![]);
    }

    let now = Utc::now();

    // Find the maximum access_count for normalization.
    let max_access = filtered_memories
        .iter()
        .map(|(m, _)| m.access_count)
        .max()
        .unwrap_or(0);

    // BM25
    let bm25_ranked = bm25_search(query, &filtered_memories);

    // Vector
    let query_embedding = embedder
        .embed_one(query)
        .map_err(|e| RecallError::Embedding(e.to_string()))?;
    let vector_ranked = vector_search(&query_embedding, &filtered_memories);

    // RRF fusion
    let fused = rrf(&bm25_ranked, &vector_ranked);
    let diagnostics = options.diagnostics.then_some(RecallDiagnostics {
        bm25_hits: bm25_ranked.len(),
        vector_hits: vector_ranked.len(),
        fused_candidates: fused.len(),
        filtered_memories: filtered_memories.len(),
    });

    // Overfetch candidates, then rerank with full score (including decay,
    // recency, and access weighting) to avoid top-K cutoff errors.
    let candidate_count = (limit.saturating_mul(CANDIDATE_MULTIPLIER)).max(MIN_CANDIDATES);
    let candidates = fused.into_iter().take(candidate_count);

    // Score = RRF × decayed_strength × recency_boost × access_weight
    let mut results: Vec<RecallResult> = candidates
        .map(|(idx, rrf_score)| {
            let mem = &filtered_memories[idx].0;
            let decayed_strength = effective_strength(mem, now);
            let recency = recency_boost(mem, now);
            let access = access_weight(mem, max_access);
            let base_score = rrf_score * decayed_strength * recency * access;
            let explanation = options.explain.then_some(RecallScoreExplanation {
                rrf_score,
                decayed_strength,
                recency_boost: recency,
                access_weight: access,
                activation_boost: 0.0,
                temporal_boost: 0.0,
                final_score: base_score,
            });
            RecallResult {
                memory: mem.clone(),
                score: base_score,
                explanation,
                diagnostics: diagnostics.clone(),
            }
        })
        .collect();

    // ── Spreading activation ─────────────────────────────────
    // For each scored Fact, boost other results that share a subject or object.
    // This is 1-hop graph traversal inspired by Collins & Loftus (1975).
    if options.explain {
        let activation_boosts = spread_activation_boosts(&results, SPREAD_FACTOR);
        apply_boosts(&mut results, &activation_boosts);
        for (result, boost) in results.iter_mut().zip(activation_boosts.iter()) {
            if let Some(explanation) = result.explanation.as_mut() {
                explanation.activation_boost = *boost;
            }
        }
    } else {
        spread_activation(&mut results, SPREAD_FACTOR);
    }

    // ── Temporal co-occurrence boost ─────────────────────────
    // Memories created near the same time as high-scoring results get a small
    // boost, implementing Tulving's encoding specificity / contextual
    // reinstatement principle.
    if options.explain {
        let temporal_boosts = temporal_cooccurrence_boosts(&results);
        apply_boosts(&mut results, &temporal_boosts);
        for (result, boost) in results.iter_mut().zip(temporal_boosts.iter()) {
            if let Some(explanation) = result.explanation.as_mut() {
                explanation.temporal_boost = *boost;
                explanation.final_score = result.score;
            }
        }
    } else {
        temporal_cooccurrence_boost(&mut results);
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
    use crate::recall_scoring::{
        access_weight, effective_strength, recency_boost, spread_activation,
        temporal_cooccurrence_boost, RECENCY_FLOOR, SPREAD_FACTOR,
    };

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
        let fact_id = store
            .remember_fact("Jared", "builds", "Gen", Some(&[1.0, 0.0]))
            .unwrap();
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

    // ── Recency boost tests ────────────────────────────────────

    #[test]
    fn recency_boost_favors_recent_over_old() {
        let store = MemoryStore::open_in_memory().unwrap();
        let now = Utc::now();

        // Two semantically identical memories, different ages
        let recent_id = store
            .remember_episode("alpha project is great", Some(&[1.0, 0.0]))
            .unwrap();
        let old_id = store
            .remember_episode("alpha project is great", Some(&[1.0, 0.0]))
            .unwrap();

        // Make the old one 30 days old
        let old_time = (now - chrono::Duration::days(30)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET created_at = ?1, last_accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![old_time, old_id],
            )
            .unwrap();

        let results = recall(&store, "alpha project", &MockEmbedder, 2).unwrap();
        assert_eq!(results.len(), 2);
        // Recent memory should score higher
        assert_eq!(results[0].memory.id, recent_id);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn recency_boost_has_floor_old_memories_still_appear() {
        let store = MemoryStore::open_in_memory().unwrap();
        let now = Utc::now();

        // Very old memory — should still appear, not be zeroed out
        let id = store
            .remember_episode("alpha ancient knowledge", Some(&[1.0, 0.0]))
            .unwrap();

        let ancient_time = (now - chrono::Duration::days(365)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET created_at = ?1, last_accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![ancient_time, id],
            )
            .unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        let boost = recency_boost(&mem, now);
        assert!(
            boost >= RECENCY_FLOOR,
            "recency boost {} should be >= floor {}",
            boost,
            RECENCY_FLOOR
        );
    }

    // ── Access weight tests ──────────────────────────────────

    #[test]
    fn access_weight_boosts_frequently_recalled_memories() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Two identical memories, one recalled many times
        let hot_id = store
            .remember_episode("alpha hot memory", Some(&[1.0, 0.0]))
            .unwrap();
        let cold_id = store
            .remember_episode("alpha cold memory", Some(&[1.0, 0.0]))
            .unwrap();

        // Bump access count on hot memory
        store
            .conn()
            .execute(
                "UPDATE memories SET access_count = 20 WHERE id = ?1",
                rusqlite::params![hot_id],
            )
            .unwrap();

        let hot = store.get_memory(hot_id).unwrap().unwrap();
        let cold = store.get_memory(cold_id).unwrap().unwrap();

        let hot_w = access_weight(&hot, 20);
        let cold_w = access_weight(&cold, 20);

        assert!(
            hot_w > cold_w,
            "hot ({}) should weigh more than cold ({})",
            hot_w,
            cold_w
        );
        assert!(
            hot_w >= 1.0 && hot_w <= 2.0,
            "access weight should be in [1.0, 2.0], got {}",
            hot_w
        );
        assert!(
            cold_w >= 1.0,
            "cold access weight should be >= 1.0, got {}",
            cold_w
        );
    }

    #[test]
    fn access_weight_is_bounded() {
        let store = MemoryStore::open_in_memory().unwrap();

        let id = store
            .remember_episode("alpha bounded test", Some(&[1.0, 0.0]))
            .unwrap();

        // Max out access count
        store
            .conn()
            .execute(
                "UPDATE memories SET access_count = 1000 WHERE id = ?1",
                rusqlite::params![id],
            )
            .unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        let w = access_weight(&mem, 1000);
        assert!(w <= 2.0, "access weight should never exceed 2.0, got {}", w);
    }

    // ── Spreading activation tests ───────────────────────────

    #[test]
    fn spreading_activation_boosts_related_facts() {
        // If "Jared has_pet Tortellini" scores high, then
        // "Tortellini is_a dog" should get a boost via shared entity "Tortellini"
        let mut results = vec![
            RecallResult {
                memory: make_fact_record(1, "Jared", "has_pet", "Tortellini"),
                score: 1.0,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_fact_record(2, "Tortellini", "is_a", "dog"),
                score: 0.1, // low initial score
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_fact_record(3, "Abby", "likes", "cats"),
                score: 0.1, // unrelated
                explanation: None,
                diagnostics: None,
            },
        ];

        let original_related = results[1].score;
        let original_unrelated = results[2].score;

        spread_activation(&mut results, SPREAD_FACTOR);

        assert!(
            results[1].score > original_related,
            "related fact should be boosted: {} > {}",
            results[1].score,
            original_related
        );
        assert_eq!(
            results[2].score, original_unrelated,
            "unrelated fact should not be boosted"
        );
    }

    #[test]
    fn spreading_activation_is_bidirectional() {
        // Both directions: A->B and B->A should boost each other
        let mut results = vec![
            RecallResult {
                memory: make_fact_record(1, "Jared", "works_at", "Microsoft"),
                score: 0.8,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_fact_record(2, "Microsoft", "located_in", "Seattle"),
                score: 0.3,
                explanation: None,
                diagnostics: None,
            },
        ];

        let score_a_before = results[0].score;
        let score_b_before = results[1].score;

        spread_activation(&mut results, SPREAD_FACTOR);

        // A boosted B via shared "Microsoft"
        assert!(results[1].score > score_b_before);
        // B boosted A via shared "Microsoft"
        assert!(results[0].score > score_a_before);
    }

    #[test]
    fn spreading_activation_does_not_self_boost() {
        let mut results = vec![RecallResult {
            memory: make_fact_record(1, "Jared", "builds", "Gen"),
            score: 1.0,
            explanation: None,
            diagnostics: None,
        }];

        spread_activation(&mut results, SPREAD_FACTOR);
        // Single result — no self-boost possible
        assert!((results[0].score - 1.0).abs() < f64::EPSILON);
    }

    // ── Temporal co-occurrence tests ─────────────────────────

    #[test]
    fn temporal_cooccurrence_boosts_same_session_memories() {
        let now = Utc::now();

        let mut results = vec![
            RecallResult {
                memory: make_timed_episode(1, "alpha anchor memory", now),
                score: 1.0,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_timed_episode(
                    2,
                    "alpha nearby memory",
                    now - chrono::Duration::minutes(5),
                ),
                score: 0.2,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_timed_episode(
                    3,
                    "alpha distant memory",
                    now - chrono::Duration::hours(3),
                ),
                score: 0.2,
                explanation: None,
                diagnostics: None,
            },
        ];

        let nearby_before = results[1].score;
        let distant_before = results[2].score;

        temporal_cooccurrence_boost(&mut results);

        assert!(
            results[1].score > nearby_before,
            "nearby memory should be boosted: {} > {}",
            results[1].score,
            nearby_before
        );
        assert_eq!(
            results[2].score, distant_before,
            "distant memory (>30min) should not be boosted"
        );
    }

    #[test]
    fn temporal_cooccurrence_scales_with_proximity() {
        let now = Utc::now();

        let mut results = vec![
            RecallResult {
                memory: make_timed_episode(1, "alpha anchor", now),
                score: 1.0,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_timed_episode(
                    2,
                    "alpha very close",
                    now - chrono::Duration::minutes(2),
                ),
                score: 0.1,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: make_timed_episode(3, "alpha further", now - chrono::Duration::minutes(25)),
                score: 0.1,
                explanation: None,
                diagnostics: None,
            },
        ];

        temporal_cooccurrence_boost(&mut results);

        // 2-min-away should get more boost than 25-min-away
        assert!(
            results[1].score > results[2].score,
            "closer memory ({}) should score higher than further one ({})",
            results[1].score,
            results[2].score
        );
    }

    // ── Integration: full pipeline test ──────────────────────

    #[test]
    fn full_recall_pipeline_ranks_recent_accessed_related_higher() {
        let store = MemoryStore::open_in_memory().unwrap();
        let now = Utc::now();

        // Create a cluster of related facts about a topic
        store
            .remember_fact("Jared", "has_pet", "Tortellini", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_fact("Tortellini", "is_a", "dog", Some(&[1.0, 0.0]))
            .unwrap();

        // Create an old, unrelated memory
        let old_id = store
            .remember_fact("weather", "is", "sunny", Some(&[0.5, 0.5]))
            .unwrap();
        let old_time = (now - chrono::Duration::days(60)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET created_at = ?1, last_accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![old_time, old_id],
            )
            .unwrap();

        let results = recall(&store, "alpha", &MockEmbedder, 10).unwrap();

        // The two Tortellini facts should be near the top (recent + related to each other)
        // The old weather fact should be lower
        if results.len() >= 3 {
            let weather_pos = results.iter().position(|r| r.memory.id == old_id);
            if let Some(pos) = weather_pos {
                assert!(pos >= 2, "old unrelated memory should rank below related recent ones, was at position {}", pos);
            }
        }
    }

    #[test]
    fn recall_order_stable_for_deterministic_case() {
        let store = MemoryStore::open_in_memory().unwrap();

        let top_id = store
            .remember_episode("alpha alpha alpha", Some(&[1.0, 0.0]))
            .unwrap();
        let middle_id = store
            .remember_episode("alpha alpha", Some(&[1.0, 0.0]))
            .unwrap();
        let lower_id = store.remember_episode("alpha", Some(&[1.0, 0.0])).unwrap();

        let results = recall(&store, "alpha", &MockEmbedder, 3).unwrap();
        let ids: Vec<i64> = results.iter().map(|r| r.memory.id).collect();

        assert_eq!(ids, vec![top_id, middle_id, lower_id]);
    }

    // ── Test helpers ─────────────────────────────────────────

    fn make_fact_record(id: i64, subj: &str, rel: &str, obj: &str) -> MemoryRecord {
        MemoryRecord {
            id,
            namespace: "default".to_string(),
            kind: MemoryKind::Fact(crate::memory::Fact {
                subject: subj.to_string(),
                relation: rel.to_string(),
                object: obj.to_string(),
            }),
            strength: 1.0,
            created_at: Utc::now(),
            last_accessed_at: Utc::now(),
            access_count: 0,
            embedding: None,
        }
    }

    fn make_timed_episode(id: i64, text: &str, time: chrono::DateTime<Utc>) -> MemoryRecord {
        MemoryRecord {
            id,
            namespace: "default".to_string(),
            kind: MemoryKind::Episode(crate::memory::Episode {
                text: text.to_string(),
            }),
            strength: 1.0,
            created_at: time,
            last_accessed_at: time,
            access_count: 0,
            embedding: None,
        }
    }

    // ── Recall kind filtering tests ──────────────────────────

    #[test]
    fn recall_filter_facts_only_excludes_episodes() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact("Jared", "builds", "conch", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_episode("Jared had coffee", Some(&[1.0, 0.0]))
            .unwrap();

        let results =
            recall_with_filter(&store, "Jared", &MockEmbedder, 10, RecallKindFilter::Facts)
                .unwrap();
        assert!(!results.is_empty());
        assert!(results
            .iter()
            .all(|r| matches!(r.memory.kind, MemoryKind::Fact(_))));
    }

    #[test]
    fn recall_filter_episodes_only_excludes_facts() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact("Jared", "builds", "conch", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_episode("Jared had coffee", Some(&[1.0, 0.0]))
            .unwrap();

        let results = recall_with_filter(
            &store,
            "Jared",
            &MockEmbedder,
            10,
            RecallKindFilter::Episodes,
        )
        .unwrap();
        assert!(!results.is_empty());
        assert!(results
            .iter()
            .all(|r| matches!(r.memory.kind, MemoryKind::Episode(_))));
    }

    #[test]
    fn recall_filter_all_returns_both_kinds() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact("Jared", "builds", "conch", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_episode("Jared had coffee", Some(&[1.0, 0.0]))
            .unwrap();

        let results =
            recall_with_filter(&store, "Jared", &MockEmbedder, 10, RecallKindFilter::All).unwrap();
        assert!(results
            .iter()
            .any(|r| matches!(r.memory.kind, MemoryKind::Fact(_))));
        assert!(results
            .iter()
            .any(|r| matches!(r.memory.kind, MemoryKind::Episode(_))));
    }

    #[test]
    fn recall_explain_includes_breakdown_and_final_score_matches() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha explainable memory", Some(&[1.0, 0.0]))
            .unwrap();

        let results = recall_with_filter_in_options(
            &store,
            DEFAULT_NAMESPACE,
            "alpha",
            &MockEmbedder,
            5,
            RecallKindFilter::All,
            RecallOptions {
                explain: true,
                diagnostics: false,
            },
        )
        .unwrap();

        assert!(!results.is_empty());
        for r in &results {
            let ex = r
                .explanation
                .as_ref()
                .expect("explanation should be present when explain=true");
            assert!((ex.final_score - r.score).abs() < 1e-9);
        }
    }

    #[test]
    fn recall_default_omits_explanation() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha default no explanation", Some(&[1.0, 0.0]))
            .unwrap();

        let results = recall(&store, "alpha", &MockEmbedder, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.explanation.is_none()));
    }

    #[test]
    fn recall_diagnostics_present_only_when_enabled() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha diagnostics memory", Some(&[1.0, 0.0]))
            .unwrap();

        let default_results = recall(&store, "alpha", &MockEmbedder, 5).unwrap();
        assert!(!default_results.is_empty());
        assert!(default_results.iter().all(|r| r.diagnostics.is_none()));

        let diagnostics_results = recall_with_filter_in_options(
            &store,
            DEFAULT_NAMESPACE,
            "alpha",
            &MockEmbedder,
            5,
            RecallKindFilter::All,
            RecallOptions {
                explain: false,
                diagnostics: true,
            },
        )
        .unwrap();
        assert!(!diagnostics_results.is_empty());
        assert!(diagnostics_results.iter().all(|r| r.diagnostics.is_some()));
    }

    #[test]
    fn recall_diagnostics_values_are_sensible() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha diagnostics one", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_episode("alpha diagnostics two", Some(&[1.0, 0.0]))
            .unwrap();

        let diagnostics_results = recall_with_filter_in_options(
            &store,
            DEFAULT_NAMESPACE,
            "alpha",
            &MockEmbedder,
            5,
            RecallKindFilter::All,
            RecallOptions {
                explain: true,
                diagnostics: true,
            },
        )
        .unwrap();

        assert!(!diagnostics_results.is_empty());
        let d = diagnostics_results[0]
            .diagnostics
            .as_ref()
            .expect("diagnostics should be present when diagnostics=true");
        assert!(d.filtered_memories >= diagnostics_results.len());
        assert!(d.filtered_memories > 0);
        assert!(d.bm25_hits > 0);
        assert!(d.vector_hits > 0);
        assert!(d.fused_candidates > 0);
        assert!(diagnostics_results.iter().all(|r| r.explanation.is_some()));
    }

    // ── Original tests ───────────────────────────────────────

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

    // ── Recall Determinism Regression Suite ─────────────────────────────────

    use crate::recall_scoring::{
        spread_activation_boosts, temporal_cooccurrence_boosts
    };

    /// Test deterministic behavior of core scoring functions with fixed inputs.
    /// These functions should produce identical outputs for identical inputs.
    #[test]
    fn recall_scoring_functions_are_deterministic() {
        let now = chrono::DateTime::from_timestamp(1640995200, 0).unwrap(); // Fixed timestamp: 2022-01-01 00:00:00 UTC
        
        // Test recency_boost determinism
        let memory_recent = MemoryRecord {
            id: 1,
            namespace: "test".to_string(),
            kind: MemoryKind::Episode(crate::memory::Episode {
                text: "Recent memory".to_string(),
            }),
            strength: 1.0,
            embedding: Some(vec![0.5; 4]),
            created_at: now - chrono::Duration::hours(6), // 6 hours ago
            last_accessed_at: now - chrono::Duration::hours(6),
            access_count: 2,
        };
        
        // Test multiple runs produce identical recency boosts
        let boosts: Vec<f64> = (0..5)
            .map(|_| recency_boost(&memory_recent, now))
            .collect();
        
        for i in 1..boosts.len() {
            assert!((boosts[0] - boosts[i]).abs() < 1e-15, 
                   "Recency boost not deterministic: {} vs {}", boosts[0], boosts[i]);
        }
        
        // Test access_weight determinism
        let max_access = 10i64;
        let weights: Vec<f64> = (0..5)
            .map(|_| access_weight(&memory_recent, max_access))
            .collect();
        
        for i in 1..weights.len() {
            assert!((weights[0] - weights[i]).abs() < 1e-15,
                   "Access weight not deterministic: {} vs {}", weights[0], weights[i]);
        }
        
        // Test effective_strength determinism
        let strengths: Vec<f64> = (0..5)
            .map(|_| effective_strength(&memory_recent, now))
            .collect();
        
        for i in 1..strengths.len() {
            assert!((strengths[0] - strengths[i]).abs() < 1e-15,
                   "Effective strength not deterministic: {} vs {}", strengths[0], strengths[i]);
        }
    }

    #[test]
    fn spreading_activation_boosts_are_deterministic() {
        let now = chrono::DateTime::from_timestamp(1640995200, 0).unwrap();
        
        // Create fixed recall results for spreading activation
        let results = vec![
            RecallResult {
                memory: MemoryRecord {
                    id: 1,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Fact(crate::memory::Fact {
                        subject: "Rust".to_string(),
                        relation: "is".to_string(),
                        object: "systems language".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.8,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 2,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Fact(crate::memory::Fact {
                        subject: "Rust".to_string(),
                        relation: "supports".to_string(),
                        object: "memory safety".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.8, 0.2, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.6,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 3,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Fact(crate::memory::Fact {
                        subject: "Python".to_string(),
                        relation: "is".to_string(),
                        object: "interpreted language".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.5, 0.5, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.4,
                explanation: None,
                diagnostics: None,
            },
        ];
        
        // Test spreading activation boost determinism across multiple runs
        let factor = 0.15;
        let boost_sets: Vec<Vec<f64>> = (0..5)
            .map(|_| spread_activation_boosts(&results, factor))
            .collect();
        
        // Verify all runs produce identical boosts
        for run_idx in 1..boost_sets.len() {
            assert_eq!(boost_sets[0].len(), boost_sets[run_idx].len());
            
            for (pos, (&first_boost, &current_boost)) in 
                boost_sets[0].iter().zip(boost_sets[run_idx].iter()).enumerate() {
                assert!((first_boost - current_boost).abs() < 1e-15,
                       "Spreading activation boost not deterministic at position {} in run {}: {} vs {}",
                       pos, run_idx, first_boost, current_boost);
            }
        }
        
        // Verify connected facts (sharing "Rust" entity) get positive boosts  
        assert!(boost_sets[0][1] > 0.0, "Connected Rust fact should get positive boost");
        assert_eq!(boost_sets[0][2], 0.0, "Unconnected Python fact should get no boost");
    }
    
    #[test]
    fn temporal_cooccurrence_boosts_are_deterministic() {
        let now = chrono::DateTime::from_timestamp(1640995200, 0).unwrap();
        
        // Create temporally clustered recall results
        let results = vec![
            RecallResult {
                memory: MemoryRecord {
                    id: 1,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "Started Rust project".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                    created_at: now - chrono::Duration::minutes(10), // Within 30min window
                    last_accessed_at: now - chrono::Duration::minutes(10),
                    access_count: 1,
                },
                score: 0.9,  // High anchor score
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 2,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "Setup Rust toolchain".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.8, 0.2, 0.0, 0.0]),
                    created_at: now - chrono::Duration::minutes(5), // Within 30min window
                    last_accessed_at: now - chrono::Duration::minutes(5),
                    access_count: 1,
                },
                score: 0.5,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 3,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "Deployed Python service".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.5, 0.5, 0.0, 0.0]),
                    created_at: now - chrono::Duration::hours(3), // Outside 30min window
                    last_accessed_at: now - chrono::Duration::hours(3),
                    access_count: 1,
                },
                score: 0.3,
                explanation: None,
                diagnostics: None,
            },
        ];
        
        // Test temporal cooccurrence boost determinism across multiple runs
        let boost_sets: Vec<Vec<f64>> = (0..5)
            .map(|_| temporal_cooccurrence_boosts(&results))
            .collect();
        
        // Verify all runs produce identical boosts
        for run_idx in 1..boost_sets.len() {
            assert_eq!(boost_sets[0].len(), boost_sets[run_idx].len());
            
            for (pos, (&first_boost, &current_boost)) in 
                boost_sets[0].iter().zip(boost_sets[run_idx].iter()).enumerate() {
                assert!((first_boost - current_boost).abs() < 1e-15,
                       "Temporal cooccurrence boost not deterministic at position {} in run {}: {} vs {}",
                       pos, run_idx, first_boost, current_boost);
            }
        }
        
        // Verify temporally clustered memories get boosts
        assert!(boost_sets[0][1] > 0.0, "Temporally close memory should get positive boost");
        assert_eq!(boost_sets[0][2], 0.0, "Temporally distant memory should get no boost");
    }
    
    #[test] 
    fn recall_result_ordering_stability_with_fixed_scores() {
        // Test that identical RecallResults are ordered consistently
        let now = chrono::DateTime::from_timestamp(1640995200, 0).unwrap();
        
        let results = vec![
            RecallResult {
                memory: MemoryRecord {
                    id: 1,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "High score memory".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.9,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 2,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "Medium score memory".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.8, 0.2, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.5,
                explanation: None,
                diagnostics: None,
            },
            RecallResult {
                memory: MemoryRecord {
                    id: 3,
                    namespace: "test".to_string(),
                    kind: MemoryKind::Episode(crate::memory::Episode {
                        text: "Low score memory".to_string(),
                    }),
                    strength: 1.0,
                    embedding: Some(vec![0.5, 0.5, 0.0, 0.0]),
                    created_at: now,
                    last_accessed_at: now,
                    access_count: 1,
                },
                score: 0.1,
                explanation: None,
                diagnostics: None,
            },
        ];
        
        // Test sorting stability across multiple runs
        let mut sorted_orders = Vec::new();
        for _ in 0..5 {
            let mut test_results = results.clone();
            test_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            let order: Vec<i64> = test_results.iter().map(|r| r.memory.id).collect();
            sorted_orders.push(order);
        }
        
        // Verify identical ordering across all runs
        for run_idx in 1..sorted_orders.len() {
            assert_eq!(sorted_orders[0], sorted_orders[run_idx],
                      "Sorting order not deterministic in run {}", run_idx);
        }
        
        // Verify expected order (highest score first)
        assert_eq!(sorted_orders[0], vec![1, 2, 3], "Results not sorted by score correctly");
    }
}
