use std::collections::HashMap;

use chrono::Utc;

use crate::embed::{cosine_similarity, Embedder};
use crate::memory::{MemoryKind, MemoryRecord};
use crate::store::MemoryStore;

/// Minimum cosine similarity threshold for vector search results.
const VECTOR_SIMILARITY_THRESHOLD: f32 = 0.3;

/// RRF constant k — standard value used by Elasticsearch, Qdrant, etc.
const RRF_K: f64 = 60.0;

/// Explainability metadata for recall ranking.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallScoreExplain {
    pub rrf_score: f64,
    pub rrf_rank: usize,
    pub rrf_rank_percentile: f64,
    pub bm25_rank: Option<usize>,
    pub bm25_score: Option<f32>,
    pub vector_rank: Option<usize>,
    pub vector_similarity: Option<f32>,
    pub modality_agreement: bool,
    pub matched_modalities: usize,
    pub decayed_strength: f64,
    pub recency_boost: f64,
    pub access_weight: f64,
    pub base_score: f64,
    pub spread_boost: f64,
    pub temporal_boost: f64,
    pub score_margin_to_next: Option<f64>,
    pub final_score: f64,
}

/// A recalled memory with its relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallResult {
    pub memory: MemoryRecord,
    pub score: f64,
    pub explain: RecallScoreExplain,
}

/// Global decay constants (lambda/day) by memory kind.
///
/// These are intentionally code-level policy constants (not stored per-memory),
/// so tuning affects all memories immediately.
const FACT_DECAY_LAMBDA_PER_DAY: f64 = 0.02;
const EPISODE_DECAY_LAMBDA_PER_DAY: f64 = 0.06;
const ACTION_DECAY_LAMBDA_PER_DAY: f64 = 0.09;

/// Reinforcement boost applied when a memory is touched.
const FACT_TOUCH_BOOST: f64 = 0.10;
const EPISODE_TOUCH_BOOST: f64 = 0.20;
const ACTION_TOUCH_BOOST: f64 = 0.25;

/// Overfetch multiplier for candidate reranking.
const CANDIDATE_MULTIPLIER: usize = 10;
const MIN_CANDIDATES: usize = 50;

/// Coefficients controlling influence of each base score component.
///
/// Final base score formula:
/// rrf^rrf_exp * decayed_strength^decay_exp * recency_boost^recency_exp * access_weight^access_exp
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RecallScoreCoefficients {
    pub rrf_exp: f64,
    pub decay_exp: f64,
    pub recency_exp: f64,
    pub access_exp: f64,
}

impl Default for RecallScoreCoefficients {
    fn default() -> Self {
        Self {
            rrf_exp: 1.0,
            decay_exp: 1.0,
            recency_exp: 1.0,
            access_exp: 1.0,
        }
    }
}

/// Spreading activation: fraction of a memory's score given to graph neighbors.
const SPREAD_FACTOR: f64 = 0.15;

/// Recency boost half-life in hours (7 days). Memories newer than this get a
/// meaningful boost; older ones taper towards a floor.
const RECENCY_HALF_LIFE_HOURS: f64 = 168.0;

/// Minimum recency multiplier so old memories aren't completely suppressed.
const RECENCY_FLOOR: f64 = 0.3;

/// Hybrid recall: BM25 + vector search fused via Reciprocal Rank Fusion,
/// enhanced with brain-inspired scoring heuristics.
///
/// Pipeline:
/// 1. BM25 search (keyword relevance)
/// 2. Vector search (semantic relevance, cosine sim > threshold)
/// 3. RRF fusion of both rankings
/// 4. Base score = RRF × decayed_strength × recency_boost × access_weight
/// 5. 1-hop spreading activation through the knowledge graph
/// 6. Temporal co-occurrence boost for memories created near top results
///
/// Recalled memories are "touched" (decay is applied, then reinforced, and
/// access count bumped).
pub fn recall(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
) -> Result<Vec<RecallResult>, RecallError> {
    recall_with_tag_filter(store, query, embedder, limit, None)
}

/// Like `recall`, but optionally filters results to only include memories
/// that have a specific tag.
pub fn recall_with_tag_filter(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    tag_filter: Option<&str>,
) -> Result<Vec<RecallResult>, RecallError> {
    recall_with_tag_filter_ns(store, query, embedder, limit, tag_filter, "default")
}

pub fn recall_with_tag_filter_ns(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    tag_filter: Option<&str>,
    namespace: &str,
) -> Result<Vec<RecallResult>, RecallError> {
    let mut all_memories = store
        .all_memories_with_text_ns(namespace)
        .map_err(RecallError::Db)?;

    // If a tag filter is specified, only keep memories that have the tag.
    if let Some(tag) = tag_filter {
        all_memories.retain(|(mem, _)| mem.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)));
    }

    if all_memories.is_empty() {
        return Ok(vec![]);
    }

    let now = Utc::now();

    // Find the maximum access_count for normalization.
    let max_access = all_memories
        .iter()
        .map(|(m, _)| m.access_count)
        .max()
        .unwrap_or(0);

    let coeffs = recall_score_coefficients_from_env();

    // Overfetch candidates, then rerank with full score (including decay,
    // recency, and access weighting) to avoid top-K cutoff errors.
    let candidate_count = (limit.saturating_mul(CANDIDATE_MULTIPLIER))
        .max(MIN_CANDIDATES)
        .min(all_memories.len());

    // BM25
    let bm25_ranked = bm25_search(query, &all_memories, candidate_count);
    let bm25_meta: HashMap<usize, (usize, f32)> = bm25_ranked
        .iter()
        .enumerate()
        .map(|(rank, (idx, score))| (*idx, (rank + 1, *score)))
        .collect();

    // Vector
    let query_embedding = embedder
        .embed_one(query)
        .map_err(|e| RecallError::Embedding(e.to_string()))?;
    let vector_ranked = vector_search(&query_embedding, &all_memories, candidate_count);
    let vector_meta: HashMap<usize, (usize, f32)> = vector_ranked
        .iter()
        .enumerate()
        .map(|(rank, (idx, sim))| (*idx, (rank + 1, *sim)))
        .collect();

    // RRF fusion
    let fused = rrf(&bm25_ranked, &vector_ranked);
    let candidates = fused.into_iter().take(candidate_count).enumerate();

    // Score = RRF × decayed_strength × recency_boost × access_weight
    let mut results: Vec<RecallResult> = candidates
        .map(|(rrf_rank, (idx, rrf_score))| {
            let mem = &all_memories[idx].0;
            let decayed_strength = effective_strength(mem, now);
            let recency = recency_boost(mem, now);
            let access = access_weight(mem, max_access);
            let base_score =
                compute_base_score(rrf_score, decayed_strength, recency, access, coeffs);
            let temporal_multiplier = temporal_relevance_multiplier(mem, now);
            let salience_multiplier = operational_salience_multiplier(mem, query, now);
            let scored = base_score * temporal_multiplier * salience_multiplier;
            let bm25 = bm25_meta.get(&idx).copied();
            let vector = vector_meta.get(&idx).copied();
            let matched_modalities = usize::from(bm25.is_some()) + usize::from(vector.is_some());
            RecallResult {
                memory: mem.clone(),
                score: scored,
                explain: RecallScoreExplain {
                    rrf_score,
                    rrf_rank: rrf_rank + 1,
                    rrf_rank_percentile: ((rrf_rank + 1) as f64 / candidate_count as f64)
                        .clamp(0.0, 1.0),
                    bm25_rank: bm25.map(|(rank, _)| rank),
                    bm25_score: bm25.map(|(_, score)| score),
                    vector_rank: vector.map(|(rank, _)| rank),
                    vector_similarity: vector.map(|(_, sim)| sim),
                    modality_agreement: bm25.is_some() && vector.is_some(),
                    matched_modalities,
                    decayed_strength,
                    recency_boost: recency,
                    access_weight: access,
                    base_score,
                    spread_boost: 0.0,
                    temporal_boost: (scored - base_score).max(0.0),
                    score_margin_to_next: None,
                    final_score: base_score,
                },
            }
        })
        .collect();

    // ── Spreading activation ─────────────────────────────────
    // For each scored Fact, boost other results that share a subject or object.
    // This is 1-hop graph traversal inspired by Collins & Loftus (1975).
    let before_spread: Vec<f64> = results.iter().map(|r| r.score).collect();
    spread_activation(&mut results, SPREAD_FACTOR);
    for (i, r) in results.iter_mut().enumerate() {
        r.explain.spread_boost = (r.score - before_spread[i]).max(0.0);
    }

    // ── Temporal co-occurrence boost ─────────────────────────
    // Memories created near the same time as high-scoring results get a small
    // boost, implementing Tulving's encoding specificity / contextual
    // reinstatement principle.
    let before_temporal: Vec<f64> = results.iter().map(|r| r.score).collect();
    temporal_cooccurrence_boost(&mut results);
    for (i, r) in results.iter_mut().enumerate() {
        r.explain.temporal_boost += (r.score - before_temporal[i]).max(0.0);
        r.explain.final_score = r.score;
    }

    sort_recall_results(&mut results);
    results.truncate(limit);
    for i in 0..results.len() {
        let margin = if i + 1 < results.len() {
            Some((results[i].score - results[i + 1].score).max(0.0))
        } else {
            None
        };
        results[i].explain.score_margin_to_next = margin;
    }

    // Touch recalled memories: apply decay first, then reinforce.
    for result in &results {
        let mem = &result.memory;
        let decayed = effective_strength(mem, now);
        let boosted = if is_expired_pending_temporal(mem, now) {
            0.0
        } else {
            (decayed + touch_boost(mem)).min(1.0)
        };
        let context = serde_json::json!({
            "query": query,
            "namespace": namespace,
            "final_score": result.score,
            "rrf_rank": result.explain.rrf_rank,
            "bm25_rank": result.explain.bm25_rank,
            "vector_rank": result.explain.vector_rank,
            "modality_agreement": result.explain.modality_agreement,
        });
        store
            .touch_memory_with_strength_context(mem.id, boosted, now, Some(&context))
            .map_err(RecallError::Db)?;
    }

    Ok(results)
}

/// Recency boost: gentle sigmoid that favours recent memories without
/// completely suppressing old ones. Independent of decay (which handles
/// forgetting); this handles *preference* when scores are close.
///
/// Returns a multiplier in [RECENCY_FLOOR, 1.0].
fn recency_boost(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    let hours_ago = (now - mem.created_at).num_seconds().max(0) as f64 / 3600.0;
    let raw = 1.0 / (1.0 + (hours_ago / RECENCY_HALF_LIFE_HOURS).powf(0.8));
    let kind_multiplier = match &mem.kind {
        MemoryKind::Action(_) | MemoryKind::Intent(_) => {
            if hours_ago <= 48.0 {
                1.35
            } else if hours_ago <= 24.0 * 7.0 {
                1.10
            } else {
                0.75
            }
        }
        _ => 1.0,
    };
    (raw.max(RECENCY_FLOOR) * kind_multiplier).max(RECENCY_FLOOR)
}

/// Access pattern weight: memories recalled more often are more consolidated
/// (Hebbian strengthening). Uses log-normalised access count so the effect
/// is gentle and bounded.
///
/// Returns a multiplier in [1.0, 2.0].
fn access_weight(mem: &MemoryRecord, max_access: i64) -> f64 {
    if max_access <= 0 {
        return 1.0;
    }
    let norm = (mem.access_count as f64 + 1.0).log2() / (max_access as f64 + 1.0).log2();
    1.0 + norm // range [1.0, 2.0]
}

/// 1-hop spreading activation through the knowledge graph.
///
/// For every Fact result, other results sharing the same subject or object
/// receive a fractional boost proportional to the parent's score. This
/// implements Collins & Loftus (1975) spreading activation: querying "Max"
/// will also boost "Jared has_pet Max" and "Max visited vet".
fn spread_activation(results: &mut Vec<RecallResult>, factor: f64) {
    // Build index: subject/object → list of result indices.
    let mut entity_index: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        if let MemoryKind::Fact(f) = &r.memory.kind {
            let subj = f.subject.to_lowercase();
            let obj = f.object.to_lowercase();
            entity_index.entry(subj).or_default().push(i);
            entity_index.entry(obj).or_default().push(i);
        }
    }

    // Accumulate boosts (don't mutate while iterating).
    let mut boosts: HashMap<usize, f64> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        if let MemoryKind::Fact(f) = &r.memory.kind {
            let entities = [f.subject.to_lowercase(), f.object.to_lowercase()];
            for entity in &entities {
                if let Some(neighbors) = entity_index.get(entity) {
                    for &ni in neighbors {
                        if ni != i {
                            *boosts.entry(ni).or_insert(0.0) += r.score * factor;
                        }
                    }
                }
            }
        }
    }

    // Apply boosts.
    for (idx, boost) in boosts {
        if idx < results.len() {
            results[idx].score += boost;
        }
    }
}

fn parse_coeff_env(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

fn recall_score_coefficients_from_env() -> RecallScoreCoefficients {
    let d = RecallScoreCoefficients::default();
    RecallScoreCoefficients {
        rrf_exp: parse_coeff_env("CONCH_RECALL_RRF_EXP", d.rrf_exp),
        decay_exp: parse_coeff_env("CONCH_RECALL_DECAY_EXP", d.decay_exp),
        recency_exp: parse_coeff_env("CONCH_RECALL_RECENCY_EXP", d.recency_exp),
        access_exp: parse_coeff_env("CONCH_RECALL_ACCESS_EXP", d.access_exp),
    }
}

fn compute_base_score(
    rrf_score: f64,
    decayed_strength: f64,
    recency: f64,
    access: f64,
    coeffs: RecallScoreCoefficients,
) -> f64 {
    rrf_score.powf(coeffs.rrf_exp)
        * decayed_strength.powf(coeffs.decay_exp)
        * recency.powf(coeffs.recency_exp)
        * access.powf(coeffs.access_exp)
}

fn sort_recall_results(results: &mut [RecallResult]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });
}

/// Temporal co-occurrence boost: memories created within 30 minutes of a
/// high-scoring result get a small boost, implementing contextual
/// reinstatement (Tulving & Thomson, 1973).
fn temporal_cooccurrence_boost(results: &mut Vec<RecallResult>) {
    if results.len() < 2 {
        return;
    }

    // Use the top 5 results as "anchors" — don't let every result boost every other.
    let mut sorted_indices: Vec<usize> = (0..results.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        results[b]
            .score
            .partial_cmp(&results[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| results[a].memory.id.cmp(&results[b].memory.id))
    });
    let anchor_count = sorted_indices.len().min(5);
    let anchors: Vec<(usize, f64, chrono::DateTime<Utc>)> = sorted_indices[..anchor_count]
        .iter()
        .map(|&i| (i, results[i].score, results[i].memory.created_at))
        .collect();

    let mut boosts: HashMap<usize, f64> = HashMap::new();
    for (ai, a_score, a_time) in &anchors {
        for (j, r) in results.iter().enumerate() {
            if j == *ai {
                continue;
            }
            let gap_minutes = (*a_time - r.memory.created_at).num_minutes().unsigned_abs() as f64;
            if gap_minutes < 30.0 {
                let proximity = 0.1 * (1.0 - gap_minutes / 30.0);
                *boosts.entry(j).or_insert(0.0) += a_score * proximity;
            }
        }
    }

    for (idx, boost) in boosts {
        if idx < results.len() {
            results[idx].score += boost;
        }
    }
}

fn kind_decay_lambda_per_day(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_DECAY_LAMBDA_PER_DAY,
        MemoryKind::Episode(_) => EPISODE_DECAY_LAMBDA_PER_DAY,
        MemoryKind::Action(_) | MemoryKind::Intent(_) => ACTION_DECAY_LAMBDA_PER_DAY,
    }
}

fn touch_boost(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_TOUCH_BOOST,
        MemoryKind::Episode(_) => EPISODE_TOUCH_BOOST,
        MemoryKind::Action(_) | MemoryKind::Intent(_) => ACTION_TOUCH_BOOST,
    }
}

fn effective_strength(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    if is_expired_pending_temporal(mem, now) {
        return 0.0;
    }
    let elapsed_secs = (now - mem.last_accessed_at).num_seconds().max(0) as f64;
    let elapsed_days = elapsed_secs / 86_400.0;
    let lambda = kind_decay_lambda_per_day(mem);
    // Importance slows decay: effective_lambda = lambda / (1 + importance)
    // importance=0 → full decay, importance=1 → half the decay rate
    let effective_lambda = lambda / (1.0 + mem.importance);
    (mem.strength * (-effective_lambda * elapsed_days).exp()).clamp(0.0, 1.0)
}

fn is_expired_pending_temporal(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> bool {
    let Some(temporal) = &mem.temporal else {
        return false;
    };
    if !temporal.status.eq_ignore_ascii_case("pending") {
        return false;
    }
    let end = temporal.resolved_end_at.unwrap_or(temporal.resolved_at);
    end <= now
}

fn temporal_relevance_multiplier(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    let Some(t) = &mem.temporal else {
        return 1.0;
    };
    if !t.status.eq_ignore_ascii_case("pending") {
        return 1.0;
    }
    if is_expired_pending_temporal(mem, now) {
        return 0.0;
    }

    // Range-like temporal intent (e.g., end-of-month windows): relevance increases
    // as we approach window start, peaks while inside the window.
    if let Some(end) = t.resolved_end_at {
        if now >= t.resolved_at && now <= end {
            return 1.6;
        }
        if now < t.resolved_at {
            let total = (t.resolved_at - t.utterance_at).num_seconds().max(1) as f64;
            let elapsed = (now - t.utterance_at).num_seconds().clamp(0, total as i64) as f64;
            let progress = (elapsed / total).clamp(0.0, 1.0);
            return 1.0 + 0.7 * progress;
        }
        return 0.0;
    }

    // Deadline-like: increase relevance as due date approaches.
    let hrs_to_due = (t.resolved_at - now).num_seconds().max(0) as f64 / 3600.0;
    if hrs_to_due <= 24.0 {
        1.8
    } else if hrs_to_due <= 24.0 * 7.0 {
        1.4
    } else if hrs_to_due <= 24.0 * 30.0 {
        1.15
    } else {
        1.0
    }
}

fn operational_salience_multiplier(
    mem: &MemoryRecord,
    query: &str,
    now: chrono::DateTime<Utc>,
) -> f64 {
    let q = query.to_ascii_lowercase();
    let text = mem.text_for_embedding().to_ascii_lowercase();

    let has_ops_signal = [
        "api key",
        "self-host",
        "self hosted",
        "credential",
        "token",
        "deploy",
        "plane",
        "dns",
        "cron",
        "server",
    ]
    .iter()
    .any(|k| text.contains(k) || q.contains(k))
        || mem.tags.iter().any(|t| {
            let t = t.to_ascii_lowercase();
            [
                "ops",
                "operational",
                "infra",
                "infrastructure",
                "credentials",
                "deployment",
                "status",
            ]
            .contains(&t.as_str())
        });

    match &mem.kind {
        MemoryKind::Fact(_) if has_ops_signal => {
            let age_days = (now - mem.created_at).num_seconds().max(0) as f64 / 86_400.0;
            if age_days <= 90.0 {
                1.35
            } else {
                1.15
            }
        }
        MemoryKind::Action(_) | MemoryKind::Intent(_) => 1.10,
        _ => 1.0,
    }
}

fn bm25_search(
    query: &str,
    memories: &[(MemoryRecord, String)],
    search_limit: usize,
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
            .b(0.5)
            .build();

    engine
        .search(query, search_limit.max(1).min(memories.len()))
        .into_iter()
        .map(|r| (r.document.id, r.score))
        .collect()
}

fn vector_search(
    query_emb: &[f32],
    memories: &[(MemoryRecord, String)],
    search_limit: usize,
) -> Vec<(usize, f32)> {
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
    scored.truncate(search_limit.max(1).min(scored.len()));
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

    fn test_explain(score: f64) -> RecallScoreExplain {
        RecallScoreExplain {
            rrf_score: score,
            rrf_rank: 1,
            rrf_rank_percentile: 1.0,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_similarity: None,
            modality_agreement: false,
            matched_modalities: 0,
            decayed_strength: 1.0,
            recency_boost: 1.0,
            access_weight: 1.0,
            base_score: score,
            spread_boost: 0.0,
            temporal_boost: 0.0,
            score_margin_to_next: None,
            final_score: score,
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

    #[test]
    fn effective_strength_zero_for_expired_pending_temporal() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store
            .remember_episode("follow up in 1 day", Some(&[1.0, 0.0]))
            .unwrap();

        let expired = serde_json::json!({
            "raw_text": "in 1 day",
            "utterance_at": (Utc::now() - chrono::Duration::days(2)).to_rfc3339(),
            "timezone": "-06:00",
            "resolved_at": (Utc::now() - chrono::Duration::days(1)).to_rfc3339(),
            "status": "pending"
        })
        .to_string();
        store
            .conn()
            .execute(
                "UPDATE memories SET temporal_json = ?1 WHERE id = ?2",
                rusqlite::params![expired, id],
            )
            .unwrap();

        let mem = store.get_memory(id).unwrap().unwrap();
        assert_eq!(effective_strength(&mem, Utc::now()), 0.0);
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
                explain: test_explain(1.0),
            },
            RecallResult {
                memory: make_fact_record(2, "Tortellini", "is_a", "dog"),
                score: 0.1, // low initial score
                explain: test_explain(0.1),
            },
            RecallResult {
                memory: make_fact_record(3, "Abby", "likes", "cats"),
                score: 0.1, // unrelated
                explain: test_explain(0.1),
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
                explain: test_explain(0.8),
            },
            RecallResult {
                memory: make_fact_record(2, "Microsoft", "located_in", "Seattle"),
                score: 0.3,
                explain: test_explain(0.3),
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
            explain: test_explain(1.0),
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
                explain: test_explain(1.0),
            },
            RecallResult {
                memory: make_timed_episode(
                    2,
                    "alpha nearby memory",
                    now - chrono::Duration::minutes(5),
                ),
                score: 0.2,
                explain: test_explain(0.2),
            },
            RecallResult {
                memory: make_timed_episode(
                    3,
                    "alpha distant memory",
                    now - chrono::Duration::hours(3),
                ),
                score: 0.2,
                explain: test_explain(0.2),
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
                explain: test_explain(1.0),
            },
            RecallResult {
                memory: make_timed_episode(
                    2,
                    "alpha very close",
                    now - chrono::Duration::minutes(2),
                ),
                score: 0.1,
                explain: test_explain(0.1),
            },
            RecallResult {
                memory: make_timed_episode(3, "alpha further", now - chrono::Duration::minutes(25)),
                score: 0.1,
                explain: test_explain(0.1),
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

    #[test]
    fn sorting_is_deterministic_for_equal_scores() {
        let now = Utc::now();
        let mut results = vec![
            RecallResult {
                memory: make_timed_episode(10, "alpha a", now),
                score: 1.0,
                explain: test_explain(1.0),
            },
            RecallResult {
                memory: make_timed_episode(3, "alpha b", now),
                score: 1.0,
                explain: test_explain(1.0),
            },
            RecallResult {
                memory: make_timed_episode(7, "alpha c", now),
                score: 1.0,
                explain: test_explain(1.0),
            },
        ];

        sort_recall_results(&mut results);
        let ids: Vec<i64> = results.iter().map(|r| r.memory.id).collect();
        assert_eq!(ids, vec![3, 7, 10]);
    }

    #[test]
    fn env_coefficients_override_defaults() {
        unsafe {
            std::env::set_var("CONCH_RECALL_RRF_EXP", "2.5");
            std::env::set_var("CONCH_RECALL_DECAY_EXP", "1.2");
            std::env::set_var("CONCH_RECALL_RECENCY_EXP", "0.8");
            std::env::set_var("CONCH_RECALL_ACCESS_EXP", "1.1");
        }

        let c = recall_score_coefficients_from_env();
        assert!((c.rrf_exp - 2.5).abs() < 1e-9);
        assert!((c.decay_exp - 1.2).abs() < 1e-9);
        assert!((c.recency_exp - 0.8).abs() < 1e-9);
        assert!((c.access_exp - 1.1).abs() < 1e-9);

        unsafe {
            std::env::remove_var("CONCH_RECALL_RRF_EXP");
            std::env::remove_var("CONCH_RECALL_DECAY_EXP");
            std::env::remove_var("CONCH_RECALL_RECENCY_EXP");
            std::env::remove_var("CONCH_RECALL_ACCESS_EXP");
        }
    }

    #[test]
    fn explainability_includes_rank_metadata() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact("Jared", "likes", "alpha", Some(&[1.0, 0.0]))
            .unwrap();
        store
            .remember_episode("alpha launch notes", Some(&[1.0, 0.0]))
            .unwrap();

        let results = recall(&store, "alpha", &MockEmbedder, 5).unwrap();
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.explain.rrf_rank >= 1);
            assert!((0.0..=1.0).contains(&r.explain.rrf_rank_percentile));
            assert!(r.explain.bm25_rank.is_some() || r.explain.vector_rank.is_some());
            assert!(r.explain.matched_modalities <= 2);
            assert_eq!(
                r.explain.modality_agreement,
                r.explain.matched_modalities == 2
            );
        }
    }

    #[test]
    fn coefficient_tuning_changes_base_score_tradeoff() {
        // Candidate A: strong recency/access, weaker semantic score
        let a = (0.70_f64, 0.95_f64, 0.95_f64, 1.80_f64);
        // Candidate B: stronger semantic/decay, weaker recency/access
        let b = (0.92_f64, 0.95_f64, 0.55_f64, 1.05_f64);

        let default = RecallScoreCoefficients::default();
        let a_default = compute_base_score(a.0, a.1, a.2, a.3, default);
        let b_default = compute_base_score(b.0, b.1, b.2, b.3, default);
        assert!(
            a_default > b_default,
            "default coeffs should favor fresher/high-access memory"
        );

        let semantic_heavy = RecallScoreCoefficients {
            rrf_exp: 2.0,
            decay_exp: 1.0,
            recency_exp: 0.4,
            access_exp: 0.4,
        };
        let a_semantic = compute_base_score(a.0, a.1, a.2, a.3, semantic_heavy);
        let b_semantic = compute_base_score(b.0, b.1, b.2, b.3, semantic_heavy);
        assert!(
            b_semantic > a_semantic,
            "semantic-heavy coeffs should flip ranking preference"
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

    // ── Test helpers ─────────────────────────────────────────

    fn make_fact_record(id: i64, subj: &str, rel: &str, obj: &str) -> MemoryRecord {
        MemoryRecord {
            id,
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
            tags: vec![],
            source: None,
            session_id: None,
            channel: None,
            importance: 0.5,
            namespace: "default".to_string(),
            checksum: None,
            temporal: None,
        }
    }

    fn make_timed_episode(id: i64, text: &str, time: chrono::DateTime<Utc>) -> MemoryRecord {
        MemoryRecord {
            id,
            kind: MemoryKind::Episode(crate::memory::Episode {
                text: text.to_string(),
            }),
            strength: 1.0,
            created_at: time,
            last_accessed_at: time,
            access_count: 0,
            embedding: None,
            tags: vec![],
            source: None,
            session_id: None,
            channel: None,
            importance: 0.5,
            namespace: "default".to_string(),
            checksum: None,
            temporal: None,
        }
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

    // ── Tag filter tests ────────────────────────────────────

    #[test]
    fn recall_with_tag_filter_returns_only_tagged_memories() {
        let store = MemoryStore::open_in_memory().unwrap();

        // Create two memories: one tagged, one not
        store
            .remember_fact_with_tags(
                "Jared",
                "likes",
                "alpha",
                Some(&[1.0, 0.0]),
                &["preference".to_string()],
            )
            .unwrap();
        store
            .remember_fact_with_tags(
                "Jared",
                "uses",
                "alpha",
                Some(&[1.0, 0.0]),
                &["technical".to_string()],
            )
            .unwrap();
        store
            .remember_fact("weather", "is", "alpha", Some(&[1.0, 0.0]))
            .unwrap();

        // Without filter: should find all 3
        let all_results = recall(&store, "alpha", &MockEmbedder, 10).unwrap();
        assert_eq!(all_results.len(), 3);

        // With "preference" filter: should find only 1
        let filtered =
            recall_with_tag_filter(&store, "alpha", &MockEmbedder, 10, Some("preference")).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].memory.tags, vec!["preference"]);

        // With "technical" filter: should find only 1
        let filtered =
            recall_with_tag_filter(&store, "alpha", &MockEmbedder, 10, Some("technical")).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].memory.tags, vec!["technical"]);
    }

    #[test]
    fn recall_with_tag_filter_is_case_insensitive() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact_with_tags(
                "Jared",
                "likes",
                "alpha",
                Some(&[1.0, 0.0]),
                &["Preference".to_string()],
            )
            .unwrap();

        let results =
            recall_with_tag_filter(&store, "alpha", &MockEmbedder, 10, Some("preference")).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn recall_with_no_tag_filter_returns_all() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_fact_with_tags(
                "Jared",
                "likes",
                "alpha",
                Some(&[1.0, 0.0]),
                &["preference".to_string()],
            )
            .unwrap();
        store
            .remember_fact("weather", "is", "alpha", Some(&[1.0, 0.0]))
            .unwrap();

        let results = recall_with_tag_filter(&store, "alpha", &MockEmbedder, 10, None).unwrap();
        assert_eq!(results.len(), 2);
    }
}
