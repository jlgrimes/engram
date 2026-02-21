//! Isomorphic retrieval: pattern-based memory recall via Mycelium integration.
//!
//! Standard recall finds memories by lexical/vector similarity.
//! Isomorphic recall first extracts the *structural pattern* of a query
//! using Mycelium's cross-domain reasoning engine, then retrieves memories
//! that match those patterns — even if they use completely different vocabulary.
//!
//! Example: "How do I handle a project that's losing momentum?" →
//!   - abstract_shape: "managing resource decline toward graceful exit"
//!   - analogies: "forest succession ecology", "deprecated API migration", "cancer remission strategy"
//!   - Retrieves: memories about any of those patterns across all domains
//!
//! Falls back gracefully to direct recall if Mycelium is unavailable.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::recall::{recall, RecallError, RecallResult};
use crate::store::MemoryStore;

/// Default Mycelium server URL.
pub const DEFAULT_MYCELIUM_URL: &str = "http://127.0.0.1:8787";

/// Mycelium `/solve` request body.
#[derive(Debug, Serialize)]
struct MyceliumRequest<'a> {
    input: &'a str,
}

/// Mycelium `/solve` response body.
/// Mirrors `mycelium-types::ProblemResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyceliumResponse {
    pub abstract_shape: String,
    pub cross_domain_matches: Vec<String>,
    pub mapping: String,
    pub synthesis: String,
}

/// How a memory was surfaced in an isomorphic search.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value")]
pub enum RetrievalSource {
    /// Matched by the original query directly.
    Direct,
    /// Matched by the abstract pattern shape extracted by Mycelium.
    AbstractShape,
    /// Matched by a cross-domain analog from Mycelium.
    Analogy(String),
}

/// A single result from isomorphic recall — a memory + how it was found.
#[derive(Debug, Clone, Serialize)]
pub struct IsomorphicResult {
    #[serde(flatten)]
    pub recall: RecallResult,
    pub source: RetrievalSource,
}

/// Full output of an isomorphic recall operation.
#[derive(Debug, Clone, Serialize)]
pub struct IsomorphicRecallResult {
    /// The original query.
    pub query: String,
    /// The structural abstraction Mycelium extracted (empty if Mycelium unavailable).
    pub abstract_shape: String,
    /// The cross-domain analogies Mycelium found (empty if Mycelium unavailable).
    pub cross_domain_matches: Vec<String>,
    /// Mycelium's synthesis / recommended framing (empty if Mycelium unavailable).
    pub synthesis: String,
    /// Whether Mycelium was reachable. If false, `results` are all `Direct`.
    pub mycelium_available: bool,
    /// Ranked recall results, deduped by memory id.
    pub results: Vec<IsomorphicResult>,
}

/// Error type for isomorphic recall.
#[derive(Debug, thiserror::Error)]
pub enum IsomorphicError {
    #[error("recall error: {0}")]
    Recall(#[from] RecallError),
}

/// Call Mycelium's `/solve` endpoint (blocking).
/// Returns `None` if the server is unreachable or returns bad data.
fn call_mycelium(query: &str, base_url: &str) -> Option<MyceliumResponse> {
    let url = format!("{}/solve", base_url.trim_end_matches('/'));
    let body = MyceliumRequest { input: query };
    match ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&body)
    {
        Ok(resp) => resp.into_json::<MyceliumResponse>().ok(),
        Err(_) => None,
    }
}

/// Isomorphic recall: combines direct recall with pattern-based retrieval.
///
/// # Algorithm
/// 1. Send query to Mycelium `/solve` to extract abstract shape + cross-domain analogies.
/// 2. Run `recall` for: original query, abstract shape, each analogy.
/// 3. Deduplicate by memory id (keep first occurrence, which has highest direct score).
/// 4. Sort by score descending, truncate to `limit`.
///
/// If Mycelium is unreachable, falls back to direct recall only.
pub fn isomorphic_recall(
    store: &MemoryStore,
    query: &str,
    embedder: &dyn Embedder,
    limit: usize,
    mycelium_url: &str,
) -> Result<IsomorphicRecallResult, IsomorphicError> {
    let mycelium = call_mycelium(query, mycelium_url);
    let mycelium_available = mycelium.is_some();

    let mut all_results: Vec<IsomorphicResult> = Vec::new();
    let mut seen_ids: HashSet<i64> = HashSet::new();

    // Helper: drain recall results into all_results, deduping by id.
    let add_results = |all: &mut Vec<IsomorphicResult>,
                       seen: &mut HashSet<i64>,
                       source: RetrievalSource,
                       items: Vec<RecallResult>| {
        for r in items {
            if seen.insert(r.memory.id) {
                all.push(IsomorphicResult { recall: r, source: source.clone() });
            }
        }
    };

    // 1. Direct recall — always runs.
    let overfetch = (limit * 3).max(20);
    let direct = recall(store, query, embedder, overfetch)?;
    add_results(&mut all_results, &mut seen_ids, RetrievalSource::Direct, direct);

    // 2. If Mycelium responded, add abstract-shape recall and analog recalls.
    let (abstract_shape, cross_domain_matches, synthesis) = if let Some(ref mres) = mycelium {
        // Abstract shape recall
        if !mres.abstract_shape.is_empty() {
            let shape_results = recall(store, &mres.abstract_shape, embedder, overfetch)?;
            add_results(
                &mut all_results,
                &mut seen_ids,
                RetrievalSource::AbstractShape,
                shape_results,
            );
        }

        // Analogy recalls
        for analogy in &mres.cross_domain_matches {
            if analogy.is_empty() {
                continue;
            }
            let analog_results = recall(store, analogy, embedder, overfetch)?;
            add_results(
                &mut all_results,
                &mut seen_ids,
                RetrievalSource::Analogy(analogy.clone()),
                analog_results,
            );
        }

        (
            mres.abstract_shape.clone(),
            mres.cross_domain_matches.clone(),
            mres.synthesis.clone(),
        )
    } else {
        (String::new(), Vec::new(), String::new())
    };

    // Sort by score descending, truncate to limit.
    all_results.sort_by(|a, b| {
        b.recall.score
            .partial_cmp(&a.recall.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(limit);

    Ok(IsomorphicRecallResult {
        query: query.to_string(),
        abstract_shape,
        cross_domain_matches,
        synthesis,
        mycelium_available,
        results: all_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{EmbedError, Embedding};
    use crate::store::MemoryStore;

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.contains("alpha") || t.contains("pattern") {
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

    /// When Mycelium is unavailable, isomorphic_recall should fall back to direct
    /// recall and return `mycelium_available = false`.
    #[test]
    fn fallback_when_mycelium_unavailable() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha important context", Some(&[1.0, 0.0]))
            .unwrap();

        let result = isomorphic_recall(
            &store,
            "alpha",
            &MockEmbedder,
            5,
            "http://127.0.0.1:19999", // nothing listening here
        )
        .unwrap();

        assert!(!result.mycelium_available, "Mycelium should be marked unavailable");
        assert_eq!(result.abstract_shape, "", "abstract_shape should be empty");
        assert!(result.cross_domain_matches.is_empty());
        assert!(!result.results.is_empty(), "should still return direct recall results");
        assert!(
            result.results.iter().all(|r| r.source == RetrievalSource::Direct),
            "all results should be Direct when Mycelium is down"
        );
    }

    /// Results should be deduped: same memory surfaced via direct AND analogy
    /// should appear only once.
    #[test]
    fn results_are_deduped_by_memory_id() {
        let store = MemoryStore::open_in_memory().unwrap();
        store
            .remember_episode("alpha pattern match", Some(&[1.0, 0.0]))
            .unwrap();

        // Simulate calling with a Mycelium mock: we test the dedup logic by
        // running the same memory through two recall passes.
        let result1 = recall(&store, "alpha", &MockEmbedder, 10).unwrap();
        let result2 = recall(&store, "alpha", &MockEmbedder, 10).unwrap();

        // Merge manually the way isomorphic_recall does.
        let mut seen: HashSet<i64> = HashSet::new();
        let mut merged: Vec<IsomorphicResult> = Vec::new();

        for r in result1 {
            if seen.insert(r.memory.id) {
                merged.push(IsomorphicResult { recall: r, source: RetrievalSource::Direct });
            }
        }
        for r in result2 {
            if seen.insert(r.memory.id) {
                merged.push(IsomorphicResult {
                    recall: r,
                    source: RetrievalSource::AbstractShape,
                });
            }
        }

        assert_eq!(merged.len(), 1, "duplicate memory should appear only once");
        assert_eq!(merged[0].source, RetrievalSource::Direct, "first-seen source wins");
    }

    /// Empty store should return empty results without panicking.
    #[test]
    fn empty_store_returns_empty_results() {
        let store = MemoryStore::open_in_memory().unwrap();

        let result = isomorphic_recall(
            &store,
            "anything",
            &MockEmbedder,
            5,
            "http://127.0.0.1:19999",
        )
        .unwrap();

        assert!(result.results.is_empty());
    }

    /// Limit is respected even when many memories match.
    #[test]
    fn limit_is_respected() {
        let store = MemoryStore::open_in_memory().unwrap();
        for i in 0..20 {
            store
                .remember_episode(&format!("alpha memory {i}"), Some(&[1.0, 0.0]))
                .unwrap();
        }

        let result = isomorphic_recall(
            &store,
            "alpha",
            &MockEmbedder,
            5,
            "http://127.0.0.1:19999",
        )
        .unwrap();

        assert!(
            result.results.len() <= 5,
            "results ({}) should be <= limit (5)",
            result.results.len()
        );
    }

    /// RetrievalSource::Analogy carries the analogy string.
    #[test]
    fn analogy_source_carries_text() {
        let analogy = "music theory: harmonic tension resolution".to_string();
        let source = RetrievalSource::Analogy(analogy.clone());
        match source {
            RetrievalSource::Analogy(s) => assert_eq!(s, analogy),
            _ => panic!("wrong variant"),
        }
    }
}
