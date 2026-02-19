use std::collections::HashMap;

use chrono::Utc;

use crate::memory::{MemoryKind, MemoryRecord};
use crate::recall::RecallResult;

/// Global decay constants (lambda/day) by memory kind.
///
/// These are intentionally code-level policy constants (not stored per-memory),
/// so tuning affects all memories immediately.
pub(crate) const FACT_DECAY_LAMBDA_PER_DAY: f64 = 0.02;
pub(crate) const EPISODE_DECAY_LAMBDA_PER_DAY: f64 = 0.06;

/// Reinforcement boost applied when a memory is touched.
pub(crate) const FACT_TOUCH_BOOST: f64 = 0.10;
pub(crate) const EPISODE_TOUCH_BOOST: f64 = 0.20;

/// Spreading activation: fraction of a memory's score given to graph neighbors.
pub(crate) const SPREAD_FACTOR: f64 = 0.15;

/// Recency boost half-life in hours (7 days). Memories newer than this get a
/// meaningful boost; older ones taper towards a floor.
pub(crate) const RECENCY_HALF_LIFE_HOURS: f64 = 168.0;

/// Minimum recency multiplier so old memories aren't completely suppressed.
pub(crate) const RECENCY_FLOOR: f64 = 0.3;

const TEMPORAL_WINDOW_MINUTES: f64 = 30.0;
const TEMPORAL_ANCHOR_LIMIT: usize = 5;
const TEMPORAL_MAX_PROXIMITY: f64 = 0.1;

/// Recency boost: gentle sigmoid that favours recent memories without
/// completely suppressing old ones. Independent of decay (which handles
/// forgetting); this handles *preference* when scores are close.
///
/// Returns a multiplier in [RECENCY_FLOOR, 1.0].
pub(crate) fn recency_boost(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    let hours_ago = (now - mem.created_at).num_seconds().max(0) as f64 / 3600.0;
    let raw = 1.0 / (1.0 + (hours_ago / RECENCY_HALF_LIFE_HOURS).powf(0.8));
    raw.max(RECENCY_FLOOR)
}

/// Access pattern weight: memories recalled more often are more consolidated
/// (Hebbian strengthening). Uses log-normalised access count so the effect
/// is gentle and bounded.
///
/// Returns a multiplier in [1.0, 2.0].
pub(crate) fn access_weight(mem: &MemoryRecord, max_access: i64) -> f64 {
    if max_access <= 0 {
        return 1.0;
    }
    let norm = (mem.access_count as f64 + 1.0).log2() / (max_access as f64 + 1.0).log2();
    1.0 + norm // range [1.0, 2.0]
}

pub(crate) fn kind_decay_lambda_per_day(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_DECAY_LAMBDA_PER_DAY,
        MemoryKind::Episode(_) => EPISODE_DECAY_LAMBDA_PER_DAY,
    }
}

pub(crate) fn touch_boost(mem: &MemoryRecord) -> f64 {
    match &mem.kind {
        MemoryKind::Fact(_) => FACT_TOUCH_BOOST,
        MemoryKind::Episode(_) => EPISODE_TOUCH_BOOST,
    }
}

pub(crate) fn effective_strength(mem: &MemoryRecord, now: chrono::DateTime<Utc>) -> f64 {
    let elapsed_secs = (now - mem.last_accessed_at).num_seconds().max(0) as f64;
    let elapsed_days = elapsed_secs / 86_400.0;
    let lambda = kind_decay_lambda_per_day(mem);
    (mem.strength * (-lambda * elapsed_days).exp()).clamp(0.0, 1.0)
}

/// 1-hop spreading activation through the knowledge graph.
///
/// For every Fact result, other results sharing the same subject or object
/// receive a fractional boost proportional to the parent's score. This
/// implements Collins & Loftus (1975) spreading activation: querying "Max"
/// will also boost "Jared has_pet Max" and "Max visited vet".
pub(crate) fn spread_activation_boosts(results: &[RecallResult], factor: f64) -> Vec<f64> {
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

    let mut out = vec![0.0; results.len()];
    for (idx, boost) in boosts {
        if idx < out.len() {
            out[idx] += boost;
        }
    }
    out
}

pub(crate) fn apply_boosts(results: &mut [RecallResult], boosts: &[f64]) {
    for (result, boost) in results.iter_mut().zip(boosts.iter()) {
        result.score += *boost;
    }
}

pub(crate) fn spread_activation(results: &mut [RecallResult], factor: f64) {
    let boosts = spread_activation_boosts(results, factor);
    apply_boosts(results, &boosts);
}

/// Temporal co-occurrence boost: memories created within 30 minutes of a
/// high-scoring result get a small boost, implementing contextual
/// reinstatement (Tulving & Thomson, 1973).
pub(crate) fn temporal_cooccurrence_boosts(results: &[RecallResult]) -> Vec<f64> {
    if results.len() < 2 {
        return vec![0.0; results.len()];
    }

    // Use the top 5 results as "anchors" — don't let every result boost every other.
    let mut sorted_indices: Vec<usize> = (0..results.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        results[b]
            .score
            .partial_cmp(&results[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let anchor_count = sorted_indices.len().min(TEMPORAL_ANCHOR_LIMIT);
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
            if gap_minutes < TEMPORAL_WINDOW_MINUTES {
                let proximity =
                    TEMPORAL_MAX_PROXIMITY * (1.0 - gap_minutes / TEMPORAL_WINDOW_MINUTES);
                *boosts.entry(j).or_insert(0.0) += a_score * proximity;
            }
        }
    }

    let mut out = vec![0.0; results.len()];
    for (idx, boost) in boosts {
        if idx < out.len() {
            out[idx] += boost;
        }
    }
    out
}

pub(crate) fn temporal_cooccurrence_boost(results: &mut [RecallResult]) {
    let boosts = temporal_cooccurrence_boosts(results);
    apply_boosts(results, &boosts);
}
