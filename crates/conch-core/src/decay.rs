use crate::store::MemoryStore;

/// Default decay factor.
const DEFAULT_DECAY_FACTOR: f64 = 0.5;

/// Default half-life in hours (24 hours = memories lose half strength per day of inactivity).
const DEFAULT_HALF_LIFE_HOURS: f64 = 24.0;

/// Minimum strength before a memory is deleted.
const MIN_STRENGTH: f64 = 0.01;

/// Result of a decay pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecayResult {
    pub decayed: usize,
    pub deleted: usize,
}

/// Run a decay pass over all memories.
///
/// For each memory, strength is reduced based on time since last access:
///   new_strength = strength * decay_factor ^ (hours_since_access / half_life_hours)
///
/// With defaults, memories lose half their strength for each 24 hours of inactivity.
/// Memories that fall below MIN_STRENGTH (0.01) are deleted.
pub fn run_decay(
    store: &MemoryStore,
    decay_factor: Option<f64>,
    half_life_hours: Option<f64>,
) -> Result<DecayResult, rusqlite::Error> {
    run_decay_ns(store, decay_factor, half_life_hours, "default")
}

pub fn run_decay_ns(
    store: &MemoryStore,
    decay_factor: Option<f64>,
    half_life_hours: Option<f64>,
    namespace: &str,
) -> Result<DecayResult, rusqlite::Error> {
    let factor = decay_factor.unwrap_or(DEFAULT_DECAY_FACTOR);
    let half_life = half_life_hours.unwrap_or(DEFAULT_HALF_LIFE_HOURS);

    let decayed = store.decay_all_ns(factor, half_life, namespace)?;

    // Delete memories that have decayed below minimum strength (namespace-scoped)
    let deleted: usize = store.conn().execute(
        "DELETE FROM memories WHERE strength < ?1 AND namespace = ?2",
        rusqlite::params![MIN_STRENGTH, namespace],
    )?;

    if deleted > 0 {
        store.log_audit("decay_delete", None, "system", Some(&format!("{{\"deleted\":{deleted},\"namespace\":{}}}", serde_json::json!(namespace))))?;
    }

    Ok(DecayResult { decayed, deleted })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_pass() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();

        // Manually set last_accessed_at to 48 hours ago
        let old_time = (chrono::Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET last_accessed_at = ?1",
                rusqlite::params![old_time],
            )
            .unwrap();

        let result = run_decay(&store, None, None).unwrap();
        assert_eq!(result.decayed, 1);

        // Strength should be reduced (0.5^(48/24) = 0.25)
        let mem = store.get_memory(1).unwrap().unwrap();
        assert!(mem.strength < 0.3);
        assert!(mem.strength > 0.2);
    }

    #[test]
    fn test_no_decay_for_fresh_memories() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();

        let result = run_decay(&store, None, None).unwrap();
        assert_eq!(result.decayed, 0);
        assert_eq!(result.deleted, 0);

        let mem = store.get_memory(1).unwrap().unwrap();
        assert!((mem.strength - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_decay_deletes_very_weak() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", None).unwrap();

        // Set strength very low and last_accessed a long time ago
        let old_time = (chrono::Utc::now() - chrono::Duration::hours(24 * 30)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET strength = 0.001, last_accessed_at = ?1",
                rusqlite::params![old_time],
            )
            .unwrap();

        let result = run_decay(&store, None, None).unwrap();
        assert!(result.deleted > 0);

        // Memory should be gone
        let mem = store.get_memory(1).unwrap();
        assert!(mem.is_none());
    }
}
