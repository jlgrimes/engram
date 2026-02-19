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
    let factor = decay_factor.unwrap_or(DEFAULT_DECAY_FACTOR);
    let half_life = half_life_hours.unwrap_or(DEFAULT_HALF_LIFE_HOURS);

    let decayed = store.decay_all(factor, half_life)?;

    // Delete memories that have decayed below minimum strength
    let deleted: usize = store.conn().execute(
        "DELETE FROM memories WHERE strength < ?1",
        rusqlite::params![MIN_STRENGTH],
    )?;

    Ok(DecayResult { decayed, deleted })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_pass() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "is", "B", None).unwrap();

        // Set importance to 0 so we get the base half-life of 24h
        store.update_importance(id, 0.0).unwrap();

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
    fn test_decay_respects_importance() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id_low = store.remember_fact("low", "importance", "mem", None).unwrap();
        let id_high = store.remember_fact("high", "importance", "mem", None).unwrap();

        // Set importance: low=0.0, high=1.0
        store.update_importance(id_low, 0.0).unwrap();
        store.update_importance(id_high, 1.0).unwrap();

        // Set both to 48 hours ago
        let old_time = (chrono::Utc::now() - chrono::Duration::hours(48)).to_rfc3339();
        store
            .conn()
            .execute(
                "UPDATE memories SET last_accessed_at = ?1",
                rusqlite::params![old_time],
            )
            .unwrap();

        let _result = run_decay(&store, None, None).unwrap();

        let low = store.get_memory(id_low).unwrap().unwrap();
        let high = store.get_memory(id_high).unwrap().unwrap();

        // High importance memory should retain more strength
        assert!(
            high.strength > low.strength,
            "high importance ({:.4}) should decay slower than low importance ({:.4})",
            high.strength, low.strength
        );
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
