use crate::memory::Association;
use crate::store::MemoryStore;

/// Create a named association between two entities.
///
/// Associations are bidirectional — querying either entity will find this link.
/// Duplicate associations (same entity_a, relation, entity_b) are silently ignored.
pub fn relate(
    store: &MemoryStore,
    entity_a: &str,
    relation: &str,
    entity_b: &str,
) -> Result<i64, rusqlite::Error> {
    store.relate(entity_a, relation, entity_b)
}

/// Find all associations for a given entity (in either position).
pub fn find_associations(
    store: &MemoryStore,
    entity: &str,
) -> Result<Vec<Association>, rusqlite::Error> {
    store.get_associations(entity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relate_and_find() {
        let store = MemoryStore::open_in_memory().unwrap();

        relate(&store, "Jared", "friend of", "Alice").unwrap();

        let found = find_associations(&store, "Jared").unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].entity_a, "Jared");
        assert_eq!(found[0].relation, "friend of");
        assert_eq!(found[0].entity_b, "Alice");

        // Also findable from the other direction
        let found_b = find_associations(&store, "Alice").unwrap();
        assert_eq!(found_b.len(), 1);
    }

    #[test]
    fn test_multiple_associations() {
        let store = MemoryStore::open_in_memory().unwrap();

        relate(&store, "Jared", "works at", "Microsoft").unwrap();
        relate(&store, "Jared", "lives in", "Seattle").unwrap();
        relate(&store, "Alice", "works at", "Google").unwrap();

        let jared_assocs = find_associations(&store, "Jared").unwrap();
        assert_eq!(jared_assocs.len(), 2);

        let microsoft_assocs = find_associations(&store, "Microsoft").unwrap();
        assert_eq!(microsoft_assocs.len(), 1);
    }

    #[test]
    fn test_duplicate_association_ignored() {
        let store = MemoryStore::open_in_memory().unwrap();

        relate(&store, "A", "knows", "B").unwrap();
        relate(&store, "A", "knows", "B").unwrap(); // duplicate

        let assocs = find_associations(&store, "A").unwrap();
        assert_eq!(assocs.len(), 1);
    }
}
