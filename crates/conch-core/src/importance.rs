use crate::memory::{MemoryKind, MemoryRecord};
use crate::store::MemoryStore;

/// Compute importance score for a memory based on heuristics.
///
/// Heuristics:
/// - Facts with high access_count → higher importance
/// - Memories with more tags → higher importance
/// - Memories with source tracking → slightly higher importance
/// - Episode length (longer = more context = more important)
///
/// Returns a score in [0.0, 1.0].
pub fn compute_importance(mem: &MemoryRecord) -> f64 {
    let mut score = 0.0;
    let mut weights = 0.0;

    // Access count: log-scaled contribution (weight: 0.35)
    // access_count=0 → 0.0, access_count=1 → ~0.3, access_count=10 → ~0.72, access_count=100 → 1.0
    let access_factor = if mem.access_count > 0 {
        ((mem.access_count as f64 + 1.0).log10() / 2.0_f64.log10()).min(1.0)
    } else {
        0.0
    };
    score += 0.35 * access_factor;
    weights += 0.35;

    // Tag count: more tags = more contextualized = more important (weight: 0.20)
    // 0 tags → 0.0, 1 tag → 0.33, 2 tags → 0.67, 3+ tags → 1.0
    let tag_factor = (mem.tags.len() as f64 / 3.0).min(1.0);
    score += 0.20 * tag_factor;
    weights += 0.20;

    // Source tracking: having a source = slightly more important (weight: 0.10)
    let source_factor = if mem.source.is_some() { 1.0 } else { 0.0 };
    score += 0.10 * source_factor;
    weights += 0.10;

    // Content richness based on memory kind (weight: 0.35)
    let content_factor = match &mem.kind {
        MemoryKind::Fact(_) => {
            // Facts are inherently structured, give a base score
            0.5
        }
        MemoryKind::Episode(e) => {
            // Longer episodes contain more context
            // <50 chars → low, 50-200 → medium, 200+ → high
            let len = e.text.len() as f64;
            (len / 200.0).min(1.0)
        }
    };
    score += 0.35 * content_factor;
    weights += 0.35;

    // Normalize to [0.0, 1.0]
    (score / weights).clamp(0.0, 1.0)
}

/// Compute and store importance scores for all memories.
/// Returns the number of memories updated.
pub fn score_all(store: &MemoryStore) -> Result<usize, rusqlite::Error> {
    let memories = store.all_memories()?;
    let mut count = 0;
    for mem in &memories {
        let importance = compute_importance(mem);
        if (importance - mem.importance).abs() > 1e-6 {
            store.update_importance(mem.id, importance)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Result for displaying importance info.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportanceInfo {
    pub id: i64,
    pub content: String,
    pub importance: f64,
    pub access_count: i64,
    pub tag_count: usize,
    pub has_source: bool,
}

/// Get importance info for all memories, sorted by importance descending.
pub fn list_importance(store: &MemoryStore) -> Result<Vec<ImportanceInfo>, rusqlite::Error> {
    let memories = store.all_memories()?;
    let mut infos: Vec<ImportanceInfo> = memories
        .iter()
        .map(|mem| {
            let content = mem.text_for_embedding();
            ImportanceInfo {
                id: mem.id,
                content,
                importance: mem.importance,
                access_count: mem.access_count,
                tag_count: mem.tags.len(),
                has_source: mem.source.is_some(),
            }
        })
        .collect();
    infos.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
    Ok(infos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::memory::{Episode, Fact, MemoryKind};

    fn make_record(kind: MemoryKind, access_count: i64, tags: Vec<String>, source: Option<String>) -> MemoryRecord {
        MemoryRecord {
            id: 1,
            kind,
            strength: 1.0,
            embedding: None,
            created_at: Utc::now(),
            last_accessed_at: Utc::now(),
            access_count,
            tags,
            source,
            session_id: None,
            channel: None,
            importance: 0.5,
        }
    }

    #[test]
    fn high_access_count_increases_importance() {
        let low = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            0, vec![], None,
        );
        let high = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            50, vec![], None,
        );
        assert!(compute_importance(&high) > compute_importance(&low));
    }

    #[test]
    fn more_tags_increases_importance() {
        let no_tags = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            0, vec![], None,
        );
        let with_tags = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            0, vec!["a".into(), "b".into(), "c".into()], None,
        );
        assert!(compute_importance(&with_tags) > compute_importance(&no_tags));
    }

    #[test]
    fn source_increases_importance() {
        let no_source = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            0, vec![], None,
        );
        let with_source = make_record(
            MemoryKind::Fact(Fact { subject: "A".into(), relation: "is".into(), object: "B".into() }),
            0, vec![], Some("cli".into()),
        );
        assert!(compute_importance(&with_source) > compute_importance(&no_source));
    }

    #[test]
    fn longer_episodes_more_important() {
        let short = make_record(
            MemoryKind::Episode(Episode { text: "hi".into() }),
            0, vec![], None,
        );
        let long = make_record(
            MemoryKind::Episode(Episode { text: "This is a very detailed episode describing a complex technical decision about database architecture and schema migration patterns that was made after careful deliberation.".into() }),
            0, vec![], None,
        );
        assert!(compute_importance(&long) > compute_importance(&short));
    }

    #[test]
    fn importance_is_bounded() {
        // Max everything out
        let maxed = make_record(
            MemoryKind::Episode(Episode { text: "x".repeat(500) }),
            1000, vec!["a".into(), "b".into(), "c".into(), "d".into()], Some("cli".into()),
        );
        let imp = compute_importance(&maxed);
        assert!(imp >= 0.0 && imp <= 1.0, "importance should be in [0, 1], got {imp}");
    }

    #[test]
    fn score_all_updates_importance_in_store() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id = store.remember_fact("A", "is", "B", None).unwrap();

        // Bump access count so importance changes from default
        store.conn().execute(
            "UPDATE memories SET access_count = 50 WHERE id = ?1",
            rusqlite::params![id],
        ).unwrap();

        let count = score_all(&store).unwrap();
        assert!(count > 0, "should have updated at least 1 memory");

        let mem = store.get_memory(id).unwrap().unwrap();
        assert!(mem.importance != 0.5, "importance should have changed from default");
    }
}
