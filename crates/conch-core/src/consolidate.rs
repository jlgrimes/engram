use crate::embed::cosine_similarity;
use crate::memory::MemoryRecord;
use crate::store::MemoryStore;

/// Minimum cosine similarity to consider two memories as belonging to the same cluster.
const CONSOLIDATION_THRESHOLD: f32 = 0.80;

/// Result of a consolidation pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidateResult {
    /// Number of clusters found.
    pub clusters: usize,
    /// Number of memories archived (strength set to 0) or deleted.
    pub archived: usize,
    /// Number of canonical memories that were boosted.
    pub boosted: usize,
}

/// A cluster of related memories found during consolidation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidateCluster {
    /// The canonical (strongest) memory in the cluster.
    pub canonical: MemoryRecord,
    /// The weaker duplicates that would be archived.
    pub duplicates: Vec<MemoryRecord>,
}

/// Find clusters of related memories by pairwise cosine similarity.
/// Returns clusters where each cluster has 2+ members with similarity > threshold.
pub fn find_clusters(
    store: &MemoryStore,
    threshold: Option<f32>,
) -> Result<Vec<ConsolidateCluster>, rusqlite::Error> {
    let threshold = threshold.unwrap_or(CONSOLIDATION_THRESHOLD);
    let all = store.all_embeddings()?;
    if all.len() < 2 {
        return Ok(vec![]);
    }

    // Build adjacency: for each pair with sim > threshold, record the link.
    let n = all.len();
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    // Simple single-linkage clustering
    for i in 0..n {
        if assigned[i] {
            continue;
        }
        let mut cluster = vec![i];
        assigned[i] = true;

        // Find all unassigned items similar to any member of this cluster
        let mut frontier = vec![i];
        while let Some(current) = frontier.pop() {
            for j in 0..n {
                if assigned[j] {
                    continue;
                }
                let sim = cosine_similarity(&all[current].1, &all[j].1);
                if sim > threshold {
                    assigned[j] = true;
                    cluster.push(j);
                    frontier.push(j);
                }
            }
        }

        if cluster.len() >= 2 {
            clusters.push(cluster);
        }
    }

    // Convert index clusters to MemoryRecord clusters
    let mut result = Vec::new();
    for cluster_indices in clusters {
        let mut members: Vec<MemoryRecord> = Vec::new();
        for &idx in &cluster_indices {
            let id = all[idx].0;
            if let Some(mem) = store.get_memory(id)? {
                members.push(mem);
            }
        }
        if members.len() < 2 {
            continue;
        }
        // Pick the strongest as canonical
        members.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        let canonical = members.remove(0);
        result.push(ConsolidateCluster {
            canonical,
            duplicates: members,
        });
    }

    Ok(result)
}

/// Run consolidation: find clusters, boost canonical, archive duplicates.
/// Merges tags from duplicates into the canonical memory.
pub fn consolidate(
    store: &MemoryStore,
    threshold: Option<f32>,
) -> Result<ConsolidateResult, rusqlite::Error> {
    let clusters = find_clusters(store, threshold)?;
    let num_clusters = clusters.len();
    let mut archived = 0;
    let mut boosted = 0;

    for cluster in &clusters {
        // Merge tags from duplicates into canonical
        let mut all_tags: Vec<String> = cluster.canonical.tags.clone();
        for dup in &cluster.duplicates {
            for tag in &dup.tags {
                if !all_tags.contains(tag) {
                    all_tags.push(tag.clone());
                }
            }
        }
        if all_tags != cluster.canonical.tags {
            store.update_tags(cluster.canonical.id, &all_tags)?;
        }

        // Boost canonical strength
        let boost = 0.1 * cluster.duplicates.len() as f64;
        store.reinforce_memory(cluster.canonical.id, boost)?;
        boosted += 1;

        // Archive duplicates (delete them)
        for dup in &cluster.duplicates {
            store.delete_memory(dup.id)?;
            archived += 1;
        }
    }

    Ok(ConsolidateResult {
        clusters: num_clusters,
        archived,
        boosted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_clusters_groups_similar_embeddings() {
        let store = MemoryStore::open_in_memory().unwrap();
        // Two very similar embeddings (same direction)
        store.remember_fact("A", "is", "B", Some(&[1.0, 0.0, 0.0])).unwrap();
        store.remember_fact("A", "is", "C", Some(&[0.99, 0.1, 0.0])).unwrap();
        // One very different embedding
        store.remember_fact("X", "is", "Y", Some(&[0.0, 0.0, 1.0])).unwrap();

        let clusters = find_clusters(&store, Some(0.80)).unwrap();
        assert_eq!(clusters.len(), 1, "should find 1 cluster of 2 similar memories");
        assert_eq!(clusters[0].duplicates.len(), 1);
    }

    #[test]
    fn find_clusters_returns_empty_for_dissimilar() {
        let store = MemoryStore::open_in_memory().unwrap();
        // All orthogonal embeddings
        store.remember_fact("A", "is", "B", Some(&[1.0, 0.0, 0.0])).unwrap();
        store.remember_fact("C", "is", "D", Some(&[0.0, 1.0, 0.0])).unwrap();
        store.remember_fact("E", "is", "F", Some(&[0.0, 0.0, 1.0])).unwrap();

        let clusters = find_clusters(&store, Some(0.80)).unwrap();
        assert!(clusters.is_empty(), "should find no clusters for orthogonal embeddings");
    }

    #[test]
    fn consolidate_merges_tags_and_archives() {
        let store = MemoryStore::open_in_memory().unwrap();
        // Two similar memories with different tags
        let _id1 = store.remember_fact_with_tags("A", "is", "B", Some(&[1.0, 0.0, 0.0]), &["tag1".to_string()]).unwrap();
        let _id2 = store.remember_fact_with_tags("A", "is", "C", Some(&[0.99, 0.1, 0.0]), &["tag2".to_string()]).unwrap();

        let result = consolidate(&store, Some(0.80)).unwrap();
        assert_eq!(result.clusters, 1);
        assert_eq!(result.archived, 1);
        assert_eq!(result.boosted, 1);

        // Check that one memory is gone
        let remaining = store.all_memories().unwrap();
        assert_eq!(remaining.len(), 1);

        // The surviving memory should have merged tags
        let survivor = &remaining[0];
        assert!(survivor.tags.contains(&"tag1".to_string()));
        assert!(survivor.tags.contains(&"tag2".to_string()));
    }

    #[test]
    fn consolidate_picks_strongest_as_canonical() {
        let store = MemoryStore::open_in_memory().unwrap();
        let id1 = store.remember_fact("weak", "is", "memory", Some(&[1.0, 0.0, 0.0])).unwrap();
        let id2 = store.remember_fact("strong", "is", "memory", Some(&[0.99, 0.1, 0.0])).unwrap();

        // Make id2 stronger
        store.conn().execute(
            "UPDATE memories SET strength = 0.9 WHERE id = ?1",
            rusqlite::params![id1],
        ).unwrap();
        store.conn().execute(
            "UPDATE memories SET strength = 1.0 WHERE id = ?1",
            rusqlite::params![id2],
        ).unwrap();

        let result = consolidate(&store, Some(0.80)).unwrap();
        assert_eq!(result.clusters, 1);

        let remaining = store.all_memories().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, id2, "strongest memory should survive");
    }

    #[test]
    fn consolidate_dry_run_via_find_clusters() {
        let store = MemoryStore::open_in_memory().unwrap();
        store.remember_fact("A", "is", "B", Some(&[1.0, 0.0, 0.0])).unwrap();
        store.remember_fact("A", "is", "C", Some(&[0.99, 0.1, 0.0])).unwrap();

        // Dry run: just find clusters without consolidating
        let clusters = find_clusters(&store, Some(0.80)).unwrap();
        assert_eq!(clusters.len(), 1);

        // Original memories should still exist
        let all = store.all_memories().unwrap();
        assert_eq!(all.len(), 2);
    }
}
