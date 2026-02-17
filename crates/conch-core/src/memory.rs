use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A fact is a subject-relation-object triple.
/// e.g., ("Jared", "works at", "Microsoft")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// An episode is a free-text event description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub text: String,
}

/// The kind of memory stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryKind {
    Fact(Fact),
    Episode(Episode),
}

/// A stored memory record with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub kind: MemoryKind,
    pub strength: f64,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: i64,
}

/// An associative link between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    pub id: i64,
    pub entity_a: String,
    pub relation: String,
    pub entity_b: String,
    pub created_at: DateTime<Utc>,
}

/// Stats about the memory database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: i64,
    pub total_facts: i64,
    pub total_episodes: i64,
    pub total_associations: i64,
    pub avg_strength: f64,
}

/// Full database export: all memories and associations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub memories: Vec<MemoryRecord>,
    pub associations: Vec<Association>,
}

impl MemoryRecord {
    /// The text content used for embedding generation.
    pub fn text_for_embedding(&self) -> String {
        match &self.kind {
            MemoryKind::Fact(f) => format!("{} {} {}", f.subject, f.relation, f.object),
            MemoryKind::Episode(e) => e.text.clone(),
        }
    }

    /// Returns the subject if this is a fact, or None for episodes.
    pub fn subject(&self) -> Option<&str> {
        match &self.kind {
            MemoryKind::Fact(f) => Some(&f.subject),
            MemoryKind::Episode(_) => None,
        }
    }
}
