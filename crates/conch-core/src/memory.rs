use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryKind {
    Fact(Fact),
    Episode(Episode),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub kind: MemoryKind,
    pub strength: f64,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: i64,
    pub total_facts: i64,
    pub total_episodes: i64,
    pub avg_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub memories: Vec<MemoryRecord>,
}

/// Result of a remember operation, indicating whether the memory was newly
/// created or deduplicated against an existing memory.
#[derive(Debug, Clone, Serialize)]
pub enum RememberResult {
    /// A new memory was created.
    Created(MemoryRecord),
    /// A duplicate was detected (cosine similarity > threshold).
    /// The existing memory was reinforced instead of creating a new one.
    Duplicate {
        existing: MemoryRecord,
        similarity: f32,
    },
    /// An existing fact with the same subject+relation was updated (upsert).
    Updated(MemoryRecord),
}

impl RememberResult {
    /// Returns the memory record regardless of whether it was new, duplicate, or updated.
    pub fn memory(&self) -> &MemoryRecord {
        match self {
            RememberResult::Created(m) => m,
            RememberResult::Duplicate { existing, .. } => existing,
            RememberResult::Updated(m) => m,
        }
    }

    /// Returns true if this was a duplicate detection.
    pub fn is_duplicate(&self) -> bool {
        matches!(self, RememberResult::Duplicate { .. })
    }

    /// Returns true if this was an upsert (existing fact updated).
    pub fn is_updated(&self) -> bool {
        matches!(self, RememberResult::Updated(_))
    }
}

impl MemoryRecord {
    pub fn text_for_embedding(&self) -> String {
        match &self.kind {
            MemoryKind::Fact(f) => format!("{} {} {}", f.subject, f.relation, f.object),
            MemoryKind::Episode(e) => e.text.clone(),
        }
    }

    pub fn subject(&self) -> Option<&str> {
        match &self.kind {
            MemoryKind::Fact(f) => Some(&f.subject),
            MemoryKind::Episode(_) => None,
        }
    }
}
