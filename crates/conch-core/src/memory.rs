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
    #[serde(default = "default_importance")]
    pub importance: f64,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

fn default_importance() -> f64 {
    0.5
}

fn default_namespace() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: i64,
    pub total_facts: i64,
    pub total_episodes: i64,
    pub avg_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WriteRetryStats {
    pub retrying_events: usize,
    pub recovered_events: usize,
    pub failed_events: usize,
    pub recovered_retries_total: usize,
    pub failed_retries_total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub memories: Vec<MemoryRecord>,
}

// ── Audit log types ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub memory_id: Option<i64>,
    pub actor: String,
    pub details_json: Option<String>,
    /// SHA-256 hash chaining this entry to the previous one (tamper evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_hash: Option<String>,
}

// ── Audit integrity types ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperedAuditEntry {
    pub id: i64,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditIntegrityResult {
    pub total: usize,
    pub valid: usize,
    pub tampered: Vec<TamperedAuditEntry>,
}

// ── Verification types ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    pub total_checked: usize,
    pub valid: usize,
    pub corrupted: Vec<CorruptedMemory>,
    pub missing_checksum: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptedMemory {
    pub id: i64,
    pub expected: String,
    pub actual: String,
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

/// A node in a graph traversal result, representing a memory at a certain hop distance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub memory: MemoryRecord,
    /// How many hops from the query subject (0 = direct match).
    pub depth: usize,
    /// The entity (subject or object) that connects this node to the previous hop.
    pub connected_via: String,
}

/// Provenance information for a single memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceInfo {
    pub memory: MemoryRecord,
    pub created_at: String,
    pub last_accessed_at: String,
    pub access_count: i64,
    pub strength: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    /// 1-hop related facts for context.
    pub related: Vec<GraphNode>,
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

    /// Returns the object of this memory if it's a Fact.
    pub fn object(&self) -> Option<&str> {
        match &self.kind {
            MemoryKind::Fact(f) => Some(&f.object),
            MemoryKind::Episode(_) => None,
        }
    }
}
