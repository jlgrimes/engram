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
    pub namespace: String,
    pub kind: MemoryKind,
    pub strength: f64,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: i64,
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
