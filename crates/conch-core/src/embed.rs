use std::sync::Arc;

pub type Embedding = Vec<f32>;

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError>;

    fn embed_one(&self, text: &str) -> Result<Embedding, EmbedError> {
        let mut results = self.embed(&[text])?;
        results
            .pop()
            .ok_or_else(|| EmbedError::Other("empty embedding result".into()))
    }

    fn dimension(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding model error: {0}")]
    Model(String),
    #[error("{0}")]
    Other(String),
}

pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dimension: usize,
}

impl FastEmbedder {
    pub fn new() -> Result<Self, EmbedError> {
        let model = fastembed::TextEmbedding::try_new(Default::default())
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        // Default model (AllMiniLML6V2) has 384 dimensions
        Ok(Self {
            model,
            dimension: 384,
        })
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, EmbedError> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.model
            .embed(owned, None)
            .map_err(|e| EmbedError::Model(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Shared embedder reference for passing across modules
pub type SharedEmbedder = Arc<dyn Embedder>;
