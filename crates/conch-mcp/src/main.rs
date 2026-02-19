use conch_core::{ConchDB, MemoryKind, RecallResult};
use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberFactParams {
    subject: String,
    relation: String,
    object: String,
    /// Optional comma-separated tags (e.g. "preference,technical")
    tags: Option<String>,
    /// Source of this memory (e.g. "mcp", "discord", "cron")
    source: Option<String>,
    /// Session identifier for grouping related memories
    session_id: Option<String>,
    /// Channel or context within the source
    channel: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberEpisodeParams {
    text: String,
    /// Optional comma-separated tags (e.g. "decision,project")
    tags: Option<String>,
    /// Source of this memory (e.g. "mcp", "discord", "cron")
    source: Option<String>,
    /// Session identifier for grouping related memories
    session_id: Option<String>,
    /// Channel or context within the source
    channel: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallParams {
    query: String,
    limit: Option<usize>,
    /// Optional tag to filter results by
    tag: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetParams {
    subject: Option<String>,
    older_than_secs: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsolidateParams {
    /// If true, only preview what would be consolidated without making changes
    dry_run: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ImportanceSetParams {
    /// Memory ID to set importance for
    id: i64,
    /// Importance value (0.0-1.0)
    importance: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ImportanceScoreParams {
    /// If true, recompute importance scores for all memories
    recompute: Option<bool>,
}

#[derive(Debug, Serialize)]
struct MemoryResponse {
    id: i64,
    kind: String,
    content: String,
    strength: f64,
    score: f64,
    created_at: String,
    last_accessed_at: String,
    access_count: i64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<String>,
}

impl From<RecallResult> for MemoryResponse {
    fn from(r: RecallResult) -> Self {
        let (kind, content) = match &r.memory.kind {
            MemoryKind::Fact(f) => ("fact".into(), format!("{} {} {}", f.subject, f.relation, f.object)),
            MemoryKind::Episode(e) => ("episode".into(), e.text.clone()),
        };
        MemoryResponse {
            id: r.memory.id,
            kind, content,
            strength: r.memory.strength,
            score: r.score,
            created_at: r.memory.created_at.to_rfc3339(),
            last_accessed_at: r.memory.last_accessed_at.to_rfc3339(),
            access_count: r.memory.access_count,
            tags: r.memory.tags.clone(),
            source: r.memory.source.clone(),
            session_id: r.memory.session_id.clone(),
            channel: r.memory.channel.clone(),
        }
    }
}

fn parse_tags_mcp(tags: Option<&str>) -> Vec<String> {
    match tags {
        Some(s) if !s.is_empty() => s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect(),
        _ => vec![],
    }
}

#[derive(Clone)]
struct ConchServer {
    conch: Arc<Mutex<ConchDB>>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ConchServer {
    fn new(conch: ConchDB) -> Self {
        Self { conch: Arc::new(Mutex::new(conch)), tool_router: Self::tool_router() }
    }

    #[tool(name = "remember_fact", description = "Store a fact as a subject-relation-object triple. Uses upsert: if a fact with the same subject+relation exists, its object is updated. Optionally tag with comma-separated categories.")]
    async fn remember_fact(&self, params: Parameters<RememberFactParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = self.conch.lock().unwrap();
        match conch.remember_fact_dedup_full(&p.subject, &p.relation, &p.object, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(result) => {
                let mem = result.memory();
                let action = match &result {
                    conch_core::RememberResult::Created(_) => "created",
                    conch_core::RememberResult::Updated(_) => "updated",
                    conch_core::RememberResult::Duplicate { .. } => "duplicate",
                };
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::json!({ "id": mem.id, "action": action, "strength": mem.strength, "tags": mem.tags, "source": mem.source }).to_string(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "remember_episode", description = "Store a free-text episode or event. Optionally tag with comma-separated categories and track source.")]
    async fn remember_episode(&self, params: Parameters<RememberEpisodeParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = self.conch.lock().unwrap();
        match conch.remember_episode_full(&p.text, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "tags": mem.tags, "source": mem.source }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "recall", description = "Search memories using natural language. BM25 + vector search, ranked by relevance × strength × recency. Optionally filter by tag.")]
    async fn recall(&self, params: Parameters<RecallParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        match conch.recall_with_tag(&p.query, p.limit.unwrap_or(5), p.tag.as_deref()) {
            Ok(results) => {
                let responses: Vec<MemoryResponse> = results.into_iter().map(MemoryResponse::from).collect();
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&responses).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "forget", description = "Delete memories by subject or by age.")]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if p.subject.is_none() && p.older_than_secs.is_none() {
            return Ok(CallToolResult::error(vec![Content::text("Provide 'subject' or 'older_than_secs'".to_string())]));
        }
        let conch = self.conch.lock().unwrap();
        let mut total = 0;
        if let Some(subject) = &p.subject {
            match conch.forget_by_subject(subject) { Ok(n) => total += n, Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])) }
        }
        if let Some(secs) = p.older_than_secs {
            match conch.forget_older_than(secs) { Ok(n) => total += n, Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])) }
        }
        Ok(CallToolResult::success(vec![Content::text(serde_json::json!({ "forgotten": total }).to_string())]))
    }

    #[tool(name = "decay", description = "Run decay pass. Memories lose strength over time; weak ones are pruned.")]
    async fn decay(&self) -> Result<CallToolResult, McpError> {
        let conch = self.conch.lock().unwrap();
        match conch.decay() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&result).unwrap())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "stats", description = "Get memory statistics.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let conch = self.conch.lock().unwrap();
        match conch.stats() {
            Ok(stats) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&stats).unwrap())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "consolidate", description = "Consolidate related memories (sleep-like memory consolidation). Finds clusters of similar memories, boosts the strongest, and archives duplicates.")]
    async fn consolidate(&self, params: Parameters<ConsolidateParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let dry_run = p.dry_run.unwrap_or(false);
        let conch = self.conch.lock().unwrap();
        if dry_run {
            match conch.consolidate_clusters() {
                Ok(clusters) => Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&clusters).unwrap(),
                )])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        } else {
            match conch.consolidate(false) {
                Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&result).unwrap(),
                )])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
    }

    #[tool(name = "importance", description = "Show or recompute importance scores for all memories. Importance affects decay rate — high-importance memories decay slower.")]
    async fn importance(&self, params: Parameters<ImportanceScoreParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        if p.recompute.unwrap_or(false) {
            match conch.score_importance() {
                Ok(count) => Ok(CallToolResult::success(vec![Content::text(
                    serde_json::json!({ "recomputed": count }).to_string(),
                )])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        } else {
            match conch.list_importance() {
                Ok(infos) => Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&infos).unwrap(),
                )])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
    }

    #[tool(name = "set_importance", description = "Set the importance score for a specific memory. Importance (0.0-1.0) affects decay rate — high-importance memories decay slower.")]
    async fn set_importance(&self, params: Parameters<ImportanceSetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        match conch.set_importance(p.id, p.importance) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": p.id, "importance": p.importance }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }
}

#[tool_handler]
impl ServerHandler for ConchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Biological memory for AI agents. Store facts and episodes, recall with natural \
                 language (hybrid BM25 + vector search). Memories strengthen with use and fade with time."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "conch-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::var("CONCH_DB").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{home}/.conch/default.db")
    });
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    eprintln!("conch-mcp: opening {db_path}");
    let conch = ConchDB::open(&db_path)?;
    eprintln!("conch-mcp: ready");
    let server = ConchServer::new(conch);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
