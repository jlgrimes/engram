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
    /// Namespace for memory isolation (default: "default")
    namespace: Option<String>,
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
    /// Namespace for memory isolation (default: "default")
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallParams {
    query: String,
    limit: Option<usize>,
    /// Optional tag to filter results by
    tag: Option<String>,
    /// Namespace for memory isolation (default: "default")
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetParams {
    subject: Option<String>,
    older_than_secs: Option<i64>,
    /// Namespace for memory isolation (default: "default")
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AuditLogParams {
    /// Number of entries to return (default: 20)
    limit: Option<usize>,
    /// Filter by memory ID
    memory_id: Option<i64>,
    /// Filter by actor
    actor: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct VerifyParams {
    /// Namespace to verify (default: "default")
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NamespaceParams {
    /// Namespace for memory isolation (default: "default")
    namespace: Option<String>,
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
    namespace: String,
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
            namespace: r.memory.namespace.clone(),
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
    db_path: String,
    tool_router: ToolRouter<Self>,
}

impl ConchServer {
    fn open_db(&self, namespace: Option<&str>) -> Result<ConchDB, conch_core::ConchError> {
        let ns = namespace.unwrap_or("default");
        ConchDB::open_with_namespace(&self.db_path, ns)
    }
}

#[tool_router]
impl ConchServer {
    fn new(db_path: String) -> Self {
        Self { db_path, tool_router: Self::tool_router() }
    }

    #[tool(name = "remember_fact", description = "Store a fact as a subject-relation-object triple. Uses upsert: if a fact with the same subject+relation exists, its object is updated. Optionally tag with comma-separated categories. Supports namespace isolation.")]
    async fn remember_fact(&self, params: Parameters<RememberFactParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.remember_fact_dedup_full(&p.subject, &p.relation, &p.object, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(result) => {
                let mem = result.memory();
                let action = match &result {
                    conch_core::RememberResult::Created(_) => "created",
                    conch_core::RememberResult::Updated(_) => "updated",
                    conch_core::RememberResult::Duplicate { .. } => "duplicate",
                };
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::json!({ "id": mem.id, "action": action, "strength": mem.strength, "tags": mem.tags, "source": mem.source, "namespace": mem.namespace }).to_string(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "remember_episode", description = "Store a free-text episode or event. Optionally tag with comma-separated categories and track source. Supports namespace isolation.")]
    async fn remember_episode(&self, params: Parameters<RememberEpisodeParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.remember_episode_full(&p.text, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "tags": mem.tags, "source": mem.source, "namespace": mem.namespace }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "recall", description = "Search memories using natural language. BM25 + vector search, ranked by relevance x strength x recency. Optionally filter by tag. Supports namespace isolation.")]
    async fn recall(&self, params: Parameters<RecallParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
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

    #[tool(name = "forget", description = "Delete memories by subject or by age. Supports namespace isolation.")]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if p.subject.is_none() && p.older_than_secs.is_none() {
            return Ok(CallToolResult::error(vec![Content::text("Provide 'subject' or 'older_than_secs'".to_string())]));
        }
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        let mut total = 0;
        if let Some(subject) = &p.subject {
            match conch.forget_by_subject(subject) { Ok(n) => total += n, Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])) }
        }
        if let Some(secs) = p.older_than_secs {
            match conch.forget_older_than(secs) { Ok(n) => total += n, Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])) }
        }
        Ok(CallToolResult::success(vec![Content::text(serde_json::json!({ "forgotten": total }).to_string())]))
    }

    #[tool(name = "decay", description = "Run decay pass. Memories lose strength over time; weak ones are pruned. Supports namespace isolation.")]
    async fn decay(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.decay() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&result).unwrap())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "stats", description = "Get memory statistics. Supports namespace isolation.")]
    async fn stats(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.stats() {
            Ok(stats) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&stats).unwrap())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "audit_log", description = "View audit trail entries. Shows recent actions (remember, forget, decay, reinforce) with timestamps, actors, and details.")]
    async fn audit_log(&self, params: Parameters<AuditLogParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.audit_log(p.limit.unwrap_or(20), p.memory_id, p.actor.as_deref()) {
            Ok(entries) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&entries).unwrap())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "verify", description = "Verify memory integrity by checking SHA-256 checksums. Reports valid, corrupted, and missing-checksum memories. Supports namespace isolation.")]
    async fn verify(&self, params: Parameters<VerifyParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.verify() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(serde_json::to_string_pretty(&result).unwrap())])),
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
                 language (hybrid BM25 + vector search). Memories strengthen with use and fade with time. \
                 Supports namespace isolation, audit trails, and integrity verification."
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
    // Validate DB opens correctly
    let _conch = ConchDB::open(&db_path)?;
    eprintln!("conch-mcp: ready");
    let server = ConchServer::new(db_path);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
