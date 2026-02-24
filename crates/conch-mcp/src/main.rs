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
struct RememberActionParams {
    text: String,
    tags: Option<String>,
    source: Option<String>,
    session_id: Option<String>,
    channel: Option<String>,
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberIntentParams {
    text: String,
    tags: Option<String>,
    source: Option<String>,
    session_id: Option<String>,
    channel: Option<String>,
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

#[derive(Debug, Deserialize, JsonSchema)]
struct RelatedParams {
    /// The subject entity to traverse from
    subject: String,
    /// Max traversal depth (1-3, default 2)
    depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WhyParams {
    /// Memory ID to inspect provenance for
    id: i64,
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
    namespace: String,
}

impl From<RecallResult> for MemoryResponse {
    fn from(r: RecallResult) -> Self {
        let (kind, content) = match &r.memory.kind {
            MemoryKind::Fact(f) => (
                "fact".into(),
                format!("{} {} {}", f.subject, f.relation, f.object),
            ),
            MemoryKind::Episode(e) => ("episode".into(), e.text.clone()),
            MemoryKind::Action(a) => ("action".into(), a.text.clone()),
            MemoryKind::Intent(i) => ("intent".into(), i.text.clone()),
        };
        MemoryResponse {
            id: r.memory.id,
            kind,
            content,
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
        Some(s) if !s.is_empty() => s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
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
        Self {
            db_path,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "remember_fact",
        description = "Store a fact as a subject-relation-object triple. Uses upsert: if a fact with the same subject+relation exists, its object is updated. Optionally tag with comma-separated categories. Supports namespace isolation."
    )]
    async fn remember_fact(
        &self,
        params: Parameters<RememberFactParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.remember_fact_dedup_full(
            &p.subject,
            &p.relation,
            &p.object,
            &tags,
            source,
            p.session_id.as_deref(),
            p.channel.as_deref(),
        ) {
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

    #[tool(
        name = "remember_episode",
        description = "Store a free-text episode or event. Optionally tag with comma-separated categories and track source. Supports namespace isolation."
    )]
    async fn remember_episode(
        &self,
        params: Parameters<RememberEpisodeParams>,
    ) -> Result<CallToolResult, McpError> {
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

    #[tool(
        name = "remember_action",
        description = "Store an executed action (free-text operational event such as files changed, deployments, or status updates). Supports namespace isolation."
    )]
    async fn remember_action(
        &self,
        params: Parameters<RememberActionParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.remember_action_full(&p.text, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "tags": mem.tags, "source": mem.source, "namespace": mem.namespace }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "remember_intent",
        description = "Store an intent (future plan/intention). Supports namespace isolation."
    )]
    async fn remember_intent(
        &self,
        params: Parameters<RememberIntentParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let tags = parse_tags_mcp(p.tags.as_deref());
        let source = Some(p.source.as_deref().unwrap_or("mcp"));
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.remember_intent_full(&p.text, &tags, source, p.session_id.as_deref(), p.channel.as_deref()) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "tags": mem.tags, "source": mem.source, "namespace": mem.namespace }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "recall",
        description = "Search memories using natural language. BM25 + vector search, ranked by relevance x strength x recency. Optionally filter by tag. Supports namespace isolation."
    )]
    async fn recall(&self, params: Parameters<RecallParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.recall_with_tag(&p.query, p.limit.unwrap_or(5), p.tag.as_deref()) {
            Ok(results) => {
                let responses: Vec<MemoryResponse> =
                    results.into_iter().map(MemoryResponse::from).collect();
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&responses).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "forget",
        description = "Delete memories by subject or by age. Supports namespace isolation."
    )]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if p.subject.is_none() && p.older_than_secs.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Provide 'subject' or 'older_than_secs'".to_string(),
            )]));
        }
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        let mut total = 0;
        if let Some(subject) = &p.subject {
            match conch.forget_by_subject(subject) {
                Ok(n) => total += n,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
        if let Some(secs) = p.older_than_secs {
            match conch.forget_older_than(secs) {
                Ok(n) => total += n,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::json!({ "forgotten": total }).to_string(),
        )]))
    }

    #[tool(
        name = "decay",
        description = "Run decay pass. Memories lose strength over time; weak ones are pruned. Supports namespace isolation."
    )]
    async fn decay(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.decay() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&result).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "related",
        description = "Graph traversal: find facts connected to a subject entity via 1-hop and multi-hop relationships. Returns a graph of related memories with hop distance."
    )]
    async fn related(&self, params: Parameters<RelatedParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let depth = p.depth.unwrap_or(2);
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.related(&p.subject, depth) {
            Ok(nodes) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&nodes).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "why",
        description = "Provenance: show full audit context for a memory — when created, by whom, access count, strength, source, session, channel, and 1-hop related facts."
    )]
    async fn why(&self, params: Parameters<WhyParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.why(p.id) {
            Ok(Some(info)) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&info).unwrap(),
            )])),
            Ok(None) => Ok(CallToolResult::error(vec![Content::text(
                serde_json::json!({ "error": "memory not found", "id": p.id }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "stats",
        description = "Get memory statistics. Supports namespace isolation."
    )]
    async fn stats(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.stats() {
            Ok(stats) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&stats).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "consolidate",
        description = "Consolidate related memories (sleep-like memory consolidation). Finds clusters of similar memories, boosts the strongest, and archives duplicates."
    )]
    async fn consolidate(
        &self,
        params: Parameters<ConsolidateParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let dry_run = p.dry_run.unwrap_or(false);
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
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

    #[tool(
        name = "importance",
        description = "Show or recompute importance scores for all memories. Importance affects decay rate — high-importance memories decay slower."
    )]
    async fn importance(
        &self,
        params: Parameters<ImportanceScoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
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

    #[tool(
        name = "set_importance",
        description = "Set the importance score for a specific memory. Importance (0.0-1.0) affects decay rate — high-importance memories decay slower."
    )]
    async fn set_importance(
        &self,
        params: Parameters<ImportanceSetParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.set_importance(p.id, p.importance) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": p.id, "importance": p.importance }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "audit_log",
        description = "View audit trail entries. Shows recent actions (remember, forget, decay, reinforce) with timestamps, actors, and details."
    )]
    async fn audit_log(
        &self,
        params: Parameters<AuditLogParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(None) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.audit_log(p.limit.unwrap_or(20), p.memory_id, p.actor.as_deref()) {
            Ok(entries) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&entries).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "verify",
        description = "Verify memory integrity by checking SHA-256 checksums. Reports valid, corrupted, and missing-checksum memories. Supports namespace isolation."
    )]
    async fn verify(&self, params: Parameters<VerifyParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = match self.open_db(p.namespace.as_deref()) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        };
        match conch.verify() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&result).unwrap(),
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
