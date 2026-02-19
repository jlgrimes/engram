use conch_core::{ConchDB, MemoryKind, RecallKindFilter, RecallOptions, RecallResult};
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
    namespace: Option<String>,
    subject: String,
    relation: String,
    object: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberEpisodeParams {
    namespace: Option<String>,
    text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallParams {
    namespace: Option<String>,
    query: String,
    limit: Option<usize>,
    kind: Option<String>,
    explain: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetParams {
    namespace: Option<String>,
    subject: Option<String>,
    older_than_secs: Option<i64>,
}

fn parse_recall_kind(kind: Option<&str>) -> Result<RecallKindFilter, String> {
    match kind.unwrap_or("all").to_ascii_lowercase().as_str() {
        "all" => Ok(RecallKindFilter::All),
        "fact" => Ok(RecallKindFilter::Facts),
        "episode" => Ok(RecallKindFilter::Episodes),
        other => Err(format!(
            "invalid kind '{other}'. Expected one of: all, fact, episode"
        )),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetByIdParams {
    namespace: Option<String>,
    id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NamespaceParams {
    namespace: Option<String>,
}

#[derive(Debug, Serialize)]
struct MemoryResponse {
    id: i64,
    kind: String,
    content: String,
    strength: f64,
    score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    rrf_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decayed_strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recency_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    activation_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal_boost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_score: Option<f64>,
    created_at: String,
    last_accessed_at: String,
    access_count: i64,
}

impl From<RecallResult> for MemoryResponse {
    fn from(r: RecallResult) -> Self {
        let (kind, content) = match &r.memory.kind {
            MemoryKind::Fact(f) => (
                "fact".into(),
                format!("{} {} {}", f.subject, f.relation, f.object),
            ),
            MemoryKind::Episode(e) => ("episode".into(), e.text.clone()),
        };
        let explanation = r.explanation;
        MemoryResponse {
            id: r.memory.id,
            kind,
            content,
            strength: r.memory.strength,
            score: r.score,
            rrf_score: explanation.as_ref().map(|e| e.rrf_score),
            decayed_strength: explanation.as_ref().map(|e| e.decayed_strength),
            recency_boost: explanation.as_ref().map(|e| e.recency_boost),
            access_weight: explanation.as_ref().map(|e| e.access_weight),
            activation_boost: explanation.as_ref().map(|e| e.activation_boost),
            temporal_boost: explanation.as_ref().map(|e| e.temporal_boost),
            final_score: explanation.as_ref().map(|e| e.final_score),
            created_at: r.memory.created_at.to_rfc3339(),
            last_accessed_at: r.memory.last_accessed_at.to_rfc3339(),
            access_count: r.memory.access_count,
        }
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
        Self {
            conch: Arc::new(Mutex::new(conch)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "remember_fact",
        description = "Store a fact as a subject-relation-object triple."
    )]
    async fn remember_fact(
        &self,
        params: Parameters<RememberFactParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.remember_fact_in(namespace, &p.subject, &p.relation, &p.object) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "remember_episode",
        description = "Store a free-text episode or event."
    )]
    async fn remember_episode(
        &self,
        params: Parameters<RememberEpisodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.remember_episode_in(namespace, &p.text) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "recall",
        description = "Search memories using natural language. BM25 + vector search, ranked by relevance × strength × recency."
    )]
    async fn recall(&self, params: Parameters<RecallParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let kind = match parse_recall_kind(p.kind.as_deref()) {
            Ok(kind) => kind,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.recall_filtered_in_with_options(
            namespace,
            &p.query,
            p.limit.unwrap_or(5),
            kind,
            RecallOptions {
                explain: p.explain.unwrap_or(false),
            },
        ) {
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

    #[tool(name = "forget", description = "Delete memories by subject or by age.")]
    async fn forget(&self, params: Parameters<ForgetParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        if p.subject.is_none() && p.older_than_secs.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Provide 'subject' or 'older_than_secs'".to_string(),
            )]));
        }
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        let mut total = 0;
        if let Some(subject) = &p.subject {
            match conch.forget_by_subject_in(namespace, subject) {
                Ok(n) => total += n,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
        if let Some(secs) = p.older_than_secs {
            match conch.forget_older_than_in(namespace, secs) {
                Ok(n) => total += n,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::json!({ "forgotten": total }).to_string(),
        )]))
    }

    #[tool(
        name = "forget_by_id",
        description = "Delete a specific memory by its ID."
    )]
    async fn forget_by_id(
        &self,
        params: Parameters<ForgetByIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.forget_by_id_in(namespace, &p.id) {
            Ok(n) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "forgotten": n }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "decay",
        description = "Run decay pass. Memories lose strength over time; weak ones are pruned."
    )]
    async fn decay(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.decay_in(namespace) {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&result).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(name = "stats", description = "Get memory statistics.")]
    async fn stats(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = p.namespace.as_deref().unwrap_or("default");
        let conch = self.conch.lock().unwrap();
        match conch.stats_in(namespace) {
            Ok(stats) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&stats).unwrap(),
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
