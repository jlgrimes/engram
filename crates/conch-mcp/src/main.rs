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

// === Tool parameter types ===

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberFactParams {
    /// The subject/entity (e.g. "Jared", "project-x")
    subject: String,
    /// The relationship/predicate (e.g. "prefers", "lives_in")
    relation: String,
    /// The object/value (e.g. "Rust", "Austin")
    object: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberEpisodeParams {
    /// Free-text description of the event or episode
    text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallParams {
    /// Natural language search query
    query: String,
    /// Max results to return (default 5)
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RelateParams {
    /// First entity
    entity_a: String,
    /// Relationship type (e.g. "works_with", "depends_on")
    relation: String,
    /// Second entity
    entity_b: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ForgetParams {
    /// Delete all memories with this subject
    subject: Option<String>,
    /// Delete memories older than this many seconds
    older_than_secs: Option<i64>,
}

// === Response types ===

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
}

impl From<RecallResult> for MemoryResponse {
    fn from(r: RecallResult) -> Self {
        let (kind, content) = match &r.memory.kind {
            MemoryKind::Fact(f) => (
                "fact".to_string(),
                format!("{} {} {}", f.subject, f.relation, f.object),
            ),
            MemoryKind::Episode(e) => ("episode".to_string(), e.text.clone()),
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
        }
    }
}

// === Server ===

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
        description = "Store a fact as a subject-relation-object triple. Use for preferences, attributes, relationships, and any structured knowledge. Examples: ('Jared', 'prefers', 'dark mode'), ('project-x', 'uses', 'Rust')"
    )]
    async fn remember_fact(
        &self,
        params: Parameters<RememberFactParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        match conch.remember_fact(&p.subject, &p.relation, &p.object) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "remember_episode",
        description = "Store a free-text episode or event. Use for things that happened, observations, or unstructured notes. The text will be embedded for semantic search."
    )]
    async fn remember_episode(
        &self,
        params: Parameters<RememberEpisodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        match conch.remember_episode(&p.text) {
            Ok(mem) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": mem.id, "strength": mem.strength }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "recall",
        description = "Search memories using natural language. Returns facts, episodes, and associations ranked by relevance using hybrid BM25 + vector search with Reciprocal Rank Fusion. Recalled memories are reinforced (strength increases)."
    )]
    async fn recall(
        &self,
        params: Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let limit = p.limit.unwrap_or(5);
        let conch = self.conch.lock().unwrap();
        match conch.recall(&p.query, limit) {
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
        name = "relate",
        description = "Create a named association between two entities. Associations are bidirectional and can be searched. Use for relationships like 'works_with', 'depends_on', 'part_of'."
    )]
    async fn relate(
        &self,
        params: Parameters<RelateParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        match conch.relate(&p.entity_a, &p.relation, &p.entity_b) {
            Ok(id) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({ "id": id }).to_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "forget",
        description = "Delete memories by subject or by age. At least one filter must be provided. Use subject to remove all facts about an entity, or older_than_secs to prune old memories."
    )]
    async fn forget(
        &self,
        params: Parameters<ForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let conch = self.conch.lock().unwrap();
        let mut total = 0usize;

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

        if p.subject.is_none() && p.older_than_secs.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one of 'subject' or 'older_than_secs' must be provided".to_string(),
            )]));
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::json!({ "forgotten": total }).to_string(),
        )]))
    }

    #[tool(
        name = "decay",
        description = "Run a decay pass over all memories. Memories lose strength based on time since last access (half-life: 24 hours). Very weak memories are deleted. Returns count of decayed and deleted memories."
    )]
    async fn decay(&self) -> Result<CallToolResult, McpError> {
        let conch = self.conch.lock().unwrap();
        match conch.decay() {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&result).unwrap(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    #[tool(
        name = "stats",
        description = "Get database statistics: total memories, facts, episodes, associations, and average memory strength."
    )]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let conch = self.conch.lock().unwrap();
        match conch.stats() {
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
                 language (hybrid BM25 + vector search), build knowledge graphs via associations. \
                 Memories strengthen with use and fade with time. All data persists in SQLite."
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

    // Ensure parent directory exists
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

#[cfg(test)]
mod tests {
    use super::*;
    use conch_core::{EmbedError, Embedder};
    use serde_json::Value;

    struct MockEmbedder;
    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|_| vec![0.5f32, 0.5, 0.5]).collect())
        }
        fn dimension(&self) -> usize {
            3
        }
    }

    fn test_server() -> ConchServer {
        let conch = ConchDB::open_in_memory_with(Box::new(MockEmbedder))
            .expect("failed to open in-memory conch");
        ConchServer::new(conch)
    }

    fn text_content(result: &CallToolResult) -> &str {
        result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.as_str())
            .expect("expected first content item to be text")
    }

    fn parse_json(result: &CallToolResult) -> Value {
        serde_json::from_str(text_content(result)).expect("tool response text should be JSON")
    }

    #[tokio::test]
    async fn remember_fact_returns_id() {
        let server = test_server();
        let result = server
            .remember_fact(Parameters(RememberFactParams {
                subject: "Jared".into(),
                relation: "prefers".into(),
                object: "Rust".into(),
            }))
            .await
            .expect("should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        assert!(body.get("id").is_some());
        assert!(body.get("strength").is_some());
    }

    #[tokio::test]
    async fn remember_episode_returns_id() {
        let server = test_server();
        let result = server
            .remember_episode(Parameters(RememberEpisodeParams {
                text: "Had a productive meeting about the roadmap".into(),
            }))
            .await
            .expect("should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        assert!(body.get("id").is_some());
    }

    #[tokio::test]
    async fn recall_returns_memories() {
        let server = test_server();

        // Store a fact first
        server
            .remember_fact(Parameters(RememberFactParams {
                subject: "Alice".into(),
                relation: "likes".into(),
                object: "recalltoken789".into(),
            }))
            .await
            .expect("remember should succeed");

        let result = server
            .recall(Parameters(RecallParams {
                query: "recalltoken789".into(),
                limit: Some(10),
            }))
            .await
            .expect("recall should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        let memories = body.as_array().expect("recall should return an array");
        assert!(!memories.is_empty(), "recall should find the stored fact");

        let first = &memories[0];
        assert!(first.get("id").is_some());
        assert!(first.get("kind").is_some());
        assert!(first.get("content").is_some());
        assert!(first.get("strength").is_some());
        assert!(first.get("score").is_some());
    }

    #[tokio::test]
    async fn relate_returns_id() {
        let server = test_server();
        let result = server
            .relate(Parameters(RelateParams {
                entity_a: "Jared".into(),
                relation: "works_on".into(),
                entity_b: "conch".into(),
            }))
            .await
            .expect("relate should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        assert!(body.get("id").is_some());
    }

    #[tokio::test]
    async fn forget_by_subject_returns_count() {
        let server = test_server();

        server
            .remember_fact(Parameters(RememberFactParams {
                subject: "forget-me".into(),
                relation: "has".into(),
                object: "data".into(),
            }))
            .await
            .expect("remember should succeed");

        let result = server
            .forget(Parameters(ForgetParams {
                subject: Some("forget-me".into()),
                older_than_secs: None,
            }))
            .await
            .expect("forget should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        let count = body.get("forgotten").and_then(Value::as_u64).unwrap();
        assert!(count >= 1);
    }

    #[tokio::test]
    async fn forget_requires_at_least_one_filter() {
        let server = test_server();
        let result = server
            .forget(Parameters(ForgetParams {
                subject: None,
                older_than_secs: None,
            }))
            .await
            .expect("forget should not return protocol error");

        assert_eq!(result.is_error, Some(true));
        assert!(text_content(&result).contains("At least one"));
    }

    #[tokio::test]
    async fn decay_returns_stats() {
        let server = test_server();

        server
            .remember_fact(Parameters(RememberFactParams {
                subject: "decay-test".into(),
                relation: "is".into(),
                object: "here".into(),
            }))
            .await
            .expect("remember should succeed");

        let result = server.decay().await.expect("decay should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        assert!(body.get("decayed").is_some());
        assert!(body.get("deleted").is_some());
    }

    #[tokio::test]
    async fn stats_returns_counts() {
        let server = test_server();

        server
            .remember_fact(Parameters(RememberFactParams {
                subject: "stats-test".into(),
                relation: "has".into(),
                object: "value".into(),
            }))
            .await
            .expect("remember should succeed");

        server
            .relate(Parameters(RelateParams {
                entity_a: "A".into(),
                relation: "knows".into(),
                entity_b: "B".into(),
            }))
            .await
            .expect("relate should succeed");

        let result = server.stats().await.expect("stats should not return protocol error");

        assert_eq!(result.is_error, Some(false));
        let body = parse_json(&result);
        assert!(body.get("total_memories").and_then(Value::as_i64).unwrap() >= 1);
        assert!(body.get("total_facts").and_then(Value::as_i64).unwrap() >= 1);
        assert!(body.get("total_associations").and_then(Value::as_i64).unwrap() >= 1);
    }

    #[tokio::test]
    async fn end_to_end_remember_recall_forget() {
        let server = test_server();
        let token = "e2e_unique_token_42";

        // Remember
        server
            .remember_fact(Parameters(RememberFactParams {
                subject: "test-user".into(),
                relation: "saved".into(),
                object: token.into(),
            }))
            .await
            .expect("remember should succeed");

        // Recall
        let recall_result = server
            .recall(Parameters(RecallParams {
                query: token.into(),
                limit: Some(5),
            }))
            .await
            .expect("recall should succeed");

        let body = parse_json(&recall_result);
        let memories = body.as_array().unwrap();
        assert!(
            memories.iter().any(|m| {
                m.get("content")
                    .and_then(Value::as_str)
                    .map(|c| c.contains(token))
                    .unwrap_or(false)
            }),
            "recalled memories should contain the token"
        );

        // Forget
        let forget_result = server
            .forget(Parameters(ForgetParams {
                subject: Some("test-user".into()),
                older_than_secs: None,
            }))
            .await
            .expect("forget should succeed");

        let forgot = parse_json(&forget_result)
            .get("forgotten")
            .and_then(Value::as_u64)
            .unwrap();
        assert!(forgot >= 1);

        // Recall again - should be empty
        let recall_after = server
            .recall(Parameters(RecallParams {
                query: token.into(),
                limit: Some(5),
            }))
            .await
            .expect("recall should succeed");

        let remaining = parse_json(&recall_after)
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(Value::as_str)
                    .map(|c| c.contains(token))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(remaining, 0, "forgotten memory should not be recalled");
    }

    #[test]
    fn remember_fact_params_require_all_fields() {
        let err = serde_json::from_value::<RememberFactParams>(serde_json::json!({
            "relation": "likes",
            "object": "rust"
        }))
        .expect_err("missing subject should fail");
        assert!(err.to_string().contains("subject"));
    }

    #[test]
    fn recall_params_require_query() {
        let err = serde_json::from_value::<RecallParams>(serde_json::json!({
            "limit": 3
        }))
        .expect_err("missing query should fail");
        assert!(err.to_string().contains("query"));
    }

    #[test]
    fn relate_params_require_all_fields() {
        let err = serde_json::from_value::<RelateParams>(serde_json::json!({
            "entity_a": "a",
            "relation": "knows"
        }))
        .expect_err("missing entity_b should fail");
        assert!(err.to_string().contains("entity_b"));
    }
}
