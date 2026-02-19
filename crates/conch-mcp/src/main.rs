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
use serde_json::Value;
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
    diagnostics: Option<bool>,
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

fn lock_conch<'a>(
    conch: &'a Arc<Mutex<ConchDB>>,
) -> Result<std::sync::MutexGuard<'a, ConchDB>, CallToolResult> {
    conch.lock().map_err(|_| {
        CallToolResult::error(vec![Content::text(
            "internal error: memory database lock poisoned".to_string(),
        )])
    })
}

fn success_json(value: Value) -> CallToolResult {
    CallToolResult::success(vec![Content::text(value.to_string())])
}

fn success_json_pretty<T: Serialize>(value: &T) -> CallToolResult {
    CallToolResult::success(vec![Content::text(
        serde_json::to_string_pretty(value).expect("serialization should not fail"),
    )])
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
    namespace: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    bm25_hits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector_hits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fused_candidates: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filtered_memories: Option<usize>,
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
        let diagnostics = r.diagnostics;
        MemoryResponse {
            id: r.memory.id,
            namespace: r.memory.namespace.clone(),
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
            bm25_hits: diagnostics.as_ref().map(|d| d.bm25_hits),
            vector_hits: diagnostics.as_ref().map(|d| d.vector_hits),
            fused_candidates: diagnostics.as_ref().map(|d| d.fused_candidates),
            filtered_memories: diagnostics.as_ref().map(|d| d.filtered_memories),
            created_at: r.memory.created_at.to_rfc3339(),
            last_accessed_at: r.memory.last_accessed_at.to_rfc3339(),
            access_count: r.memory.access_count,
        }
    }
}

#[derive(Clone)]
struct ConchServer {
    conch: Arc<Mutex<ConchDB>>,
    default_namespace: String,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ConchServer {
    fn new(conch: ConchDB, default_namespace: String) -> Self {
        Self {
            conch: Arc::new(Mutex::new(conch)),
            default_namespace,
            tool_router: Self::tool_router(),
        }
    }

    fn namespace_or_default<'a>(&'a self, namespace: Option<&'a str>) -> &'a str {
        namespace.unwrap_or(self.default_namespace.as_str())
    }

    async fn with_conch_blocking<R, F>(&self, f: F) -> Result<Result<R, CallToolResult>, McpError>
    where
        R: Send + 'static,
        F: FnOnce(&ConchDB) -> Result<R, String> + Send + 'static,
    {
        let conch = Arc::clone(&self.conch);
        match tokio::task::spawn_blocking(move || {
            let conch = match lock_conch(&conch) {
                Ok(guard) => guard,
                Err(result) => return Err(result),
            };
            f(&conch).map_err(|msg| CallToolResult::error(vec![Content::text(msg)]))
        })
        .await
        {
            Ok(result) => Ok(result),
            Err(_) => Ok(Err(CallToolResult::error(vec![Content::text(
                "internal error: blocking task failed".to_string(),
            )]))),
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
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();
        let subject = p.subject;
        let relation = p.relation;
        let object = p.object;

        match self
            .with_conch_blocking(move |conch| {
                conch
                    .remember_fact_in(&namespace, &subject, &relation, &object)
                    .map_err(|e| e.to_string())
            })
            .await?
        {
            Ok(mem) => Ok(success_json(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "namespace": response_namespace }),
            )),
            Err(error) => Ok(error),
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
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();
        let text = p.text;

        match self
            .with_conch_blocking(move |conch| {
                conch
                    .remember_episode_in(&namespace, &text)
                    .map_err(|e| e.to_string())
            })
            .await?
        {
            Ok(mem) => Ok(success_json(
                serde_json::json!({ "id": mem.id, "strength": mem.strength, "namespace": response_namespace }),
            )),
            Err(error) => Ok(error),
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
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let query = p.query;
        let limit = p.limit.unwrap_or(5);
        let explain = p.explain.unwrap_or(false);
        let diagnostics = p.diagnostics.unwrap_or(false);

        match self
            .with_conch_blocking(move |conch| {
                conch
                    .recall_filtered_in_with_options(
                        &namespace,
                        &query,
                        limit,
                        kind,
                        RecallOptions {
                            explain,
                            diagnostics,
                        },
                    )
                    .map_err(|e| e.to_string())
            })
            .await?
        {
            Ok(results) => {
                let responses: Vec<MemoryResponse> =
                    results.into_iter().map(MemoryResponse::from).collect();
                Ok(success_json_pretty(&responses))
            }
            Err(error) => Ok(error),
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
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();
        let subject = p.subject;
        let older_than_secs = p.older_than_secs;

        match self
            .with_conch_blocking(move |conch| {
                let mut total = 0;
                if let Some(subject) = &subject {
                    total += conch
                        .forget_by_subject_in(&namespace, subject)
                        .map_err(|e| e.to_string())?;
                }
                if let Some(secs) = older_than_secs {
                    total += conch
                        .forget_older_than_in(&namespace, secs)
                        .map_err(|e| e.to_string())?;
                }
                Ok(total)
            })
            .await?
        {
            Ok(total) => Ok(success_json(
                serde_json::json!({ "forgotten": total, "namespace": response_namespace }),
            )),
            Err(error) => Ok(error),
        }
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
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();
        let id = p.id;

        match self
            .with_conch_blocking(move |conch| {
                conch
                    .forget_by_id_in(&namespace, &id)
                    .map_err(|e| e.to_string())
            })
            .await?
        {
            Ok(n) => Ok(success_json(
                serde_json::json!({ "forgotten": n, "namespace": response_namespace }),
            )),
            Err(error) => Ok(error),
        }
    }

    #[tool(
        name = "decay",
        description = "Run decay pass. Memories lose strength over time; weak ones are pruned."
    )]
    async fn decay(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();

        match self
            .with_conch_blocking(move |conch| conch.decay_in(&namespace).map_err(|e| e.to_string()))
            .await?
        {
            Ok(result) => Ok(success_json(serde_json::json!({
                "namespace": response_namespace,
                "decayed": result.decayed,
                "deleted": result.deleted
            }))),
            Err(error) => Ok(error),
        }
    }

    #[tool(name = "stats", description = "Get memory statistics.")]
    async fn stats(&self, params: Parameters<NamespaceParams>) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let namespace = self
            .namespace_or_default(p.namespace.as_deref())
            .to_string();
        let response_namespace = namespace.clone();

        match self
            .with_conch_blocking(move |conch| conch.stats_in(&namespace).map_err(|e| e.to_string()))
            .await?
        {
            Ok(stats) => Ok(success_json(serde_json::json!({
                "namespace": response_namespace,
                "total_memories": stats.total_memories,
                "total_facts": stats.total_facts,
                "total_episodes": stats.total_episodes,
                "avg_strength": stats.avg_strength
            }))),
            Err(error) => Ok(error),
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

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopEmbedder;

    impl conch_core::Embedder for NoopEmbedder {
        fn embed(
            &self,
            texts: &[&str],
        ) -> Result<Vec<conch_core::embed::Embedding>, conch_core::EmbedError> {
            Ok(texts.iter().map(|_| vec![0.0_f32, 0.0_f32]).collect())
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[test]
    fn parse_recall_kind_rejects_invalid_kind() {
        let err = parse_recall_kind(Some("bogus")).expect_err("invalid kind should error");
        assert!(err.contains("invalid kind"));
    }

    #[test]
    fn lock_conch_returns_error_when_poisoned() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let lock = Arc::new(Mutex::new(db));

        let lock_for_thread = Arc::clone(&lock);
        let _ = std::thread::spawn(move || {
            let _guard = lock_for_thread.lock().expect("acquire lock");
            panic!("poison lock for test");
        })
        .join();

        let result = lock_conch(&lock);
        assert!(
            result.is_err(),
            "poisoned lock should produce tool error, not panic"
        );
    }

    #[test]
    fn namespace_falls_back_to_server_default() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let server = ConchServer::new(db, "team-a".to_string());

        assert_eq!(server.namespace_or_default(None), "team-a");
        assert_eq!(server.namespace_or_default(Some("team-b")), "team-b");
    }

    #[tokio::test]
    async fn with_conch_blocking_returns_successful_result() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let server = ConchServer::new(db, "default-ns".to_string());

        let result = server
            .with_conch_blocking(|conch| {
                conch
                    .remember_episode_in("team-c", "implemented async helper")
                    .map_err(|e| e.to_string())
            })
            .await
            .expect("helper should return outer Ok");

        let memory = result.expect("helper should return inner success");
        assert_eq!(memory.namespace, "team-c");
    }

    fn tool_result_json_text(result: &CallToolResult) -> serde_json::Value {
        let outer = serde_json::to_value(result).expect("serialize tool result");
        let text = outer["content"][0]["text"]
            .as_str()
            .expect("text content payload");
        serde_json::from_str(text).expect("inner json payload")
    }

    #[tokio::test]
    async fn remember_fact_success_includes_namespace() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let server = ConchServer::new(db, "default-ns".to_string());

        let result = server
            .remember_fact(Parameters(RememberFactParams {
                namespace: Some("team-a".to_string()),
                subject: "sky".to_string(),
                relation: "is".to_string(),
                object: "blue".to_string(),
            }))
            .await
            .expect("tool call should succeed");

        let value = tool_result_json_text(&result);
        assert_eq!(value["namespace"], "team-a");
        assert!(value["id"].is_number());
    }

    #[tokio::test]
    async fn stats_success_includes_namespace() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let server = ConchServer::new(db, "default-ns".to_string());

        let result = server
            .stats(Parameters(NamespaceParams {
                namespace: Some("team-b".to_string()),
            }))
            .await
            .expect("tool call should succeed");

        let value = tool_result_json_text(&result);
        assert_eq!(value["namespace"], "team-b");
        assert!(value["total_memories"].is_number());
    }

    #[tokio::test]
    async fn concurrent_remember_episode_and_recall_succeeds_without_internal_errors() {
        let db = ConchDB::open_in_memory_with(Box::new(NoopEmbedder)).expect("db");
        let server = ConchServer::new(db, "default-ns".to_string());
        let namespace = "team-concurrency".to_string();

        let seed = server
            .remember_episode(Parameters(RememberEpisodeParams {
                namespace: Some(namespace.clone()),
                text: "seed episode".to_string(),
            }))
            .await
            .expect("seed write should succeed");
        let seed_outer = serde_json::to_value(&seed).expect("serialize seed result");
        assert_ne!(seed_outer["isError"], serde_json::Value::Bool(true));

        // Guards against runtime blocking / HOL regressions from inline sync DB work in MCP tools.
        let remember_server = server.clone();
        let remember_ns = namespace.clone();
        let remember_a = tokio::spawn(async move {
            remember_server
                .remember_episode(Parameters(RememberEpisodeParams {
                    namespace: Some(remember_ns),
                    text: "parallel episode a".to_string(),
                }))
                .await
        });

        let recall_server = server.clone();
        let recall_ns = namespace.clone();
        let recall_a = tokio::spawn(async move {
            recall_server
                .recall(Parameters(RecallParams {
                    namespace: Some(recall_ns),
                    query: "episode".to_string(),
                    limit: Some(5),
                    kind: Some("episode".to_string()),
                    explain: Some(false),
                    diagnostics: Some(false),
                }))
                .await
        });

        let remember_b = server.remember_episode(Parameters(RememberEpisodeParams {
            namespace: Some(namespace.clone()),
            text: "parallel episode b".to_string(),
        }));
        let recall_b = server.recall(Parameters(RecallParams {
            namespace: Some(namespace),
            query: "parallel".to_string(),
            limit: Some(5),
            kind: Some("episode".to_string()),
            explain: Some(false),
            diagnostics: Some(false),
        }));

        let (remember_b_res, recall_b_res, remember_a_join, recall_a_join) =
            tokio::join!(remember_b, recall_b, remember_a, recall_a);

        let remember_b_res = remember_b_res.expect("inline remember should complete");
        let recall_b_res = recall_b_res.expect("inline recall should complete");
        let remember_a_res = remember_a_join
            .expect("spawned remember task should join")
            .expect("spawned remember should complete");
        let recall_a_res = recall_a_join
            .expect("spawned recall task should join")
            .expect("spawned recall should complete");

        for result in [
            &remember_b_res,
            &recall_b_res,
            &remember_a_res,
            &recall_a_res,
        ] {
            let outer = serde_json::to_value(result).expect("serialize tool result");
            assert_ne!(
                outer["isError"],
                serde_json::Value::Bool(true),
                "tool call should not return MCP error payload"
            );
            let text = outer["content"][0]["text"]
                .as_str()
                .expect("tool payload should include text");
            assert!(
                !text.contains("internal error"),
                "tool payload should not contain internal lock/runtime errors"
            );
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
    let default_namespace = std::env::var("CONCH_NAMESPACE").unwrap_or_else(|_| "default".into());
    eprintln!("conch-mcp: opening {db_path} (namespace default: {default_namespace})");
    let conch = ConchDB::open(&db_path)?;
    eprintln!("conch-mcp: ready");
    let server = ConchServer::new(conch, default_namespace);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
