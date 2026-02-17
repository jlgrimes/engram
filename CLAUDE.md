# Conch — Codebase Guide

Rust workspace. Biological memory for AI agents — memories strengthen with use, fade with time, connect associatively.

## Architecture

```
Cargo.toml                    — workspace root
crates/
  conch-core/                 — library crate (all logic)
    src/
      lib.rs                  — ConchDB high-level API (open, remember, recall, decay, etc.)
      memory.rs               — types: Fact, Episode, MemoryKind, MemoryRecord, Association, ExportData
      store.rs                — SQLite storage layer (MemoryStore, schema, CRUD, import/export)
      embed.rs                — Embedder trait + FastEmbedder (fastembed, AllMiniLML6V2, 384-dim)
      decay.rs                — temporal decay engine (half-life 24h, min strength 0.01)
      recall.rs               — hybrid BM25 + vector search with Reciprocal Rank Fusion
      relate.rs               — associative links between entities
  conch-cli/                  — binary crate
    src/
      main.rs                 — clap CLI: remember, recall, relate, forget, decay, stats, embed, export, import
  conch-mcp/                  — MCP server (Model Context Protocol for LLM tool use)
    src/
      main.rs                 — rmcp-based server exposing conch operations as MCP tools
```

## Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ConchDB` | `lib.rs` | Main API. Wraps `MemoryStore` + `Embedder`. Entry point for all operations. |
| `MemoryStore` | `store.rs` | SQLite layer. Raw CRUD. Schema init. Embedding blob serialization. |
| `MemoryRecord` | `memory.rs` | A stored memory with id, kind, strength, embedding, timestamps, access_count. |
| `MemoryKind` | `memory.rs` | Enum: `Fact(Fact)` or `Episode(Episode)`. |
| `Fact` | `memory.rs` | Subject-relation-object triple. |
| `Episode` | `memory.rs` | Free-text event description. |
| `Association` | `memory.rs` | Named link between two entities (bidirectional). |
| `ExportData` | `memory.rs` | Full database dump: `Vec<MemoryRecord>` + `Vec<Association>`. |
| `RecallResult` | `recall.rs` | A recalled memory with its relevance score. |
| `DecayResult` | `decay.rs` | Stats from a decay pass: `decayed` + `deleted` counts. |
| `Embedder` | `embed.rs` | Trait for embedding generation. `FastEmbedder` is the default impl. |

## How Search Works

Hybrid BM25 + vector search, fused via Reciprocal Rank Fusion (RRF):

1. **BM25**: All memories searched with `bm25` crate. `b=0.5` for short docs. Proper IDF weighting.
2. **Vector**: Query embedded, cosine similarity computed against all embeddings. Threshold: `> 0.3`.
3. **RRF fusion**: Rankings combined with `k=60`. Items appearing in both lists score highest.
4. **Weighting**: Fused scores multiplied by `strength * recency_factor`.
5. **Relations**: Associations searched separately via BM25, scored at 0.8x discount.
6. **Reinforcement**: Recalled memories are "touched" (strength += 0.2, access_count++).

## Memory Lifecycle

1. **Store**: `remember` or `remember-episode` → embedding generated → strength = 1.0
2. **Recall**: Semantic search finds it → touch (reinforce strength, bump access count)
3. **Decay**: `decay` pass → strength *= 0.5^(hours_since_access / 24)
4. **Death**: Strength falls below 0.01 → memory deleted during decay

## SQLite Schema

```sql
memories (
  id, kind ['fact'|'episode'], subject, relation, object, episode_text,
  strength REAL DEFAULT 1.0, embedding BLOB,
  created_at TEXT, last_accessed_at TEXT, access_count INTEGER DEFAULT 0
)
-- Indexes on: subject, kind

associations (
  id, entity_a, relation, entity_b, created_at,
  UNIQUE(entity_a, relation, entity_b)
)
-- Indexes on: entity_a, entity_b
```

Embeddings stored as little-endian f32 blobs. Timestamps as RFC 3339 strings.

## CLI Commands

All commands support `--json` and `--quiet`. Database: `--db <path>` (default `~/.conch/default.db`).

```
conch remember "<subject>" "<relation>" "<object>"    # store a fact
conch remember-episode "<text>"                       # store an episode
conch recall "<query>" [--limit N]                    # hybrid semantic search
conch relate "<entity_a>" "<relation>" "<entity_b>"   # create association
conch forget --subject "<subject>"                    # delete by subject
conch forget --older-than <duration>                  # delete by age (s/m/h/d/w)
conch decay                                           # run temporal decay pass
conch stats                                           # database statistics
conch embed                                           # batch-generate missing embeddings
conch export                                          # dump all data as JSON to stdout
conch import                                          # read JSON from stdin into db
```

## MCP Server

The `conch-mcp` crate exposes conch operations as MCP tools via `rmcp`. Runs on stdio transport.

**Tools**: `remember_fact`, `remember_episode`, `recall`, `relate`, `forget`, `decay`, `stats`

**Key types**: `ConchServer` (wraps `Arc<Mutex<ConchDB>>`), parameter structs with `schemars::JsonSchema` for schema generation.

**Config**: Database path via `CONCH_DB` env var (defaults to `~/.conch/default.db`).

## Dependencies

**conch-core**:
- `rusqlite` (bundled SQLite)
- `fastembed` (local embeddings, AllMiniLML6V2, 384-dim)
- `bm25` (BM25 search scoring)
- `serde` + `serde_json` (serialization)
- `chrono` (timestamps)
- `thiserror` (error types)

**conch-cli**: `clap` (derive feature)

**conch-mcp**: `rmcp` (server + transport-io), `tokio`, `schemars`

## Build & Test

```bash
cargo build
cargo test
cargo install --path crates/conch-cli
```

## Adding New Features

- **New memory operations**: Add method to `MemoryStore` (store.rs), wrap in `ConchDB` (lib.rs), add CLI subcommand (main.rs)
- **New search strategies**: Modify `recall.rs`. The RRF pattern makes it easy to add new signal sources.
- **New embedding backends**: Implement the `Embedder` trait (embed.rs)
- **Tests**: Each module has `#[cfg(test)] mod tests`. Use `MemoryStore::open_in_memory()` for test databases. Mock embedders with a simple struct implementing `Embedder`.
