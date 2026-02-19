# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-02-19

### Added

- **Deduplication** — cosine similarity threshold (0.95) prevents duplicate memories. Existing memories are reinforced instead of duplicated. New methods: `remember_fact_dedup()`, `remember_episode_dedup()`.
- **Upsert** — facts with the same subject+relation are updated in place via `upsert_fact()`.
- **Tags** — categorize memories with comma-separated tags. Filter recall results by tag with `--tag`.
- **Source tracking** — track memory origin with `--source`, `--session-id`, and `--channel` metadata fields.
- **Brain-inspired scoring** in recall:
  - Recency boost (7-day half-life, 0.3 floor)
  - Access weighting (log-normalized frequency, range 1.0-2.0)
  - Spreading activation (1-hop graph traversal through shared subjects/objects)
  - Temporal co-occurrence (context reinstatement for memories created within 30 minutes)
- **MCP server** (`conch-mcp`) — Model Context Protocol server exposing conch operations as LLM tools via rmcp.
- Comprehensive test suite (50 tests across all modules).
- crates.io publish metadata for all crates.

### Changed

- Upgraded `fastembed` from v4 to v5.
- Recall scoring now incorporates all brain-inspired layers by default.

### Fixed

- Strength clamping ensures values stay within [0.0, 1.0] range after reinforcement.

## [0.1.0] - 2025-01-01

### Added

- Initial release.
- Fact and episode memory types.
- Hybrid BM25 + vector search with Reciprocal Rank Fusion.
- Temporal decay engine (half-life 24h).
- Local embeddings via FastEmbed (AllMiniLM-L6-V2, 384-dim).
- SQLite storage with single-file database.
- CLI with remember, recall, forget, decay, stats, embed, export, import commands.
