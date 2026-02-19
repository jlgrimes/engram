# 🐚 Conch

**Biological memory for AI agents.** Security-aware, production-reliable.

Memories strengthen with use, fade with time — just like the real thing.

[![Crates.io](https://img.shields.io/crates/v/conch-core.svg)](https://crates.io/crates/conch-core)
[![docs.rs](https://docs.rs/conch-core/badge.svg)](https://docs.rs/conch-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/jlgrimes/conch/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgrimes/conch/actions)

---

## Features

- **Biological decay** — memories weaken over time with configurable half-life curves. Facts persist longer than episodes, just like human memory.
- **Hybrid search** — BM25 keyword + vector semantic search, fused via Reciprocal Rank Fusion (RRF).
- **Deduplication** — cosine similarity threshold (0.95) prevents duplicate memories. Existing memories are reinforced instead.
- **Upsert** — facts with the same subject+relation are updated in place, not duplicated.
- **Graph traversal** — spreading activation boosts related memories through shared subjects/objects.
- **Temporal co-occurrence** — memories created in the same session context get recall boosts.
- **Tags & source tracking** — categorize memories with tags, track origin via source/session/channel metadata.
- **MCP support** — Model Context Protocol server for direct LLM tool integration.
- **Local embeddings** — FastEmbed (AllMiniLM-L6-V2, 384-dim). No API keys, no network calls.
- **Single-file SQLite** — zero infrastructure. One portable database file.

## Quick Start

```bash
# Install from crates.io
cargo install conch

# Store a fact
conch remember "Jared" "works at" "Microsoft"

# Store an episode
conch remember-episode "Deployed v2.0 to production"

# Recall
conch recall "where does Jared work?"
# → [fact] Jared works at Microsoft (score: 0.847)

# Check database health
conch stats
```

## How It Works

```
Store → Embed → Search → Decay → Reinforce
```

1. **Store** — `remember` creates a fact (subject-relation-object triple) or episode (free text). Embedding generated locally via FastEmbed.
2. **Search** — `recall` runs hybrid BM25 + vector search. Results fused via RRF, weighted by decayed strength.
3. **Decay** — strength diminishes over time. Facts decay slowly (λ=0.02/day), episodes decay faster (λ=0.06/day).
4. **Reinforce** — recalled memories are "touched": decay applied first, then reinforcement boost added. Frequently accessed memories survive longer.
5. **Death** — memories below strength 0.01 are pruned during decay passes.

### Scoring

```
score = RRF(BM25_rank, vector_rank) × recency_boost × access_weight × effective_strength
```

Brain-inspired scoring layers:
- **Recency boost** — 7-day half-life, floor of 0.3
- **Access weighting** — log-normalized frequency boost (range 1.0–2.0)
- **Spreading activation** — 1-hop graph traversal through shared subjects/objects
- **Temporal co-occurrence** — context reinstatement for memories created within 30 minutes of each other

## Comparison

| Feature | Conch | Mem0 | Zep | Raw Vector DB |
|---------|-------|------|-----|---------------|
| Biological decay | Yes | No | No | No |
| Deduplication | Cosine 0.95 | Basic | Basic | Manual |
| Graph traversal | Spreading activation | No | Graph edges | No |
| Local embeddings | FastEmbed (no API) | API required | API required | Varies |
| Infrastructure | SQLite (zero-config) | Cloud/Redis | Postgres | Server required |
| MCP support | Built-in | No | No | No |
| Source tracking | source/session/channel | No | Session | No |
| Tags | Yes | Metadata | Metadata | Varies |

## Commands

```
conch remember <subject> <relation> <object>   # store a fact
conch remember-episode <text>                   # store an event
conch recall <query> [--limit N] [--tag T]     # semantic search
conch forget --id <id>                          # delete by ID
conch forget --subject <name>                   # delete by subject
conch forget --older-than <duration>            # prune old (e.g. 30d)
conch decay                                     # run decay maintenance pass
conch stats                                     # database health
conch embed                                     # generate missing embeddings
conch export                                    # JSON dump to stdout
conch import                                    # JSON load from stdin
```

All commands support `--json` and `--quiet`. Database path: `--db <path>` (default `~/.conch/default.db`).

### Tags & Source Tracking

```bash
# Store with tags
conch remember "API" "uses" "REST" --tags "architecture,backend"

# Store with source tracking
conch remember-episode "Fixed auth bug" --source "slack" --session-id "abc123"

# Filter recall by tag
conch recall "architecture decisions" --tag "architecture"
```

## Architecture

```
conch-core     Library crate. All logic: storage, search, decay, embeddings.
conch          CLI binary. Clap-based interface to conch-core.
conch-mcp      MCP server. Exposes conch operations as LLM tools via rmcp.
```

### conch-core (library)

Use directly in Rust projects:

```rust
use conch_core::ConchDB;

let db = ConchDB::open("my_agent.db")?;

// Store
db.remember_fact("Jared", "works at", "Microsoft")?;
db.remember_episode("Deployed v2.0 to production")?;

// Store with dedup (reinforces if duplicate found)
db.remember_fact_dedup("Jared", "works at", "Microsoft")?;

// Recall
let results = db.recall("where does Jared work?", 5)?;
for r in &results {
    println!("{}: {:.3}", r.record.id, r.score);
}

// Decay pass
let stats = db.decay()?;
println!("Decayed: {}, Deleted: {}", stats.decayed, stats.deleted);
```

### conch-mcp (MCP server)

Add to your MCP client config:

```json
{
  "mcpServers": {
    "conch": {
      "command": "conch-mcp",
      "env": {
        "CONCH_DB": "~/.conch/default.db"
      }
    }
  }
}
```

**MCP tools**: `remember_fact`, `remember_episode`, `recall`, `forget`, `decay`, `stats`

## Import / Export

```bash
# Full backup
conch export > backup.json

# Restore
conch import < backup.json
```

## OpenClaw Integration

Tell your OpenClaw agent:
> Read https://raw.githubusercontent.com/jlgrimes/conch/master/skill/SKILL.md and install conch.

### Memory redirect trick

Put this in your workspace `MEMORY.md` to redirect OpenClaw's built-in memory search to Conch:

```markdown
# Memory

Do not use this file. Use Conch for all memory operations.

conch recall "your query"        # search memory
conch remember "s" "r" "o"       # store a fact
conch remember-episode "what"    # store an event

This file exists only to redirect you. All real memory lives in Conch.
```

## Storage

Single SQLite file at `~/.conch/default.db`. Embeddings stored as little-endian f32 blobs. Timestamps as RFC 3339. Override path with `--db <path>` or `CONCH_DB` env var.

## Build & Test

```bash
cargo build
cargo test
cargo install --path crates/conch-cli
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Run `cargo test` and ensure all tests pass
4. Submit a pull request

## License

MIT — see [LICENSE](LICENSE).
