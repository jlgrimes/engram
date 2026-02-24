# 🐚 Conch

**Biological memory for AI agents.** Semantic search + decay, no API keys needed.

[![Crates.io](https://img.shields.io/crates/v/conch.svg)](https://crates.io/crates/conch)
[![docs.rs](https://docs.rs/conch-core/badge.svg)](https://docs.rs/conch-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/jlgrimes/conch/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgrimes/conch/actions)

---

## The Problem

Most AI agents use a flat `memory.md` file. It doesn't scale:

- **Loads the whole file into context** — bloats every prompt as memory grows
- **No semantic recall** — `grep` finds keywords, not meaning
- **No decay** — stale facts from months ago are weighted equally to today's
- **No deduplication** — the same thing gets stored 10 times in slightly different words

You end up with an ever-growing, expensive-to-query, unreliable mess.

## Why Conch

Conch replaces the flat file with a **biologically-inspired memory engine**:

- **Recall by meaning** — hybrid BM25 + vector search finds semantically relevant memories, not just keyword matches
- **Decay over time** — old memories fade unless reinforced; frequently-accessed ones survive longer
- **Deduplicate on write** — cosine similarity (0.95) detects near-duplicates and reinforces instead of cloning
- **No infrastructure** — SQLite file, local embeddings (FastEmbed, no API key), zero config
- **Scales silently** — 10,000 memories in your DB, 5 returned in context. Prompt stays small.

```
memory.md after 6 months: 4,000 lines, loaded every prompt
Conch after 6 months: 10,000 memories, 5 relevant ones returned per recall
```

## Install

```bash
cargo install conch
```

**No Cargo?** See the [Installation Guide](docs/install.md) for prebuilt binaries and build-from-source instructions.

## Quick Start

```bash
# Store a fact
conch remember "Jared" "works at" "Microsoft"

# Store an episode
conch remember-episode "Deployed v2.0 to production"

# Store an action (executed operation)
conch remember-action "Edited Caddyfile and restarted caddy"

# Store an intent (future plan)
conch remember-intent "Plan to rotate API keys this Friday"

# Recall by meaning (not keyword)
conch recall "where does Jared work?"
# → [fact] Jared works at Microsoft (score: 0.847)

# Run decay maintenance
conch decay

# Database health
conch stats
```

## How It Works

```
Store → Embed → Search → Decay → Reinforce
```

1. **Store** — facts (subject-relation-object) or episodes (free text). Embedding generated locally via FastEmbed.
2. **Search** — hybrid BM25 + vector recall, fused via Reciprocal Rank Fusion (RRF), weighted by decayed strength.
3. **Decay** — strength diminishes over time. Facts decay slowly (λ=0.02/day), episodes faster (λ=0.06/day), actions/intents fastest (λ=0.09/day).
4. **Reinforce** — recalled memories get a boost. Frequently accessed ones survive longer.
5. **Death** — memories below strength 0.01 are pruned during decay passes.

### Scoring

```
score = RRF(BM25_rank, vector_rank) × recency_boost × access_weight × effective_strength
```

- **Recency boost** — 7-day half-life, floor of 0.3
- **Access weighting** — log-normalized frequency boost (1.0–2.0×)
- **Spreading activation** — 1-hop graph traversal through shared subjects/objects
- **Temporal co-occurrence** — memories created in the same session get context boosts

## Features

- **Hybrid search** — BM25 + vector semantic search via Reciprocal Rank Fusion
- **Biological decay** — configurable half-life curves per memory type
- **Deduplication** — cosine similarity threshold prevents duplicates; reinforces instead
- **Graph traversal** — spreading activation through shared subjects/objects
- **Tags & source tracking** — tag memories, track origin via source/session/channel
- **MCP support** — Model Context Protocol server for direct LLM tool integration
- **Local embeddings** — FastEmbed (AllMiniLM-L6-V2, 384-dim). No API keys, no network calls
- **Single-file SQLite** — zero infrastructure. One portable DB file

## Comparison

| Feature | Conch | Mem0 | Zep | Raw Vector DB |
|---------|-------|------|-----|---------------|
| Biological decay | ✅ | ❌ | ❌ | ❌ |
| Deduplication | Cosine 0.95 | Basic | Basic | Manual |
| Graph traversal | Spreading activation | ❌ | Graph edges | ❌ |
| Local embeddings | FastEmbed (no API) | API required | API required | Varies |
| Infrastructure | SQLite (zero-config) | Cloud/Redis | Postgres | Server required |
| MCP support | Built-in | ❌ | ❌ | ❌ |

## Commands

```
conch remember <subject> <relation> <object>   # store a fact
conch remember-episode <text>                   # store an event
conch remember-action <text>                    # store an executed action
conch remember-intent <text>                    # store a future intent/plan
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
conch remember "API" "uses" "REST" --tags "architecture,backend"
conch remember-episode "Fixed auth bug" --source "slack" --session-id "abc123"
conch recall "architecture decisions" --tag "architecture"
```

## Architecture

```
conch-core     Library crate. All logic: storage, search, decay, embeddings.
conch          CLI binary. Clap-based interface to conch-core.
conch-mcp      MCP server. Exposes conch operations as LLM tools via rmcp.
```

### Use as a Library

```rust
use conch_core::ConchDB;

let db = ConchDB::open("my_agent.db")?;
db.remember_fact("Jared", "works at", "Microsoft")?;
db.remember_episode("Deployed v2.0 to production")?;
let results = db.recall("where does Jared work?", 5)?;
let stats = db.decay()?;
```

### MCP Server

```json
{
  "mcpServers": {
    "conch": {
      "command": "conch-mcp",
      "env": { "CONCH_DB": "~/.conch/default.db" }
    }
  }
}
```

**MCP tools**: `remember_fact`, `remember_episode`, `remember_action`, `remember_intent`, `recall`, `forget`, `decay`, `stats`

## OpenClaw Integration (One-Click)

If setup is not one-click, it will fail in practice. Use this:

```bash
curl -fsSL https://raw.githubusercontent.com/jlgrimes/conch/master/scripts/openclaw-one-click.sh | bash
```

What this script does automatically:

1. Installs `conch` if missing
2. Configures `~/.openclaw/workspace/MEMORY.md` redirect to Conch
3. Adds mandatory Conch storage triggers to `AGENTS.md` (idempotent)
4. Fixes OpenClaw gateway PATH so `conch` is discoverable from cron/runtime
5. Restarts gateway service (if present) and runs remember/recall smoke test

### Result you should expect

- Agent memory is redirected to Conch
- Runtime can invoke Conch without ENOENT/PATH issues
- Session continuity writes are deterministic via trigger rules
- Smoke test validates write + recall immediately

### If you need manual setup (fallback)

Tell your OpenClaw agent:
> Read https://raw.githubusercontent.com/jlgrimes/conch/master/skill/SKILL.md and install conch.

Then manually apply the same pieces (MEMORY redirect + AGENTS triggers + gateway PATH + smoke test).

## Import / Export

```bash
conch export > backup.json
conch import < backup.json
```

## Storage

Single SQLite file at `~/.conch/default.db`. Embeddings stored as little-endian f32 blobs. Timestamps as RFC 3339. Override with `--db <path>` or `CONCH_DB` env var.

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

## Web Apps

- Internal dashboard: `dashboard/` (internal tooling)
- Customer-facing app: `customer-app/` (external site for `app.conch.so`)
- Deployment and DNS guide: `docs/customer-app-deploy.md`
