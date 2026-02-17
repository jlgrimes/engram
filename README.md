# Conch

**Biological memory for AI agents. Memories that strengthen, fade, and connect.**

Give your [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or [OpenClaw](https://github.com/openclaw/openclaw) agent a memory system inspired by how biological memory actually works: memories strengthen with use, decay with time, and form associative links.

## Install

Tell your OpenClaw agent:

> Read https://raw.githubusercontent.com/jlgrimes/conch/master/skill/SKILL.md and install conch as a skill.

Or manually:

```bash
git clone https://github.com/jlgrimes/conch.git /tmp/conch
cd /tmp/conch && cargo install --path crates/conch-cli
rm -rf /tmp/conch

# Install the OpenClaw skill
cp -r skill/ ~/.openclaw/workspace/skills/conch/
```

## How it works

```
┌──────────────────────────────────────────────────┐
│                     Agent                        │
│  "remember Jared works at Microsoft"             │
│  "what do I know about Jared?"                   │
└──────────┬───────────────────────────┬───────────┘
           │ CLI                       │ MCP
    ┌──────▼──────┐            ┌───────▼──────┐
    │  conch-cli  │            │  conch-mcp   │
    └──────┬──────┘            └───────┬──────┘
           └───────────┬───────────────┘
                ┌──────▼──────┐
                │  conch-core │
                │  ┌────────┐ │
                │  │ SQLite │ │  ← memories + associations
                │  └────────┘ │
                │  ┌────────┐ │
                │  │fastembed│ │  ← local embeddings (no API key)
                │  └────────┘ │
                └─────────────┘
```

### Memory types

- **Facts** — Subject-relation-object triples: `"Jared" "works at" "Microsoft"`
- **Episodes** — Free-text events: `"Migrated from nginx to Caddy"`

### What makes it biological

- **Strength decay** — Unused memories fade (half-life: 24h). Recalled memories get reinforced.
- **Semantic search** — Hybrid BM25 + vector search with Reciprocal Rank Fusion. Finds memories by meaning, not just keywords.
- **Associative links** — Named relationships between entities, like a knowledge graph.
- **Natural forgetting** — Memories below 0.01 strength are automatically deleted during decay.

## Usage

```bash
# Remember facts
conch remember "Jared" "works at" "Microsoft"
conch remember "Tortellini" "is a" "dog"

# Remember events
conch remember-episode "Migrated from nginx to Caddy for reverse proxy"

# Recall by meaning
conch recall "what does Jared do for work" --json

# Create associations
conch relate "Jared" "owns" "Tortellini"

# Maintenance
conch decay              # run temporal decay pass
conch embed              # generate missing embeddings
conch stats              # database statistics

# Data management
conch export > backup.json          # export all data
conch import < backup.json          # import from backup

# Forget
conch forget --subject "Jared"      # delete by subject
conch forget --older-than 90d       # delete old memories
```

All commands support `--json` for structured output and `--quiet` for minimal output. Database defaults to `~/.conch/default.db`, override with `--db <path>`.

## MCP Server

Conch includes an MCP (Model Context Protocol) server for direct LLM tool integration:

```bash
cargo install --path crates/conch-mcp
```

Exposes `remember_fact`, `remember_episode`, `recall`, `relate`, `forget`, `decay`, and `stats` as MCP tools.

## Requirements

- Rust toolchain
- No API keys needed (uses local embeddings via fastembed)

## License

MIT
