# 🐚 Conch

Biological memory for AI agents. Memories strengthen with use, fade with time.

## Install

**One-liner (downloads pre-built binary):**
```bash
curl -fsSL https://raw.githubusercontent.com/jlgrimes/conch/master/install.sh | bash
```

**From source:**
```bash
cargo install --git https://github.com/jlgrimes/conch conch
```

**For OpenClaw agents** — tell your agent:
> Read https://raw.githubusercontent.com/jlgrimes/conch/master/skill/SKILL.md and install conch.

## How it works

```
Agent → conch remember "Jared" "works at" "Microsoft"
Agent → conch recall "where does Jared work?"
        → [fact] Jared works at Microsoft (score: 0.847)
```

- **Facts** — subject-relation-object triples
- **Episodes** — free-text events
- **Embeddings** — local FastEmbed (no API key needed)
- **Recall** — BM25 + vector search, fused via RRF, weighted by strength × recency
- **Decay** — memories fade over time; recalled memories get stronger

## Quick start (new user)

```bash
# 1) Add your first memory
conch remember "Jared" "builds" "Gen"

# 2) Recall it
conch recall "what does Jared build"

# 3) Check DB health
conch stats
```

## Import / export memories

```bash
# Export
conch export > backup.json

# Import
conch import < backup.json

# Or use helper script
bash scripts/import-memories.sh backup.json
```

## Commands

```
conch remember <subject> <relation> <object>   # store a fact
conch remember-episode <text>                   # store an event
conch recall <query> [--limit N]               # semantic search
conch forget --id <id>                          # delete by ID
conch forget --subject <name>                   # delete by subject
conch forget --older-than <duration>            # prune old (e.g. 30d)
conch decay                                     # fade old memories
conch stats                                     # check health
conch embed                                     # generate missing embeddings
conch export                                    # JSON to stdout
conch import                                    # JSON from stdin
```

## Scoring

```
score = RRF(BM25_rank, vector_rank) × strength × recency
```

- **BM25** — keyword relevance
- **Vector** — semantic similarity (384-dim FastEmbed)
- **Strength** — reinforced on recall (+0.2, max 1.0), decays over time
- **Recency** — exponential decay based on last access time

## Storage

Single SQLite file at `~/.conch/default.db`. Override with `--db <path>`.

## License

MIT
