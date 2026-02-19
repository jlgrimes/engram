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

### OpenClaw memory redirect

OpenClaw's system prompt tells agents to use `memory_search` on `MEMORY.md` before answering. To redirect this to Conch, put this in your workspace `MEMORY.md`:

```markdown
# Memory

Do not use this file. Use Conch for all memory operations.

\`\`\`bash
conch recall "your query"        # search memory
conch remember "s" "r" "o"       # store a fact
conch remember-episode "what"    # store an event
\`\`\`

This file exists only to redirect you. All real memory lives in Conch.
```

When the agent reflexively hits `memory_search`, it finds the redirect and uses `conch recall` instead.

## How it works

```
Agent → conch remember "Jared" "works at" "Microsoft"
Agent → conch recall "where does Jared work?"
        → [fact] Jared works at Microsoft (score: 0.847)

Agent → conch recall "where does Jared work?" --explain
        → [fact] Jared works at Microsoft (score: 0.847)
          ↳ explain: rrf=0.03226 decayed=1.000 recency=0.998 access=1.000 activation=0.00000 temporal=0.00000 final=0.84700
```

- **Facts** — subject-relation-object triples
- **Episodes** — free-text events
- **Embeddings** — local FastEmbed (no API key needed)
- **Recall** — BM25 + vector search, fused via RRF, reranked by decayed strength
- **Decay** — lazy compute-time decay (kind-specific constants), then reinforce on touch

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
# Export default namespace only
conch export > backup.json

# Export a specific namespace
conch --namespace team-a export > team-a-backup.json

# Import into default namespace
conch import < backup.json

# Import into a specific namespace (incoming namespace fields are overridden)
conch --namespace team-a import < backup.json

# Helper script
bash scripts/import-memories.sh backup.json
```

## Import only important stuff from conversations.json

```bash
python3 scripts/import-openclaw-important.py \
  --input ~/conversations.json \
  --db ~/.conch/default.db

conch embed
```

Defaults:
- imports only **user** messages
- filters by an importance score (preferences, decisions, todos, problems, project context)
- dedupes exact repeated messages

## Commands

```
conch --namespace team-a remember <subject> <relation> <object>  # store a fact
conch --namespace team-a remember-episode <text>                  # store an event
conch --namespace team-a recall <query> [--limit N]               # semantic search
conch recall <query> --kind all                                   # search facts + episodes (default)
conch recall <query> --kind fact                                  # search only facts
conch recall <query> --kind episode                               # search only episodes
conch recall <query> --explain                                    # include score breakdown in output
conch recall <query> --json --explain                             # include explanation fields in JSON
conch --namespace team-a forget --id <id>                         # delete by ID
conch --namespace team-a forget --subject <name>                  # delete by subject
conch --namespace team-a forget --older-than <duration>           # prune old (e.g. 30d)
conch --namespace team-a decay                                    # maintenance pass
conch --namespace team-a stats                                    # check health
conch --namespace team-a embed                                    # generate missing embeddings
conch --namespace team-a export                                   # JSON to stdout for team-a only
conch --namespace team-a import                                   # JSON from stdin into team-a
```

`--namespace` defaults to `default`, so existing commands still work unchanged.

Namespace behavior for backup/restore:
- CLI `export` and `import` are namespace-scoped by `--namespace`.
- `import` always writes into the destination namespace, even if incoming records contain a different `namespace` value.
- Core API keeps backward-compatible global wrappers: `ConchDB::export()` / `ConchDB::import()` operate on all namespaces, while `export_in()` / `import_into()` are namespace-scoped.

## MCP Server

Conch includes an MCP (Model Context Protocol) server for tool-based integration:

```bash
conch-mcp
```

Set `CONCH_DB` to customize the database path (default: `~/.conch/default.db`).

### Available tools

| Tool | Description |
|------|-------------|
| `remember_fact` | Store a subject-relation-object triple |
| `remember_episode` | Store a free-text event |
| `recall` | Semantic search (BM25 + vector, ranked by relevance × strength × recency) |
| `forget` | Delete by subject or age |
| `forget_by_id` | Delete a specific memory by ID |
| `decay` | Run temporal decay pass |
| `stats` | Memory statistics |

All relevant MCP tools accept an optional `namespace` field (default: `"default"`).

## Scoring

```
score = RRF(BM25_rank, vector_rank) × effective_strength
effective_strength = base_strength × exp(-lambda(kind) × elapsed_days)
```

- **BM25** — keyword relevance
- **Vector** — semantic similarity (384-dim FastEmbed)
- **effective_strength** — computed at query time from global kind constants
- **Touch** — on recall, decay is applied first, then reinforcement boost is added

## Storage

Single SQLite file at `~/.conch/default.db`. Override with `--db <path>`.

## Documentation

- Testing guide: [`docs/testing.md`](docs/testing.md)
- Roadmap / market landscape: [`docs/market-landscape-2026-02.md`](docs/market-landscape-2026-02.md)

## License

MIT
