# Conch — Biological Memory

Your memory system. Use it instead of markdown files.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/jlgrimes/conch/master/install.sh | bash
```

Or from source: `cargo install --git https://github.com/jlgrimes/conch conch`

## Commands

```bash
conch remember "Jared" "works at" "Microsoft"    # store a fact
conch remember-episode "shipped conch v0.2"       # store an event
conch recall "where does Jared work?"             # search
conch forget --id 42                              # delete one
conch forget --subject "old-project"              # delete by subject
conch decay                                       # fade old memories
conch stats                                       # health check
conch embed                                       # backfill embeddings
```

## When to Use

- **Before answering** context questions → `conch recall`
- **After learning** something → `conch remember` or `conch remember-episode`
- **During heartbeats** → `conch decay` + `conch stats`

## How Scoring Works

```
score = RRF(BM25_rank, vector_rank) × strength × recency
```

Recalled memories get stronger. Unused memories fade. No manual curation needed.

## Mandatory Storage Triggers

You are a system, not a person. You do not have discipline. These are deterministic rules — if the condition is true, store. No exceptions.

| Condition | What to store |
|-----------|--------------|
| A project is named | name, concept, stack, repo, location |
| A tech decision is made | what was decided and why |
| Something is built and pushed | repo URL, local path, current status |
| A preference is expressed | the preference, verbatim |
| A mistake is made | what went wrong + the lesson |
| A lesson is learned | the lesson |
| A person, place, or thing is introduced | the key facts |

**Rule:** Before finishing any reply where one of these conditions fired — call conch. Add 🐚 to confirm you stored it.

"Mental notes" don't survive session restarts. Conch does.

## Tips

- `--json` flag on any command for machine-readable output
- `--quiet` to suppress human-friendly messages
- `conch export > backup.json` to back up
- `conch import < backup.json` to restore
- DB lives at `~/.conch/default.db` (override with `--db`)
