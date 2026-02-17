# Conch — Biological Memory Skill

Conch is your primary memory system. Use it instead of MEMORY.md or memory/*.md files.

**Binary:** `~/.cargo/bin/conch`
**Repo:** `~/projects/conch`

## When to Use

- **Before answering** anything about prior work, decisions, preferences, people, dates, or context → `conch recall`
- **After learning** something new or important → `conch remember` or `conch remember-episode`
- **Periodically** (during heartbeats) → `conch decay` to let old memories fade naturally

## Commands

### Recall (search memories)
```bash
conch recall "query here"
```
Returns ranked memories with salience and relevance scores. Always do this before answering memory-dependent questions.

### Remember a fact (structured: subject-relation-object)
```bash
conch remember "subject" "relation" "object"
```
Examples:
```bash
conch remember "Jared" "works at" "Microsoft"
conch remember "Tortellini" "is a" "dog"
conch remember "grotto" "uses" "lobster mascot"
```

### Remember an episode (free-text event)
```bash
conch remember-episode "Built grotto daemon with 3 agents — they invented emergent file locking"
```
Use for events, decisions, observations, lessons learned.

### Relate (link entities)
```bash
conch relate "Jared" "owner of" "Tortellini"
```

### Forget
```bash
conch forget <memory-id>
```

### Decay (run temporal fade)
```bash
conch decay
```
Memories weaken over time (24h half-life). Frequently recalled memories stay strong. Run during heartbeats.

### Stats
```bash
conch stats
```

## Memory Strategy

### What to remember as FACTS:
- People, relationships, preferences
- Project details, tech stacks, decisions
- Locations, schedules, recurring info

### What to remember as EPISODES:
- Conversations, events, milestones
- Lessons learned, mistakes, insights
- Decisions and their reasoning

### Salience
Memories have salience (0-1). Recalling a memory strengthens it. Time weakens it. This is automatic — just use recall naturally and the system self-organizes.

## Important
- Conch is the **single source of truth** for memory
- Do NOT use MEMORY.md or memory/*.md for new memories
- Always `conch recall` before answering memory questions
- Always `conch remember` or `conch remember-episode` when learning something worth keeping
