# Task Board

✅ **task-1** - DEDUP ON INSERT: Before storing a new memory (fact or episode), compute its embedding and check cosine similarity against existing memories. If similarity > 0.95, skip the insert (or merge by reinforcing the existing memory's strength). Add this check in conch-core/src/store.rs or lib.rs. The CLI commands (remember, remember-episode) should report when a duplicate was detected.

✅ **task-2** - UPSERT FACTS: When storing a fact with the same subject+relation as an existing fact, update the object instead of creating a duplicate. Add a store method like upsert_fact() that checks for existing subject+relation match first. Update the CLI to use upsert by default.

✅ **task-3** - TAGS/CATEGORIES: Add an optional tags field to memories. Schema change: add a 'tags' TEXT column (comma-separated or JSON array). Update MemoryRecord, store methods, CLI commands (--tags flag), and recall (--tag filter). Suggested categories: preference, decision, person, project, emotional, technical, meta.

✅ **task-4** - SOURCE TRACKING: Add source/context fields to memories: source (string, e.g. 'discord', 'cli', 'cron'), session_id (optional string), channel (optional string). Schema migration in init_schema(). Update all insert methods, MemoryRecord struct, CLI (--source flag, auto-detect 'cli'), export/import.
