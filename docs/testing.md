# Testing

Conch has two primary layers of test coverage:

- **Core/unit tests (`conch-core`)**
  - Storage and migration behavior (including legacy DB schema upgrade paths)
  - Namespace isolation and import/export behavior
  - Recall ranking and decay behavior with deterministic mock embeddings
- **CLI-level command handling tests (`conch`)**
  - High-level command behavior around namespace-aware `remember`/`recall --kind`
  - Namespace-scoped export/import flows via command-level helpers

## Run all tests

```bash
cargo test --workspace
```

## Run specific suites

```bash
# Core/unit tests
cargo test -p conch-core

# CLI command-level tests
cargo test -p conch

# MCP server tests
cargo test -p conch-mcp
```

### MCP concurrency regression coverage

The MCP crate includes a regression test that exercises concurrent read/write tool calls to prevent lock/runtime regressions in the async server:

- `concurrent_mixed_remember_and_recall_regression`

Run just MCP tests with:

```bash
cargo test -p conch-mcp
```

For a lightweight latency trend check of concurrent MCP behavior, run:

```bash
scripts/benchmark-mcp-concurrency.sh
# optional: RUNS=10 scripts/benchmark-mcp-concurrency.sh
```

## Lint/format checks used in CI

```bash
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
```
