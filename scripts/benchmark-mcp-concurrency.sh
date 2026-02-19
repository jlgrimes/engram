#!/usr/bin/env bash
set -euo pipefail

# Lightweight MCP concurrency benchmark fixture.
# Runs the MCP concurrency regression test repeatedly and prints timing.

RUNS="${RUNS:-5}"
TEST_NAME="concurrent_mixed_remember_and_recall_regression"

if command -v hyperfine >/dev/null 2>&1; then
  echo "Using hyperfine (${RUNS} runs)"
  hyperfine --warmup 1 --runs "$RUNS" \
    "cargo test -p conch-mcp ${TEST_NAME} -- --nocapture"
else
  echo "hyperfine not found; using built-in timing loop (${RUNS} runs)"
  for i in $(seq 1 "$RUNS"); do
    echo "Run $i/$RUNS"
    /usr/bin/time -f 'elapsed=%E user=%U sys=%S' \
      cargo test -p conch-mcp "${TEST_NAME}" -- --nocapture >/dev/null
  done
fi
