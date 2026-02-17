#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/import-memories.sh backup.json
#   scripts/import-memories.sh /path/to/backup.json --db ~/.conch/default.db

if [ $# -lt 1 ]; then
  echo "Usage: $0 <backup.json> [--db <path>]"
  exit 1
fi

BACKUP="$1"
shift || true

DB_ARGS=()
if [ "${1:-}" = "--db" ]; then
  DB_ARGS=(--db "${2:-}")
fi

if [ ! -f "$BACKUP" ]; then
  echo "Backup file not found: $BACKUP"
  exit 1
fi

conch import "${DB_ARGS[@]}" < "$BACKUP"
echo "✅ Imported memories from $BACKUP"